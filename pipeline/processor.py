"""process_split — the shared iteration loop used by every dataset module.

Responsibilities:
  - Iterate every clip in a split
  - Run AudioQualityFilter on each
  - Write passing clips straight to disk via CorpusWriter
  - Log every rejection with its stage and reason
  - Resume by clip id, so a restart re-does no work

Two changes from the previous design.

**Passing clips are no longer accumulated in memory.** They were kept in a list
with their decoded 24 kHz float32 audio and then copied again by
`Dataset.from_list`; at Common Voice scale that is roughly 14 GB before
encoding, doubling during it. Only stats stay in memory now.

**Resume is keyed by clip id, not by batch index.** The old scheme skipped
`(last_checkpoint + 1) * batch_size` rows, which is only correct if the dataset
enumerates in exactly the same order every run, and it pickled float32 audio
into the checkpoint directory. The written manifest is now the checkpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .calibrate import Calibration
from .checkpoint import flush_gpu_cache
from .clip_result import ClipResult
from .constants import FILTER_POLICY_VERSION, OUTPUT_DIR, SAMPLE_RATE
from .corpus import CorpusWriter
from .recovery import split_at_silence
from .stats import CleaningStats, RejectionLog

# Only a type here; importing it at runtime would pull the whole model stack
# into any process that merely wants to read this module.
if TYPE_CHECKING:
    from .audio_filter import AudioQualityFilter

log = logging.getLogger(__name__)

_FLUSH_EVERY = 500


def process_split(
    split_dataset,
    filt: AudioQualityFilter,
    writer: CorpusWriter,
    *,
    audio_field: str,
    text_field: str,
    dataset_name: str,
    split_name: str,
    extra_fields: list[str],
    field_renames: dict[str, str] | None = None,
    resume: bool = True,
    limit: int | None = None,
    calibration: Calibration | None = None,
) -> CleaningStats:
    """Filter one split into `writer`. Returns the split's stats.

    `limit` caps how many clips are processed -- enough to read pass rates
    before committing to a full pass. `calibration`, when supplied, scores every
    gate instead of stopping at the first failure, which is far slower but is
    the only way to get comparable per-gate rejection rates.
    """
    run_name = f"{dataset_name}_{split_name}_{FILTER_POLICY_VERSION}"
    stats = (
        _load_stats(run_name, f"{dataset_name}/{split_name}")
        if resume
        else CleaningStats(f"{dataset_name}/{split_name}")
    )

    reject_log = RejectionLog(
        OUTPUT_DIR / "logs" / f"rejected_{run_name}.csv", append=resume
    )

    total = len(split_dataset)
    if limit is not None:
        total = min(total, limit)
        log.info("Limited to the first %d clips", total)
    log.info("Processing %s/%s (%d clips)", dataset_name, split_name, total)

    processed = 0
    for idx in range(total):
        item = split_dataset[idx]
        clip_id = _clip_id(item, dataset_name, split_name, idx)

        if clip_id in writer:
            continue

        ground_truth = item.get(text_field, "") or ""
        try:
            result = filt.process_clip(
                item[audio_field], ground_truth, measure_all=calibration is not None
            )
        except Exception as exc:
            log.warning("Clip %s crashed: %s", clip_id, exc)
            result = ClipResult(passed=False, reject_stage="crash", reject_reason=str(exc))

        # Only a too-long clip is a splitting candidate -- every other
        # rejection stage is final, and a segment must never reach here
        # itself (this branch only fires for the source clip, once).
        segments = None
        if (
            not result.passed
            and result.reject_stage == "duration"
            and result.reject_reason.startswith("too_long")
        ):
            segments = _split_clip(filt, item[audio_field], ground_truth)

        if segments is not None:
            # The source clip is counted as the rejection it already is; the
            # corpus gets only what the split produces, scored on its own
            # merits. Not counting both would hide how much of the corpus
            # started life as a rejected clip.
            stats.record(result)
            if calibration is not None:
                calibration.record(result)
            reject_log.record(clip_id, result.reject_stage, result.reject_reason, ground_truth)

            for i, (seg_audio, seg_sr, seg_text) in enumerate(segments):
                seg_id = f"{clip_id}_p{i}"
                seg_result = filt.process_clip(
                    {"array": seg_audio, "sampling_rate": seg_sr}, seg_text
                )
                seg_result.recovered_by = "split_at_silence"
                _finalize_clip(
                    seg_id, seg_text, seg_result, item, filt, stats, writer,
                    reject_log, extra_fields, field_renames, calibration,
                )
        else:
            _finalize_clip(
                clip_id, ground_truth, result, item, filt, stats, writer,
                reject_log, extra_fields, field_renames, calibration,
            )

        processed += 1
        if processed % _FLUSH_EVERY == 0:
            _save_stats(run_name, stats)
            log.info("  [%d/%d] passed so far: %d", idx + 1, total, stats.passed)
            flush_gpu_cache()

    _save_stats(run_name, stats)
    reject_log.close()
    return stats


def _clip_id(item: dict, dataset_name: str, split_name: str, idx: int) -> str:
    """Stable identity for resume, unique across datasets sharing a corpus."""
    raw = item.get("path") or item.get("id") or f"{split_name}_{idx}"
    return f"{dataset_name}_{Path(str(raw)).stem}"


def _split_clip(filt: AudioQualityFilter, audio_input, ground_truth: str):
    """Attempt the one legal repair for a clip rejected as too long.

    Needs the decoded audio a second time -- `process_clip` decoded and then
    discarded its own copy on the way to rejecting the clip, and nothing short
    of decoding again gets it back. Passing an empty `speech_spans` gives up
    the VAD as a corroborating signal for cut candidates; `split_at_silence`
    treats that signal as optional, not required, so it still refuses on its
    own terms rather than cutting somewhere unverified.
    """
    audio, err = filt._load_audio(audio_input)
    if audio is None:
        return None
    return split_at_silence(
        audio, SAMPLE_RATE, ground_truth, aligner=filt._aligner, speech_spans=[]
    )


def _finalize_clip(
    clip_id: str,
    ground_truth: str,
    result: ClipResult,
    item: dict,
    filt: AudioQualityFilter,
    stats: CleaningStats,
    writer: CorpusWriter,
    reject_log: RejectionLog,
    extra_fields: list[str],
    field_renames: dict[str, str] | None,
    calibration: Calibration | None,
) -> None:
    """Resolve one clip's text, record it, and write or log it.

    Shared by the ordinary path and by each segment a split produces, so a
    segment is gated, normalised and recorded exactly the way any other clip
    is -- splitting earns a clip nothing beyond a chance at the same gates.
    """
    # The normalised text is also what CER was scored against, so the
    # corpus and the score describe one string. The normaliser refuses
    # constructions it cannot expand without guessing (oron-tts
    # docs/normaliser-review.md); the CER and alignment gates reject those
    # already, so this is the belt to their braces. Resolved *before*
    # stats.record so a refusal is counted as the rejection it is rather
    # than as a pass -- and never published with unexpanded digits.
    text = ""
    if result.passed:
        try:
            text = filt.normalized_text(ground_truth)
        except Exception as exc:
            result = ClipResult(
                passed=False, reject_stage="normalize", reject_reason=str(exc)
            )

    stats.record(result)
    if calibration is not None:
        calibration.record(result)

    if result.passed:
        writer.add(
            clip_id,
            result.audio_normalized,
            text,
            _metadata(result, item, extra_fields, field_renames),
        )
    else:
        reject_log.record(clip_id, result.reject_stage, result.reject_reason, ground_truth)


def _metadata(
    result: ClipResult,
    item: dict,
    extra_fields: list[str],
    field_renames: dict[str, str] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "snr_db":           float(result.snr_db),
        "bandwidth_hz":     float(result.bandwidth_hz),
        "mean_f0_hz":       float(result.mean_f0_hz),
        "pitch_confidence": float(result.pitch_confidence),
        "dnsmos_sig":       float(result.dnsmos_sig),
        "dnsmos_bak":       float(result.dnsmos_bak),
        "dnsmos_ovr":       float(result.dnsmos_ovr),
        "dnsmos_p808":      float(result.dnsmos_p808),
        "align_score":      float(result.align_score),
        "cer":              float(result.cer),
        "len_ratio":        float(result.len_ratio),
        "asr_transcript":   result.asr_transcript,
        "duration_s":       float(result.duration_s),
        "recovered_by":     result.recovered_by,
    }
    for field in extra_fields:
        dest = field_renames[field] if (field_renames and field in field_renames) else field
        value = item.get(field)
        # Manifest rows are JSON, so anything exotic is stringified rather than
        # failing the write half way through a long run.
        record[dest] = (
            value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        )
    return record


def _stats_path(run_name: str) -> Path:
    return OUTPUT_DIR / "checkpoints" / f"{run_name}.json"


def _save_stats(run_name: str, stats: CleaningStats) -> None:
    path = _stats_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "total": stats.total,
        "passed": stats.passed,
        "stage_counts": stats.stage_counts,
        "total_duration_s": stats.total_duration_s,
        "sum_dnsmos_ovr": stats.sum_dnsmos_ovr,
        "sum_snr": stats.sum_snr,
        "sum_cer": stats.sum_cer,
    }), encoding="utf-8")


def _load_stats(run_name: str, display_name: str) -> CleaningStats:
    stats = CleaningStats(display_name)
    path = _stats_path(run_name)
    if not path.exists():
        return stats
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("Ignoring corrupt stats checkpoint %s", path)
        return stats
    stats.total = d.get("total", 0)
    stats.passed = d.get("passed", 0)
    stats.stage_counts = d.get("stage_counts", {})
    stats.total_duration_s = d.get("total_duration_s", 0.0)
    stats.sum_dnsmos_ovr = d.get("sum_dnsmos_ovr", 0.0)
    stats.sum_snr = d.get("sum_snr", 0.0)
    stats.sum_cer = d.get("sum_cer", 0.0)
    return stats
