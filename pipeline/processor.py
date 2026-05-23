"""
process_split — shared iteration loop used by all three dataset modules.

Responsibilities:
  - Iterate every clip in a HuggingFace split
  - Run AudioQualityFilter on each clip
  - Checkpoint passing records every `batch_size` clips
  - Resume from the latest checkpoint when requested
  - Return the full list of passing records and per-split stats
"""

import logging
from pathlib import Path
from typing import Any

from .audio_filter import AudioQualityFilter
from .checkpoint import (
    flush_gpu_cache,
    latest_checkpoint_idx,
    load_prior_batches,
    save_batch,
)
from .clip_result import ClipResult
from .constants import FILTER_POLICY_VERSION, OUTPUT_DIR, OUTPUT_SAMPLE_RATE
from .stats import CleaningStats, RejectionLog

log = logging.getLogger(__name__)


def process_split(
    split_dataset,
    filt: AudioQualityFilter,
    *,
    audio_field: str,
    text_field: str,
    dataset_name: str,
    split_name: str,
    extra_fields: list[str],
    field_renames: dict[str, str] | None = None,
    batch_size: int = 500,
    resume: bool = True,
) -> tuple[list[dict], CleaningStats]:
    """
    Returns (passing_records, stats).

    extra_fields lists original-item keys to copy into each passing record.
    field_renames maps original key → destination key for conflict-free copying
    (used by WorldSpeech to avoid overwriting freshly computed metrics).
    """
    ckpt_name = f"{dataset_name}_{split_name}_{FILTER_POLICY_VERSION}"
    last_ckpt = latest_checkpoint_idx(ckpt_name) if resume else -1
    skip_until = (last_ckpt + 1) * batch_size if last_ckpt >= 0 else 0

    log.info(
        "Processing %s/%s  (%d clips, starting at idx %d)",
        dataset_name, split_name, len(split_dataset), skip_until,
    )

    stats = CleaningStats(f"{dataset_name}/{split_name}")
    reject_log = RejectionLog(
        OUTPUT_DIR / "logs" / f"rejected_{ckpt_name}.csv",
        append=resume and last_ckpt >= 0,
    )

    passing, prior_stats = load_prior_batches(ckpt_name, last_ckpt)
    stats.merge(prior_stats)

    batch_passing: list[dict] = []
    current_batch_idx = last_ckpt + 1
    batch_stats = CleaningStats(f"{ckpt_name}/batch_{current_batch_idx:06d}")

    for idx in range(skip_until, len(split_dataset)):
        item = split_dataset[idx]

        clip_id = str(item.get("path") or item.get("id") or idx)
        ground_truth = item.get(text_field, "")

        try:
            result = filt.process_clip(item[audio_field], ground_truth)
        except Exception as exc:
            log.warning("Clip %s crashed: %s", clip_id, exc)
            result = ClipResult(passed=False, reject_stage="crash", reject_reason=str(exc))

        stats.record(result)
        batch_stats.record(result)

        if not result.passed:
            reject_log.record(clip_id, result.reject_stage, result.reject_reason, ground_truth)
        else:
            batch_passing.append(_build_record(result, item, extra_fields, field_renames))

        if (idx + 1) % batch_size == 0 or idx == len(split_dataset) - 1:
            save_batch(ckpt_name, current_batch_idx, batch_passing, batch_stats)
            passing.extend(batch_passing)
            batch_passing = []
            current_batch_idx += 1
            batch_stats = CleaningStats(f"{ckpt_name}/batch_{current_batch_idx:06d}")
            log.info("  [%d/%d] checkpoint saved — passed so far: %d", idx + 1, len(split_dataset), stats.passed)
            flush_gpu_cache()

    reject_log.close()
    return passing, stats


def _build_record(
    result: ClipResult,
    item: dict,
    extra_fields: list[str],
    field_renames: dict[str, str] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "audio": {"array": result.audio_normalized, "sampling_rate": OUTPUT_SAMPLE_RATE},
        "snr_db":           float(result.snr_db),
        "mean_f0_hz":       float(result.mean_f0_hz),
        "pitch_confidence": float(result.pitch_confidence),
        "dnsmos_sig":       float(result.dnsmos_sig),
        "dnsmos_bak":       float(result.dnsmos_bak),
        "dnsmos_ovr":       float(result.dnsmos_ovr),
        "dnsmos_p808":      float(result.dnsmos_p808),
        "cer":              float(result.cer),
        "asr_transcript":   result.asr_transcript,
        "duration_s":       float(result.duration_s),
    }
    for field in extra_fields:
        dest = field_renames[field] if (field_renames and field in field_renames) else field
        record[dest] = item.get(field)
    return record
