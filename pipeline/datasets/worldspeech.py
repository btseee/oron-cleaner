"""WorldSpeech Mongolian — NON-COMMERCIAL, opt-in only.

`disco-eth/WorldSpeech` config `mn_mn` is by far the largest Mongolian corpus
that exists: 138,529 clips, **~221 hours, 24 kHz native**, with human
transcripts and precomputed CER, WADA-SNR and DNSMOS. Sources are
`parliament_mn` and `lds_general_conference`.

It is also the only Mongolian source with meaningful full-band content, so it
would roughly quadruple the corpus and lift the ~8 kHz bandwidth ceiling that
every other source imposes.

**It is CC-BY-NC-4.0.** Under the project's commercial-safe licence decision it
is excluded, so this loader is not in `ALL_SOURCES` and must be requested
explicitly:

    python clean_pipeline.py --datasets ws --allow-non-commercial

Sampled quality, for planning if that constraint is ever revisited:

    source                  CER median   SNR median   DNSMOS OVR >= 3.0
    lds_general_conference     0.050        11.2 dB          35%
    parliament_mn              0.071         8.7 dB          20%
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..calibrate import Calibration
from ..corpus import CorpusWriter
from ..processor import process_split
from ..stats import CleaningStats

# AudioQualityFilter is only referenced as a type here. Importing it at
# runtime would drag Silero VAD, transformers, torchmetrics and the MMS_FA
# aligner into any process that merely wants to inspect a loader.
if TYPE_CHECKING:
    from ..audio_filter import AudioQualityFilter

# `datasets` is imported inside the loader, not here: a process that only
# wants to inspect this module should not need the model stack, and an
# eager import broke four test files in an environment without one.
log = logging.getLogger(__name__)

LICENCE = "CC-BY-NC-4.0"

# Upstream's own quality metrics, kept under original_* so they cannot collide
# with the metrics this pipeline computes. Comparing the two is useful: they were
# produced by a different toolchain on the untrimmed audio.
_EXTRA_FIELDS = [
    "human_transcript", "snr", "duration", "source", "source_url",
    "source_start_s", "source_end_s", "session_date", "segment_id",
    "language", "country",
    "asr_transcript", "cer",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "dnsmos_p808",
]

_FIELD_RENAMES = {
    "asr_transcript": "original_asr_transcript",
    "cer": "original_cer",
    "dnsmos_sig": "original_dnsmos_sig",
    "dnsmos_bak": "original_dnsmos_bak",
    "dnsmos_ovr": "original_dnsmos_ovr",
    "dnsmos_p808": "original_dnsmos_p808",
    "snr": "original_snr",
    "duration": "original_duration",
}


def process_worldspeech(
    filt: AudioQualityFilter,
    writer: CorpusWriter,
    *,
    resume: bool = True,
    limit: int | None = None,
    calibration: Calibration | None = None,
    allow_non_commercial: bool = False,
) -> CleaningStats:
    if not allow_non_commercial:
        raise SystemExit(
            f"WorldSpeech is {LICENCE}. Including it makes the trained model "
            "non-commercial, which the project's licence decision excludes. "
            "Pass --allow-non-commercial to override deliberately."
        )

    log.warning("Including WorldSpeech (%s) — the resulting model is NOT "
                "commercially usable.", LICENCE)
    from datasets import load_dataset

    ws = load_dataset("disco-eth/WorldSpeech", "mn_mn")

    all_stats = CleaningStats("worldspeech_mn")
    for split_name in ws:
        all_stats.merge(process_split(
            ws[split_name],
            filt,
            writer,
            audio_field="audio",
            text_field="human_transcript",
            dataset_name="ws",
            split_name=split_name,
            extra_fields=_EXTRA_FIELDS,
            field_renames=_FIELD_RENAMES,
            resume=resume,
            limit=limit,
            calibration=calibration,
        ))
    return all_stats
