from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClipResult:
    passed: bool
    reject_stage: str = ""
    reject_reason: str = ""
    # Every gate this clip failed, not just the first. Empty in normal
    # operation, where processing stops at the first failure; populated by
    # calibration runs, which is how per-gate rejection rates are obtained
    # rather than only the rate of whichever gate fires first.
    failed_gates: list[str] = field(default_factory=list)
    snr_db: float = 0.0
    # True when the clip is spoken end to end, leaving under 0.1 s of
    # non-speech to compute a noise floor from. Recorded rather than
    # rejected: unmeasurable is not the same as noisy, and DNSMOS-BAK
    # judges the background directly. Kept in the manifest so a corpus
    # can be audited for how much of it was never SNR-checked.
    snr_unmeasurable: bool = False
    mean_f0_hz: float = 0.0
    pitch_confidence: float = 0.0
    dnsmos_sig: float = 0.0
    dnsmos_bak: float = 0.0
    dnsmos_ovr: float = 0.0
    dnsmos_p808: float = 0.0
    # Primary transcript gate. Constrained to the given transcript, so a low
    # score means the audio does not contain those words -- unlike CER, which
    # also moves when the recogniser simply struggles.
    align_score: float = 0.0
    cer: float = 0.0
    # ASR characters over ground-truth characters. Previously computed and then
    # discarded, so a truncated reading could not be diagnosed after the fact.
    len_ratio: float = 0.0
    asr_transcript: str = ""
    # Measured lowpass shelf. Recorded rather than only gated on, because
    # reference-voice selection needs the brightest clips available and no
    # Mongolian source is full-band.
    bandwidth_hz: float = 0.0
    # Duration of the audio actually shipped, after edge-trimming.
    duration_s: float = 0.0
    audio_normalized: np.ndarray = field(default_factory=lambda: np.zeros(1))
