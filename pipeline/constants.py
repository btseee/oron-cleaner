"""Filter thresholds for the strict Mongolian TTS corpus.

Every gate here is data, not code: `FILTER_POLICY_VERSION` is derived from a hash
of the values below, so changing any threshold automatically invalidates cached
checkpoints instead of silently reusing clips filtered under the old policy.

Measurements behind these numbers live in `oron-tts/docs/phase0-findings.md`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SAMPLE_RATE: int = 16_000
OUTPUT_SAMPLE_RATE: int = 24_000

OUTPUT_DIR: Path = Path("output")

# ── Duration ──────────────────────────────────────────────────────────────────
# F5-TTS's DynamicBatchSampler silently drops any clip longer than
# batch_size_per_gpu frames, and clips over ~20 s eat the whole frame budget for
# a single sample. Measured on the corpus, 20 s loses very little.
MIN_DURATION_S: float = 1.0
MAX_DURATION_S: float = 20.0

# ── Voice activity ────────────────────────────────────────────────────────────
# Silero defaults are deliberately overridden. speech_pad_ms keeps plosive onsets
# and final consonants that a tight boundary clips off; min_silence_duration_ms
# stops the detector fragmenting normal Mongolian inter-word pauses.
VAD_SPEECH_THRESHOLD: float = 0.5
VAD_MIN_SPEECH_MS: int = 250
VAD_MIN_SILENCE_MS: int = 300
VAD_SPEECH_PAD_MS: int = 30
# A clip that is mostly silence is usually a failed recording, not a slow speaker.
VAD_MIN_SPEECH_RATIO: float = 0.35

# ── SNR ───────────────────────────────────────────────────────────────────────
# Measured as speech-region RMS against true non-speech-region RMS, on the
# untrimmed signal. The previous implementation measured the *spliced* speech-only
# signal, where every genuine noise region had already been deleted, so it
# reported speech dynamic range rather than SNR. It was also justified by a
# DeepFilterNet pass that never actually ran in the training path.
SNR_MIN_DB: float = 15.0

# ── Clipping / DC ─────────────────────────────────────────────────────────────
# Consecutive samples at full scale indicate a clipped recording; clipping
# survives every other gate and teaches the model to reproduce distortion.
CLIP_SAMPLE_THRESHOLD: float = 0.999
MAX_CLIPPED_RATIO: float = 0.001
# A DC offset wastes headroom and biases the mel filterbank.
MAX_DC_OFFSET: float = 0.01

# ── Bandwidth ─────────────────────────────────────────────────────────────────
# Highest frequency whose mean band power is within BANDWIDTH_DROP_DB of the
# spectral peak. Measured corpus medians: Common Voice 7.1 kHz, FLEURS 7.7 kHz
# (capped), MBSpeech 7.6 kHz (capped). No Mongolian source is full-band, so a
# 10 kHz gate would discard 77% of the corpus; 7 kHz keeps 59% of Common Voice
# and effectively all of FLEURS/MBSpeech.
BANDWIDTH_DROP_DB: float = 40.0
MIN_BANDWIDTH_HZ: float = 6_000.0

# ── DNSMOS P.835 ──────────────────────────────────────────────────────────────
# Previously 2.2/2.4/2.0, justified by a downstream denoiser that does not run.
# ~2.0 on a 1-5 MOS scale is "poor". These are also scored on the edge-trimmed
# signal now rather than a spliced one, which no longer depresses the score.
DNSMOS_MIN_OVR: float = 2.8
DNSMOS_MIN_SIG: float = 3.0
DNSMOS_MIN_BAK: float = 2.5

# ── Transcript agreement ──────────────────────────────────────────────────────
# Scored with bayartsogt/wav2vec2-large-xlsr-mongolian, whose CER floor on clean
# correctly-transcribed Mongolian is 0.123 median (whisper-large-v3 is 0.311, so
# the old MAX_CER of 0.35 sat *below* its own error floor and gated on ASR noise
# rather than data quality). Both sides are normalised through oron_tts.text
# first, so digits and abbreviations no longer inflate the score.
MAX_CER: float = 0.20
# Forced alignment is the primary gate: it is constrained to the given
# transcript, so a low score is real evidence of mismatch rather than ASR error.
# Calibrated on real Mongolian audio -- correct against deliberately mismatched
# transcripts, both corpora separating cleanly:
#     FLEURS        correct min 0.829   mismatched max 0.443
#     Common Voice  correct min 0.722   mismatched max 0.547
# Worst-case gap is 0.547..0.722. 0.65 sits above the worst mismatch with margin
# and is biased toward rejection: a mismatched clip teaches a wrong text-to-audio
# mapping, a rejected good clip only costs data.
MIN_ALIGN_SCORE: float = 0.65
# Guards against an ASR that stopped early or ran away.
MIN_LEN_RATIO: float = 0.60
MAX_LEN_RATIO: float = 1.60

# ── Speaker balance ───────────────────────────────────────────────────────────
# The top 10 of 511 Common Voice speakers hold 45.7% of validated clips; the
# largest single contributor has 1,956. Without a cap the model collapses toward
# a handful of voices.
#
# Counted in hours rather than clips. Common Voice validated averages 5.07 s
# (33,331 clips / 46.9 h), so the previous 400-clip cap was 0.56 h for that
# source -- but the same number meant something different for every other
# source, because it silently tracked clip length.
MAX_SPEAKER_HOURS: float = 0.6

# A cap buys voice diversity. A source with one narrator has none to buy, so
# capping it only deletes audio. MBSpeech is 3,846 clips / 6.3 h under a single
# male narrator: at 400 clips the cap kept 0.66 h and discarded 5.64 h of the
# cleanest male speech available, in a corpus whose binding acceptance criterion
# is male hours (Common Voice supplies 10.7 h male before any gate). Declared
# single-narrator sources get a budget set by how much one voice the corpus can
# tolerate, not by parity with a crowd-sourced contributor.
MAX_NARRATOR_HOURS: float = 8.0


def _policy_version() -> str:
    """Hash the gate values so a threshold change invalidates old checkpoints.

    Checkpoints are namespaced by this string. Previously it was a hand-edited
    label, so tuning a threshold and resuming silently mixed clips filtered under
    two different policies.
    """
    payload = {
        k: v for k, v in sorted(globals().items())
        if k.isupper() and isinstance(v, (int, float, str))
        and k not in {"OUTPUT_DIR", "FILTER_POLICY_VERSION"}
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"v4_{digest}"


FILTER_POLICY_VERSION: str = _policy_version()
