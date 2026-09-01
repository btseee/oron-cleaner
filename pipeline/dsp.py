"""Pure signal-measurement and decision functions.

Deliberately free of model imports. `audio_filter` pulls in Silero VAD,
transformers and torchmetrics at module scope, which meant none of this logic
could be exercised without a full ML stack installed -- so in practice it was
not tested at all. Everything here takes arrays and numbers and returns numbers.
"""

from __future__ import annotations

import re
import unicodedata

import librosa
import numpy as np

from .constants import (
    BANDWIDTH_DROP_DB,
    CLIP_SAMPLE_THRESHOLD,
    MAX_CER,
    MAX_LEN_RATIO,
    MIN_LEN_RATIO,
    SAMPLE_RATE,
)


def clipped_ratio(audio: np.ndarray) -> float:
    """Fraction of samples at or beyond full scale.

    Clipping survives every other gate -- it does not lower DNSMOS much and does
    not confuse an ASR -- but it teaches a generative model to reproduce
    distortion.
    """
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= CLIP_SAMPLE_THRESHOLD))


def dc_offset(audio: np.ndarray) -> float:
    """Absolute mean sample value. A DC offset wastes headroom and biases mel."""
    return 0.0 if audio.size == 0 else float(abs(np.mean(audio)))


def edge_trim_bounds(timestamps: list[dict]) -> tuple[int, int] | None:
    """Sample bounds spanning the first to last speech segment, pauses included.

    This is the whole difference between trimming and splicing. Concatenating
    the segments -- what the pipeline used to publish -- deletes every interior
    pause and butt-joins the pieces, which removes prosodic pausing and leaves a
    discontinuity at each join.
    """
    if not timestamps:
        return None
    return timestamps[0]["start"], timestamps[-1]["end"]


def estimate_snr(
    audio: np.ndarray, timestamps: list[dict], sr: int = SAMPLE_RATE
) -> float:
    """Speech-region RMS against non-speech-region RMS, in dB.

    Must run on the *untrimmed* signal: the non-speech regions are the
    measurement. The previous implementation ran on the spliced speech-only
    audio and took the quietest 10% of frames as its noise floor, which measures
    speech dynamic range -- a uniformly noisy clip with steady delivery scored
    well, a clean clip with wide dynamics scored badly.

    Returns NaN when there is no usable silence, rather than inventing a passing
    value the way the old `noise_floor < 1e-10 -> 40.0` branch did.
    """
    if audio.size == 0 or not timestamps:
        return float("nan")

    speech = np.zeros(len(audio), dtype=bool)
    for t in timestamps:
        speech[t["start"]:t["end"]] = True

    noise = ~speech
    if not speech.any() or noise.sum() < sr // 10:
        return float("nan")

    speech_rms = float(np.sqrt(np.mean(audio[speech] ** 2)))
    noise_rms = float(np.sqrt(np.mean(audio[noise] ** 2)))
    if speech_rms <= 0.0 or noise_rms <= 0.0:
        return float("nan")
    return float(20.0 * np.log10(speech_rms / noise_rms))


def measure_bandwidth(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Highest frequency within BANDWIDTH_DROP_DB of the spectral peak.

    Detects a real lowpass shelf, which is what an mp3 encoder or a 16 kHz
    source leaves behind. Cumulative-energy rolloff was tried and rejected:
    speech energy is dominated by sub-1 kHz formants, so it underestimates by
    1-2 kHz. Validated against 16 kHz-native corpora, which correctly measure
    ~7.7 kHz.
    """
    n = min(len(audio), sr * 5)
    if n < sr // 2:
        return 0.0
    spec = np.abs(librosa.stft(audio[:n], n_fft=2048)) ** 2
    power = np.maximum(spec.mean(axis=1), 1e-20)
    db = 10.0 * np.log10(power / power.max())
    above = np.where(db > -BANDWIDTH_DROP_DB)[0]
    if above.size == 0:
        return 0.0
    return float(np.fft.rfftfreq(2048, 1 / sr)[above[-1]])


def median_f0(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[float, float]:
    """Median F0 over voiced frames, and the voiced fraction. Never rejects.

    librosa pyin rather than CREPE: CREPE was the most expensive stage in the
    pipeline and could not reject anything, so it was pure cost. pyin separates
    male from female cleanly enough for the gender inference this metadata
    supports -- measured 115 Hz male against 239 Hz female on self-declared
    Common Voice labels.
    """
    try:
        f0, voiced, _ = librosa.pyin(
            audio, fmin=60.0, fmax=400.0, sr=sr, frame_length=1024
        )
    except Exception:
        return 0.0, 0.0
    if f0 is None or voiced is None:
        return 0.0, 0.0
    valid = f0[voiced & ~np.isnan(f0)]
    if valid.size < 10:
        return 0.0, 0.0
    return float(np.median(valid)), float(valid.size / max(len(f0), 1))


def for_comparison(text: str) -> str:
    """Casefold and strip punctuation, so only phonetic content is scored."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def reading_passes(cer: float, length_ratio: float) -> tuple[bool, str]:
    """Single CER threshold, no rescue band.

    The old policy rescued clips up to CER 0.50 when the ASR output happened to
    be a similar *length* -- half the characters wrong, admitted on a character
    count. That existed to work around whisper-large-v3's 0.311 CER floor on
    Mongolian. With a 0.123 floor the rescue is unnecessary, and it was admitting
    clips where the speaker said something substantively different.
    """
    if not (MIN_LEN_RATIO <= length_ratio <= MAX_LEN_RATIO):
        return False, f"length_ratio_{length_ratio:.2f}"
    if cer > MAX_CER:
        return False, f"high_cer_{cer:.3f}"
    return True, ""
