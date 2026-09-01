"""Signal measurement and gate decisions.

These are the functions that decide what enters the training corpus, so they are
tested against synthetic signals with known properties rather than mocked.

None of this needs a model, which is the point of `pipeline.dsp` existing:
`pipeline.audio_filter` imports Silero VAD, transformers and torchmetrics at
module scope, so nothing inside it could be tested without the full ML stack.
"""

import numpy as np
import pytest

from pipeline.constants import MAX_CER, SAMPLE_RATE
from pipeline.dsp import (
    clipped_ratio,
    dc_offset,
    edge_trim_bounds,
    estimate_snr,
    for_comparison,
    measure_bandwidth,
    reading_passes,
)

RNG = np.random.default_rng(0)


def _tone(freq: float, seconds: float, sr: int = SAMPLE_RATE, amp: float = 0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ── clipping and DC ───────────────────────────────────────────────────────────

def test_clean_audio_is_not_flagged_as_clipped():
    assert clipped_ratio(_tone(200, 1.0, amp=0.5)) == 0.0


def test_clipping_is_detected():
    audio = np.clip(_tone(200, 1.0, amp=2.0), -1.0, 1.0)
    assert clipped_ratio(audio) > 0.01


def test_clipped_ratio_of_empty_audio_is_zero():
    assert clipped_ratio(np.array([], dtype=np.float32)) == 0.0


def test_dc_offset_is_measured():
    assert dc_offset(_tone(200, 1.0)) == pytest.approx(0.0, abs=1e-3)
    assert dc_offset(_tone(200, 1.0) + 0.2) == pytest.approx(0.2, abs=1e-3)


# ── edge trimming: the defect this rewrite exists to fix ──────────────────────

def test_edge_trim_spans_first_to_last_speech():
    ts = [{"start": 1000, "end": 2000}, {"start": 5000, "end": 7000}]
    assert edge_trim_bounds(ts) == (1000, 7000)


def test_edge_trim_preserves_interior_pauses():
    """The whole point: interior silence must survive into the published audio.

    Splicing the segments would yield 3000 samples and delete the 3000-sample
    pause between them, removing prosodic pausing and leaving a discontinuity.
    """
    ts = [{"start": 1000, "end": 2000}, {"start": 5000, "end": 7000}]
    start, end = edge_trim_bounds(ts)
    spliced = sum(t["end"] - t["start"] for t in ts)
    assert end - start == 6000
    assert spliced == 3000
    assert end - start > spliced


def test_edge_trim_of_no_speech_is_none():
    assert edge_trim_bounds([]) is None


# ── SNR ───────────────────────────────────────────────────────────────────────

def _speech_plus_silence(speech_amp: float, noise_amp: float):
    """1 s of noise, 2 s of speech-over-noise, 1 s of noise."""
    sr = SAMPLE_RATE
    noise = (RNG.normal(0, noise_amp, 4 * sr)).astype(np.float32)
    audio = noise.copy()
    audio[sr:3 * sr] += _tone(200, 2.0, amp=speech_amp)
    return audio, [{"start": sr, "end": 3 * sr}]


def test_snr_recovers_a_known_ratio():
    audio, ts = _speech_plus_silence(speech_amp=0.3, noise_amp=0.003)
    # speech RMS ~= 0.3/sqrt(2) = 0.212; noise RMS = 0.003 -> ~37 dB
    assert estimate_snr(audio, ts) == pytest.approx(37.0, abs=3.0)


def test_snr_falls_when_noise_rises():
    quiet, ts = _speech_plus_silence(speech_amp=0.3, noise_amp=0.003)
    noisy, _ = _speech_plus_silence(speech_amp=0.3, noise_amp=0.05)
    assert estimate_snr(quiet, ts) > estimate_snr(noisy, ts) + 10


def test_snr_is_nan_without_silence_to_measure():
    """No non-speech region means SNR is unmeasurable.

    The previous implementation returned a hardcoded 40.0 dB when the noise floor
    was near zero, so digitally-silent and unmeasurable clips auto-passed the gate.
    """
    audio = _tone(200, 2.0)
    ts = [{"start": 0, "end": len(audio)}]
    assert np.isnan(estimate_snr(audio, ts))


def test_snr_of_empty_input_is_nan():
    assert np.isnan(estimate_snr(np.array([], dtype=np.float32), []))


# ── bandwidth ─────────────────────────────────────────────────────────────────

def test_bandwidth_recovers_a_known_lowpass_cutoff():
    """Sum of tones up to 5 kHz should measure ~5 kHz, not the Nyquist rate."""
    audio = sum(_tone(f, 2.0, amp=0.2) for f in (200, 800, 2000, 3500, 5000))
    measured = measure_bandwidth(np.asarray(audio, dtype=np.float32))
    assert 4500 <= measured <= 5800, measured


def test_bandwidth_separates_narrow_from_wide():
    narrow = np.asarray(sum(_tone(f, 2.0, amp=0.2) for f in (200, 1000, 3000)), np.float32)
    wide = np.asarray(sum(_tone(f, 2.0, amp=0.2) for f in (200, 1000, 3000, 7000)), np.float32)
    assert measure_bandwidth(wide) > measure_bandwidth(narrow) + 2000


def test_bandwidth_of_too_short_audio_is_zero():
    assert measure_bandwidth(np.zeros(100, dtype=np.float32)) == 0.0


# ── transcript agreement ──────────────────────────────────────────────────────

def test_good_reading_passes():
    assert reading_passes(cer=0.05, length_ratio=1.0) == (True, "")


def test_high_cer_is_rejected():
    ok, reason = reading_passes(cer=0.45, length_ratio=1.0)
    assert not ok and reason.startswith("high_cer")


def test_there_is_no_rescue_band():
    """CER 0.46 at a plausible length used to be rescued.

    That existed to work around whisper-large-v3's 0.311 CER floor on Mongolian.
    With wav2vec2-xlsr's 0.123 floor it is unnecessary, and it was admitting
    clips where the speaker said something substantively different -- half the
    characters wrong, accepted on a character count.
    """
    ok, _ = reading_passes(cer=0.46, length_ratio=0.98)
    assert not ok


def test_truncated_and_runaway_readings_are_rejected():
    for ratio in (0.30, 2.00):
        ok, reason = reading_passes(cer=0.01, length_ratio=ratio)
        assert not ok and reason.startswith("length_ratio"), ratio


def test_cer_threshold_is_above_the_asr_error_floor():
    """A threshold below the scorer's own floor gates on ASR noise, not data.

    whisper-large-v3 measures 0.311 median CER on clean Mongolian, so the old
    0.35 threshold had almost no headroom. wav2vec2-xlsr measures 0.123.
    """
    assert MAX_CER > 0.123, "threshold must leave headroom above the ASR floor"
    assert MAX_CER < 0.311, "threshold must be tighter than Whisper's error floor"


# ── comparison normalisation ──────────────────────────────────────────────────

def test_comparison_strips_case_and_punctuation():
    assert for_comparison("Сайн, байна уу?") == "сайн байна уу"


def test_comparison_keeps_cyrillic_letters():
    assert for_comparison("Өнөөдөр үүлшинэ.") == "өнөөдөр үүлшинэ"
