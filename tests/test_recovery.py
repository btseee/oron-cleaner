"""Repairs may fix a recording. They may not change what was said.

The pipeline discards a clip at the first gate it fails, and most of those
failures are final -- you cannot invent bandwidth or audio nobody recorded. But
a clip whose speech is clean and whose transcript is right can still be thrown
away for carrying eight seconds of room tone, or for a DC offset, or for being
recorded too quietly for the recogniser. Those are repairable, and the repair
does not touch the speech.

Every test here holds one line: after a repair, the speech must be the same
speech.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.constants import (
    RECOVERY_MIN_DC,
    RECOVERY_QUIET_PEAK,
    RECOVERY_TARGET_PEAK,
    SAMPLE_RATE,
)
from pipeline.recovery import (
    HOMOGLYPHS,
    normalise_gain,
    remove_dc_offset,
    repair_homoglyphs,
)

TEXT = "Сайн байна уу"


def speech(seconds: float = 3.0, peak: float = 0.5, seed: int = 0) -> np.ndarray:
    """Speech-shaped noise. The content is irrelevant; the shape is not."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    sig = rng.standard_normal(n).astype("float32")
    sig = np.convolve(sig, np.hanning(64), mode="same").astype("float32")
    return (sig / np.max(np.abs(sig)) * peak).astype("float32")


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── DC offset ─────────────────────────────────────────────────────────────────

def test_dc_offset_is_removed_exactly():
    audio = speech() + 0.05
    out, sr, text = remove_dc_offset(audio, SAMPLE_RATE, TEXT)
    assert abs(float(out.mean())) < 1e-6
    assert sr == SAMPLE_RATE and text == TEXT


def test_dc_removal_does_not_change_the_speech():
    """Subtracting a constant is the whole repair. The waveform's shape -- which
    is what anyone hears and what every gate measures -- must survive it."""
    audio = speech()
    out, _, _ = remove_dc_offset(audio + 0.05, SAMPLE_RATE, TEXT)
    assert correlation(out, audio) == pytest.approx(1.0, abs=1e-6)


def test_a_clip_without_a_dc_offset_is_not_repaired():
    """`None` means "not the kind of clip this fixes". Returning a changed clip
    anyway would put a second gate pass on the bill for nothing."""
    assert remove_dc_offset(speech(), SAMPLE_RATE, TEXT) is None


def test_dither_sized_offsets_are_left_alone():
    audio = speech() + RECOVERY_MIN_DC / 2
    assert remove_dc_offset(audio, SAMPLE_RATE, TEXT) is None


# ── gain ──────────────────────────────────────────────────────────────────────

def test_a_quiet_clip_is_brought_up_to_the_target_peak():
    audio = speech(peak=0.02)
    out, _, _ = normalise_gain(audio, SAMPLE_RATE, TEXT)
    assert float(np.max(np.abs(out))) == pytest.approx(RECOVERY_TARGET_PEAK, abs=1e-4)


def test_gain_is_a_scalar_multiply_and_nothing_else():
    """This is the claim that makes gain a legal repair: no information changes,
    so the model hears the same recording, louder."""
    audio = speech(peak=0.02)
    out, _, _ = normalise_gain(audio, SAMPLE_RATE, TEXT)
    ratio = out / np.where(audio == 0, np.nan, audio)
    assert np.nanstd(ratio) == pytest.approx(0.0, abs=1e-5)
    assert correlation(out, audio) == pytest.approx(1.0, abs=1e-6)


def test_a_clip_at_a_normal_level_is_left_alone():
    """Somebody chose that level. Above the quiet threshold it is not ours."""
    assert normalise_gain(speech(peak=RECOVERY_QUIET_PEAK + 0.1), SAMPLE_RATE, TEXT) is None


def test_silence_is_not_amplified():
    """Dividing by a zero peak would produce inf, and there is no speech to save."""
    assert normalise_gain(np.zeros(SAMPLE_RATE, "float32"), SAMPLE_RATE, TEXT) is None


# ── homoglyphs ────────────────────────────────────────────────────────────────

def test_a_latin_letter_inside_a_cyrillic_word_is_corrected():
    """`о` U+006F in an otherwise-Cyrillic word has exactly one correct reading.
    It is an encoding error, not an ambiguity, so fixing it guesses nothing."""
    audio = speech()
    out, _, text = repair_homoglyphs(audio, SAMPLE_RATE, "Mонгол хэл")
    assert text == "Монгол хэл"
    assert out is audio, "the audio is untouched by a text repair"


def test_the_ukrainian_i_is_corrected():
    """U+0456 passes `is_representable` because it is in the vocabulary, so it
    reaches the model as a distinct embedding row for a letter nobody typed."""
    _, _, text = repair_homoglyphs(speech(), SAMPLE_RATE, "саін")
    assert text == "сайн"


def test_an_all_latin_word_is_left_alone():
    """An English proper noun in a Mongolian sentence is not an encoding error."""
    assert repair_homoglyphs(speech(), SAMPLE_RATE, "Google-ийн") is None


def test_clean_cyrillic_is_not_repaired():
    assert repair_homoglyphs(speech(), SAMPLE_RATE, TEXT) is None


def test_every_homoglyph_maps_to_a_cyrillic_letter():
    """A mapping that produced another Latin letter would move the problem."""
    for latin, cyrillic in HOMOGLYPHS.items():
        assert ord(latin) < 0x400, f"{latin!r} is not the Latin side"
        assert 0x400 <= ord(cyrillic) <= 0x4FF, f"{cyrillic!r} is not Cyrillic"
