"""Forced alignment: romanisation, and the discrimination the gate relies on.

The romanisation tests are cheap and run everywhere. The discrimination test
needs the 1.18 GB MMS_FA checkpoint and is opt-in via RUN_SLOW_TESTS=1 -- but it
is the test that actually justifies MIN_ALIGN_SCORE, so it should be run
whenever that threshold is touched.
"""

import os

import numpy as np
import pytest

from pipeline.constants import MIN_ALIGN_SCORE, SAMPLE_RATE

pytest.importorskip("uroman", reason="uroman not installed")
pytest.importorskip("torchaudio", reason="torchaudio not installed")

SLOW = os.environ.get("RUN_SLOW_TESTS") == "1"


@pytest.fixture(scope="module")
def aligner():
    from pipeline.alignment import ForcedAligner

    return ForcedAligner(device="cpu")


# ── romanisation ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def romanize():
    """Romanisation alone, without loading the 1.18 GB acoustic model."""
    import re

    import uroman

    u = uroman.Uroman()
    non_roman = re.compile(r"[^a-z'\s]")

    def _romanize(text: str) -> list[str]:
        return [w for w in non_roman.sub(" ", u.romanize_string(text, lcode="mon").lower()).split() if w]

    return _romanize


def test_romanization_produces_only_mms_fa_labels(romanize):
    """MMS_FA's label set is a-z plus apostrophe; anything else fails to tokenize."""
    allowed = set("abcdefghijklmnopqrstuvwxyz'")
    words = romanize("Сайн байна уу? Өнөөдөр үүлшинэ, хоёр мянга хорин дөрөв.")
    assert words
    for w in words:
        assert set(w) <= allowed, w


def test_romanization_preserves_word_count(romanize):
    assert len(romanize("Сайн байна уу")) == 3
    assert len(romanize("Улаанбаатар")) == 1


def test_mongolian_specific_vowels_survive(romanize):
    """ө and ү must not vanish -- they are ordinary Mongolian vowels."""
    assert romanize("өө") and romanize("үү")
    assert romanize("Өнөөдөр") != romanize("өдөр")


def test_empty_and_punctuation_only_text_yields_no_words(romanize):
    assert romanize("") == []
    assert romanize("!?.,") == []


# ── the gate itself ───────────────────────────────────────────────────────────

def test_unalignable_input_returns_nan(aligner):
    """NaN means "no evidence", never "passes".

    Returning a passing number when a stage cannot run is how the previous SNR
    gate let digitally silent clips through.
    """
    audio = np.zeros(16000, dtype=np.float32)
    assert np.isnan(aligner.score(audio, ""))
    assert np.isnan(aligner.score(np.zeros(10, dtype=np.float32), "сайн байна"))


@pytest.mark.skipif(not SLOW, reason="set RUN_SLOW_TESTS=1 (downloads 1.18 GB)")
def test_alignment_separates_correct_from_mismatched_transcripts(aligner):
    """The measurement MIN_ALIGN_SCORE is derived from.

    Synthesised here from a fixed clip pair rather than the corpus, so it runs
    offline once the checkpoint is cached. Real-corpus numbers, both separating
    cleanly:
        FLEURS        correct min 0.829   mismatched max 0.443
        Common Voice  correct min 0.722   mismatched max 0.547
    """
    from pathlib import Path

    import soundfile as sf

    fixture = Path(__file__).parent / "fixtures" / "align_pair.wav"
    if not fixture.exists():
        pytest.skip("fixture audio not present")
    audio, _ = sf.read(fixture, dtype="float32")
    correct = (fixture.with_suffix(".txt")).read_text(encoding="utf-8").strip()

    good = aligner.score(audio, correct)
    bad = aligner.score(audio, "огт өөр өгүүлбэр энд байна гэж бодъё")

    assert good > MIN_ALIGN_SCORE > bad, (good, bad)


# ── word timings ──────────────────────────────────────────────────────────────
#
# `word_timings` computes exactly what `word_scores` computes and keeps the
# span boundaries `word_scores` throws away, so it needs the real model to
# exercise the frame-to-second conversion -- there is no way to fake torchaudio's
# span objects without reimplementing them. Not SLOW-gated, matching
# `test_unalignable_input_returns_nan` above: it needs the same cached
# checkpoint and no network call, and is fast once that checkpoint is warm.

def _noisy_speech(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.1).astype(np.float32)


def test_word_timings_is_empty_for_unalignable_input(aligner):
    """Mirrors word_scores's own guard clauses: no evidence, no timings."""
    assert aligner.word_timings(np.zeros(16000, dtype=np.float32), "") == []
    assert aligner.word_timings(np.zeros(10, dtype=np.float32), "сайн байна") == []


def test_word_timings_word_count_matches_the_transcript(aligner):
    text = "сайн байна уу"
    timings = aligner.word_timings(_noisy_speech(3.0), text)
    assert len(timings) == len(aligner.romanize(text)) == 3


def test_word_timings_are_non_decreasing_and_each_word_ends_at_or_after_it_starts(aligner):
    timings = aligner.word_timings(_noisy_speech(3.0), "сайн байна уу")
    assert timings
    prev_end = 0.0
    for _word, start, end, _score in timings:
        assert start >= prev_end - 1e-6
        assert end >= start
        prev_end = end


# ── word_timings returns the source word, not the romanised one ──────────────
#
# The transcript this feeds is published, scored for CER and trained on --
# Latin output here would corrupt all three. A fixture that hands back
# Cyrillic words directly (as FakeAligner in test_recovery_split.py does)
# can't catch a method that quietly returns the romanised word instead, so
# these drive the *real* romanisation and check what comes out the other
# side. The acoustic model is stubbed -- it is the one part of the aligner
# that word identity does not depend on -- so no checkpoint download is
# needed.

class _FakeSpan:
    def __init__(self, score: float, start: int, end: int) -> None:
        self.score = score
        self.start = start
        self.end = end


def _aligner_with_stubbed_model():
    """A real `ForcedAligner` -- real uroman romanisation, real `word_timings`
    code path -- with only the expensive MMS_FA model swapped for a stub that
    hands back one dummy span per input token."""
    import uroman

    from pipeline.alignment import ForcedAligner

    inst = ForcedAligner.__new__(ForcedAligner)
    inst.device = "cpu"
    inst._uroman = uroman.Uroman()

    def fake_model(waveform):
        import torch

        return torch.zeros(1, waveform.size(1), 1), None

    inst._model = fake_model
    inst._tokenizer = lambda words: list(range(len(words)))
    inst._aligner = lambda emission0, tokens: [
        [_FakeSpan(0.9, i, i + 1)] for i in range(len(tokens))
    ]
    return inst


def test_word_timings_returns_the_original_cyrillic_word_not_the_romanised_one():
    aligner = _aligner_with_stubbed_model()
    text = "Сайн байна уу"
    timings = aligner.word_timings(_noisy_speech(3.0), text)
    assert [word for word, *_ in timings] == text.split()


def test_word_timings_refuses_when_romanisation_changes_the_word_count():
    """A digit like `2024` romanises to nothing -- uroman's Latin output is
    filtered to a-z -- so a naive positional zip would pair the wrong
    Cyrillic word with the wrong span instead of failing loudly."""
    aligner = _aligner_with_stubbed_model()
    assert aligner.word_timings(_noisy_speech(3.0), "хоёр мянга 2024 он") == []
