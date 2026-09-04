"""WorldSpeech transcripts run past the end of their audio; this cuts them back.

Measured on 800 WorldSpeech clips: among those that PASS the gates, 80% have
audio shorter than their transcript and 53% end mid-word.
"""
from __future__ import annotations

import numpy as np

from pipeline.trimming import MAX_TRIM_FRACTION, trim_to_audio

AUDIO = np.zeros(24000, dtype="float32")


class Stub:
    """Only word_scores is used, so MMS-FA never has to load in a test."""

    def __init__(self, scores):
        self._scores = scores

    def word_scores(self, audio, text):
        return self._scores


def test_trailing_unsupported_words_are_cut():
    text = "энэ бол сайн өгүүлбэр байна улс ор"
    # Last two words have no acoustic support: the audio stopped before them.
    got = trim_to_audio(Stub([0.9, 0.88, 0.91, 0.87, 0.9, 0.05, 0.02]), AUDIO, text)
    assert got.trimmed
    assert got.text == "энэ бол сайн өгүүлбэр байна"
    assert got.discarded == "улс ор"
    assert got.words_removed == 2


def test_a_clean_clip_is_returned_untouched():
    """Safe to run on every source, so it needs no per-corpus flag."""
    text = "энэ бол сайн өгүүлбэр байна"
    got = trim_to_audio(Stub([0.9, 0.92, 0.88, 0.91, 0.9]), AUDIO, text)
    assert not got.trimmed
    assert got.text == text
    assert got.reason == "aligned"


def test_a_weak_word_in_the_middle_is_not_trimmed():
    """A bad word mid-transcript means the text is wrong, not long. That clip
    should fail the gates, not be silently rewritten."""
    text = "энэ бол сайн өгүүлбэр байна"
    got = trim_to_audio(Stub([0.9, 0.02, 0.88, 0.91, 0.9]), AUDIO, text)
    assert not got.trimmed
    assert got.text == text


def test_mostly_unsupported_transcript_is_refused_not_trimmed():
    text = "нэг хоёр гурав дөрөв тав зургаа долоо найм"
    scores = [0.9, 0.9] + [0.01] * 6          # 75% unsupported
    got = trim_to_audio(Stub(scores), AUDIO, text)
    assert not got.trimmed
    assert got.reason == "too_much_unsupported"
    assert MAX_TRIM_FRACTION < 6 / 8


def test_token_mismatch_refuses_rather_than_guessing():
    """romanize() can split or drop tokens. Without a 1:1 mapping the cut point
    cannot be located, and guessing corrupts the transcript."""
    text = "энэ бол сайн өгүүлбэр байна"
    got = trim_to_audio(Stub([0.9, 0.9]), AUDIO, text)
    assert not got.trimmed
    assert got.reason == "token_mismatch"


def test_unalignable_clip_is_left_alone():
    got = trim_to_audio(Stub([]), AUDIO, "энэ бол сайн өгүүлбэр байна")
    assert not got.trimmed
    assert got.reason == "unalignable"


def test_aligner_exposes_per_word_scores():
    """score() averages these away; trimming needs the sequence."""
    import inspect

    from pipeline.alignment import ForcedAligner

    assert hasattr(ForcedAligner, "word_scores")
    src = inspect.getsource(ForcedAligner.word_scores)
    assert "return []" in src, "must report unalignable rather than a fake score"
