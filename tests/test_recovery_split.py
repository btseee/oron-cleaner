"""A split is two new transcripts. A cut at the wrong word is wrong three times.

This project publishes the corpus text, scores CER against it and trains on it --
one string, three uses. So a split is refused unless the cut is confident and
both halves stand on their own.
"""
from __future__ import annotations

import numpy as np

from pipeline.constants import SAMPLE_RATE
from pipeline.recovery import split_at_silence


class FakeAligner:
    """Word timings without loading MMS_FA."""

    def __init__(self, timings):
        self.timings = timings

    def word_timings(self, audio, text):
        return self.timings


def speech(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.3).astype("float32")


def test_a_clip_is_split_at_the_silence_between_sentences():
    audio = speech(24.0)
    timings = [("Сайн", 0.0, 4.0, 0.9), ("байна", 4.0, 10.0, 0.9),
               ("Өнөөдөр", 14.0, 18.0, 0.9), ("сайхан", 18.0, 23.0, 0.9)]
    parts = split_at_silence(audio, SAMPLE_RATE, "Сайн байна Өнөөдөр сайхан",
                             aligner=FakeAligner(timings),
                             speech_spans=[(0.0, 10.0), (14.0, 23.0)])
    assert parts is not None and len(parts) == 2
    assert parts[0][2] == "Сайн байна"
    assert parts[1][2] == "Өнөөдөр сайхан"


def test_the_transcripts_concatenate_back_to_the_original():
    """If they do not, a word was lost or duplicated at the cut."""
    audio = speech(24.0)
    text = "Сайн байна Өнөөдөр сайхан"
    timings = [("Сайн", 0.0, 4.0, 0.9), ("байна", 4.0, 10.0, 0.9),
               ("Өнөөдөр", 14.0, 18.0, 0.9), ("сайхан", 18.0, 23.0, 0.9)]
    parts = split_at_silence(audio, SAMPLE_RATE, text, aligner=FakeAligner(timings),
                             speech_spans=[(0.0, 10.0), (14.0, 23.0)])
    assert " ".join(p[2] for p in parts) == text


def test_a_weak_alignment_at_the_cut_refuses_the_split():
    """The cut point is the one place the alignment has to be right."""
    audio = speech(24.0)
    timings = [("Сайн", 0.0, 4.0, 0.9), ("байна", 4.0, 10.0, 0.05),
               ("Өнөөдөр", 14.0, 18.0, 0.9), ("сайхан", 18.0, 23.0, 0.9)]
    assert split_at_silence(audio, SAMPLE_RATE, "Сайн байна Өнөөдөр сайхан",
                            aligner=FakeAligner(timings),
                            speech_spans=[(0.0, 10.0), (14.0, 23.0)]) is None


def test_a_clip_within_the_length_limit_is_not_split():
    assert split_at_silence(speech(8.0), SAMPLE_RATE, "Сайн байна",
                            aligner=FakeAligner([("Сайн", 0.0, 4.0, 0.9),
                                                 ("байна", 4.0, 8.0, 0.9)]),
                            speech_spans=[(0.0, 8.0)]) is None


def test_a_clip_with_no_usable_silence_is_not_split():
    """Cutting mid-word to satisfy a length limit would be the defect."""
    assert split_at_silence(speech(24.0), SAMPLE_RATE, "Сайн байна",
                            aligner=FakeAligner([("Сайн", 0.0, 12.0, 0.9),
                                                 ("байна", 12.0, 24.0, 0.9)]),
                            speech_spans=[(0.0, 24.0)]) is None


def test_a_timing_gap_not_confirmed_by_the_vad_is_not_used_as_a_cut():
    """Word-timing jitter can look like a pause that never happened. The VAD
    measures silence directly from the audio, so a cut needs both to agree --
    otherwise `speech_spans` would be an argument nothing actually checks."""
    audio = speech(24.0)
    timings = [("Сайн", 0.0, 4.0, 0.9), ("байна", 4.0, 10.0, 0.9),
               ("Өнөөдөр", 14.0, 18.0, 0.9), ("сайхан", 18.0, 23.0, 0.9)]
    # One continuous VAD speech span across the whole clip contradicts the
    # apparent 10..14s gap in the word timings.
    assert split_at_silence(audio, SAMPLE_RATE, "Сайн байна Өнөөдөр сайхан",
                            aligner=FakeAligner(timings),
                            speech_spans=[(0.0, 24.0)]) is None
