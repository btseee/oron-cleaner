"""Repair the clips that are genuinely repairable, and only those.

The pipeline discards a clip at the first gate it fails. Most of those failures
are real and final: you cannot invent bandwidth, or audio that was never
recorded, or a transcript nobody wrote. But some are not failures of the
recording at all. A clip whose speech is clean and whose transcript is right can
still be thrown away for carrying eight seconds of room tone before the sentence
starts, for a microphone's DC offset, or for being recorded too quietly for the
recogniser to read.

Every repair here obeys one rule: **it may not alter the speech signal.** A
scalar gain, a subtracted constant, a trimmed edge and a corrected encoding
error all leave the speech exactly as recorded. Denoising, de-clipping and
bandwidth extension do not, and they are not here -- a denoised clip teaches the
model the denoiser's artifacts, and worse, its SNR and DNSMOS then measure our
processing rather than the recording, so the gate that judges quality is judging
us.

A repair returns `None` when it does not apply. That is the common case: most
clips are not the kind any given repair fixes, and saying so costs nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np

from .constants import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    RECOVERY_MIN_DC,
    RECOVERY_QUIET_PEAK,
    RECOVERY_TARGET_PEAK,
)
from .trimming import WEAK_WORD_SCORE

Repair = Callable[[np.ndarray, int, str], "tuple[np.ndarray, int, str] | None"]

# Latin letters with an identical Cyrillic twin. Inside an otherwise-Cyrillic
# word each has exactly one correct reading, so this corrects an encoding error
# rather than guessing at ambiguity.
HOMOGLYPHS: dict[str, str] = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "i": "и",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х",
}

# `і` U+0456 (Ukrainian/Belarusian Cyrillic i) passes `is_representable` because
# it is in the vocabulary, so it reaches the model as its own embedding row for
# a letter nobody typed -- but it is not the Latin side of anything, it already
# sits inside the Cyrillic block. HOMOGLYPHS' contract (asserted by the test
# that walks it) is Latin key to Cyrillic value, so this single substitution
# lives outside that dict instead of breaking it.
_EXTRA_FIXES: dict[str, str] = {"і": "й"}

_CYRILLIC = range(0x400, 0x500)


def remove_dc_offset(audio: np.ndarray, sr: int, text: str):
    """Subtract the mean.

    A DC offset is an additive constant with exactly one correct removal, so
    this is the least ambiguous repair there is. It matters because the offset
    eats headroom and shifts the clipping ratio, failing a gate that has nothing
    to say about the recording's quality.
    """
    if audio.size == 0:
        # Matches dsp.dc_offset()'s own guard: `.mean()` on an empty array is a
        # RuntimeWarning ("Mean of empty slice"), not a value worth repairing.
        return None
    # NaN and inf compare False against everything -- `abs(nan) < x` is False,
    # same as `nan <= 0.0` below in normalise_gain -- so a single poisoned
    # sample would otherwise slip past every guard, get "repaired", and leave
    # the whole clip NaN while looking like a success. Refuse instead.
    if not np.isfinite(audio).all():
        return None
    offset = float(audio.mean())
    if abs(offset) < RECOVERY_MIN_DC:
        return None
    return (audio - offset).astype("float32"), sr, text


def normalise_gain(audio: np.ndarray, sr: int, text: str):
    """Scale a too-quiet clip up to a target peak.

    A scalar multiply changes no information: the model hears the same
    recording, louder. Crowd-sourced clips are routinely quiet enough that the
    VAD finds no speech and the recogniser misreads, for a reason that belongs
    to the microphone rather than the speaker.

    Only clips below `RECOVERY_QUIET_PEAK` are touched. Above it, the level is
    somebody's deliberate choice.
    """
    if audio.size == 0:
        return None
    # See remove_dc_offset: NaN/inf comparisons are always False, so without
    # this guard a poisoned sample reaches `peak`, propagates through the
    # multiply, and the clip comes back as an all-NaN "repair".
    if not np.isfinite(audio).all():
        return None
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0 or peak >= RECOVERY_QUIET_PEAK:
        return None
    return (audio * (RECOVERY_TARGET_PEAK / peak)).astype("float32"), sr, text


def repair_homoglyphs(audio: np.ndarray, sr: int, text: str):
    """Correct Latin letters sitting inside Cyrillic words.

    The decision needs context: `о` in `Mонгол` is a typo for `О`, while
    `Google` is a word that is simply Latin. Repair runs per hyphen-joined
    segment rather than per whole space-separated word, because a hyphen
    routinely joins a foreign word to a Mongolian suffix (`Google-ийн`) and the
    Latin segment must not be repaired just for sitting next to a Cyrillic one.
    A segment is repaired only when it already contains Cyrillic, which is what
    makes this a correction rather than a guess.
    """
    words = text.split(" ")
    changed = False
    for i, word in enumerate(words):
        segments = word.split("-")
        word_changed = False
        for j, segment in enumerate(segments):
            if not any(ord(c) in _CYRILLIC for c in segment):
                continue
            fixed = "".join(
                HOMOGLYPHS.get(c, _EXTRA_FIXES.get(c, c)) for c in segment
            )
            if fixed != segment:
                segments[j] = fixed
                word_changed = True
        if word_changed:
            words[i] = "-".join(segments)
            changed = True
    if not changed:
        return None
    return audio, sr, " ".join(words)


# Pins the three functions above to the `Repair` contract: if a signature ever
# drifts (an extra required argument, a return type outside the tuple-or-None
# shape), this line is where a type checker catches it, rather than nothing
# noticing until a caller iterating `Repair`s breaks at runtime.
_REPAIRS: tuple[Repair, ...] = (remove_dc_offset, normalise_gain, repair_homoglyphs)


def split_at_silence(audio: np.ndarray, sr: int, text: str, *, aligner,
                     speech_spans: list[tuple[float, float]]):
    """Cut an over-length clip into segments at the silences between sentences.

    Each segment is the original audio, unmodified; only the boundaries are new.
    That is what makes this a legal repair -- but it is the one repair that can
    create a defect rather than fail cleanly, because a split is two new
    transcripts, and this project publishes the text, scores CER against it and
    trains on it. A cut at the wrong word is wrong three times.

    So it refuses unless the alignment is confident on the words either side of
    the cut, every segment lands inside the duration limits, and the segments'
    transcripts concatenate back to the original.
    """
    duration = len(audio) / sr
    if duration <= MAX_DURATION_S:
        return None
    timings = aligner.word_timings(audio, text)
    if not timings:
        return None

    gaps = []
    for i in range(len(timings) - 1):
        _, _, end, score_a = timings[i]
        _, start, _, score_b = timings[i + 1]
        if start - end < MIN_DURATION_S:
            continue
        if min(score_a, score_b) < WEAK_WORD_SCORE:
            # The cut point is the one place the alignment has to be right.
            return None
        gaps.append((i, (end + start) / 2.0))
    if not gaps:
        return None

    bounds = [0.0] + [t for _, t in gaps] + [duration]
    words = [w for w, _, _, _ in timings]
    parts: list[tuple[np.ndarray, int, str]] = []
    cut_at = [i for i, _ in gaps]
    first = 0
    for k, (lo, hi) in enumerate(pairwise(bounds)):
        if not MIN_DURATION_S <= hi - lo <= MAX_DURATION_S:
            return None
        last = cut_at[k] + 1 if k < len(cut_at) else len(words)
        segment_text = " ".join(words[first:last])
        if not segment_text:
            return None
        parts.append((audio[int(lo * sr):int(hi * sr)].astype("float32"), sr,
                      segment_text))
        first = last

    if " ".join(p[2] for p in parts) != " ".join(words):
        return None                      # a word was lost or duplicated
    return parts
