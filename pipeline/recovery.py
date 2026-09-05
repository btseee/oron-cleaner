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

import numpy as np

from .constants import RECOVERY_MIN_DC, RECOVERY_QUIET_PEAK, RECOVERY_TARGET_PEAK

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
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
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
