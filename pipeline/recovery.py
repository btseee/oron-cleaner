"""Repair the clips that are genuinely repairable, and only those.

The pipeline discards a clip at the first gate it fails. Most of those failures
are real and final: you cannot invent bandwidth, or audio that was never
recorded, or a transcript nobody wrote. But some are not failures of the
recording at all. A clip whose speech is clean and whose transcript is right can
still be thrown away for running past a length limit that exists to suit a
batch sampler.

Every repair here obeys one rule: **it may not alter the speech signal.** A
trimmed edge and a new boundary leave the speech exactly as recorded. Denoising,
de-clipping and bandwidth extension do not, and they are not here -- a denoised
clip teaches the model the denoiser's artifacts, and worse, its SNR and DNSMOS
then measure our processing rather than the recording, so the gate that judges
quality is judging us.

Splitting an over-length clip is the only repair left in this module. Three
others -- DC-offset removal, gain normalisation and homoglyph correction --
were written, measured, and found to recover approximately nothing on all
three corpora; a recovery that recovers nothing gets deleted rather than
shipped. Homoglyph correction survives, but as what it always was: a text
normalisation, applied to every transcript in `audio_filter.py` rather than
offered to rejected clips as a second chance.

A repair returns `None` when it does not apply. That is the common case: most
clips are not the kind any given repair fixes, and saying so costs nothing.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from .constants import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    RECOVERY_MIN_CUT_SCORE,
    RECOVERY_MIN_SILENCE_S,
)


def split_at_silence(audio: np.ndarray, sr: int, text: str, *, aligner,
                     speech_spans: list[tuple[float, float]],
                     cut_audio: np.ndarray | None = None, cut_sr: int = 0):
    """Cut an over-length clip into segments at the silences between sentences.

    Each segment is the original audio, unmodified; only the boundaries are new.
    That is what makes this a legal repair -- but it is the one repair that can
    create a defect rather than fail cleanly, because a split is two new
    transcripts, and this project publishes the text, scores CER against it and
    trains on it. A cut at the wrong word is wrong three times.

    So it refuses unless the alignment is confident on the words either side of
    the cut, every segment lands inside the duration limits, and the segments'
    transcripts concatenate back to the original.

    `audio`/`sr` is the 16 kHz signal the aligner and the VAD must see. Pass
    `cut_audio`/`cut_sr` to slice the segments out of the source-rate signal
    instead: cut points are decided in seconds, so they carry across rates
    unchanged, and a split clip then keeps the bandwidth its source had rather
    than inheriting the aligner's 16 kHz.

    A word-timing gap alone is not enough evidence of silence: it can be the
    alignment jittering at a boundary rather than an actual pause. `speech_spans`
    comes straight from the VAD, in seconds, so it is a second, independently-
    measured silence signal -- a candidate cut is only trusted when both agree
    there is silence there. With no spans there is no corroboration and nothing
    can be cut, which is a refusal, not a fallback.
    """
    duration = len(audio) / sr
    if duration <= MAX_DURATION_S:
        return None
    timings = aligner.word_timings(audio, text)
    if not timings:
        return None

    # The silence between one VAD-detected speech span and the next -- the
    # gaps a word-timing candidate must fall inside to count as real silence.
    vad_silences = [(a_end, b_start) for (_, a_end), (b_start, _) in pairwise(speech_spans)]

    gaps = []
    for i in range(len(timings) - 1):
        _, _, end, score_a = timings[i]
        _, start, _, score_b = timings[i + 1]
        if start - end < RECOVERY_MIN_SILENCE_S:
            continue
        midpoint = (end + start) / 2.0
        if not any(vs <= midpoint <= ve for vs, ve in vad_silences):
            # Word timings alone suggest a pause here, but the VAD -- looking
            # directly at the audio -- disagrees; not a real cut candidate.
            continue
        if min(score_a, score_b) < RECOVERY_MIN_CUT_SCORE:
            # The cut point is the one place the alignment has to be right.
            return None
        gaps.append((i, midpoint))
    if not gaps:
        return None

    src, src_sr = (cut_audio, cut_sr) if cut_audio is not None and cut_sr else (audio, sr)
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
        parts.append((src[int(lo * src_sr):int(hi * src_sr)].astype("float32"),
                      src_sr, segment_text))
        first = last

    # Against `text`, the transcript that came in -- not against `words`, which
    # the parts were sliced out of and so can only ever agree with themselves.
    # The aligner is the untrusted party here: it is what decides which words
    # exist and in what order, and a segment transcript is only publishable if
    # the segments still say what the source clip said.
    if " ".join(p[2] for p in parts) != " ".join(text.split()):
        return None                      # a word was lost, added or reordered
    return parts
