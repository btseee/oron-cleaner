"""Cut a transcript back to the span its audio actually covers.

WorldSpeech's Mongolian segments are misaligned with their text. Measured on 800
clips: 6.9% pass the gates, and among clips that *pass*, 80% have audio shorter
than their transcript and 53% end mid-word. The transcript runs past the end of
the recording.

That is not noise, and no gate tuning fixes it. Training on it teaches the model
to speak words that are not in the audio, and the CER gate is structurally blind
because the wrong string is the reference it scores against.

Forced alignment already runs as the transcript gate and already computes a
score per word. A transcript that overruns its audio shows up as a collapse in
the trailing words: the aligner has nothing to match them to. Cutting at that
collapse recovers the clip.

Deliberately conservative:

  * only a **trailing** run is removed. A bad word in the middle means the
    transcript is wrong, not long, and that clip should fail the gates.
  * removing more than `max_trim_fraction` is refused. A clip whose transcript
    is mostly unsupported is broken, not trimmable.
  * a clip that aligns cleanly is returned untouched, so this is safe to run on
    every source rather than needing a per-corpus flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# A word scoring below this has essentially no acoustic support. MMS-FA scores
# sit near 1.0 for confidently aligned words and fall off sharply.
WEAK_WORD_SCORE = 0.30

# Refuse to remove more than this share of the words: past it the transcript is
# wrong rather than long.
MAX_TRIM_FRACTION = 0.40

# Below this many words the trailing-collapse signal is not distinguishable from
# ordinary variation.
MIN_WORDS = 4


@dataclass
class TrimResult:
    text: str
    trimmed: bool
    discarded: str
    words_removed: int
    reason: str


def trim_to_audio(aligner, audio: np.ndarray, text: str,
                  weak_score: float = WEAK_WORD_SCORE,
                  max_trim_fraction: float = MAX_TRIM_FRACTION) -> TrimResult:
    """Return `text` cut back to the words the audio supports.

    `aligner` is a `ForcedAligner`; only its `word_scores` method is used, so a
    stub satisfies this in tests without loading MMS-FA.
    """
    words = text.split()
    if len(words) < MIN_WORDS:
        return TrimResult(text, False, "", 0, "too_short_to_judge")

    scores = aligner.word_scores(audio, text)
    if not scores:
        return TrimResult(text, False, "", 0, "unalignable")
    if len(scores) != len(words):
        # romanize() can split or drop tokens, so the score list need not line up
        # with whitespace words. Without a 1:1 mapping a cut point cannot be
        # located safely, and guessing one would silently corrupt the transcript.
        return TrimResult(text, False, "", 0, "token_mismatch")

    # Walk back over the trailing weak run only.
    cut = len(words)
    while cut > 0 and scores[cut - 1] < weak_score:
        cut -= 1

    if cut == len(words):
        return TrimResult(text, False, "", 0, "aligned")

    removed = len(words) - cut
    if removed > max_trim_fraction * len(words):
        return TrimResult(text, False, "", 0, "too_much_unsupported")
    if cut < MIN_WORDS:
        return TrimResult(text, False, "", 0, "nothing_left")

    kept = " ".join(words[:cut])
    discarded = " ".join(words[cut:])
    log.debug("trimmed %d word(s): %r", removed, discarded)
    return TrimResult(kept, True, discarded, removed, "trimmed_trailing")
