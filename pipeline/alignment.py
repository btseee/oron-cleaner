"""CTC forced alignment — the primary transcript-agreement gate.

Free-running ASR is the wrong instrument for this job in Mongolian. Even the
best available model has a CER floor of 0.123 on clean, correctly-transcribed
speech (whisper-large-v3 is 0.311), so any absolute CER threshold sits close to
the model's own error and rejects good clips while admitting bad ones.

Forced alignment is *constrained to the transcript you give it*, so a low score
is evidence that the audio does not contain those words -- not evidence that the
recogniser struggled. Measured separation on real Mongolian audio:

    corpus          correct (min)   mismatched (max)
    FLEURS               0.829            0.443
    Common Voice         0.722            0.547

Clean separation on both, worst-case gap 0.547 .. 0.722. MIN_ALIGN_SCORE sits at
0.65: above the worst mismatch with margin, and biased toward rejection because
a mismatched clip teaches the model a wrong text-to-audio mapping, whereas a
rejected good clip only costs data -- and there are ~59 h of it.

The model is MMS_FA (torchaudio), trained on romanised text across 1,100+
languages, so Mongolian Cyrillic is romanised with uroman before alignment.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .constants import SAMPLE_RATE

log = logging.getLogger(__name__)

# MMS_FA's label set is romanised Latin plus apostrophe.
_NON_ROMAN = re.compile(r"[^a-z'\s]")


class ForcedAligner:
    """Score how well an audio clip matches a given transcript, in [0, 1].

    Not thread-safe -- use one instance per process.
    """

    def __init__(self, device: str = "cpu") -> None:
        import uroman
        from torchaudio.pipelines import MMS_FA

        log.info("Loading MMS_FA aligner …")
        self.device = device
        self._uroman = uroman.Uroman()
        self._model = MMS_FA.get_model().to(device).eval()
        self._tokenizer = MMS_FA.get_tokenizer()
        self._aligner = MMS_FA.get_aligner()

    def romanize(self, text: str) -> list[str]:
        """Mongolian Cyrillic to the romanised word list MMS_FA expects."""
        romanized = self._uroman.romanize_string(text, lcode="mon").lower()
        return [w for w in _NON_ROMAN.sub(" ", romanized).split() if w]

    def score(self, audio: np.ndarray, text: str) -> float:
        """Mean per-token alignment score, or NaN if it cannot be computed.

        NaN means "no evidence", not "bad": the caller must decide. Returning a
        passing number when a stage cannot run is how the old SNR gate let
        digitally silent clips through.
        """
        import torch

        words = self.romanize(text)
        if not words:
            return float("nan")
        if audio.size < SAMPLE_RATE // 10:
            return float("nan")

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                emission, _ = self._model(waveform)
                spans = self._aligner(emission[0], self._tokenizer(words))
        except Exception as exc:
            # Most often the transcript is longer than the audio can support,
            # which is itself a mismatch -- but report it as unknown and let the
            # caller log the reason rather than silently scoring it 0.
            log.debug("alignment failed: %s", exc)
            return float("nan")

        scores = [s.score for span in spans for s in span]
        return float(np.mean(scores)) if scores else float("nan")

    def word_scores(self, audio: np.ndarray, text: str) -> list[float]:
        """Per-word alignment score, in transcript order. Empty if unalignable.

        `score()` averages these away, which is what a gate needs but not what
        trimming needs: a transcript that runs past the end of its audio shows
        up as a collapse in the LAST words, and the mean only says the clip is
        slightly worse overall. Returning the sequence lets a caller find where
        the audio stops supporting the text.
        """
        import torch

        words = self.romanize(text)
        if not words or audio.size < SAMPLE_RATE // 10:
            return []
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                emission, _ = self._model(waveform)
                spans = self._aligner(emission[0], self._tokenizer(words))
        except Exception as exc:
            log.debug("alignment failed: %s", exc)
            return []
        out = []
        for span in spans:
            sub = [s.score for s in span]
            out.append(float(np.mean(sub)) if sub else 0.0)
        return out

    def word_timings(self, audio: np.ndarray, text: str) -> list[tuple[str, float, float, float]]:
        """Per-word `(word, start_seconds, end_seconds, mean_score)`, in transcript order.

        `word_scores` computes exactly this from the same spans and then
        discards the timing information, keeping only the mean score. A caller
        that wants to *cut* a clip rather than just judge it needs to know
        where each word sits in time, so this keeps what `word_scores` throws
        away. Empty if unalignable -- mirrors `word_scores`'s own guard
        clauses and except path exactly.

        The word returned is the transcript's own -- romanisation is an input
        format MMS_FA needs, not something a caller publishing, CER-scoring or
        training on this text should ever see. `romanize` normally emits one
        Latin token per source word, so the alignment result (ordered the same
        way) can be zipped back against `text.split()`. But romanisation is not
        guaranteed length-preserving: a digit-only word (uroman's Latin output
        is filtered to a-z, so `2024` romanises to nothing) or a hyphenated
        word split into two tokens (`Google-ийн` -> `google`, `iyn`) changes
        the count. A mismatch means the positional pairing is no longer
        trustworthy, so this refuses rather than risk zipping the wrong
        original word to the wrong span.
        """
        import torch

        words = self.romanize(text)
        source_words = text.split()
        if not words or len(words) != len(source_words) or audio.size < SAMPLE_RATE // 10:
            return []
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                emission, _ = self._model(waveform)
                spans = self._aligner(emission[0], self._tokenizer(words))
        except Exception as exc:
            log.debug("alignment failed: %s", exc)
            return []

        # The emission is a downsampled view of the waveform -- MMS_FA emits
        # roughly one frame per 20 ms of audio, not one frame per sample -- so
        # a span's frame indices are not sample indices. This ratio rescales a
        # frame index back to a sample index at the waveform's own rate;
        # dividing by SAMPLE_RATE then turns that sample index into seconds.
        ratio = waveform.size(1) / emission.size(1)
        out = []
        for word, span in zip(source_words, spans, strict=True):
            sub = [s.score for s in span]
            if not sub:
                out.append((word, 0.0, 0.0, 0.0))
                continue
            start_s = span[0].start * ratio / SAMPLE_RATE
            end_s = span[-1].end * ratio / SAMPLE_RATE
            out.append((word, float(start_s), float(end_s), float(np.mean(sub))))
        return out
