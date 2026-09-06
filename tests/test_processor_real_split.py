"""`_split_clip` driven through the real `split_at_silence`.

These live in their own module because they need uroman and torchaudio, and a
module-scope `pytest.importorskip` aborts collection of the entire file it sits
in. Both guards used to sit at the *bottom* of tests/test_processor.py, after 26
model-free tests were already defined, so on CI -- which installs neither -- that
whole file collapsed to a single skip and `process_split` had no coverage at
all: not the resume guard, not the crash guard, not clip-id stability, nothing.
Splitting the module is what keeps the model-free tests running where the model
stack is absent.

Every wiring test in tests/test_processor.py monkeypatches `split_at_silence`
with a fake that ignores `speech_spans`, and tests/test_recovery_split.py calls
the real function but always hands it spans of its own. The one argument that
differed between the two was the one production got wrong: `_split_clip` passed
`[]`, which makes every cut candidate uncorroborated, so the split path returned
None for every clip ever offered to it while the suite stayed green. These drive
the real function through the real caller.

No models: the aligner is a real `ForcedAligner` -- real uroman romanisation,
the real `word_timings` code path -- with only the 1.18 GB acoustic model
stubbed, the way tests/test_alignment.py already does it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("uroman", reason="uroman not installed")
pytest.importorskip("torchaudio", reason="torchaudio not installed")

from pipeline.constants import (  # noqa: E402
    MAX_DURATION_S,
    MIN_DURATION_S,
    SAMPLE_RATE,
)
from pipeline.processor import _split_clip  # noqa: E402

OVER_LENGTH_S = 24.0
# Speech, silence, speech -- the shape of a two-sentence clip that ran past the
# 20 s limit, which is the whole population this repair exists for.
SPEECH_A = (0.0, 9.5)
SILENCE = (9.5, 14.0)
SPEECH_B = (14.0, 23.0)


class _FakeSpan:
    def __init__(self, score: float, start: int, end: int) -> None:
        self.score = score
        self.start = start
        self.end = end


def _lay_out(n_tokens: int) -> list[tuple[float, float]]:
    """Place `n_tokens` words evenly across the two speech regions."""
    half = max(1, n_tokens // 2)
    spans = []
    for lo, hi, count in ((*SPEECH_A, half), (*SPEECH_B, n_tokens - half)):
        step = (hi - lo) / max(count, 1)
        spans += [(lo + i * step, lo + (i + 1) * step) for i in range(count)]
    return spans


def _real_aligner():
    """A real ForcedAligner with a stubbed acoustic model.

    The stub emits one frame per sample, so `word_timings`' frame-to-sample
    ratio is 1 and the spans below are sample indices -- which is what makes
    the seconds it returns predictable.
    """
    import torch
    import uroman

    from pipeline.alignment import ForcedAligner

    inst = ForcedAligner.__new__(ForcedAligner)
    inst.device = "cpu"
    inst._uroman = uroman.Uroman()
    inst._model = lambda waveform: (torch.zeros(1, waveform.size(1), 1), None)
    inst._tokenizer = lambda words: list(range(len(words)))
    inst._aligner = lambda emission0, tokens: [
        [_FakeSpan(0.9, int(lo * SAMPLE_RATE), int(hi * SAMPLE_RATE))]
        for lo, hi in _lay_out(len(tokens))
    ]
    return inst


class RealSplitFilter:
    """The three hooks `_split_clip` reaches into, and nothing else.

    `_run_vad` returns what Silero returns: sample indices, and the speech
    ratio path keeps them, so a clip padded with room tone still yields spans.
    """

    def __init__(self) -> None:
        from oron_tts.text import MongolianNormalizer

        self._aligner = _real_aligner()
        self._normalizer = MongolianNormalizer()

    NATIVE_SR = 48_000

    def _load_audio(self, audio_input):
        """16 kHz for the aligner, 48 kHz for the cut -- the real shape.

        Deliberately not the same array at the same rate: a split has to slice
        the source-rate signal, and a fake that returns one 16 kHz array for
        both would pass whether or not that wiring exists.
        """
        rng = np.random.default_rng(0)
        n = int(OVER_LENGTH_S * SAMPLE_RATE)
        work = (rng.standard_normal(n) * 0.3).astype(np.float32)
        native = np.repeat(work, self.NATIVE_SR // SAMPLE_RATE)
        return work, native, self.NATIVE_SR, ""

    def _run_vad(self, audio):
        spans = [
            {"start": int(SPEECH_A[0] * SAMPLE_RATE), "end": int(SPEECH_A[1] * SAMPLE_RATE)},
            {"start": int(SPEECH_B[0] * SAMPLE_RATE), "end": int(SPEECH_B[1] * SAMPLE_RATE)},
        ]
        return audio, spans, ""

    def normalized_text(self, text: str) -> str:
        return self._normalizer.normalize(text, strict=False)


def test_an_over_length_clip_really_splits():
    """The claim the whole feature rests on, with nothing faked between the
    caller and the cut."""
    text = "Сайн байна уу Өнөөдөр цаг агаар сайхан байна"
    parts = _split_clip(RealSplitFilter(), None, text)

    assert parts is not None, "the split path returned None for a splittable clip"
    assert len(parts) == 2
    assert " ".join(p[2] for p in parts) == text
    for audio, sr, _ in parts:
        # Source rate, not the aligner's 16 kHz. The cut is decided in seconds
        # on the working signal and applied to the native one, so a split clip
        # keeps the bandwidth its source had instead of inheriting the 16 kHz
        # the forced aligner happens to require.
        assert sr == RealSplitFilter.NATIVE_SR
        assert MIN_DURATION_S <= len(audio) / sr <= MAX_DURATION_S


def test_the_vad_spans_are_seconds_by_the_time_the_split_sees_them():
    """`_run_vad` is asked for sample indices; `split_at_silence` compares them
    against word timings, which are seconds. Passing the raw indices puts every
    silence tens of thousands of seconds past the end of a 24 s clip, so no cut
    candidate ever falls inside one and nothing splits."""
    filt = RealSplitFilter()
    text = "Сайн байна уу Өнөөдөр цаг агаар сайхан байна"
    assert _split_clip(filt, None, text) is not None

    raw = filt._run_vad

    def unconverted(audio):
        audio, spans, reason = raw(audio)
        return audio, [{"start": s["start"] * SAMPLE_RATE,
                        "end": s["end"] * SAMPLE_RATE} for s in spans], reason

    filt._run_vad = unconverted
    assert _split_clip(filt, None, text) is None


def test_a_clip_containing_a_digit_is_still_splittable():
    """Over-length clips are the long ones, so they are disproportionately the
    ones carrying dates and numbers. `word_timings` refuses when romanisation
    does not emit one token per source word, and a digit romanises to nothing,
    so on the raw transcript exactly the targeted population refused."""
    filt = RealSplitFilter()
    text = "Тэр 1990 онд төрсөн бөгөөд одоо энд ажиллаж байна"

    assert filt._aligner.word_timings(
        filt._load_audio(None)[0], text
    ) == [], "the raw transcript should be unalignable -- that is the bug"

    parts = _split_clip(filt, None, text)
    assert parts is not None
    assert " ".join(p[2] for p in parts) == filt.normalized_text(text)
    assert not any(c.isdigit() for p in parts for c in p[2])


def test_a_split_is_refused_when_the_vad_hears_no_silence():
    """Not weakened, only reachable: one continuous speech span still refuses."""
    filt = RealSplitFilter()
    filt._run_vad = lambda audio: (
        audio, [{"start": 0, "end": int(OVER_LENGTH_S * SAMPLE_RATE)}], ""
    )
    assert _split_clip(filt, None, "Сайн байна уу Өнөөдөр цаг агаар сайхан байна") is None


def test_a_clip_the_normaliser_refuses_is_not_split():
    """A transcript that cannot be published is not one to spend a split on --
    and the refusal must not escape into the 24-48 h loop as a crash."""
    filt = RealSplitFilter()
    filt.normalized_text = _refuse
    assert _split_clip(filt, None, "Сайн байна уу Өнөөдөр цаг агаар сайхан байна") is None


def _refuse(text: str) -> str:
    raise ValueError(f"no verified form for {text!r}")
