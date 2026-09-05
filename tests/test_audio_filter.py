"""The gate cascade in `process_clip`.

M7: the two functions that actually build the corpus had no tests, only the
pure helpers they call. `process_split` is covered in test_processor.py; this
covers the other one.

`AudioQualityFilter.__init__` loads roughly 3 GB of models, so the instance is
built without it and the seven methods `process_clip` uses are stubbed. What is
under test is the cascade -- ordering, short-circuiting, which failure is
reported, and the calibration mode that must *not* short-circuit -- none of
which is about the models.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.audio_filter import AudioQualityFilter  # noqa: E402
from pipeline.constants import (  # noqa: E402
    DNSMOS_MIN_OVR,
    MAX_DURATION_S,
    MIN_DURATION_S,
    SAMPLE_RATE,
)


def _speech(seconds: float = 5.0) -> np.ndarray:
    """Tone in the middle, near-silence at the edges.

    The silence is load-bearing: SNR is measured against the *true* non-speech
    regions, so a buffer that is speech end to end has no noise floor and is
    rejected -- correctly, and it is the code saying so, not the stub.
    """
    # Broadband, not a tone: the bandwidth gate wants a lowpass shelf above
    # 6 kHz, and a 180 Hz sine measures 211 Hz -- rejected, correctly, by the
    # code rather than by the stub.
    n = int(seconds * SAMPLE_RATE)
    audio = (0.2 * np.random.default_rng(0).standard_normal(n)).astype(np.float32)
    edge = n // 10
    audio[:edge] *= 0.001
    audio[-edge:] *= 0.001
    return audio


def _speech_bounds(audio: np.ndarray) -> list[dict]:
    edge = len(audio) // 10
    return [{"start": edge, "end": len(audio) - edge}]


@pytest.fixture
def filt():
    """A filter with no models: every stage answers "fine" until a test says
    otherwise, so each test changes exactly one thing."""
    f = object.__new__(AudioQualityFilter)
    f.calls = []

    def load(audio_input):
        f.calls.append("load")
        return (audio_input, "") if audio_input is not None else (None, "decode_failed")

    def vad(audio):
        f.calls.append("vad")
        # Silero's shape, which dsp.speech_ratio indexes by name -- a tuple here
        # passed the stub and failed the code, which is the wrong way round.
        return audio, _speech_bounds(audio), ""

    def dnsmos(audio):
        f.calls.append("dnsmos")
        # (ok, scores, reason) -- the signature's order, not (ok, reason, scores).
        return True, {"dnsmos_sig": 4.0, "dnsmos_bak": 4.0,
                      "dnsmos_ovr": DNSMOS_MIN_OVR + 0.5, "dnsmos_p808": 4.0}, ""

    def verify(audio, text):
        f.calls.append("verify")
        return True, 0.05, 1.0, "тест", ""

    f._load_audio = load
    f._run_vad = vad
    f._score_dnsmos = dnsmos
    f._verify_reading = verify
    class _Aligner:
        def score(self, audio, text):
            f.calls.append("align")
            return 0.95

    from oron_tts.text import MongolianNormalizer

    f._prepare_output_audio = lambda a: np.asarray(a, dtype=np.float32)
    f._aligner = _Aligner()
    f._normalizer = MongolianNormalizer()
    return f


# ── the happy path ────────────────────────────────────────────────────────────

def test_a_clean_clip_passes_and_carries_its_audio(filt):
    result = filt.process_clip(_speech(), "Сайн байна уу")
    assert result.passed
    assert result.reject_stage == ""
    assert result.audio_normalized.size > 1


# ── short-circuiting ──────────────────────────────────────────────────────────

def test_a_load_failure_stops_before_anything_else_runs(filt):
    """The 24-48 h pass is bounded by how early it can stop."""
    result = filt.process_clip(None, "текст")
    assert not result.passed
    assert result.reject_stage == "load"
    assert filt.calls == ["load"]


def test_a_short_clip_never_reaches_the_models(filt):
    result = filt.process_clip(_speech(MIN_DURATION_S / 2), "текст")
    assert result.reject_stage == "duration"
    assert "vad" not in filt.calls and "dnsmos" not in filt.calls


def test_a_long_clip_is_rejected_too(filt):
    """DynamicBatchSampler silently drops anything over the frame budget, so a
    long clip that survives here disappears later with nothing logged."""
    result = filt.process_clip(_speech(MAX_DURATION_S + 5), "текст")
    assert result.reject_stage == "duration"


def test_the_first_failure_is_the_one_reported(filt):
    """So the rejection log says which gate to loosen."""
    result = filt.process_clip(_speech(0.1), "текст")
    assert result.reject_stage == "duration"
    assert result.reject_reason.startswith("too_short")


# ── calibration mode ──────────────────────────────────────────────────────────

def test_measure_all_scores_every_gate_instead_of_stopping(filt):
    """The whole point of --calibrate: in a normal run a clip rejected for
    duration is never scored for DNSMOS, so the per-gate rates are not
    comparable."""
    result = filt.process_clip(_speech(MIN_DURATION_S / 2), "текст", measure_all=True)
    assert not result.passed
    assert "dnsmos" in filt.calls
    assert "verify" in filt.calls


def test_measure_all_records_every_failure_not_just_the_first(filt):
    def bad_dnsmos(audio):
        filt.calls.append("dnsmos")
        return False, {"dnsmos_sig": 1.0, "dnsmos_bak": 1.0,
                       "dnsmos_ovr": 1.0, "dnsmos_p808": 1.0}, "ovr_1.0"

    filt._score_dnsmos = bad_dnsmos
    result = filt.process_clip(_speech(0.1), "текст", measure_all=True)
    stages = {g.split(":")[0] for g in result.failed_gates}
    assert {"duration", "dnsmos"} <= stages


def test_a_vad_failure_is_terminal_even_in_calibration_mode(filt):
    """Without speech bounds there is nothing downstream can measure, so this
    one cannot be scored past."""
    def no_speech(audio):
        filt.calls.append("vad")
        return None, [], "no_speech_detected"

    filt._run_vad = no_speech
    result = filt.process_clip(_speech(), "текст", measure_all=True)
    assert result.reject_stage == "vad"
    assert "dnsmos" not in filt.calls


# ── a failing clip carries no audio ──────────────────────────────────────────

def test_a_rejected_clip_does_not_carry_audio(filt):
    """`CorpusWriter.add` writes whatever it is handed; a rejected clip must not
    arrive with a usable buffer."""
    result = filt.process_clip(None, "текст")
    assert result.audio_normalized.size == 1


def test_unmeasurable_snr_is_not_a_rejection():
    """estimate_snr returns NaN when there is under 0.1 s of non-speech to
    compute a noise floor from. On continuous narration that means the clip is
    spoken end to end -- a property of the reading, not of the recording.

    Treating it as a failure dropped 41% of a 200-clip MBSpeech sample whose
    DNSMOS-BAK was 3.22 at the 5th percentile against a 2.5 floor. The noise
    judgement belongs to DNSMOS-BAK, which measures background directly."""
    import numpy as np

    from pipeline.dsp import estimate_snr

    sr = 24000
    speech = np.random.default_rng(0).normal(0, 0.1, sr * 2).astype("float32")
    # Speech covering the whole clip: no non-speech region to measure.
    wall_to_wall = [{"start": 0, "end": len(speech)}]
    assert np.isnan(estimate_snr(speech, wall_to_wall, sr)), (
        "a clip with no silence must report NaN, not a number"
    )

    src = (ROOT / "pipeline" / "audio_filter.py").read_text(encoding="utf-8")
    i = src.index("snr = estimate_snr(audio, timestamps)")
    block = src[i:i + 1400]
    nan_branch = block[block.index("if np.isnan(snr):"):block.index("else:")]
    assert 'note("snr"' not in nan_branch, (
        "an unmeasurable SNR must not be recorded as an SNR rejection"
    )
    assert "return done()" not in nan_branch, (
        "an unmeasurable SNR must not drop the clip before DNSMOS has judged it"
    )


def test_a_path_torchaudio_cannot_decode_falls_back_to_ffmpeg(tmp_path, monkeypatch):
    """A missing codec must not read as an unusable corpus.

    torchaudio 2.9 dropped its own backends for torchcodec, which needs
    FFmpeg's shared libraries rather than the ffmpeg binary. On a machine with
    the binary and not the libraries, every mp3 raised here and the pipeline
    reported it as a `load` rejection -- 250 of 250 clips, which reads as a
    corpus that is 100% unusable rather than as a decoder that is absent.
    """
    import numpy as np

    import pipeline.audio_filter as af

    calls = {"ffmpeg": 0}

    def refuse(*a, **k):
        raise RuntimeError("no torchaudio backend")

    def fake_ffmpeg(path):
        calls["ffmpeg"] += 1
        return np.zeros(SAMPLE_RATE, dtype="float32"), SAMPLE_RATE

    import torchaudio
    monkeypatch.setattr(torchaudio, "load", refuse)
    monkeypatch.setattr(af, "_decode_with_ffmpeg", fake_ffmpeg)

    filt = af.AudioQualityFilter.__new__(af.AudioQualityFilter)
    audio, err = af.AudioQualityFilter._load_audio(filt, tmp_path / "clip.mp3")
    assert err == "", f"fallback did not run: {err}"
    assert calls["ffmpeg"] == 1
    assert audio is not None and len(audio) == SAMPLE_RATE


# ── homoglyphs ────────────────────────────────────────────────────────────────
#
# A Latin letter inside a Cyrillic word is an encoding error, and it reaches the
# model as its own embedding row whether or not the clip carrying it was ever
# rejected. So this is normalisation, not recovery: it belongs on the path every
# transcript takes, and these drive it there rather than through a repair table.

def test_a_latin_letter_inside_a_cyrillic_word_is_corrected():
    """`о` U+006F in an otherwise-Cyrillic word has exactly one correct reading.
    It is an encoding error, not an ambiguity, so fixing it guesses nothing."""
    from pipeline.audio_filter import repair_homoglyphs

    assert repair_homoglyphs("Mонгол хэл") == "Монгол хэл"


def test_the_ukrainian_i_is_corrected():
    """U+0456 passes `is_representable` because it is in the vocabulary, so it
    reaches the model as a distinct embedding row for a letter nobody typed."""
    from pipeline.audio_filter import repair_homoglyphs

    assert repair_homoglyphs("саін") == "сайн"


def test_an_all_latin_word_is_left_alone():
    """An English proper noun in a Mongolian sentence is not an encoding error.

    Hyphen-joined segments are judged one at a time: `Google-ийн` puts a Latin
    word next to a Cyrillic suffix, and proximity is not evidence."""
    from pipeline.audio_filter import repair_homoglyphs

    assert repair_homoglyphs("Google-ийн") == "Google-ийн"


def test_clean_cyrillic_is_returned_unchanged():
    from pipeline.audio_filter import repair_homoglyphs

    assert repair_homoglyphs("Сайн байна уу") == "Сайн байна уу"


def test_every_homoglyph_maps_to_a_cyrillic_letter():
    """A mapping that produced another Latin letter would move the problem."""
    from pipeline.audio_filter import HOMOGLYPHS

    for latin, cyrillic in HOMOGLYPHS.items():
        assert ord(latin) < 0x400, f"{latin!r} is not the Latin side"
        assert 0x400 <= ord(cyrillic) <= 0x4FF, f"{cyrillic!r} is not Cyrillic"


def test_the_published_text_has_its_homoglyphs_repaired(filt):
    """The wiring, not the function: `normalized_text` is what the corpus
    writer publishes and what the CER gate scored against, so a repair that
    only existed in a repair table would never reach either."""
    assert "M" not in filt.normalized_text("Mонгол хэл")


def test_the_alignment_gate_scores_the_repaired_text(filt):
    """One string, three uses -- published, CER-scored, aligned against. A
    repair reaching only the published copy would score the corpus against
    text it does not contain."""
    seen = []

    class _Recording:
        def score(self, audio, text):
            seen.append(text)
            return 0.95

    filt._aligner = _Recording()
    filt.process_clip(_speech(), "Mонгол хэл")
    assert seen and "M" not in seen[0]
