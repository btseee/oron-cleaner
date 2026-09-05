"""The loop that actually builds the corpus.

`process_split` had no tests. Every pure function it calls did -- which is the
familiar shape: the parts are covered and the wiring between them is not, and
the wiring is where two fatal defects already lived.

The filter is stubbed. What is under test here is resume, the limit, clip-id
stability, rejection accounting and the normalisation path, none of which needs
a model.

**Nothing heavier than soundfile may be imported at module scope here, and no
`pytest.importorskip` may name anything CI does not install.** A module-scope
guard aborts collection of the whole file, so two such lines at the bottom --
below 26 already-defined tests -- reduced this entire module to one skip on CI,
which installs neither uroman nor torchaudio. The tests that genuinely need the
aligner live in tests/test_processor_real_split.py, where the guard costs only
themselves.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("soundfile")

from pipeline.clip_result import ClipResult  # noqa: E402
from pipeline.constants import OUTPUT_SAMPLE_RATE  # noqa: E402
from pipeline.corpus import CorpusWriter, read_manifest  # noqa: E402
from pipeline.processor import process_split  # noqa: E402


class FakeFilter:
    """Passes everything except clip ids listed in `reject`.

    `normalize` may raise for a given text, which is how the normaliser's
    refusal reaches this loop.
    """

    def __init__(self, reject=(), raise_on=()):
        self.reject = set(reject)
        self.raise_on = set(raise_on)
        self.seen: list[str] = []

    def process_clip(self, audio, ground_truth, measure_all=False):
        self.seen.append(ground_truth)
        if ground_truth in self.reject:
            return ClipResult(passed=False, reject_stage="snr", reject_reason="too noisy")
        return ClipResult(
            passed=True,
            audio_normalized=np.zeros(OUTPUT_SAMPLE_RATE // 2, dtype=np.float32),
            duration_s=0.5,
            snr_db=20.0,
            align_score=0.9,
        )

    def normalized_text(self, text: str) -> str:
        if text in self.raise_on:
            raise ValueError(f"no verified form for {text!r}")
        return text.upper()


def split(n: int):
    return [{"audio": None, "raw_transcription": f"text {i}", "path": f"clip{i}"}
            for i in range(n)]


class SplittableFilter:
    """A too-long rejection path plus the `_load_audio`/`_run_vad`/`_aligner`
    hooks `_split_clip` needs. `split_at_silence` itself is monkeypatched in
    the wiring tests below, so `_aligner` is never called through there -- it
    only has to exist. The tests that drive the real function build a real
    aligner instead.
    """

    def __init__(self, too_long=(), reject=()):
        self.too_long = set(too_long)
        self.reject = set(reject)
        self.seen: list[str] = []
        self._aligner = object()

    def process_clip(self, audio, ground_truth, measure_all=False):
        self.seen.append(ground_truth)
        if ground_truth in self.too_long:
            return ClipResult(
                passed=False, reject_stage="duration", reject_reason="too_long_25.00s"
            )
        if ground_truth in self.reject:
            return ClipResult(passed=False, reject_stage="snr", reject_reason="too noisy")
        return ClipResult(
            passed=True,
            audio_normalized=np.zeros(OUTPUT_SAMPLE_RATE // 2, dtype=np.float32),
            duration_s=0.5,
            snr_db=20.0,
            align_score=0.9,
        )

    def _load_audio(self, audio_input):
        return np.zeros(1, dtype=np.float32), ""

    def _run_vad(self, audio):
        # Silero's shape: sample indices, two speech spans with a silence
        # between them. `_split_clip` converts these to seconds before the
        # split sees them.
        return audio, [{"start": 0, "end": 16_000},
                       {"start": 32_000, "end": 48_000}], ""

    def normalized_text(self, text: str) -> str:
        return text.upper()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """process_split writes stats and rejection logs under OUTPUT_DIR."""
    monkeypatch.setattr("pipeline.processor.OUTPUT_DIR", tmp_path / "out")
    yield


def run(corpus: Path, dataset, filt, **kw):
    with CorpusWriter(corpus) as w:
        return process_split(
            dataset, filt, w,
            audio_field="audio", text_field="raw_transcription",
            dataset_name="fake", split_name="train",
            extra_fields=[], **kw,
        )


def test_passing_clips_are_written(tmp_path):
    stats = run(tmp_path / "corpus", split(5), FakeFilter())
    assert stats.passed == 5
    assert len(read_manifest(tmp_path / "corpus")) == 5


def test_the_published_text_is_the_normalised_one(tmp_path):
    """The corpus text, the CER reference and the training target are one
    string -- so the writer must store what the filter normalised, not the raw
    input re-derived."""
    run(tmp_path / "corpus", split(2), FakeFilter())
    assert {r["text"] for r in read_manifest(tmp_path / "corpus")} == {"TEXT 0", "TEXT 1"}


def test_rejected_clips_are_not_written_but_are_counted(tmp_path):
    stats = run(tmp_path / "corpus", split(4), FakeFilter(reject={"text 1", "text 2"}))
    assert stats.passed == 2
    assert stats.total == 4
    assert len(read_manifest(tmp_path / "corpus")) == 2


def test_a_normalisation_refusal_is_a_rejection_not_a_pass(tmp_path):
    """The normaliser refuses numeral suffixes it cannot expand without
    guessing. Counting those as passes would inflate the yield, and publishing
    the raw text would put unexpanded digits in the corpus."""
    stats = run(tmp_path / "corpus", split(3), FakeFilter(raise_on={"text 1"}))
    assert stats.passed == 2
    assert stats.total == 3
    assert "text 1" not in {r["text"].lower() for r in read_manifest(tmp_path / "corpus")}


def test_a_refusal_is_logged_under_its_own_stage(tmp_path):
    run(tmp_path / "corpus", split(3), FakeFilter(raise_on={"text 1"}))
    log = next((tmp_path / "out" / "logs").glob("rejected_*.csv"))
    with open(log, encoding="utf-8") as f:
        stages = [row["stage"] for row in csv.DictReader(f)]
    assert "normalize" in stages


def test_a_crashing_clip_does_not_stop_the_run(tmp_path):
    """A 24-48 h pass must not die on one bad file."""
    class Exploding(FakeFilter):
        def process_clip(self, audio, ground_truth, measure_all=False):
            if ground_truth == "text 1":
                raise RuntimeError("decode failed")
            return super().process_clip(audio, ground_truth, measure_all)

    stats = run(tmp_path / "corpus", split(4), Exploding())
    assert stats.total == 4
    assert stats.passed == 3


def test_limit_caps_the_work(tmp_path):
    filt = FakeFilter()
    run(tmp_path / "corpus", split(50), filt, limit=7)
    assert len(filt.seen) == 7


def test_resume_skips_clips_already_written(tmp_path):
    """Keyed by clip id, so a restart re-does no work regardless of the order
    the dataset enumerates in."""
    corpus = tmp_path / "corpus"
    run(corpus, split(5), FakeFilter())
    second = FakeFilter()
    run(corpus, split(8), second)
    assert second.seen == ["text 5", "text 6", "text 7"]
    assert len(read_manifest(corpus)) == 8


def test_clip_ids_are_stable_across_runs(tmp_path):
    """Resume compares ids, so an id derived from enumeration order would make
    a restart re-process everything."""
    a, b = tmp_path / "a", tmp_path / "b"
    run(a, split(3), FakeFilter())
    run(b, split(3), FakeFilter())
    assert ([r["clip_id"] for r in read_manifest(a)]
            == [r["clip_id"] for r in read_manifest(b)])


def test_clip_ids_are_namespaced_by_dataset(tmp_path):
    """Sources share one corpus, so two clips named the same must not collide."""
    corpus = tmp_path / "corpus"
    run(corpus, split(2), FakeFilter())
    assert all(r["clip_id"].startswith("fake_") for r in read_manifest(corpus))


def test_the_stats_checkpoint_survives_a_restart(tmp_path):
    corpus = tmp_path / "corpus"
    run(corpus, split(4), FakeFilter())
    resumed = run(corpus, split(6), FakeFilter())
    # The first run's four are counted from the checkpoint, not re-processed.
    assert resumed.total == 6
    assert resumed.passed == 6


def test_nothing_is_written_outside_the_temporary_directory(tmp_path):
    """Guards the fixture itself. process_split writes stats checkpoints and
    rejection logs under OUTPUT_DIR; unpatched, a test run litters the repo."""
    run(tmp_path / "corpus", split(3), FakeFilter(reject={"text 0"}))
    assert (tmp_path / "out" / "logs").is_dir()
    assert (tmp_path / "out" / "checkpoints").is_dir()
    assert not list(Path("output").glob("logs/rejected_fake_*"))


# ── split_at_silence wiring ────────────────────────────────────────────────


def test_a_clean_split_produces_its_segments_and_not_the_source(tmp_path, monkeypatch):
    def fake_split(audio, sr, text, *, aligner, speech_spans):
        return [
            (np.zeros(1, dtype=np.float32), 16000, f"{text} a"),
            (np.zeros(1, dtype=np.float32), 16000, f"{text} b"),
        ]

    monkeypatch.setattr("pipeline.processor.split_at_silence", fake_split)
    filt = SplittableFilter(too_long={"text 0"})
    stats = run(tmp_path / "corpus", split(1), filt)

    manifest = read_manifest(tmp_path / "corpus")
    assert {r["clip_id"] for r in manifest} == {"fake_clip0_p0", "fake_clip0_p1"}
    # The source clip failed on its own merits and is counted as that
    # rejection; the two segments are counted separately, on theirs.
    assert stats.total == 3
    assert stats.passed == 2


def test_a_failing_segment_does_not_block_its_passing_siblings(tmp_path, monkeypatch):
    def fake_split(audio, sr, text, *, aligner, speech_spans):
        return [
            (np.zeros(1, dtype=np.float32), 16000, "text 0 a"),
            (np.zeros(1, dtype=np.float32), 16000, "text 0 b"),
        ]

    monkeypatch.setattr("pipeline.processor.split_at_silence", fake_split)
    filt = SplittableFilter(too_long={"text 0"}, reject={"text 0 b"})
    run(tmp_path / "corpus", split(1), filt)

    manifest = read_manifest(tmp_path / "corpus")
    assert {r["clip_id"] for r in manifest} == {"fake_clip0_p0"}


def test_a_rejection_for_any_other_reason_is_never_split(tmp_path, monkeypatch):
    calls = []

    def fake_split(audio, sr, text, *, aligner, speech_spans):
        calls.append(text)
        return None

    monkeypatch.setattr("pipeline.processor.split_at_silence", fake_split)
    filt = SplittableFilter(reject={"text 0"})
    stats = run(tmp_path / "corpus", split(1), filt)

    assert calls == []
    assert stats.passed == 0
    assert read_manifest(tmp_path / "corpus") == []


def test_a_refused_split_leaves_the_clip_rejected(tmp_path, monkeypatch):
    """split_at_silence returning None is a refusal, not a partial success --
    the source clip must stay exactly as rejected as it already was."""
    monkeypatch.setattr("pipeline.processor.split_at_silence", lambda *a, **k: None)
    filt = SplittableFilter(too_long={"text 0"})
    stats = run(tmp_path / "corpus", split(1), filt)

    assert stats.total == 1
    assert stats.passed == 0
    assert read_manifest(tmp_path / "corpus") == []


def test_a_segment_is_never_split_again(tmp_path, monkeypatch):
    calls = []

    def fake_split(audio, sr, text, *, aligner, speech_spans):
        calls.append(text)
        return [(np.zeros(1, dtype=np.float32), 16000, "text 0 a")]

    monkeypatch.setattr("pipeline.processor.split_at_silence", fake_split)
    # The segment's own ground truth would itself look like a too-long
    # rejection if it were ever fed back through the split path -- proving
    # it is not is exactly what this asserts.
    filt = SplittableFilter(too_long={"text 0", "text 0 a"})
    run(tmp_path / "corpus", split(1), filt)

    # Normalised, because that is the text the split works on and publishes;
    # SplittableFilter.normalized_text upper-cases.
    assert calls == ["TEXT 0"]
    assert read_manifest(tmp_path / "corpus") == []


def test_a_segment_carries_its_recovery_provenance(tmp_path, monkeypatch):
    def fake_split(audio, sr, text, *, aligner, speech_spans):
        return [(np.zeros(1, dtype=np.float32), 16000, f"{text} a")]

    monkeypatch.setattr("pipeline.processor.split_at_silence", fake_split)
    filt = SplittableFilter(too_long={"text 0"})
    run(tmp_path / "corpus", split(1), filt)

    manifest = read_manifest(tmp_path / "corpus")
    assert manifest[0]["recovered_by"] == "split_at_silence"


@pytest.mark.parametrize(
    ("rejected_segments", "kept"),
    [
        (set(), 2),
        ({"text 0 a"}, 1),
        ({"text 0 b"}, 1),
        ({"text 0 a", "text 0 b"}, 0),
    ],
    ids=["every-segment-passes", "first-segment-fails",
         "last-segment-fails", "every-segment-fails"],
)
def test_a_split_source_is_not_re_split_on_every_restart(
    tmp_path, monkeypatch, rejected_segments, kept
):
    """Resume skips a clip whose id is in the writer, and a split source's own
    id is never written -- only `_p0`, `_p1`, and only for the segments that
    passed. So each restart re-decoded the source, re-split it, re-ran the model
    stack on every segment and appended a second copy of every count.

    Recognising `_p0` in the writer covered only the case where segment 0
    happened to pass. Measured over three runs of one clip with `_p0` rejected
    and `_p1` passing: total 3, 6, 9 and passed 1, 2, 3 -- and with every
    segment rejected, nothing named after the source exists at all. Hence all
    three cases here: what has to survive a restart is that the source was
    *consumed*, which is independent of what its segments earned."""
    def fake_split(audio, sr, text, *, aligner, speech_spans):
        return [
            (np.zeros(1, dtype=np.float32), 16000, "text 0 a"),
            (np.zeros(1, dtype=np.float32), 16000, "text 0 b"),
        ]

    monkeypatch.setattr("pipeline.processor.split_at_silence", fake_split)
    corpus = tmp_path / "corpus"

    totals, passes, work = [], [], []
    for _ in range(3):
        filt = SplittableFilter(too_long={"text 0"}, reject=rejected_segments)
        stats = run(corpus, split(1), filt)
        totals.append(stats.total)
        passes.append(stats.passed)
        work.append(len(filt.seen))

    assert totals == [3, 3, 3]
    assert passes == [kept, kept, kept]
    assert work[1:] == [0, 0], "a restart re-ran the model stack on a split clip"
    assert len(read_manifest(corpus)) == kept


def test_resume_false_re_splits_a_source_it_already_consumed(tmp_path, monkeypatch):
    """The record of consumed sources is a resume artifact, so `resume=False`
    must not let a previous run's copy of it suppress work in this one."""
    def fake_split(audio, sr, text, *, aligner, speech_spans):
        return [(np.zeros(1, dtype=np.float32), 16000, "text 0 a")]

    monkeypatch.setattr("pipeline.processor.split_at_silence", fake_split)
    corpus = tmp_path / "corpus"
    run(corpus, split(1), SplittableFilter(too_long={"text 0"}))

    second = SplittableFilter(too_long={"text 0"})
    run(tmp_path / "fresh", split(1), second, resume=False)
    assert second.seen == ["text 0", "text 0 a"]


def test_a_segment_keeps_its_provenance_when_normalisation_refuses(tmp_path, monkeypatch):
    """The refusal path builds a fresh ClipResult, so the field has to be
    carried across by hand. The rejection log is where the yield of a repair is
    read off -- silently dropping it there understates what splitting cost."""
    def fake_split(audio, sr, text, *, aligner, speech_spans):
        return [(np.zeros(1, dtype=np.float32), 16000, "text 0 a")]

    monkeypatch.setattr("pipeline.processor.split_at_silence", fake_split)

    # Recorded from the stats, because the refusal path *replaces* the result
    # object -- watching the one handed in would show the field it never lost.
    from pipeline.stats import CleaningStats

    recorded = []
    real_record = CleaningStats.record

    def spy(self, result):
        recorded.append((result.reject_stage, result.recovered_by))
        return real_record(self, result)

    monkeypatch.setattr(CleaningStats, "record", spy)

    filt = SplittableFilter(too_long={"text 0"})
    filt.normalized_text = _raise_for("text 0 a")
    run(tmp_path / "corpus", split(1), filt)

    assert ("normalize", "split_at_silence") in recorded


def _raise_for(bad: str):
    def normalized_text(text: str) -> str:
        if text == bad:
            raise ValueError(f"no verified form for {text!r}")
        return text.upper()
    return normalized_text


def test_a_segment_is_measured_on_every_gate_during_calibration(tmp_path, monkeypatch):
    """calibrate.py's whole value is that every clip is scored by every gate.
    A segment stopped at its first failure biases the per-gate rates it feeds,
    silently, in the one mode whose output is used to move thresholds."""
    def fake_split(audio, sr, text, *, aligner, speech_spans):
        return [(np.zeros(1, dtype=np.float32), 16000, "text 0 a")]

    monkeypatch.setattr("pipeline.processor.split_at_silence", fake_split)

    class Recording(SplittableFilter):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.modes: list[tuple[str, bool]] = []

        def process_clip(self, audio, ground_truth, measure_all=False):
            self.modes.append((ground_truth, measure_all))
            return super().process_clip(audio, ground_truth, measure_all)

    from pipeline.calibrate import Calibration

    filt = Recording(too_long={"text 0"})
    run(tmp_path / "corpus", split(1), filt, calibration=Calibration())
    assert filt.modes == [("text 0", True), ("text 0 a", True)]
