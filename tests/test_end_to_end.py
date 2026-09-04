"""The whole chain, across both repos, with nothing stubbed but the audio.

    CorpusWriter  ->  finalize  ->  select_splits  ->  metadata.csv  ->  preflight

Every unit in this chain had tests before, and two fatal defects still lived in
the wiring between them: `split` and `gender_resolved` were computed, written to
a parquet, and never read by the tools that needed them. Unit tests could not
see that, because each side was tested against its own idea of the contract.

This is also not hypothetical maintenance. Running this chain by hand found a
third defect the unit tests missed -- the evaluation splits were taking half the
corpus on a small one -- which is now pinned by
`test_training_keeps_the_majority_of_a_small_corpus`.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("soundfile")
pytest.importorskip("pandas")

# oron-tts sits beside this repo locally and is checked out as .oron-tts in CI.
ORON_TTS = next(
    (p for p in (ROOT.parent / "oron-tts", ROOT / ".oron-tts") if (p / "scripts").is_dir()),
    None,
)
if ORON_TTS is None:
    pytest.skip("oron-tts checkout not found", allow_module_level=True)
sys.path.insert(0, str(ORON_TTS / "scripts"))

from build_f5_dataset import select_splits, write_metadata_csv  # noqa: E402
from preflight import check_epochs, check_vocab  # noqa: E402

from clean_pipeline import finalize  # noqa: E402
from pipeline.constants import OUTPUT_SAMPLE_RATE  # noqa: E402
from pipeline.corpus import CorpusWriter, read_manifest  # noqa: E402

# Real Mongolian, pulled from Wikipedia and split by whether the normaliser
# accepts it. Synthetic strings never exercise a refusal, and the refusal path
# is where the corpus writer and the text layer actually meet: a clip whose
# transcript cannot be expanded has to be dropped rather than published with
# digits in it.
_FIXTURES = json.loads(
    (Path(__file__).parent / "data" / "mn_sentences.json").read_text(encoding="utf-8")
)
SENTENCES: list[str] = _FIXTURES["normalises"]
REFUSED_SENTENCES: list[str] = _FIXTURES["refuses"]


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * OUTPUT_SAMPLE_RATE)) / OUTPUT_SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype("float32")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    """A corpus written and finalised exactly the way the pipeline does it."""
    root = tmp_path_factory.mktemp("e2e") / "corpus"
    with CorpusWriter(root) as w:
        for s in range(12):
            for c in range(20):
                duration = 4.0 + (c % 5)
                w.add(
                    f"cv_spk{s}_{c}", _tone(duration),
                    SENTENCES[(s * 20 + c) % len(SENTENCES)],
                    {"client_id": f"speaker{s}",
                     "gender": "male_masculine" if s % 2 else "female_feminine",
                     "duration_s": duration, "cer": 0.1, "align_score": 0.9,
                     "dnsmos_ovr": 3.6, "bandwidth_hz": 7500.0},
                )
    finalize(root)
    return root


def test_the_corpus_finalises(corpus):
    assert len(read_manifest(corpus)) == 240


def test_training_keeps_the_clear_majority(corpus):
    """The defect this test was written after finding."""
    rows = read_manifest(corpus)
    train = [r for r in rows if r["split"] == "train"]
    assert len(train) / len(rows) > 0.5


def test_the_split_filter_the_trainer_uses_actually_filters(corpus):
    """F0a. The guard degraded to "keep everything", so raw.arrow was built from
    the whole corpus including test, and nothing said so."""
    rows = read_manifest(corpus)
    train = select_splits(rows, "train")
    assert 0 < len(train) < len(rows)
    assert {r["split"] for r in train} == {"train"}


def test_no_evaluation_clip_reaches_training(corpus):
    """The property the whole split exists for, checked on the real output."""
    rows = read_manifest(corpus)
    train_ids = {r["clip_id"] for r in select_splits(rows, "train")}
    for split in ("validation", "test", "withheld"):
        held = {r["clip_id"] for r in rows if r["split"] == split}
        assert not (held & train_ids), split


def test_no_speaker_spans_train_and_an_evaluation_split(corpus):
    rows = read_manifest(corpus)
    train = {r["client_id"] for r in rows if r["split"] == "train"}
    for split in ("validation", "test"):
        held = {r["client_id"] for r in rows if r["split"] == split}
        assert not (held & train), split


def test_no_evaluation_sentence_occurs_in_training(corpus):
    """F2. Measured at 99.6% before the text holdout existed."""
    from pipeline.speakers import text_key

    rows = read_manifest(corpus)
    train_texts = {text_key(r["text"]) for r in rows if r["split"] == "train"}
    held = [ln.strip() for ln in
            (corpus / "eval_sentences.txt").read_text(encoding="utf-8").splitlines()]
    assert held
    assert not ({text_key(t) for t in held if t} & train_texts)


def test_the_metadata_csv_the_trainer_reads_is_the_train_split(corpus, tmp_path):
    """F5-TTS validates this header exactly and rejects relative paths."""
    rows = select_splits(read_manifest(corpus), "train")
    out = tmp_path / "metadata.csv"
    written = write_metadata_csv(corpus, rows, out)
    assert written == len(rows)
    with open(out, encoding="utf-8-sig") as f:
        table = list(csv.reader(f, delimiter="|"))
    assert table[0] == ["audio_file", "text"]
    for path, text in table[1:]:
        assert Path(path).is_absolute() and Path(path).exists()
        assert text.strip()


def test_preflight_accepts_the_vocabulary_the_build_installs(corpus, tmp_path):
    """build_f5_dataset replaces the 2545-line vocab prepare_csv_wavs copies.
    Without that, unknown ids map to 0 -- the SPACE token."""
    import shutil

    data = tmp_path / "oron_mn_pinyin"
    data.mkdir()
    shutil.copy(ORON_TTS / "data" / "oron_mn_pinyin" / "vocab.txt", data / "vocab.txt")
    problems: list[str] = []
    check_vocab(data / "vocab.txt", problems, [])
    assert not problems, problems


def test_preflight_refuses_the_shipped_epochs_for_this_corpus(corpus, tmp_path):
    """An epoch count carried over from another corpus must be refused.

    `epochs` sets the LR decay horizon, so a stale value does not fail loudly --
    it trains on the wrong schedule. The shipped config therefore holds the
    string `PLACEHOLDER`, which cannot be mistaken for a number; this substitutes
    a plausible one and checks preflight still refuses it."""
    import yaml

    data = tmp_path / "oron_mn_pinyin"
    data.mkdir(exist_ok=True)
    durations = [float(r["duration_s"]) for r in read_manifest(corpus)
                 if r["split"] == "train"]
    (data / "duration.json").write_text(json.dumps({"duration": durations}),
                                        encoding="utf-8")
    config = yaml.safe_load(
        (ORON_TTS / "configs" / "oron.yaml").read_text(encoding="utf-8")
        .replace("epochs: PLACEHOLDER", "epochs: 51")
    )
    problems: list[str] = []
    check_epochs(config, data, problems, [])
    assert problems and "LR decay length" in problems[0]


def test_provenance_identifies_this_corpus(corpus):
    """Two builds agreeing on the content hash produced the same corpus; two
    agreeing only on the policy version did not."""
    payload = json.loads((corpus / "provenance.json").read_text(encoding="utf-8"))
    assert payload["corpus_content_hash"]
    assert payload["normaliser_fingerprint"]
    assert payload["clips"] == 240
    assert set(payload["splits"]) == {"train", "validation", "test", "withheld"}


def test_a_refusable_transcript_never_reaches_the_corpus(tmp_path, monkeypatch):
    """The junction the unit tests cannot see.

    `oron_tts` refuses numeral suffixes it cannot expand without guessing, and
    `process_split` has to turn that into a dropped clip. Publishing the raw
    text instead would put digits into the transcript, which is then the
    published corpus, the CER reference *and* the training target -- one string
    by design, so one wrong string three times.

    These sentences are real Mongolian that the normaliser genuinely refuses,
    not constructed ones.
    """
    from pipeline.clip_result import ClipResult
    from pipeline.processor import process_split

    assert REFUSED_SENTENCES, "fixture lost its refusing sentences"

    class RealFilter:
        """The real normaliser, everything else stubbed."""

        def __init__(self) -> None:
            from oron_tts.text import MongolianNormalizer

            self._n = MongolianNormalizer()

        def process_clip(self, audio, ground_truth, measure_all=False):
            return ClipResult(passed=True, audio_normalized=_tone(4.0), duration_s=4.0,
                              align_score=0.9, snr_db=20.0)

        def normalized_text(self, text: str) -> str:
            return self._n.normalize(text, strict=False)

    mixed = SENTENCES[:8] + REFUSED_SENTENCES
    dataset = [{"audio": None, "raw_transcription": t, "path": f"c{i}"}
               for i, t in enumerate(mixed)]

    # process_split writes stats checkpoints and rejection logs under
    # OUTPUT_DIR, and resumes from them. Unpatched, this test littered the repo
    # and then read its own previous run's totals back on the next invocation.
    monkeypatch.setattr("pipeline.processor.OUTPUT_DIR", tmp_path / "out")

    root = tmp_path / "corpus"
    with CorpusWriter(root) as writer:
        stats = process_split(
            dataset, RealFilter(), writer,
            audio_field="audio", text_field="raw_transcription",
            dataset_name="cv", split_name="train", extra_fields=[],
        )

    published = [r["text"] for r in read_manifest(root)]
    assert len(published) == 8, "a refused transcript was published"
    assert stats.total == len(mixed)
    assert stats.passed == 8
    # And nothing published still carries a digit, which is what the refusal
    # was protecting against.
    assert not [t for t in published if any(ch.isdigit() for ch in t)]
