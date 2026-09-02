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

SENTENCES = [
    "Сайн байна уу", "Өнөөдөр цаг агаар сайхан байна", "Монгол улс Азид оршдог",
    "Улаанбаатар бол нийслэл хот", "Тэр ном уншиж байна", "Би сургуульдаа явлаа",
    "Ус бол амьдралын үндэс", "Морь хурдан гүйдэг", "Хүүхдүүд гадаа тоглож байна",
    "Энэ жил ургац сайн боллоо",
]


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
                    f"{SENTENCES[(s * 20 + c) % len(SENTENCES)]} {s}{c}",
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
    """The placeholder is computed for a corpus that does not exist, so it must
    fail against any real one rather than quietly setting a wrong LR decay."""
    import yaml

    data = tmp_path / "oron_mn_pinyin"
    data.mkdir(exist_ok=True)
    durations = [float(r["duration_s"]) for r in read_manifest(corpus)
                 if r["split"] == "train"]
    (data / "duration.json").write_text(json.dumps({"duration": durations}),
                                        encoding="utf-8")
    config = yaml.safe_load(
        (ORON_TTS / "configs" / "f5tts_mn.yaml").read_text(encoding="utf-8")
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
