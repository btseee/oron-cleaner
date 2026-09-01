"""The finalize step must persist what it derives.

Every other test in this suite hands `speaker_disjoint_split` and the summary a
list of dicts that already contains `gender_resolved` and `split`. That is the
shape of the bug rather than a test of it: the production path is
`CorpusWriter.add` writing a row, and `add` writes only `{clip_id, audio_path,
text, **meta}` -- neither derived field is in `meta`, because neither exists yet
when the clip is written.

So these tests never construct a record. They drive the real writer with the
fields the dataset loaders actually supply, run the real `finalize`, and read the
manifest back with the same function the downstream tools use. Two fatal defects
survived 205 passing tests by living in exactly the gap between those two things.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.constants import OUTPUT_SAMPLE_RATE  # noqa: E402
from pipeline.corpus import CorpusWriter, read_manifest  # noqa: E402

sf = pytest.importorskip("soundfile")
pytest.importorskip("pandas")  # finalize writes the parquet manifest

from clean_pipeline import finalize  # noqa: E402


def _audio(seconds=1.0):
    t = np.arange(int(seconds * OUTPUT_SAMPLE_RATE)) / OUTPUT_SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _write_corpus(root: Path, speakers=6, clips_per_speaker=4) -> Path:
    """A corpus written the way the pipeline writes one.

    `meta` carries only what a dataset loader knows: the declared gender, the
    speaker id, the measured duration. Nothing derived.
    """
    with CorpusWriter(root) as w:
        for s in range(speakers):
            for c in range(clips_per_speaker):
                w.add(
                    f"spk{s}_clip{c}",
                    _audio(),
                    f"өгүүлбэр {s}-{c}",
                    {
                        "client_id": f"speaker{s}",
                        # Common Voice's own vocabulary, not the resolved value.
                        "gender": "male_masculine" if s % 2 else "female_feminine",
                        "duration_s": 4.0,
                    },
                )
    return root


def test_finalize_persists_split_to_the_jsonl(tmp_path):
    """F0a: build_f5_dataset filters the JSONL on `split`.

    Written only to the parquet, the key is absent from every row the trainer's
    dataset builder reads, its filter matches nothing, and training silently
    consumes the test split.
    """
    finalize(_write_corpus(tmp_path))
    rows = read_manifest(tmp_path)
    assert rows, "finalize produced an empty manifest"
    assert all("split" in r for r in rows)
    assert set(r["split"] for r in rows) <= {"train", "validation", "test"}


def test_finalize_persists_gender_resolved_to_the_jsonl(tmp_path):
    """F0b: eval_mn.pick_reference and select_voices filter on this field.

    Absent, both match zero candidates -- the evaluation exits before
    synthesising anything and the shipped voices cannot be built.
    """
    finalize(_write_corpus(tmp_path))
    rows = read_manifest(tmp_path)
    assert all("gender_resolved" in r for r in rows)
    assert {r["gender_resolved"] for r in rows} == {"male", "female"}


def test_declared_gender_is_mapped_not_copied(tmp_path):
    """`male_masculine` is what Common Voice >= v17 emits, not what we store."""
    finalize(_write_corpus(tmp_path))
    for r in read_manifest(tmp_path):
        assert r["gender"] in {"male_masculine", "female_feminine"}   # untouched
        assert r["gender_resolved"] in {"male", "female"}             # derived
        assert r["gender_source"] == "declared"


def test_a_split_filter_over_the_manifest_is_not_a_no_op(tmp_path):
    """The consumer's actual operation, on the pipeline's actual output.

    `build_f5_dataset.py` reduces to this line. Before the manifest rewrite it
    returned every row, so the assertion that matters is that it now returns
    fewer.
    """
    finalize(_write_corpus(tmp_path))
    rows = read_manifest(tmp_path)
    train = [r for r in rows if r["split"] == "train"]
    assert 0 < len(train) < len(rows)


def test_the_jsonl_and_the_csv_splits_agree(tmp_path):
    """Both are written from the same in-memory splits; drift means one is stale."""
    import csv

    finalize(_write_corpus(tmp_path))
    rows = read_manifest(tmp_path)
    for split in ("train", "validation", "test"):
        name = "metadata.csv" if split == "train" else f"metadata_{split}.csv"
        with open(tmp_path / name, encoding="utf-8-sig") as f:
            csv_rows = list(csv.reader(f, delimiter="|"))[1:]
        assert len(csv_rows) == sum(1 for r in rows if r["split"] == split)


def test_rewriting_preserves_every_clip_and_its_text(tmp_path):
    """A rewrite that loses rows would be worse than the bug it fixes."""
    _write_corpus(tmp_path)
    before = {r["clip_id"]: r["text"] for r in read_manifest(tmp_path)}
    finalize(tmp_path)
    after = {r["clip_id"]: r["text"] for r in read_manifest(tmp_path)}
    assert after == before


def test_finalize_is_idempotent(tmp_path):
    """The runbook says to re-run `--finalize-only` to re-split.

    The second pass reads rows that already carry `split` and `gender_resolved`;
    it must not duplicate them or nest them.
    """
    finalize(_write_corpus(tmp_path))
    first = read_manifest(tmp_path)
    finalize(tmp_path)
    second = read_manifest(tmp_path)
    assert len(second) == len(first)
    assert {r["clip_id"] for r in second} == {r["clip_id"] for r in first}


def test_the_manifest_stays_one_json_object_per_line(tmp_path):
    """The rewrite goes through a temp file; a partial write would corrupt it."""
    finalize(_write_corpus(tmp_path))
    text = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8")
    for line in text.splitlines():
        assert isinstance(json.loads(line), dict)
    assert not list(tmp_path.glob("*.jsonl.tmp"))
