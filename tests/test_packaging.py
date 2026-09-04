"""Every one of these failures was invisible until the dataset page was opened."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402

from pipeline.packaging import ROW_GROUP_ROWS, VIEWER_SCAN_LIMIT, pack_corpus  # noqa: E402


def _corpus(tmp_path: Path, n: int = 12, splits=("train", "withheld")) -> Path:
    d = tmp_path / "corpus"
    (d / "wavs").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        name = f"clip_{i:03d}.wav"
        sf.write(d / "wavs" / name, rng.normal(0, 0.1, 24000).astype("float32"), 24000)
        rows.append({
            "clip_id": f"clip_{i:03d}",
            "audio_path": f"wavs/{name}",
            "text": f"өгүүлбэр {i}",
            "client_id": f"spk{i % 3}",
            "gender_resolved": "male" if i % 2 else "female",
            "duration_s": 1.0,
            "up_votes": i,
            "split": splits[i % len(splits)],
        })
    (d / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return d


def test_every_manifest_field_becomes_a_column(tmp_path):
    """An earlier packer hardcoded its column list and would have dropped
    sentence, up_votes, age, accents and locale."""
    out = tmp_path / "pq"
    pack_corpus(_corpus(tmp_path), out)
    table = pq.read_table(sorted(out.glob("train-*.parquet"))[0])
    for col in ("audio", "text", "client_id", "gender_resolved", "up_votes", "split"):
        assert col in table.column_names, f"{col} was dropped"
    assert table.column("text")[0].as_py().startswith("өгүүлбэр")


def test_splits_become_separate_files(tmp_path):
    """The Hub reads splits from the file layout, not from a column. Shipping
    everything as train-*.parquet showed one split of 15.1k rows."""
    out = tmp_path / "pq"
    counts = pack_corpus(_corpus(tmp_path), out)
    assert set(counts) == {"train", "withheld"}
    assert sorted(p.name.split("-")[0] for p in out.glob("*.parquet")) == ["train", "withheld"]


def test_row_groups_stay_under_the_viewer_scan_limit(tmp_path):
    """450 MB shards with one row group each were refused with
    TooBigContentError against a 300 MB limit."""
    out = tmp_path / "pq"
    # More rows per split than ROW_GROUP_ROWS, or there is only ever one group
    # and the test asserts nothing about the bug it exists for.
    pack_corpus(_corpus(tmp_path, n=2 * ROW_GROUP_ROWS + 40), out)
    f = pq.ParquetFile(sorted(out.glob("*.parquet"))[0])
    biggest = max(f.metadata.row_group(i).total_byte_size
                  for i in range(f.metadata.num_row_groups))
    assert biggest < VIEWER_SCAN_LIMIT
    assert f.metadata.num_row_groups > 1, (
        "one row group per file is the whole bug: the shard size becomes the "
        "minimum read")


def test_audio_survives_as_playable_wav(tmp_path):
    import io
    out = tmp_path / "pq"
    pack_corpus(_corpus(tmp_path), out)
    table = pq.read_table(sorted(out.glob("train-*.parquet"))[0])
    info = sf.info(io.BytesIO(table.column("audio")[0].as_py()["bytes"]))
    assert info.samplerate == 24000
    assert info.duration > 0


def test_refuses_a_corpus_with_no_manifest_rows(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "manifest.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        pack_corpus(d, tmp_path / "out")


def test_upload_packs_rather_than_sending_loose_wavs():
    """upload_large_folder on 15,092 clips is 15,092 requests; the Hub allows
    1000 per 5 minutes."""
    src = (ROOT / "pipeline" / "upload.py").read_text(encoding="utf-8")
    assert "pack_corpus" in src
    assert "upload_large_folder" not in src
