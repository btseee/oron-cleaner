"""Pack a finished corpus into parquet shards the Hub can actually serve.

Uploading the corpus directory as it sits on disk does not work, and the failure
modes are all silent until someone opens the dataset page.

  * **One request per clip.** `upload_large_folder` on a 15,092-clip corpus blew
    HuggingFace's 1000-requests-per-5-minutes quota; 9,994 wavs landed and the
    card and metadata never did.
  * **Loose wavs are not a dataset.** The viewer does not join `manifest.jsonl`
    to a directory of wav files, so the page showed an `audio` column and
    nothing else -- no text, no speaker, no measurements.
  * **One row group per file.** `pq.write_table` defaults to a single row group,
    so a 450 MB shard is a 450 MB minimum read and the viewer refused it with
    `TooBigContentError` against its 300 MB scan limit.
  * **One file set for every split.** The Hub takes splits from the file layout,
    not from a column, so shipping everything as `train-*.parquet` showed one
    split of 15.1k rows however many the corpus actually had.

Each of those is fixed here and asserted on read-back, because every one of them
was invisible until publication.
"""

from __future__ import annotations

import io
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ~200 KB per clip at 24 kHz mono, so 100 rows is a ~20 MB row group -- an order
# of magnitude under the viewer's 300 MB scan limit.
ROW_GROUP_ROWS = 100
SHARD_BYTES = 450 * 1_000_000
VIEWER_SCAN_LIMIT = 300_000_000


def _arrow_type(pa, kinds: Counter) -> Any:
    if not kinds:
        return pa.string()
    top = kinds.most_common(1)[0][0]
    return {"bool": pa.bool_(), "int": pa.int64(), "float": pa.float64()}.get(top, pa.string())


def pack_corpus(corpus_dir: Path | str, out_dir: Path | str) -> dict[str, int]:
    """Write `out_dir/<split>-NNNNN.parquet`, audio inline, every field a column.

    Returns clips written per split. Raises rather than writing a shard whose
    row groups the viewer would refuse.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf

    corpus_dir, out_dir = Path(corpus_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in
            (corpus_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"{corpus_dir}/manifest.jsonl is empty")

    # Scan every row: a field can be absent on the first clip and present later,
    # and an earlier hardcoded column list silently dropped sentence, up_votes,
    # age, accents and locale while looking for a field CV does not have.
    kinds: dict[str, Counter] = {}
    for r in rows:
        for k, v in r.items():
            if v is not None:
                kinds.setdefault(k, Counter())[type(v).__name__] += 1
    fields = sorted(kinds)
    types = {k: _arrow_type(pa, kinds[k]) for k in fields}
    schema = pa.schema(
        [("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())]))]
        + [(k, types[k]) for k in fields]
    )

    def coerce(k: str, v: Any) -> Any:
        if v is None:
            return None
        t = types[k]
        if t == pa.int64():
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        if t == pa.float64():
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        if t == pa.bool_():
            return bool(v)
        return str(v)

    def flush(buf: list[dict], split: str, idx: int) -> int:
        if not buf:
            return idx
        path = out_dir / f"{split}-{idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buf, schema=schema), path,
                       compression="snappy",
                       row_group_size=ROW_GROUP_ROWS, write_page_index=True)
        log.info("  %s  %d clips  %.1f MB", path.name, len(buf), path.stat().st_size / 1e6)
        return idx + 1

    by_split: dict[str, list[dict]] = {}
    for r in rows:
        by_split.setdefault(r.get("split") or "train", []).append(r)

    written: dict[str, int] = {}
    missing = 0
    for split, split_rows in sorted(by_split.items()):
        buf: list[dict] = []
        size = idx = 0
        for r in split_rows:
            wav = corpus_dir / (r.get("audio_path") or "")
            if not wav.is_file():
                missing += 1
                continue
            raw = wav.read_bytes()
            rec: dict[str, Any] = {"audio": {"bytes": raw, "path": wav.name}}
            for k in fields:
                rec[k] = coerce(k, r.get(k))
            buf.append(rec)
            size += len(raw)
            if size >= SHARD_BYTES:
                idx = flush(buf, split, idx)
                buf, size = [], 0
        idx = flush(buf, split, idx)
        written[split] = len(split_rows) - (0 if idx else len(split_rows))
    if missing:
        log.warning("%d manifest rows had no wav on disk", missing)

    _verify(out_dir, pq, sf)
    log.info("packed %s", {k: len(v) for k, v in sorted(by_split.items())})
    return {k: len(v) for k, v in by_split.items()}


def _verify(out_dir: Path, pq, sf) -> None:
    """A shard the viewer refuses, or one that lost the text, is worse than none."""
    shards = sorted(out_dir.glob("*.parquet"))
    if not shards:
        raise ValueError(f"no shards written to {out_dir}")
    probe = pq.ParquetFile(shards[0])
    biggest = max(probe.metadata.row_group(i).total_byte_size
                  for i in range(probe.metadata.num_row_groups))
    if biggest >= VIEWER_SCAN_LIMIT:
        raise ValueError(
            f"{shards[0].name}: largest row group is {biggest/1e6:.0f} MB, over the "
            f"viewer's {VIEWER_SCAN_LIMIT/1e6:.0f} MB scan limit -- lower ROW_GROUP_ROWS")
    table = pq.read_table(shards[0])
    if "text" not in table.column_names:
        raise ValueError("packed shard has no text column")
    if not table.column("text")[0].as_py():
        raise ValueError("packed shard has an empty text column")
    audio = table.column("audio")[0].as_py()["bytes"]
    info = sf.info(io.BytesIO(audio))
    log.info("  verified: %d row groups, largest %.1f MB, first clip %.2f s @ %d Hz",
             probe.metadata.num_row_groups, biggest / 1e6, info.duration, info.samplerate)
