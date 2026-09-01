"""Incremental corpus writer and finaliser.

Two problems solved by the same change.

**Memory.** `process_split` accumulated every passing record -- including the
decoded 24 kHz float32 audio -- in a list, then `Dataset.from_list` materialised
a second copy. At Common Voice scale that is roughly 29k clips x 5 s x 24000 x
4 B ~ 14 GB before encoding, and it would OOM before finishing. Clips are now
written to disk the moment they pass, and only their metadata stays in memory.

**Output contract.** F5-TTS's `prepare_csv_wavs.py` wants a `metadata.csv` with
a literal `audio_file|text` header and absolute paths. Producing it directly
removes a conversion step that would otherwise have to re-read every clip.

Layout:

    <root>/
      wavs/<clip_id>.wav          24 kHz mono, edge-trimmed, peak-normalised
      manifest.jsonl              one row per clip, written as clips pass
      metadata.csv                audio_file|text  (train split, absolute paths)
      metadata_<split>.csv        the other splits
      manifest.parquet            full metadata for analysis and voice selection
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .constants import OUTPUT_SAMPLE_RATE

log = logging.getLogger(__name__)

# prepare_csv_wavs.py validates this header exactly and rejects anything else.
F5_HEADER = ["audio_file", "text"]


class CorpusWriter:
    """Append clips to an on-disk corpus, holding no audio in memory."""

    def __init__(self, root: Path | str, *, resume: bool = True) -> None:
        self.root = Path(root)
        self.wav_dir = self.root / "wavs"
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"

        self._seen: set[str] = set()
        if resume and self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        self._seen.add(json.loads(line)["clip_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            log.info("Resuming corpus at %s with %d clips", self.root, len(self._seen))

        self._file = open(  # noqa: SIM115 - open for the run, closed in close()
            self.manifest_path, "a" if resume else "w", encoding="utf-8", newline="\n"
        )

    def __contains__(self, clip_id: str) -> bool:
        return clip_id in self._seen

    def add(self, clip_id: str, audio: np.ndarray, text: str, meta: dict[str, Any]) -> None:
        """Write one clip's audio and append its manifest row.

        `text` must already be normalised: it is what gets published, what CER
        was scored against, and what training will read.
        """
        if clip_id in self._seen:
            return
        wav_path = self.wav_dir / f"{clip_id}.wav"
        sf.write(wav_path, audio, OUTPUT_SAMPLE_RATE, subtype="PCM_16")

        row = {
            "clip_id": clip_id,
            # Relative on disk so the corpus stays movable; metadata.csv is
            # written with absolute paths later, as F5-TTS requires.
            "audio_path": str(wav_path.relative_to(self.root)).replace("\\", "/"),
            "text": text,
            **meta,
        }
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()
        self._seen.add(clip_id)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> CorpusWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_manifest(root: Path | str) -> list[dict]:
    """Load the metadata only -- no audio, so this stays small."""
    root = Path(root)
    path = root / "manifest.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_f5_metadata(root: Path | str, records: list[dict], split: str = "train") -> Path:
    """Emit the `audio_file|text` CSV that prepare_csv_wavs.py consumes.

    Paths are absolute because prepare_csv_wavs.py raises otherwise, and the
    delimiter is "|" with that exact header because it validates both.
    """
    root = Path(root).resolve()
    name = "metadata.csv" if split == "train" else f"metadata_{split}.csv"
    out = root / name
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(F5_HEADER)
        for r in records:
            writer.writerow([str(root / r["audio_path"]), r["text"]])
    log.info("Wrote %s (%d clips)", out, len(records))
    return out


def rewrite_manifest(root: Path | str, splits: dict[str, list[dict]]) -> Path:
    """Rewrite manifest.jsonl with the fields finalize() derived.

    `split` and `gender_resolved` are computed in memory by `speaker_disjoint_split`
    and `propagate_gender`, and used to be written **only** into the parquet. Every
    downstream consumer reads the JSONL:

      * `build_f5_dataset.py` filters on `split` -- with the key absent its guard
        silently fell through and training consumed the whole corpus, test included;
      * `eval_mn.py:pick_reference` and `select_voices.py` filter on
        `gender_resolved` -- with the key absent they matched nothing;
      * `upload.py` reports per-gender hours -- always 0.0.

    Writing the derived fields back is what makes the split real.
    """
    root = Path(root)
    path = root / "manifest.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for name, records in splits.items():
            for r in records:
                f.write(json.dumps({**r, "split": name}, ensure_ascii=False) + "\n")
                n += 1
    tmp.replace(path)
    log.info("Rewrote %s (%d rows, split and gender persisted)", path, n)
    return path


def write_parquet_manifest(root: Path | str, splits: dict[str, list[dict]]) -> Path:
    """Full metadata for analysis, quality auditing and voice selection."""
    import pandas as pd

    root = Path(root)
    rows = [{**r, "split": name} for name, rs in splits.items() for r in rs]
    frame = pd.DataFrame(rows)
    out = root / "manifest.parquet"
    frame.to_parquet(out, index=False)
    log.info("Wrote %s (%d rows, %d columns)", out, len(frame), len(frame.columns))
    return out


def summarise(splits: dict[str, list[dict]]) -> str:
    """Human-readable corpus summary, including the acceptance criteria."""
    lines = ["=== Corpus summary ===", ""]
    lines.append(f"{'split':<12}{'clips':>8}{'hours':>9}{'speakers':>10}"
                 f"{'male h':>9}{'female h':>10}")

    def hours(rs, gender=None):
        return sum(
            float(r.get("duration_s") or 0.0) for r in rs
            if gender is None or r.get("gender_resolved") == gender
        ) / 3600.0

    total_male = total_female = total_h = 0.0
    for name, rs in splits.items():
        spk = len({str(r.get("client_id") or "") for r in rs})
        mh, fh, h = hours(rs, "male"), hours(rs, "female"), hours(rs)
        total_male += mh
        total_female += fh
        total_h += h
        lines.append(f"{name:<12}{len(rs):>8,}{h:>9.1f}{spk:>10}{mh:>9.1f}{fh:>10.1f}")

    male_speakers = len({
        str(r.get("client_id")) for rs in splits.values() for r in rs
        if r.get("gender_resolved") == "male"
    })
    lines += [
        "",
        f"total: {total_h:.1f} h   male {total_male:.1f} h   female {total_female:.1f} h",
        "",
        "Acceptance criteria:",
        f"  total >= 25 h            {'PASS' if total_h >= 25 else 'FAIL'}  ({total_h:.1f} h)",
        f"  male  >= 5 h             {'PASS' if total_male >= 5 else 'FAIL'}  ({total_male:.1f} h)",
        f"  male speakers >= 3       {'PASS' if male_speakers >= 3 else 'FAIL'}  ({male_speakers})",
    ]
    return "\n".join(lines)
