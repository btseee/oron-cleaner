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

# Lower case only: coverage is about which sounds the model has seen, and
# case is orthographic. Spelled out because a naive [a-ya] range is
# U+0410-U+044F and excludes o U+04E9 and y U+04AF.
MN_LETTERS_LOWER = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"


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


def write_eval_sentences(
    root: Path | str, records: list[dict], reserved: set[str]
) -> Path:
    """Write the sentences withheld from training, one per line.

    This is the CER target text. It is deliberately not `metadata_test.csv`:
    that file's job is reference prompts and the ground-truth topline, which
    need an unseen *speaker*, while this needs unseen *text*. Keeping them in
    separate files is what lets both be held out without intersecting to
    nothing.
    """
    from .speakers import text_key

    root = Path(root)
    seen: set[str] = set()
    lines: list[str] = []
    for r in records:
        text = (r.get("text") or "").strip()
        key = text_key(text)
        if key in reserved and key not in seen:
            seen.add(key)
            lines.append(text)

    out = root / "eval_sentences.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log.info("Wrote %s (%d held-out sentences)", out, len(lines))
    return out


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


def text_diversity(splits: dict[str, list[dict]]) -> str:
    """How much of the language the corpus actually shows the model.

    Audio hours are the number everyone quotes and they say nothing about this.
    Common Voice mn has 28,858 usable clips over 6,062 distinct sentences --
    4.76x repetition -- so a corpus can be 40 h and still show the model a
    narrow slice of Mongolian orthography.

    Character coverage is the floor: a letter that never appears in training
    cannot be pronounced, and the model has only a barely-trained embedding row
    for it. Bigram coverage is the more honest measure, since Mongolian
    phonotactics live in the transitions -- vowel harmony is a constraint
    between adjacent vowels, not a property of one.
    """
    from collections import Counter

    from .speakers import text_key

    texts = [r.get("text") or "" for rs in splits.values() for r in rs]
    if not texts:
        return ""

    distinct = len({text_key(t) for t in texts})
    chars = Counter(c for t in texts for c in t.lower() if c in MN_LETTERS_LOWER)
    bigrams = Counter(
        t[i:i + 2].lower() for t in texts for i in range(len(t) - 1)
        if t[i].lower() in MN_LETTERS_LOWER and t[i + 1].lower() in MN_LETTERS_LOWER
    )
    missing = sorted(set(MN_LETTERS_LOWER) - set(chars))
    # A letter seen a handful of times is nearly as bad as one never seen: the
    # model cannot learn its realisation from a dozen examples.
    rare = sorted(c for c, n in chars.items() if n < 100)

    lines = [
        "Text diversity:",
        f"  clips                 {len(texts):,}",
        f"  distinct sentences    {distinct:,}  "
        f"({len(texts) / max(1, distinct):.2f}x repetition)",
        f"  letters covered       {len(chars)}/{len(MN_LETTERS_LOWER)}",
        f"  letter bigrams        {len(bigrams):,} distinct",
    ]
    if missing:
        lines.append(f"  NEVER APPEARS         {' '.join(missing)}")
    if rare:
        lines.append(f"  under 100 occurrences {' '.join(rare)}")
    if not missing and not rare:
        lines.append("  every Mongolian letter appears at least 100 times")
    return "\n".join(lines)


def _per_source_cer(splits: dict[str, list[dict]]) -> str:
    """CER by source, because the gate's scorer is not neutral between them.

    `bayartsogt/wav2vec2-large-xlsr-mongolian` is, per its own model card,
    fine-tuned on Common Voice Mongolian -- and it is both the corpus's CER gate
    and the evaluation scorer. So it has seen Common Voice's speakers and
    sentences and has not seen FLEURS' or MBSpeech's: the gate is systematically
    lenient on one source and strict on the others, which biases corpus
    composition by source rather than by quality.

    That cannot be fixed with a threshold. It can be *seen*: if Common Voice's
    median CER sits well below the others on audio of comparable quality, the
    gap is the contamination, not the recording.
    """
    import statistics

    by_source: dict[str, list[float]] = {}
    for rs in splits.values():
        for r in rs:
            source = str(r.get("clip_id", "")).split("_")[0] or "?"
            cer = r.get("cer")
            if cer is not None:
                by_source.setdefault(source, []).append(float(cer))
    if not by_source:
        return ""

    out = ["CER by source (the gate's recogniser trained on Common Voice):",
           f"  {'source':<12}{'clips':>8}{'median':>9}{'mean':>8}"]
    for source, values in sorted(by_source.items()):
        out.append(f"  {source:<12}{len(values):>8}{statistics.median(values):>9.3f}"
                   f"{statistics.fmean(values):>8.3f}")
    out.append("  A markedly lower median for cv is contamination, not quality.")
    return "\n".join(out)


def _recovered_share(splits: dict[str, list[dict]]) -> str:
    """How much of the corpus exists only because a repair ran.

    A corpus that is 30% repaired clips has a different character from one that
    is not, even when every clip passed the same gates. `recovered_by` has
    reached the manifest since splitting shipped, so the number was always
    derivable -- but nothing derived it, and a number nobody prints is a number
    nobody checks. Printed even when it is zero, so "none was recovered" is a
    measurement rather than a line that happened not to appear.
    """
    from collections import Counter

    rows = [r for rs in splits.values() for r in rs]
    counts = Counter(name for r in rows if (name := r.get("recovered_by")))
    lines = ["Recovered clips (repaired, then re-gated on the same thresholds):"]
    if not counts:
        return lines[0] + "\n  none — every kept clip passed as recorded"
    for name, n in sorted(counts.items()):
        lines.append(f"  {name:<22}{n:>8,}  ({100.0 * n / max(len(rows), 1):.1f}% of kept)")
    return "\n".join(lines)


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

    lines += ["", text_diversity(splits), "", _per_source_cer(splits),
              "", _recovered_share(splits)]

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
