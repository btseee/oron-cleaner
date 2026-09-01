
from __future__ import annotations

import csv
import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from ..constants import OUTPUT_DIR
from ..corpus import CorpusWriter
from ..processor import process_split
from ..stats import CleaningStats

# AudioQualityFilter is only referenced as a type here. Importing it at
# runtime would drag Silero VAD, transformers, torchmetrics and the MMS_FA
# aligner into any process that merely wants to inspect a loader.
if TYPE_CHECKING:
    from ..audio_filter import AudioQualityFilter

log = logging.getLogger(__name__)

# Common Voice Scripted Speech 26.0 - Mongolian (CC0-1.0, 2.87 GB, MP3).
# Each language is now its own dataset on Mozilla Data Collective, so this ID is
# release- and language-specific. The previous ID returned "Dataset with id ...
# not found"; there is no list or search endpoint, so a dead ID can only be
# replaced by hand from the dataset's page URL. Downloads also require the
# account to have accepted the dataset terms in the web UI, and are capped at
# 30/day per organisation.
_DATASET_ID = "cmqinq6zs00x8nr07elg0nyrr"
_DATASET_NAME = "cv-corpus-26.0-2026-06-12"
_API_URL = f"https://mozilladatacollective.com/api/datasets/{_DATASET_ID}/download"
_CACHE_DIR = OUTPUT_DIR / "cv_cache"

# Cloudflare rejects requests' default User-Agent with error 1010 before the
# request ever reaches the API.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# validated only, deliberately.
#
#   validated    33,258 clips  46.9 h  520 speakers   <- human-confirmed
#   other        58,244 clips  77.1 h                 <- NOT confirmed; misreads
#   invalidated   2,963 clips   4.5 h                 <- rejected by voters
#   train/dev/test                                    <- subsets of validated
#
# train/dev/test are drawn from validated, so processing them alongside it runs
# every clip through the pipeline twice and publishes it twice. `other` is the
# unvalidated pool: including it is a volume-over-purity trade that a strict
# corpus cannot make. Splits for training are rebuilt speaker-disjoint later,
# which Common Voice's own splits are not.
_SPLITS = ["validated"]

_EXTRA_FIELDS = [
    "client_id", "path", "sentence", "up_votes", "down_votes",
    "age", "gender", "accents", "variant", "segment", "locale",
    "duration_tsv",
]


class _CvSplit:
    """Minimal dataset-like wrapper around rows parsed from a Common Voice TSV."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


def _get_download_url(api_key: str) -> str:
    resp = requests.post(
        _API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": _UA,
        },
        timeout=30,
    )
    if resp.status_code == 403:
        raise RuntimeError(
            f"Mozilla Data Collective refused dataset {_DATASET_ID}: {resp.text[:200]}\n"
            "Either the ID is stale (each Common Voice release publishes a new "
            "per-language dataset) or the account has not accepted this dataset's "
            "terms. Open https://mozilladatacollective.com/datasets/"
            f"{_DATASET_ID} to check, and take the current ID from that URL."
        )
    resp.raise_for_status()
    return resp.json()["downloadUrl"]


def _download_archive(api_key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = _CACHE_DIR / f"{_DATASET_NAME}-mn.tar.gz"

    if archive_path.exists():
        log.info("Archive already cached: %s", archive_path)
        return archive_path

    log.info("Fetching presigned URL from Mozilla Data Collective …")
    url = _get_download_url(api_key)

    log.info("Downloading %s Mongolian …", _DATASET_NAME)
    with requests.get(url, stream=True, timeout=600, headers={"User-Agent": _UA}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        next_log = 100 * 1024 * 1024  # log every 100 MB
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_log:
                    pct = f" ({100.0 * downloaded / total:.1f}%)" if total else ""
                    log.info("  %.0f MB downloaded%s", downloaded / 1024 ** 2, pct)
                    next_log += 100 * 1024 * 1024

    log.info("Download complete: %s", archive_path)
    return archive_path


def _extract_archive(archive_path: Path) -> Path:
    extract_dir = _CACHE_DIR / "extracted"

    if extract_dir.exists():
        existing = list(extract_dir.rglob("validated.tsv"))
        if existing:
            lang_dir = existing[0].parent
            log.info("Already extracted: %s", lang_dir)
            return lang_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s …", archive_path.name)
    with tarfile.open(archive_path, "r:gz") as tar:
        # filter="data" is the default from Python 3.14, but state it: without it
        # a tarball can write outside extract_dir via absolute paths or "..".
        tar.extractall(extract_dir, filter="data")

    lang_dir = next(extract_dir.rglob("validated.tsv")).parent
    log.info("Extracted to %s", lang_dir)
    return lang_dir


def _load_clip_durations(lang_dir: Path) -> dict[str, float]:
    """Exact per-clip durations shipped by Common Voice, in seconds.

    Preferred over decoding every mp3 to measure length: it is free, and it lets
    a split be sized before any audio is touched.
    """
    path = lang_dir / "clip_durations.tsv"
    if not path.exists():
        log.warning("clip_durations.tsv missing; durations will be unknown")
        return {}
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                out[row["clip"]] = int(row["duration[ms]"]) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _load_split(lang_dir: Path, split: str) -> _CvSplit | None:
    tsv_path = lang_dir / f"{split}.tsv"
    if not tsv_path.exists():
        return None

    durations = _load_clip_durations(lang_dir)
    clips_dir = lang_dir / "clips"
    rows: list[dict] = []
    skipped_downvoted = 0
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip_path = clips_dir / row["path"]
            if not clip_path.exists():
                continue
            # A down-vote means a listener judged the clip wrong. 13.2% of
            # validated clips carry one, and it is a far more reliable signal
            # than anything the audio filters can infer -- a human heard it.
            if int(row.get("down_votes") or 0) > 0:
                skipped_downvoted += 1
                continue
            rows.append({
                "duration_tsv": durations.get(row["path"], 0.0),
                "client_id":  row.get("client_id", ""),
                "path":       row.get("path", ""),
                "sentence":   row.get("sentence", ""),
                "up_votes":   int(row.get("up_votes") or 0),
                "down_votes": int(row.get("down_votes") or 0),
                "age":        row.get("age", ""),
                "gender":     row.get("gender", ""),
                "accents":    row.get("accents", ""),
                "variant":    row.get("variant", ""),
                "segment":    row.get("segment", ""),
                "locale":     row.get("locale", ""),
                "audio":      str(clip_path),
            })

    hours = sum(r["duration_tsv"] for r in rows) / 3600
    log.info(
        "  %s: %d clips (%.1f h) kept; %d dropped for down_votes>0",
        split, len(rows), hours, skipped_downvoted,
    )
    return _CvSplit(rows) if rows else None


def process_common_voice(
    filt: AudioQualityFilter, writer: CorpusWriter, *, api_key: str, resume: bool = True
) -> CleaningStats:
    log.info("Loading %s Mongolian from Mozilla Data Collective …", _DATASET_NAME)
    archive = _download_archive(api_key)
    lang_dir = _extract_archive(archive)

    all_stats = CleaningStats("common_voice_26_mn")
    for split_name in _SPLITS:
        split = _load_split(lang_dir, split_name)
        if split is None:
            continue
        all_stats.merge(process_split(
            split,
            filt,
            writer,
            audio_field="audio",
            text_field="sentence",
            dataset_name="cv",
            split_name=split_name,
            extra_fields=_EXTRA_FIELDS,
            resume=resume,
        ))
    return all_stats
