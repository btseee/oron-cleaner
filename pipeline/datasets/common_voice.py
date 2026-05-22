import csv
import logging
import tarfile
from pathlib import Path

import requests
from datasets import Audio, Dataset, DatasetDict, Features, Value

from ..audio_filter import AudioQualityFilter
from ..constants import OUTPUT_DIR, SAMPLE_RATE
from ..processor import process_split
from ..stats import CleaningStats

log = logging.getLogger(__name__)

_DATASET_ID = "cmn2e7nxs01k6mm07you99zve"
_API_URL = f"https://mozilladatacollective.com/api/datasets/{_DATASET_ID}/download"
_CACHE_DIR = OUTPUT_DIR / "cv_cache"
_SPLITS = ["train", "dev", "test", "validated", "other"]

_EXTRA_FIELDS = [
    "client_id", "path", "sentence", "up_votes", "down_votes",
    "age", "gender", "accents", "variant", "segment", "locale",
]

_FEATURES = Features({
    "client_id":        Value("string"),
    "path":             Value("string"),
    "audio":            Audio(sampling_rate=SAMPLE_RATE),
    "sentence":         Value("string"),
    "up_votes":         Value("int32"),
    "down_votes":       Value("int32"),
    "age":              Value("string"),
    "gender":           Value("string"),
    "accents":          Value("string"),
    "variant":          Value("string"),
    "segment":          Value("string"),
    "locale":           Value("string"),
    "snr_db":           Value("float32"),
    "mean_f0_hz":       Value("float32"),
    "pitch_confidence": Value("float32"),
    "dnsmos_sig":       Value("float32"),
    "dnsmos_bak":       Value("float32"),
    "dnsmos_ovr":       Value("float32"),
    "dnsmos_p808":      Value("float32"),
    "cer":              Value("float32"),
    "asr_transcript":   Value("string"),
    "duration_s":       Value("float32"),
})


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
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["downloadUrl"]


def _download_archive(api_key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = _CACHE_DIR / "common-voice-25-mn.tar.gz"

    if archive_path.exists():
        log.info("Archive already cached: %s", archive_path)
        return archive_path

    log.info("Fetching presigned URL from Mozilla Data Collective …")
    url = _get_download_url(api_key)

    log.info("Downloading Common Voice 25 Mongolian …")
    with requests.get(url, stream=True, timeout=600) as r:
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
        tar.extractall(extract_dir)

    lang_dir = next(extract_dir.rglob("validated.tsv")).parent
    log.info("Extracted to %s", lang_dir)
    return lang_dir


def _load_split(lang_dir: Path, split: str) -> _CvSplit | None:
    tsv_path = lang_dir / f"{split}.tsv"
    if not tsv_path.exists():
        return None

    clips_dir = lang_dir / "clips"
    rows: list[dict] = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip_path = clips_dir / row["path"]
            if not clip_path.exists():
                continue
            rows.append({
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

    log.info("  %s: %d clips found", split, len(rows))
    return _CvSplit(rows) if rows else None


def process_common_voice(
    filt: AudioQualityFilter, *, api_key: str, resume: bool = True
) -> tuple[DatasetDict, CleaningStats]:
    log.info("Loading Common Voice 25 Mongolian from Mozilla Data Collective …")
    archive = _download_archive(api_key)
    lang_dir = _extract_archive(archive)

    all_stats = CleaningStats("common_voice_25_mn")
    split_map: dict[str, Dataset] = {}

    for split_name in _SPLITS:
        split = _load_split(lang_dir, split_name)
        if split is None:
            continue

        passing, stats = process_split(
            split,
            filt,
            audio_field="audio",
            text_field="sentence",
            dataset_name="cv",
            split_name=split_name,
            extra_fields=_EXTRA_FIELDS,
            resume=resume,
        )
        all_stats.merge(stats)

        if passing:
            out_name = "other_clean" if split_name == "other" else split_name
            split_map[out_name] = Dataset.from_list(passing, features=_FEATURES)
            log.info("  %s → %d clips passed", out_name, len(passing))

    return DatasetDict(split_map), all_stats
