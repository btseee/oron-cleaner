import logging
from pathlib import Path

from datasets import DatasetDict
from huggingface_hub import HfApi

from .constants import OUTPUT_DIR
from .stats import CleaningStats

log = logging.getLogger(__name__)

# Maps the short dataset key used throughout the pipeline to its HF repo,
# dataset card content, and commit message.
_REPO_CONFIG: dict[str, tuple[str, Path, str]] = {
    "cv": (
        "btsee/common-voices-25-mn",
        Path(__file__).parent / "cards" / "common_voice.md",
        "Initial upload: cleaned Common Voice 25 Mongolian",
    ),
    "fleurs": (
        "btsee/fleurs-mn",
        Path(__file__).parent / "cards" / "fleurs.md",
        "Initial upload: cleaned FLEURS Mongolian",
    ),
    "ws": (
        "btsee/worldspeech-mn",
        Path(__file__).parent / "cards" / "worldspeech.md",
        "Initial upload: cleaned WorldSpeech Mongolian",
    ),
}


def upload_dataset(key: str, cleaned_ds: DatasetDict, stats: CleaningStats) -> None:
    if not cleaned_ds:
        log.warning("No clean clips for %s — skipping upload.", key)
        return

    # Drop any splits that are empty so HF never sees a format mismatch
    # (empty splits produce no parquet files, causing FileFormatMismatchBetweenSplitsError)
    non_empty = DatasetDict({
        split: ds for split, ds in cleaned_ds.items() if len(ds) > 0
    })
    if not non_empty:
        log.warning("All splits empty for %s — skipping upload.", key)
        return

    skipped = set(cleaned_ds.keys()) - set(non_empty.keys())
    if skipped:
        log.warning("Skipping empty split(s) for %s: %s", key, skipped)

    repo_id, card_path, commit_msg = _REPO_CONFIG[key]
    api = HfApi()

    log.info("Creating / verifying repo: %s", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    log.info("Pushing %d split(s) to %s …", len(non_empty), repo_id)
    non_empty.push_to_hub(repo_id, commit_message=commit_msg, max_shard_size="500MB")

    log.info("Uploading dataset card …")
    api.upload_file(
        path_or_fileobj=card_path.read_bytes(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    report_text = stats.report()
    report_filename = f"cleaning_report_{key}.txt"
    (OUTPUT_DIR / report_filename).write_text(report_text, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=report_text.encode("utf-8"),
        path_in_repo=report_filename,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add cleaning report",
    )

    log.info("Upload complete: https://huggingface.co/datasets/%s", repo_id)
