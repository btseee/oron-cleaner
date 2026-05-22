import gc
import logging
import pickle
from pathlib import Path

from .constants import OUTPUT_DIR
import torch

from .stats import CleaningStats

log = logging.getLogger(__name__)

_CHECKPOINT_ROOT = OUTPUT_DIR / "checkpoints"


def checkpoint_dir(name: str, batch_idx: int) -> Path:
    p = _CHECKPOINT_ROOT / name / f"batch_{batch_idx:06d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_checkpoint_idx(name: str) -> int:
    """Return the index of the last saved batch, or -1 if none exist."""
    p = _CHECKPOINT_ROOT / name
    if not p.exists():
        return -1
    indices = sorted(
        int(d.name.split("_")[1]) for d in p.iterdir() if d.is_dir()
    )
    return indices[-1] if indices else -1


def save_batch(name: str, batch_idx: int, records: list[dict]) -> None:
    path = checkpoint_dir(name, batch_idx) / "records.pkl"
    with open(path, "wb") as f:
        pickle.dump(records, f)


def load_prior_batches(name: str, last_idx: int) -> tuple[list[dict], CleaningStats]:
    """
    Load all checkpoint batches up to and including last_idx.
    Returns the combined record list and partial stats (passed clips only —
    rejection counts for prior batches are not stored in checkpoints).
    """
    stats = CleaningStats(f"{name}/resumed")
    records: list[dict] = []

    for batch_idx in range(last_idx + 1):
        path = checkpoint_dir(name, batch_idx) / "records.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as f:
            batch = pickle.load(f)
        records.extend(batch)
        for rec in batch:
            stats.passed += 1
            stats.total += 1
            stats.total_duration_s += float(rec.get("duration_s") or 0.0)
            stats.sum_dnsmos_ovr   += float(rec.get("dnsmos_ovr") or 0.0)
            stats.sum_snr          += float(rec.get("snr_db")     or 0.0)
            stats.sum_cer          += float(rec.get("cer")        or 0.0)

    return records, stats


def flush_gpu_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
