"""Checkpoint discovery and the filter-policy namespace.

Checkpoints are namespaced by FILTER_POLICY_VERSION so that resuming a run after
changing a threshold cannot silently mix clips filtered under two policies.
"""

from pathlib import Path

from pipeline import checkpoint
from pipeline.constants import FILTER_POLICY_VERSION


def test_latest_checkpoint_ignores_malformed_directories(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    run_dir = root / f"fleurs_train_{FILTER_POLICY_VERSION}"
    (run_dir / "batch_000001").mkdir(parents=True)
    (run_dir / "tmp").mkdir()
    (run_dir / "batch_backup").mkdir()

    monkeypatch.setattr(checkpoint, "_CHECKPOINT_ROOT", root)

    assert checkpoint.latest_checkpoint_idx(f"fleurs_train_{FILTER_POLICY_VERSION}") == 1


def test_latest_checkpoint_takes_the_highest_index(tmp_path, monkeypatch):
    """Ordering must come from the index, not from directory iteration order.

    Path.iterdir() has no ordering guarantee. NTFS and ext4 happen to return
    sorted names, so an implementation using indices[-1] passes locally and
    resumes from the wrong batch elsewhere. The iteration order is reversed here
    so the test fails against indices[-1] rather than passing by luck.
    """
    root = tmp_path / "checkpoints"
    run_dir = root / "run"
    for i in (1, 2, 10):
        (run_dir / f"batch_{i:06d}").mkdir(parents=True)

    monkeypatch.setattr(checkpoint, "_CHECKPOINT_ROOT", root)

    real_iterdir = Path.iterdir
    monkeypatch.setattr(
        Path, "iterdir", lambda self: reversed(sorted(real_iterdir(self)))
    )

    assert checkpoint.latest_checkpoint_idx("run") == 10


def test_policy_version_changes_when_a_threshold_changes(monkeypatch):
    """A threshold change must invalidate cached checkpoints.

    The version used to be a hand-edited label, so tuning a gate and resuming
    reused clips accepted under the previous policy.
    """
    from pipeline import constants

    before = constants._policy_version()
    monkeypatch.setattr(constants, "MAX_CER", constants.MAX_CER + 0.01)
    assert constants._policy_version() != before
