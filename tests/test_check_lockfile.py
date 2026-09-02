"""The lockfile drift check.

It exists because a lockfile that has drifted from pyproject.toml is worse than
none: it is believed. The check is coverage, not re-resolution -- recompiling
and diffing would turn "some package released today" into a red build, which
teaches people to ignore the signal.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_lockfile import canonical, declared, locked, satisfies  # noqa: E402

# ── name normalisation ────────────────────────────────────────────────────────

def test_underscores_and_hyphens_are_one_name():
    """pyproject says huggingface_hub, the lockfile says huggingface-hub."""
    assert canonical("huggingface_hub") == canonical("huggingface-hub")
    assert canonical("Torch") == "torch"


# ── version comparison ────────────────────────────────────────────────────────

def test_a_version_inside_its_range_satisfies():
    assert satisfies("2.6.0", [(">=", "2.6.0")])
    assert satisfies("2.13.0", [(">=", "2.6.0")])


def test_a_version_below_the_floor_does_not():
    assert not satisfies("2.5.0", [(">=", "2.6.0")])


def test_shorter_and_longer_versions_compare_correctly():
    """"2.6" is not below "2.6.0", and "2.10" is above "2.9"."""
    assert satisfies("2.6", [(">=", "2.6.0")])
    assert satisfies("2.10.0", [(">=", "2.9.0")])
    assert not satisfies("2.9.0", [(">=", "2.10.0")])


def test_a_local_or_suffixed_version_is_read_by_its_numbers():
    """torch ships "2.13.0+cpu"."""
    assert satisfies("2.13.0+cpu", [(">=", "2.6.0")])
    assert satisfies("1.3.1.1", [(">=", "1.3.1")])


def test_an_upper_bound_is_honoured():
    assert satisfies("2.6.0", [(">=", "2.0.0"), ("<", "3.0.0")])
    assert not satisfies("3.0.0", [(">=", "2.0.0"), ("<", "3.0.0")])


# ── reading the real files ────────────────────────────────────────────────────

def test_the_shipped_lockfile_covers_the_shipped_pyproject():
    """The check the CI job runs, against the real artifacts."""
    want = declared(ROOT / "pyproject.toml")
    have = locked(ROOT / "requirements.lock")
    unlockable = {"oron-tts"}
    for name, specs in want.items():
        if name in unlockable:
            continue
        assert name in have, f"{name} declared but not locked"
        assert satisfies(have[name], specs), f"{name} locked at {have[name]}, outside {specs}"


def test_comments_and_blank_lines_are_not_read_as_packages():
    assert "#" not in "".join(locked(ROOT / "requirements.lock"))


def test_the_lockfile_pins_rather_than_ranges():
    """Every entry read is an ==; a range in a lockfile is not a lock."""
    have = locked(ROOT / "requirements.lock")
    assert len(have) > 50
    assert all(v and v[0].isdigit() for v in have.values())
