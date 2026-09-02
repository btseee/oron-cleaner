"""The cleaning report's accounting.

These are the numbers an operator reads to decide whether a 24-48 h pass
worked, and whether a gate is too strict. A rejection that lands in the "other"
bucket is a clip lost without an explanation.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.clip_result import ClipResult  # noqa: E402
from pipeline.stats import CleaningStats  # noqa: E402


def passed(**kw) -> ClipResult:
    base = {"passed": True, "duration_s": 5.0, "dnsmos_ovr": 3.5,
            "snr_db": 20.0, "cer": 0.1}
    return ClipResult(**{**base, **kw})


def rejected(stage: str) -> ClipResult:
    return ClipResult(passed=False, reject_stage=stage, reject_reason="because")


# ── every stage the pipeline can emit must be labelled ───────────────────────

def test_every_emitted_reject_stage_appears_in_the_report():
    """The module's own rule, checked against the source rather than a list.

    It was already broken once: "normalize" was added to processor.py as a
    reject stage and not to the report, so a refused transcript was counted
    under "other rejections" with nothing naming it.
    """
    sources = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in ("pipeline/audio_filter.py", "pipeline/processor.py")
    )
    emitted = set(re.findall(r'reject_stage=["\']([a-z_]+)', sources))
    emitted |= set(re.findall(r'\bnote\("([a-z_]+)"', sources))
    emitted.discard("")

    stats = CleaningStats("x")
    for stage in sorted(emitted):
        stats.record(rejected(stage))
    report = stats.report()
    assert "other" not in report.lower() or "0" in report
    unlabelled = [s for s in emitted if f"{stats.stage_counts[s]:>8,}" not in report]
    assert not unlabelled, f"emitted but unlabelled: {sorted(unlabelled)}"
    assert emitted, "found no reject stages at all -- the pattern stopped matching"


def test_the_normalize_stage_is_named(tmp_path):
    """The specific regression."""
    stats = CleaningStats("x")
    stats.record(rejected("normalize"))
    assert "normalise" in stats.report() or "normalize" in stats.report()


# ── accounting ────────────────────────────────────────────────────────────────

def test_only_passing_clips_contribute_to_the_sums():
    """A rejected clip has no measurements worth averaging, and counting its
    zeros would drag every mean down."""
    stats = CleaningStats("x")
    stats.record(passed(duration_s=10.0, snr_db=30.0))
    stats.record(rejected("snr"))
    assert stats.total == 2
    assert stats.passed == 1
    assert stats.total_duration_s == 10.0
    assert stats.sum_snr == 30.0


def test_rejections_are_counted_by_stage():
    stats = CleaningStats("x")
    for stage in ("snr", "snr", "dnsmos"):
        stats.record(rejected(stage))
    assert stats.stage_counts == {"snr": 2, "dnsmos": 1}


def test_merge_adds_both_the_totals_and_the_stages():
    """Splits are merged into one report per source."""
    a, b = CleaningStats("a"), CleaningStats("b")
    a.record(passed(duration_s=4.0))
    a.record(rejected("snr"))
    b.record(passed(duration_s=6.0))
    b.record(rejected("snr"))
    b.record(rejected("cer"))
    a.merge(b)
    assert a.total == 5
    assert a.passed == 2
    assert a.total_duration_s == 10.0
    assert a.stage_counts == {"snr": 2, "cer": 1}


def test_an_empty_report_does_not_divide_by_zero():
    """The first log line of a run, before anything has been processed."""
    assert CleaningStats("empty").report()


def test_the_report_names_the_dataset():
    assert "FLEURS" in CleaningStats("fleurs").report().upper()
