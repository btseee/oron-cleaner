"""Threshold calibration.

Every gate in constants.py was set from published figures, small samples, or
reasoning -- none against this corpus's actual distribution. This is the tool
that replaces that guess with a measurement, so its arithmetic has to be right.
"""

import json

import pytest

from pipeline.calibrate import Calibration, _fmt
from pipeline.clip_result import ClipResult


def result(passed=True, failed=(), **metrics):
    return ClipResult(passed=passed, failed_gates=list(failed), **metrics)


def _cal(n=100, snr_start=0.0):
    cal = Calibration()
    for i in range(n):
        cal.record(result(snr_db=snr_start + i, align_score=i / n))
    return cal


# ── independent per-gate rates ────────────────────────────────────────────────

def test_every_failed_gate_is_counted_not_just_the_first():
    """The whole reason calibration mode exists.

    In a normal run a clip rejected for SNR is never scored for DNSMOS, so the
    DNSMOS rate is only the rate among clips that already passed SNR -- which
    makes the gates incomparable.
    """
    cal = Calibration()
    cal.record(result(passed=False, failed=["snr:x", "dnsmos:y", "cer:z"]))
    assert cal.gate_failures == {"snr": 1, "dnsmos": 1, "cer": 1}


def test_first_failure_is_tracked_separately():
    """So the report can show which gate a normal run would have blamed."""
    cal = Calibration()
    cal.record(result(passed=False, failed=["snr:x", "dnsmos:y"]))
    cal.record(result(passed=False, failed=["dnsmos:y"]))
    assert cal.first_failure == {"snr": 1, "dnsmos": 1}
    assert cal.gate_failures["dnsmos"] == 2


def test_pass_count_tracks_clips_failing_nothing():
    cal = Calibration()
    cal.record(result(passed=True))
    cal.record(result(passed=False, failed=["snr:x"]))
    assert (cal.total, cal.passed) == (2, 1)


# ── distributions ─────────────────────────────────────────────────────────────

def test_unmeasured_metrics_are_excluded():
    """0.0 is the dataclass default, meaning "not measured".

    Counting it would drag every percentile toward zero and make a gate look
    far stricter than it is.
    """
    cal = Calibration()
    cal.record(result(snr_db=20.0))          # bandwidth never measured
    cal.record(result(snr_db=22.0))
    assert "bandwidth_hz" not in cal.values
    assert cal.values["snr_db"] == [20.0, 22.0]


def test_quantiles_track_the_distribution():
    cal = _cal(101)
    assert cal._quantile("snr_db", 50) == pytest.approx(50, abs=2)
    assert cal._quantile("snr_db", 5) == pytest.approx(5, abs=2)


# ── yields and suggestions ────────────────────────────────────────────────────

def test_yield_for_a_minimum_gate():
    """Higher passes: SNR >= 50 keeps roughly the top half of 0..99."""
    assert _cal(100).yield_at("snr_db", 50.0) == pytest.approx(0.5, abs=0.02)


def test_yield_for_a_maximum_gate():
    """Lower passes: CER <= 0.5 keeps roughly the bottom half."""
    cal = Calibration()
    for i in range(100):
        cal.record(result(cer=i / 100))
    assert cal.yield_at("cer", 0.5) == pytest.approx(0.5, abs=0.02)


def test_suggested_threshold_hits_the_requested_yield():
    cal = _cal(200)
    threshold = cal.threshold_for_yield("snr_db", 0.75)
    assert cal.yield_at("snr_db", threshold) == pytest.approx(0.75, abs=0.03)


def test_suggestion_respects_gate_direction():
    """For a max-gate a looser threshold is higher, not lower."""
    cal = Calibration()
    for i in range(100):
        cal.record(result(cer=i / 100))
    assert cal.threshold_for_yield("cer", 0.9) > cal.threshold_for_yield("cer", 0.5)


def test_unmeasured_metric_yields_nothing_rather_than_dividing_by_zero():
    cal = Calibration()
    assert cal.yield_at("snr_db", 10.0) == 0.0
    assert cal.threshold_for_yield("snr_db", 0.75) is None


# ── report ────────────────────────────────────────────────────────────────────

def test_report_warns_when_almost_nothing_passes():
    cal = Calibration()
    for _ in range(100):
        cal.record(result(passed=False, failed=["snr:x"], snr_db=1.0))
    assert "Under 10% of clips pass" in cal.report()


def test_report_is_quiet_at_a_healthy_pass_rate():
    cal = Calibration()
    for _ in range(100):
        cal.record(result(snr_db=25.0))
    assert "Under 10%" not in cal.report()


def test_report_survives_an_empty_run():
    """A crash here would hide whatever went wrong upstream."""
    assert "clips measured" in Calibration().report()


def test_save_writes_both_the_text_and_the_json(tmp_path):
    cal = _cal(50)
    out = tmp_path / "calibration_report.txt"
    cal.save(out)
    assert out.exists()
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["total"] == 50
    assert "snr_db" in data["distributions"]
    assert "thresholds" in data


# ── formatting ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (11200.0, "11,200"),   # Hz: scientific notation is unreadable here
    (15.0, "15.0"),
    (0.6523, "0.652"),     # MOS/CER: four decimals are noise
])
def test_number_formatting_suits_each_scale(value, expected):
    assert _fmt(value) == expected
