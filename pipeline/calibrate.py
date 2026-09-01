"""Threshold calibration from a measured slice.

Every gate in `constants.py` was set from published figures, small samples, or
reasoning about what a strict corpus needs. None of them has been checked
against the actual distribution of this corpus, and a threshold that is 10%
too strict silently discards hours of usable audio while looking like it worked.

A calibration run processes a slice with `measure_all=True`, so each clip is
scored by *every* gate rather than stopping at the first failure. That gives two
things a normal run cannot:

* **Independent per-gate rejection rates.** In normal operation a clip rejected
  for SNR is never scored for DNSMOS, so the reported DNSMOS rate is only the
  rate among clips that already passed SNR. Comparing gates that way is
  meaningless.
* **The full distribution of each metric**, so a threshold can be chosen to hit
  a target yield instead of guessed.

    python clean_pipeline.py --datasets cv --calibrate --limit 500 --no-upload
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import constants
from .clip_result import ClipResult

# Metric -> (constants attribute, direction). "min" means higher passes.
GATES: dict[str, tuple[str, str]] = {
    "snr_db": ("SNR_MIN_DB", "min"),
    "bandwidth_hz": ("MIN_BANDWIDTH_HZ", "min"),
    "dnsmos_ovr": ("DNSMOS_MIN_OVR", "min"),
    "dnsmos_sig": ("DNSMOS_MIN_SIG", "min"),
    "dnsmos_bak": ("DNSMOS_MIN_BAK", "min"),
    "align_score": ("MIN_ALIGN_SCORE", "min"),
    "cer": ("MAX_CER", "max"),
    "duration_s": ("MAX_DURATION_S", "max"),
}

_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def _fmt(value: float) -> str:
    """Readable across the range these metrics span, from 0.02 CER to 11200 Hz.

    Scientific notation is unreadable for a bandwidth in Hz, and four decimals
    are noise for a MOS score.
    """
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


@dataclass
class Calibration:
    """Accumulates measurements across a slice."""

    values: dict[str, list[float]] = field(default_factory=dict)
    gate_failures: Counter = field(default_factory=Counter)
    first_failure: Counter = field(default_factory=Counter)
    total: int = 0
    passed: int = 0

    def record(self, result: ClipResult) -> None:
        self.total += 1
        if result.passed:
            self.passed += 1
        for metric in GATES:
            value = getattr(result, metric, None)
            # 0.0 is the dataclass default, i.e. "not measured", and including
            # it would drag every percentile toward zero.
            if value:
                self.values.setdefault(metric, []).append(float(value))
        for entry in result.failed_gates:
            self.gate_failures[entry.split(":", 1)[0]] += 1
        if result.failed_gates:
            self.first_failure[result.failed_gates[0].split(":", 1)[0]] += 1

    def _quantile(self, metric: str, p: float) -> float | None:
        vals = sorted(self.values.get(metric, []))
        if not vals:
            return None
        return vals[min(len(vals) - 1, max(0, int(p / 100 * (len(vals) - 1))))]

    def yield_at(self, metric: str, threshold: float) -> float:
        """Fraction of measured clips a threshold would keep, gate alone."""
        vals = self.values.get(metric, [])
        if not vals:
            return 0.0
        direction = GATES[metric][1]
        kept = sum(1 for v in vals if (v >= threshold if direction == "min" else v <= threshold))
        return kept / len(vals)

    def threshold_for_yield(self, metric: str, target: float) -> float | None:
        """Threshold that would keep `target` of clips on this gate alone."""
        vals = sorted(self.values.get(metric, []))
        if not vals:
            return None
        if GATES[metric][1] == "min":
            return vals[min(len(vals) - 1, int((1 - target) * (len(vals) - 1)))]
        return vals[min(len(vals) - 1, int(target * (len(vals) - 1)))]

    def report(self, target_yield: float = 0.75) -> str:
        lines = [
            "=== Threshold calibration ===",
            "",
            f"clips measured        {self.total:,}",
            f"pass all gates        {self.passed:,} ({self.passed / max(self.total, 1):.1%})",
            "",
            "Per-gate rejection, each evaluated independently.",
            "In a normal run a clip stops at its first failure, so these rates",
            "are not comparable to the counts in the cleaning report.",
            "",
            f"  {'gate':<14}{'rejected':>10}{'rate':>9}{'first':>9}",
        ]
        for gate, n in self.gate_failures.most_common():
            lines.append(f"  {gate:<14}{n:>10,}{n / max(self.total, 1):>8.1%}"
                         f"{self.first_failure.get(gate, 0):>9,}")

        lines += ["", "Metric distributions and what the current threshold keeps.", ""]
        # Wide enough for a 6-significant-figure value like 11200; too narrow and
        # adjacent columns run together, which is worse than an unaligned table.
        col = 9
        header = "  " + f"{'metric':<14}" + "".join(f"{'p' + str(p):<{col}}" for p in _PERCENTILES)
        lines += [header + f"{'current':>10}{'keeps':>8}{'for ' + f'{target_yield:.0%}':>10}"]
        for metric, (attr, _direction) in GATES.items():
            if metric not in self.values:
                continue
            row = f"  {metric:<14}"
            for p in _PERCENTILES:
                q = self._quantile(metric, p)
                row += f"{_fmt(q):<{col}}" if q is not None else f"{'-':<{col}}"
            current = getattr(constants, attr)
            suggested = self.threshold_for_yield(metric, target_yield)
            row += f"{_fmt(current):>10}{self.yield_at(metric, current):>8.0%}"
            row += f"{_fmt(suggested):>10}" if suggested is not None else f"{'-':>10}"
            lines.append(row)

        lines += [
            "",
            "'keeps' is that gate in isolation; gates are correlated, so the",
            "combined pass rate is lower than any single column suggests.",
            "'for N%' is the threshold that would keep N% on that gate alone --",
            "a starting point to weigh against quality, not a recommendation.",
        ]

        if self.total and self.passed / self.total < 0.10:
            lines += ["", "[!] Under 10% of clips pass. Check the gates with the",
                      "    highest independent rejection rate before running the",
                      "    full pass; the corpus may not support these thresholds."]
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.report(), encoding="utf-8")
        (path.with_suffix(".json")).write_text(json.dumps({
            "total": self.total,
            "passed": self.passed,
            "gate_failures": dict(self.gate_failures),
            "first_failure": dict(self.first_failure),
            "distributions": {
                metric: {
                    "n": len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "median": statistics.median(vals),
                    **{f"p{p}": self._quantile(metric, p) for p in _PERCENTILES},
                }
                for metric, vals in self.values.items() if vals
            },
            "thresholds": {
                metric: getattr(constants, attr) for metric, (attr, _) in GATES.items()
            },
        }, indent=2), encoding="utf-8")
