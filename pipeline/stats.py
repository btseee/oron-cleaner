from __future__ import annotations

import csv
import logging
from pathlib import Path

from .clip_result import ClipResult

log = logging.getLogger(__name__)


class RejectionLog:
    """Append-only CSV log of rejected clips. Not thread-safe."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if path.stat().st_size == 0:
            self._writer.writerow(["clip_id", "stage", "reason", "ground_truth"])

    def record(self, clip_id: str, stage: str, reason: str, ground_truth: str = "") -> None:
        self._writer.writerow([clip_id, stage, reason, ground_truth[:100]])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class CleaningStats:
    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name
        self.total: int = 0
        self.passed: int = 0
        self.stage_counts: dict[str, int] = {}
        self.total_duration_s: float = 0.0
        self.sum_dnsmos_ovr: float = 0.0
        self.sum_snr: float = 0.0
        self.sum_cer: float = 0.0

    def record(self, result: ClipResult) -> None:
        self.total += 1
        if result.passed:
            self.passed += 1
            self.total_duration_s += result.duration_s
            self.sum_dnsmos_ovr += result.dnsmos_ovr
            self.sum_snr += result.snr_db
            self.sum_cer += result.cer
        else:
            self.stage_counts[result.reject_stage] = (
                self.stage_counts.get(result.reject_stage, 0) + 1
            )

    def merge(self, other: CleaningStats) -> None:
        """Accumulate another split's stats into this aggregate."""
        self.total += other.total
        self.passed += other.passed
        self.total_duration_s += other.total_duration_s
        self.sum_dnsmos_ovr += other.sum_dnsmos_ovr
        self.sum_snr += other.sum_snr
        self.sum_cer += other.sum_cer
        for stage, count in other.stage_counts.items():
            self.stage_counts[stage] = self.stage_counts.get(stage, 0) + count

    def report(self) -> str:
        lines = [
            f"=== {self.name.upper()} — Cleaning Report ===",
            f"Total input clips:          {self.total:>8,}",
        ]
        stage_labels = {
            "load":     "Rejected — load error:      ",
            "duration": "Rejected — too short/long:  ",
            "vad":      "Rejected — VAD (no speech): ",
            "snr":      "Rejected — SNR too low:     ",
            "pitch":    "Rejected — pitch/mumbling:  ",
            "dnsmos":   "Rejected — DNSMOS too low:  ",
            "cer":      "Rejected — sentence verify: ",
        }
        for stage, label in stage_labels.items():
            lines.append(f"{label}{self.stage_counts.get(stage, 0):>8,}")
        lines.append("─" * 50)
        pct = 100.0 * self.passed / max(self.total, 1)
        hours = self.total_duration_s / 3600.0
        lines.append(f"Total PASSED (clean):       {self.passed:>8,}  ({pct:.1f}% of input)")
        lines.append(f"Total hours (clean):        {hours:>10.2f} hours")
        if self.passed > 0:
            lines.append(f"Average DNSMOS OVR:         {self.sum_dnsmos_ovr/self.passed:>10.3f}")
            lines.append(f"Average SNR:                {self.sum_snr/self.passed:>10.1f} dB")
            lines.append(f"Average CER:                {self.sum_cer/self.passed:>10.3f}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        report = self.report()
        path.write_text(report, encoding="utf-8")
        log.info("Report saved to %s", path)
        print("\n" + report + "\n")
