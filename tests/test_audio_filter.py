import numpy as np
import torch

import pipeline.checkpoint as checkpoint
from pipeline.audio_filter import AudioQualityFilter


def test_pitch_detector_shortage_is_diagnostic_not_rejection(monkeypatch):
    filt = object.__new__(AudioQualityFilter)
    filt.device = "cpu"

    def fake_predict(*args, **kwargs):
        frequency = torch.full((1, 20), 180.0)
        periodicity = torch.zeros((1, 20))
        return frequency, periodicity

    monkeypatch.setattr("pipeline.audio_filter.torchcrepe.predict", fake_predict)

    passed, mean_f0, mean_confidence, reason = filt._check_pitch(
        np.zeros(16_000, dtype=np.float32)
    )

    assert passed is True
    assert mean_f0 == 0.0
    assert mean_confidence == 0.0
    assert reason == ""


def test_near_miss_cer_passes_only_when_length_is_aligned():
    passed, reason = AudioQualityFilter._reading_passes(cer=0.46, length_ratio=0.98)

    assert passed is True
    assert reason == ""


def test_near_miss_cer_rejects_when_audio_is_probably_truncated():
    passed, reason = AudioQualityFilter._reading_passes(cer=0.46, length_ratio=0.52)

    assert passed is False
    assert reason == "uncertain_reading_cer_0.460_ratio_0.52"


def test_high_cer_still_rejects_even_when_length_is_aligned():
    passed, reason = AudioQualityFilter._reading_passes(cer=0.57, length_ratio=1.01)

    assert passed is False
    assert reason == "high_cer_0.570"


def test_cer_rescue_boundaries_are_inclusive():
    assert AudioQualityFilter._reading_passes(cer=0.50, length_ratio=0.75) == (True, "")
    assert AudioQualityFilter._reading_passes(cer=0.50, length_ratio=1.25) == (True, "")


def test_cer_rescue_rejects_overlong_asr():
    passed, reason = AudioQualityFilter._reading_passes(cer=0.46, length_ratio=1.40)

    assert passed is False
    assert reason == "uncertain_reading_cer_0.460_ratio_1.40"


def test_latest_checkpoint_ignores_malformed_directories(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    run_dir = root / "fleurs_train_v3_pitch_diagnostic_cer_rescue"
    (run_dir / "batch_000001").mkdir(parents=True)
    (run_dir / "tmp").mkdir()
    (run_dir / "batch_backup").mkdir()

    monkeypatch.setattr(checkpoint, "_CHECKPOINT_ROOT", root)

    assert checkpoint.latest_checkpoint_idx("fleurs_train_v3_pitch_diagnostic_cer_rescue") == 1