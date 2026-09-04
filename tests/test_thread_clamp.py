"""Silero must not leave torch single-threaded for the rest of the run.

`import silero_vad` calls `torch.set_num_threads(1)` at module scope. The clamp
is process-wide and permanent, so every later torch CPU op -- resampling,
DNSMOS, the CTC forward on CPU -- inherits it. Measured on a 48-core node,
MBSpeech cleaning ran at 7.7 clips a minute instead of ~46.
"""
from __future__ import annotations

import sys
import types

import torch

import pipeline.audio_filter as af


def test_filter_restores_the_thread_count(monkeypatch):
    before = torch.get_num_threads()
    target = max(4, before)
    torch.set_num_threads(target)

    # A silero_vad whose import clamps threads, exactly like the real one.
    fake = types.ModuleType("silero_vad")

    def load_silero_vad(*a, **k):
        torch.set_num_threads(1)
        return object()

    fake.load_silero_vad = load_silero_vad
    fake.get_speech_timestamps = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "silero_vad", fake)

    # Stop __init__ immediately after the VAD block. Everything below it
    # downloads and loads real recognisers, which this test has no use for.
    class StopAfterVad(Exception):
        pass

    transformers = types.ModuleType("transformers")

    class _Raises:
        @staticmethod
        def from_pretrained(*a, **k):
            raise StopAfterVad

    transformers.AutoModelForCTC = _Raises
    transformers.AutoProcessor = _Raises
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    try:
        af.AudioQualityFilter(device="cpu")
    except StopAfterVad:
        pass
    else:
        raise AssertionError("__init__ did not reach the ASR load; test is not "
                             "exercising the path it claims to")

    assert torch.get_num_threads() == target, (
        "silero left torch at %d threads; the whole pipeline runs single-threaded"
        % torch.get_num_threads())
    torch.set_num_threads(before)
