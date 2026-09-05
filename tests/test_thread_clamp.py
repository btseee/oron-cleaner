"""Torch stays on one CPU thread, whatever silero_vad happens to do.

`import silero_vad` sets it to 1 as a side effect. Relying on that would make
the pipeline's speed depend on an unrelated library's internals, and an earlier
attempt to "fix" the side effect by restoring the previous count made alignment
twelve times slower: 0.84 s per 6 s clip on 48 threads against 0.07 s on one.
So the count is set deliberately, and this test pins it.
"""
from __future__ import annotations

import sys
import types

import pytest

# The whole module is about torch's thread count, so it has nothing to test
# where torch is absent. Guarding at module scope is right *here* -- and only
# because every test in the file needs it. A bare `import torch` was a hard
# collection error on CI, which installs no model stack.
torch = pytest.importorskip("torch", reason="torch not installed")

import pipeline.audio_filter as af  # noqa: E402


def test_filter_pins_torch_to_one_thread(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(max(4, before))

    # A silero_vad that does NOT clamp, so the test proves the filter sets the
    # count itself rather than inheriting someone else's side effect.
    fake = types.ModuleType("silero_vad")

    def load_silero_vad(*a, **k):
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

    assert torch.get_num_threads() == af.TORCH_THREADS == 1, (
        f"torch is on {torch.get_num_threads()} threads; measured, that makes "
        "alignment up to twelve times slower")
    torch.set_num_threads(before)
