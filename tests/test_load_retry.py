"""The Hub loader must survive a lost working directory, and must retry.

Both properties were bought with a dead 221-hour pass: `datasets` resolves data
files through `posixpath.relpath(f, start="hf://")`, which reaches `os.getcwd()`,
which raises FileNotFoundError once the process's cwd has been unlinked.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

from pipeline.datasets import _load


@pytest.fixture
def fake_datasets(monkeypatch):
    """Stand in for `datasets`, recording each call, without touching the Hub."""
    module = types.ModuleType("datasets")
    module.calls = []

    def load_dataset(*args, **kwargs):
        module.calls.append((args, kwargs))
        behaviour = module.behaviour.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    monkeypatch.setattr(_load, "BACKOFF_S", 0.0)
    return module


def test_returns_the_dataset(fake_datasets):
    fake_datasets.behaviour = ["corpus"]
    assert _load.load_hub_dataset("a/b", "cfg", revision="deadbeef") == "corpus"
    assert fake_datasets.calls == [(("a/b", "cfg"), {"revision": "deadbeef"})]


def test_retries_then_succeeds(fake_datasets):
    fake_datasets.behaviour = [FileNotFoundError(2, "No such file or directory"),
                               "corpus"]
    assert _load.load_hub_dataset("a/b") == "corpus"
    assert len(fake_datasets.calls) == 2


def test_gives_up_after_attempts_and_reraises(fake_datasets):
    fake_datasets.behaviour = [FileNotFoundError(2, "boom")] * _load.ATTEMPTS
    with pytest.raises(FileNotFoundError):
        _load.load_hub_dataset("a/b")
    assert len(fake_datasets.calls) == _load.ATTEMPTS


@pytest.mark.skipif(os.name != "posix",
                    reason="Windows refuses to unlink a directory that is a cwd; "
                           "the failure this guards against is POSIX-only and CI is Linux")
def test_loads_with_the_cwd_deleted_underneath(fake_datasets):
    """The exact production failure: cwd unlinked, so os.getcwd() raises."""
    doomed = Path(tempfile.mkdtemp(prefix="oron-doomed-"))
    original = os.getcwd()
    os.chdir(doomed)
    try:
        doomed.rmdir()
        with pytest.raises(FileNotFoundError):
            os.getcwd()          # the precondition the traceback showed

        seen = {}

        def load_dataset(*args, **kwargs):
            seen["cwd"] = os.getcwd()        # must not raise inside the helper
            return "corpus"

        fake_datasets.load_dataset = load_dataset
        assert _load.load_hub_dataset("a/b") == "corpus"
        assert os.path.isdir(seen["cwd"])
    finally:
        os.chdir(original)


def test_restores_the_caller_cwd(fake_datasets, tmp_path):
    fake_datasets.behaviour = ["corpus"]
    original = os.getcwd()
    os.chdir(tmp_path)
    try:
        _load.load_hub_dataset("a/b")
        assert Path(os.getcwd()).resolve() == tmp_path.resolve()
    finally:
        os.chdir(original)
