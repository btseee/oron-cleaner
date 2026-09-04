"""Load a Hub dataset in a way that survives a transient environment fault.

Written after a 221-hour WorldSpeech pass died at the very first call, with a
traceback that names no dataset problem at all:

    File "<frozen posixpath>", line 423, in abspath
    FileNotFoundError: [Errno 2] No such file or directory

`datasets` resolves each data file through
`posixpath.relpath(data_file, start="hf://")`. `"hf://"` is not an absolute
POSIX path, so `relpath` calls `abspath`, which calls `os.getcwd()` -- and
`getcwd` raises exactly that error when the process's working directory has been
unlinked. The identical load succeeded minutes later. Nothing about the corpus
was wrong; the run simply lost its cwd.

Two consequences, both cheap to prevent and expensive to hit unattended:

* **Run from a directory that outlives the call.** The caller's cwd is
  irrelevant to reading a Hub dataset, so it is not worth depending on. This
  chdirs to a directory it knows exists for the duration.
* **Retry.** A single network hiccup or a resolver fault should cost a minute,
  not a stage. The pass that failed was the last one before publication, and
  the publish step downstream had no gate of its own -- so a 19-clip leftover
  corpus was packaged and pushed as the release.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

ATTEMPTS = 3
BACKOFF_S = 60.0


@contextmanager
def _stable_cwd():
    """Work from a directory that exists, restoring the old one if it still does."""
    previous: str | None
    try:
        previous = os.getcwd()
    except OSError:
        previous = None          # already lost; nothing to restore
    with tempfile.TemporaryDirectory(prefix="oron-load-") as tmp:
        os.chdir(tmp)
        try:
            yield
        finally:
            if previous is not None and os.path.isdir(previous):
                os.chdir(previous)


def load_hub_dataset(*args: Any, **kwargs: Any):
    """`datasets.load_dataset`, retried, from a working directory that exists."""
    from datasets import load_dataset

    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with _stable_cwd():
                return load_dataset(*args, **kwargs)
        except Exception as exc:                      # noqa: BLE001 - re-raised below
            last = exc
            if attempt == ATTEMPTS:
                break
            log.warning("load_dataset(%s) failed on attempt %d/%d: %s: %s; "
                        "retrying in %.0fs", args[0] if args else kwargs.get("path"),
                        attempt, ATTEMPTS, type(exc).__name__, exc, BACKOFF_S)
            time.sleep(BACKOFF_S)
    assert last is not None
    raise last
