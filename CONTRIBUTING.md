# Contributing to oron-cleaner

Thank you for taking the time to contribute. This document covers everything you need to get a change from idea to merged PR.

---

## Table of contents

1. [Ground rules](#ground-rules)
2. [Development setup](#development-setup)
3. [Project structure](#project-structure)
4. [Making changes](#making-changes)
5. [Code style](#code-style)
6. [Tests](#tests)
7. [Submitting a pull request](#submitting-a-pull-request)
8. [Reporting bugs](#reporting-bugs)
9. [Suggesting features](#suggesting-features)

---

## Ground rules

- Be respectful. Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- One concern per PR. A PR that cleans up formatting AND adds a new feature is harder to review and slower to merge.
- If your change is large or architectural, open an issue first so we can discuss the approach before you write the code.

---

## Development setup

**Python 3.14** and **ffmpeg** are required.

```bash
git clone https://github.com/BBadral/oron-cleaner.git
cd oron-cleaner

python3.14 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # installs the package + dev dependencies
```

Add a `[project.optional-dependencies]` `dev` extra to `pyproject.toml` when adding new dev tools, e.g.:

```toml
[project.optional-dependencies]
dev = ["ruff", "mypy", "pytest"]
```

---

## Project structure

```
pipeline/constants.py     ← quality thresholds only (change a threshold → edit here)
pipeline/clip_result.py   ← ClipResult dataclass
pipeline/audio_filter.py  ← AudioQualityFilter class (7 pipeline stages)
pipeline/stats.py         ← CleaningStats + RejectionLog
pipeline/checkpoint.py    ← save / load checkpoint helpers
pipeline/processor.py     ← shared per-split iteration loop
pipeline/upload.py        ← HuggingFace upload
pipeline/datasets/        ← one file per source dataset
pipeline/cards/           ← HuggingFace dataset card markdown
```

**Single-responsibility rule:** each file has one job. If your change touches more than two files for a single logical concern, consider whether a new file or a refactor of an existing one is needed.

---

## Making changes

### Changing a quality threshold

Edit `pipeline/constants.py` only. All filter logic reads from there.

### Adding a new filter stage

1. Add the stage method to `AudioQualityFilter` in `pipeline/audio_filter.py`.
2. Call it in `process_clip()` in the same file, following the existing early-return pattern.
3. Add a new `stage_labels` entry in `CleaningStats.report()` in `pipeline/stats.py`.
4. Update the pipeline table in `README.md`.

### Adding a new source dataset

1. Create `pipeline/datasets/{name}.py` following the existing files as a template.
2. Create `pipeline/cards/{name}.md` with the HuggingFace dataset card.
3. Add an entry to `_REPO_CONFIG` in `pipeline/upload.py`.
4. Add the dataset key to `resolve_datasets()` in `clean_pipeline.py`.

---

## Code style

This project uses **Ruff** for linting and formatting, and **mypy** for type checking.

```bash
# Format + lint
ruff format .
ruff check . --fix

# Type check
mypy pipeline/ clean_pipeline.py
```

Style rules (enforced by Ruff, configured in `pyproject.toml`):

- Line length: 100 characters.
- No `from __future__ import annotations` — Python 3.14 defers annotations by default (PEP 649).
- Explicit over implicit: avoid `**kwargs` in internal APIs.
- No comments explaining *what* code does — only *why* when it's non-obvious.
- No docstrings on private methods (prefix `_`).

---

## Tests

Tests live in `tests/`. Run them with:

```bash
pytest
```

For changes to the pipeline stages, add a unit test in `tests/test_audio_filter.py`. Tests that require model inference should be marked `@pytest.mark.slow` and skipped in CI unless the `RUN_SLOW_TESTS=1` environment variable is set.

```python
import pytest

@pytest.mark.slow
def test_whisper_transcription():
    ...
```

---

## Submitting a pull request

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b fix/snr-threshold-edge-case
   ```
2. Make your changes, following the guidelines above.
3. Run `ruff format . && ruff check . && mypy pipeline/ && pytest` — all must pass.
4. Push and open a PR against `main`.
5. Fill in the [PR template](.github/PULL_REQUEST_TEMPLATE.md).
6. A maintainer will review within a few days. Address review comments by pushing new commits to the same branch (do not force-push after review begins).

### Branch naming

| Type | Prefix | Example |
|---|---|---|
| Bug fix | `fix/` | `fix/vad-empty-audio` |
| New feature | `feat/` | `feat/add-khalkha-filter` |
| Refactor | `refactor/` | `refactor/checkpoint-module` |
| Documentation | `docs/` | `docs/update-readme` |
| CI / tooling | `chore/` | `chore/upgrade-ruff` |

---

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml). Include:

- Python version (`python --version`)
- OS
- Full traceback
- Minimal reproducer (dataset key, clip index or audio path, and the command you ran)

---

## Suggesting features

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml). Describe:

- What problem the feature solves
- The proposed interface (new CLI flag, new constant, etc.)
- Any trade-offs you are aware of
