"""What was actually used to build a corpus.

`FILTER_POLICY_VERSION` hashes the gate *thresholds*, and only those. It does
not capture:

  * the ASR, aligner, VAD or DNSMOS model revisions;
  * the torchaudio release that ships the MMS_FA weights;
  * the source dataset revisions;
  * the `oron_tts.text` version.

That last one is the sharpest. Every fix to the normaliser changes every
published transcript -- and changes nothing about the version string, so two
corpora built months apart are indistinguishable while containing different
text. This module records all of it, plus a content hash of the corpus itself,
so a run can be identified after the fact.
"""

from __future__ import annotations

import hashlib
import json
import logging
from importlib import metadata
from pathlib import Path
from typing import Any

from .constants import FILTER_POLICY_VERSION

log = logging.getLogger(__name__)

# Resolved 2026-09-02 against the Hub. Pinning these is what makes a rebuild
# reproducible: `main` moves, and a recogniser that changed under the CER gate
# silently changes which clips enter the corpus.
PINNED_REVISIONS: dict[str, str] = {
    "bayartsogt/wav2vec2-large-xlsr-mongolian": "b615fb8829bce4a3921bc2f2b9984e894f192951",
    "google/fleurs": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
    # btsee/mbspeech_mn was deleted after its cleaned corpus was published;
    # a pin to a repo that cannot be fetched documents nothing.
    "disco-eth/WorldSpeech": "7fc2c2f19528b3d3972110a04e500098f6fc7f24",
}

_TRACKED_PACKAGES = (
    "torch", "torchaudio", "torchmetrics", "transformers", "datasets",
    "librosa", "soundfile", "silero-vad", "uroman", "numpy", "oron-tts",
)


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def normaliser_fingerprint() -> str:
    """Hash of the text-normalisation source that produced every transcript.

    The package version is too coarse: the normaliser is edited far more often
    than oron-tts is released, and a change to it rewrites the corpus.
    """
    import oron_tts.text.normalizer as normalizer
    import oron_tts.text.numbers as numbers

    digest = hashlib.sha256()
    for module in (numbers, normalizer):
        source = Path(module.__file__)
        digest.update(source.read_bytes())
    return digest.hexdigest()[:12]


def corpus_content_hash(records: list[dict]) -> str:
    """Identify the corpus by what is in it, not by when it was built.

    Over (clip_id, text, duration) sorted by clip id -- the three fields that
    decide what the model sees. Two runs that produce this hash produced the
    same corpus.
    """
    digest = hashlib.sha256()
    for r in sorted(records, key=lambda x: str(x.get("clip_id", ""))):
        digest.update(
            f"{r.get('clip_id')}\x1f{r.get('text')}\x1f"
            f"{float(r.get('duration_s') or 0.0):.3f}\x1e".encode()
        )
    return digest.hexdigest()[:16]


def build(records: list[dict], splits: dict[str, list[dict]]) -> dict[str, Any]:
    return {
        "filter_policy_version": FILTER_POLICY_VERSION,
        "normaliser_fingerprint": normaliser_fingerprint(),
        "corpus_content_hash": corpus_content_hash(records),
        "pinned_revisions": dict(PINNED_REVISIONS),
        "package_versions": _versions(),
        "clips": len(records),
        "hours": round(
            sum(float(r.get("duration_s") or 0.0) for r in records) / 3600.0, 3
        ),
        "splits": {name: len(rs) for name, rs in splits.items()},
    }


def write(root: Path | str, records: list[dict], splits: dict[str, list[dict]]) -> Path:
    root = Path(root)
    out = root / "provenance.json"
    payload = build(records, splits)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    log.info(
        "Wrote %s (policy %s, normaliser %s, corpus %s)",
        out, payload["filter_policy_version"],
        payload["normaliser_fingerprint"], payload["corpus_content_hash"],
    )
    return out
