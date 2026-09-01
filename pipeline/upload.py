"""Publish the strict corpus to HuggingFace.

One repository, not one per source: the corpus is merged before the speaker cap
and the speaker-disjoint split, so publishing the sources separately would ship
splits that leak speakers into each other.

The dataset card is generated from the manifest and the live threshold values.
The previous cards were hand-maintained static markdown and had drifted -- they
claimed Common Voice 17 provenance while the loader used the Data Collective
API, and described "6 stages" where there were 7.
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import HfApi

from . import constants
from .corpus import read_manifest

log = logging.getLogger(__name__)

REPO_ID = "btsee/oron-mn-strict"


def _gate_table() -> str:
    rows = [
        ("duration", f"{constants.MIN_DURATION_S:g}–{constants.MAX_DURATION_S:g} s"),
        ("clipping", f"≤ {constants.MAX_CLIPPED_RATIO:.3%} of samples at full scale"),
        ("DC offset", f"≤ {constants.MAX_DC_OFFSET}"),
        ("voice activity", f"≥ {constants.VAD_MIN_SPEECH_RATIO:.0%} speech, edge-trimmed"),
        ("SNR", f"≥ {constants.SNR_MIN_DB:g} dB (speech vs true non-speech regions)"),
        ("bandwidth", f"≥ {constants.MIN_BANDWIDTH_HZ / 1000:g} kHz lowpass shelf"),
        ("DNSMOS P.835", f"OVR ≥ {constants.DNSMOS_MIN_OVR} · SIG ≥ "
                         f"{constants.DNSMOS_MIN_SIG} · BAK ≥ {constants.DNSMOS_MIN_BAK}"),
        ("forced alignment", f"≥ {constants.MIN_ALIGN_SCORE} (MMS_FA, primary gate)"),
        ("CER", f"≤ {constants.MAX_CER} (wav2vec2-xlsr-mongolian)"),
        ("per-speaker cap", f"{constants.MAX_CLIPS_PER_SPEAKER} clips"),
    ]
    return "\n".join(f"| {name} | {value} |" for name, value in rows)


def build_card(corpus_dir: Path) -> str:
    """Generate the dataset card from the manifest and the live thresholds."""
    records = read_manifest(corpus_dir)
    hours = sum(float(r.get("duration_s") or 0.0) for r in records) / 3600.0

    def gender_hours(g: str) -> float:
        return sum(
            float(r.get("duration_s") or 0.0)
            for r in records if r.get("gender_resolved") == g
        ) / 3600.0

    speakers = len({str(r.get("client_id") or "") for r in records})
    sources = sorted({str(r.get("clip_id", "")).split("_")[0] for r in records})

    return f"""---
language:
- mn
license: cc0-1.0
task_categories:
- text-to-speech
- automatic-speech-recognition
pretty_name: "Oron MN — strict Mongolian TTS corpus"
tags:
- mongolian
- khalkha
- speech
- tts
---

# Oron MN — strict Mongolian TTS corpus

{len(records):,} clips · **{hours:.1f} hours** · {speakers} speakers
(male {gender_hours('male'):.1f} h, female {gender_hours('female'):.1f} h)

Built for finetuning [F5-TTS](https://github.com/SWivid/F5-TTS) `F5TTS_v1_Base`
on Mongolian (Khalkha Cyrillic). Sources: {', '.join(sources)}.

## Licensing

Every source is commercially usable: Common Voice is CC0-1.0, FLEURS is
CC-BY-4.0, MBSpeech is MIT. WorldSpeech is deliberately excluded — it is by far
the largest Mongolian corpus (~221 h, 24 kHz native) but CC-BY-NC-4.0.

## Quality gates

| gate | threshold |
|---|---|
{_gate_table()}

Policy version `{constants.FILTER_POLICY_VERSION}`, derived from a hash of these
values, so a threshold change invalidates cached work rather than silently
mixing policies.

**Forced alignment is the primary transcript gate.** Free-running ASR is the
wrong instrument in Mongolian: the best available model has a CER floor of 0.123
on clean, correctly-transcribed speech (whisper-large-v3 is 0.311), so any
absolute CER threshold sits near the recogniser's own error. Alignment is
constrained to the given transcript, and measured cleanly separated correct from
mismatched transcripts on both corpora (worst correct 0.722, worst mismatched
0.547).

## Known limitation: bandwidth

**No Mongolian source is full-band.** Measured lowpass shelves: Common Voice
median 7.1 kHz (its 48 kHz container is low-bitrate mp3), FLEURS capped at
7.7 kHz, MBSpeech at 7.7 kHz. A model trained on this is a **wideband ~8 kHz**
voice, not a full-band one. `bandwidth_hz` is recorded per clip so the brightest
clips can be selected for reference voices.

## Layout

```
wavs/<clip_id>.wav        24 kHz mono, edge-trimmed, peak-normalised to −1 dBFS
metadata.csv              audio_file|text  (train; F5-TTS prepare_csv_wavs.py)
metadata_validation.csv   likewise
metadata_test.csv         likewise
manifest.parquet          full per-clip metrics and speaker metadata
manifest.jsonl            the same rows, as written
```

Splits are **speaker-disjoint**. Common Voice's own train/dev/test are not, and
neither is a row-level random split — evaluating a voice-cloning model on
speakers it trained on measures memorisation.

Audio is edge-trimmed, never spliced: interior pauses are preserved, so prosodic
pausing survives into the corpus.

## Text

`text` is normalised Mongolian — numbers, dates, currency and abbreviations
expanded to words, case preserved. It is produced by `oron_tts.text`, the same
code that normalises text at training time, and is the exact string the CER gate
scored against.
"""


def upload_corpus(corpus_dir: Path | str, repo_id: str = REPO_ID) -> None:
    corpus_dir = Path(corpus_dir)
    if not (corpus_dir / "manifest.jsonl").exists():
        log.warning("No manifest at %s — skipping upload.", corpus_dir)
        return

    api = HfApi()
    log.info("Creating / verifying %s", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    card = corpus_dir / "README.md"
    card.write_text(build_card(corpus_dir), encoding="utf-8")

    log.info("Uploading %s … (this transfers the whole corpus)", corpus_dir)
    api.upload_large_folder(
        folder_path=str(corpus_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    log.info("Uploaded: https://huggingface.co/datasets/%s", repo_id)
