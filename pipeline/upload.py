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
        ("per-speaker cap", f"{constants.MAX_SPEAKER_HOURS:g} h "
                            f"({constants.MAX_NARRATOR_HOURS:g} h for a "
                            f"single-narrator source)"),
    ]
    return "\n".join(f"| {name} | {value} |" for name, value in rows)


# What each source actually grants, and what it demands in return. The merged
# corpus is governed by all of them at once, not by the most permissive one.
SOURCE_LICENCES: dict[str, dict[str, str]] = {
    "cv": {
        "name": "Common Voice 26.0 mn",
        "licence": "CC0-1.0",
        "url": "https://commonvoice.mozilla.org/",
        "requires": "nothing (public domain dedication)",
    },
    "fleurs": {
        "name": "FLEURS mn_mn",
        "licence": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/google/fleurs",
        "requires": "**attribution**",
    },
    "mbspeech": {
        "name": "MBSpeech mn",
        "licence": "MIT",
        "url": "https://huggingface.co/datasets/btsee/mbspeech_mn",
        "requires": "**licence and copyright notice**",
    },
    "ws": {
        "name": "WorldSpeech mn_mn",
        "licence": "CC-BY-NC-4.0",
        "url": "https://huggingface.co/datasets/disco-eth/WorldSpeech",
        "requires": "**attribution, and non-commercial use only**",
    },
}


def resolve_licence(sources: list[str]) -> tuple[str, str]:
    """The merged corpus's licence tag, and why.

    A merge is bound by every source's terms simultaneously. Declaring
    `cc0-1.0` for a corpus containing FLEURS strips the CC-BY-4.0 attribution
    obligation from every downstream user, which the card's own next section
    then contradicts eleven lines later.

    Only a corpus that is genuinely all-CC0 gets the CC0 tag. Anything mixed is
    `other`, with the components named, so a reader has to look rather than
    assume.
    """
    known = [s for s in sources if s in SOURCE_LICENCES]
    licences = {SOURCE_LICENCES[s]["licence"] for s in known}
    if not licences:
        return "other", "unknown-source-mix"
    if licences == {"CC0-1.0"}:
        return "cc0-1.0", "CC0-1.0"
    if "CC-BY-NC-4.0" in licences:
        # The strongest term wins, and this one is a use restriction, not just
        # an attribution one: a model trained here cannot be used commercially.
        return "other", "mixed-non-commercial (" + " + ".join(sorted(licences)) + ")"
    return "other", "mixed (" + " + ".join(sorted(licences)) + ")"


def _licence_table(sources: list[str]) -> str:
    rows = [
        f"| [{SOURCE_LICENCES[s]['name']}]({SOURCE_LICENCES[s]['url']}) "
        f"| {SOURCE_LICENCES[s]['licence']} | {SOURCE_LICENCES[s]['requires']} |"
        for s in sources if s in SOURCE_LICENCES
    ]
    return "\n".join(rows)


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

    licence_tag, licence_name = resolve_licence(sources)

    return f"""---
language:
- mn
license: {licence_tag}
license_name: {licence_name}
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

This corpus is a **merge**, so it is bound by every source's terms at once —
not by the most permissive one. The tag above is `{licence_tag}`
(`{licence_name}`) for exactly that reason.

| source | licence | using this corpus requires |
|---|---|---|
{_licence_table(sources)}

Attribution obligations are **not** waived by the merge. If your use is
commercial, confirm that no CC-BY-NC source is listed above: WorldSpeech
(~221 h, 24 kHz native — by far the largest Mongolian corpus) is CC-BY-NC-4.0
and is excluded from the default build for this reason.

## Intended use, and what this corpus should not be used for

Built to finetune a Mongolian text-to-speech model. It is published so that
Khalkha Cyrillic — a language with no commercially-usable open TTS corpus — has
one.

**The consent basis is narrower than the technical capability.** Common Voice
contributors dedicated their recordings CC0 for speech research, and FLEURS and
MBSpeech speakers recorded for read-speech benchmarks. None of them consented to
having their individual voice cloned. A zero-shot TTS model finetuned on this
corpus can reproduce a recognisable voice from roughly ten seconds of audio, and
`client_id` groups every clip by contributor — so the corpus supports building a
per-speaker voice whether or not that speaker would agree to it.

Out of scope, and asked of anyone who uses this:

- **do not** synthesise a named or identifiable person's voice without that
  person's explicit consent;
- **do not** present synthetic audio as a real recording of anyone;
- **do not** use it for voice-biometric spoofing, or to attack a system that
  authenticates people by voice;
- **do not** redistribute per-speaker subsets in a form that targets an
  individual contributor.

There is **no watermarking** in this corpus or in the models trained on it, so
audio produced from it cannot be detected as synthetic by any downstream tool.
Anyone shipping a voice built from this should say so where listeners can see
it. If you are a contributor to any source corpus and want your clips removed,
open an issue on this dataset and they will be dropped from the next build.

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
