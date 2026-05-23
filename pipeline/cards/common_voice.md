---
language:
- mn
license: cc0-1.0
task_categories:
- automatic-speech-recognition
task_ids: []
pretty_name: "Common Voice 25 — Mongolian (Clean)"
tags:
- mongolian
- speech
- audio
- asr
- common-voice
---

# Common Voice 25 — Mongolian (Quality-Filtered)

This is a **quality-filtered** subset of the [Mozilla Common Voice 25.0 (2026-03-09)](https://commonvoice.mozilla.org/) Mongolian (`mn`) dataset, cleaned for use with [oron-tts](https://github.com/btseee/oron-tts) (F5-TTS / Flow Matching TTS).

## Source

Derived from `mozilla-foundation/common_voice_17_0` config `mn` (cv-corpus-25.0-2026-03-09).
Original dataset: 95,905 clips · 130.32 hours · 609 speakers · 6,114 sentences.

## Cleaning Pipeline

6-stage automated quality filter, thresholds calibrated for Mongolian TTS training (low-resource language; DeepFilterNet denoising applied downstream in oron-tts):

| Stage | Method | Threshold |
|---|---|---|
| 1. Format normalization | librosa | mono · 16 kHz |
| 2. Voice activity detection | Silero VAD | ≥25 % speech frames |
| 3. SNR filter | RMS-based SNR | ≥8 dB |
| 4. Pitch metadata | CREPE F0 | recorded when available; not a rejection gate |
| 5. AI quality score | DNSMOS P.835 | OVR ≥2.2 · SIG ≥2.4 · BAK ≥2.0 |
| 6. Full sentence verification | Whisper large-v3 + CER | CER ≤0.35, or ≤0.50 when length ratio is 0.75–1.25 |

Stage 6 confirms that the speaker **actually read the full sentence**. CER above 0.35 can still pass only when it is at most 0.50 and the ASR output length stays close to the target sentence, which compensates for Whisper's weaker Mongolian accuracy without accepting truncated clips.

Clips are kept between **1–30 seconds** to match oron-tts training limits. All passing clips are peak-normalized to −1 dBFS and resampled to **24 kHz**.

## Schema

| Field | Type | Description |
|---|---|---|
| `client_id` | string | Speaker UUID hash |
| `path` | string | Original clip filename |
| `audio` | Audio(24000) | Cleaned audio resampled to 24 kHz |
| `sentence` | string | Ground-truth Mongolian sentence |
| `up_votes` | int32 | Community up-votes |
| `down_votes` | int32 | Community down-votes |
| `age` | string | Speaker age group |
| `gender` | string | Speaker gender |
| `accents` | string | Speaker accent |
| `variant` | string | Language variant |
| `segment` | string | Custom dataset segment |
| `locale` | string | Locale code (`mn`) |
| `snr_db` | float32 | Signal-to-noise ratio (dB) |
| `mean_f0_hz` | float32 | Mean fundamental frequency (Hz) |
| `pitch_confidence` | float32 | CREPE pitch confidence (0–1) |
| `dnsmos_sig` | float32 | DNSMOS P.835 signal quality |
| `dnsmos_bak` | float32 | DNSMOS P.835 background noise |
| `dnsmos_ovr` | float32 | DNSMOS P.835 overall MOS |
| `dnsmos_p808` | float32 | DNSMOS P.808 MOS |
| `cer` | float32 | Character error rate vs. ground truth |
| `asr_transcript` | string | Whisper large-v3 transcription |
| `duration_s` | float32 | Clip duration in seconds |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("btsee/common-voices-25-mn")
sample = ds["train"][0]
audio = sample["audio"]["array"]
text  = sample["sentence"]
```

## Language

Mongolian (`mn`) — script: Cyrillic (Монгол Кирилл үсэг)

## License

[CC0 1.0 Public Domain](https://creativecommons.org/public-domain/cc0/)
