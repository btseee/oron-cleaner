---
language:
- mn
license: cc-by-nc-4.0
task_categories:
- automatic-speech-recognition
pretty_name: "WorldSpeech — Mongolian (Clean)"
tags:
- mongolian
- speech
- audio
- parliament
---

# WorldSpeech — Mongolian (Quality-Filtered)

A quality-filtered version of the Mongolian (`mn_mn`) subset of [WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech), cleaned for use with [oron-tts](https://github.com/btseee/oron-tts) (F5-TTS / Flow Matching TTS).

## Source

Derived from `disco-eth/WorldSpeech` config `mn_mn` (~181 hours, avg DNSMOS OVR 2.80).
Source: Mongolian Parliament sessions (public record) + Latter-day Saints addresses (CC0).

## Cleaning Pipeline

6-stage automated quality filter, thresholds calibrated for Mongolian TTS training (low-resource language; DeepFilterNet denoising applied downstream in oron-tts). Same thresholds as `btsee/common-voices-25-mn` and `btsee/fleurs-mn`:

| Stage | Method | Threshold |
|---|---|---|
| 1. Format normalization | librosa | mono · 16 kHz |
| 2. Voice activity detection | Silero VAD | ≥25 % speech frames |
| 3. SNR filter | RMS-based SNR | ≥8 dB |
| 4. Pitch & clarity | CREPE F0 | F0 ≥50 Hz · confidence ≥0.25 |
| 5. AI quality score | DNSMOS P.835 | OVR ≥2.2 · SIG ≥2.4 · BAK ≥2.0 |
| 6. Full sentence verification | Whisper large-v3 + CER | CER ≤0.35 · length ratio ≥0.40 |

Ground truth: `human_transcript` field.
Pre-filter: clips with original `snr < 8 dB` skipped before processing.

All passing clips peak-normalized to −1 dBFS and resampled to **24 kHz**.

## Schema

All original WorldSpeech fields preserved. Original `asr_transcript` and `cer` renamed with `original_` prefix. New quality metrics prefixed `clean_`:

| Field | Type | Description |
|---|---|---|
| `audio` | Audio(16000) | Decoded audio at 16 kHz |
| `human_transcript` | string | Human-verified transcript |
| `original_asr_transcript` | string | WorldSpeech ASR output |
| `original_cer` | float32 | WorldSpeech CER |
| `snr` | float32 | WorldSpeech WADA-SNR estimate |
| `dnsmos_sig` | float32 | WorldSpeech DNSMOS signal |
| `dnsmos_bak` | float32 | WorldSpeech DNSMOS background |
| `dnsmos_ovr` | float32 | WorldSpeech DNSMOS overall |
| `dnsmos_p808` | float32 | WorldSpeech DNSMOS P.808 |
| `duration` | float32 | Original segment duration (s) |
| `source` | string | Source identifier |
| `source_url` | string | URL to original recording |
| `source_start_s` | float32 | Start offset (s) |
| `source_end_s` | float32 | End offset (s) |
| `session_date` | string | ISO-8601 recording date |
| `segment_id` | string | Unique segment ID |
| `language` | string | BCP-47 language tag (`mn-MN`) |
| `country` | string | ISO 3166-1 country code (`MN`) |
| `clean_snr_db` | float32 | Re-measured SNR (dB) |
| `clean_mean_f0_hz` | float32 | Mean F0 (Hz) |
| `clean_pitch_confidence` | float32 | CREPE pitch confidence |
| `clean_dnsmos_sig` | float32 | Re-scored DNSMOS signal |
| `clean_dnsmos_bak` | float32 | Re-scored DNSMOS background |
| `clean_dnsmos_ovr` | float32 | Re-scored DNSMOS overall |
| `clean_dnsmos_p808` | float32 | Re-scored DNSMOS P.808 |
| `clean_cer` | float32 | CER vs. human_transcript |
| `clean_asr_transcript` | string | Whisper large-v3 output |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("btsee/worldspeech-mn", "mn_mn")
sample = ds["train"][0]
```

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
Parliamentary proceedings sources: public record.
