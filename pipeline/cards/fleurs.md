---
language:
- mn
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
- audio-classification
task_ids:
- automatic-speech-recognition
- language-identification
pretty_name: "FLEURS — Mongolian (Clean)"
tags:
- mongolian
- speech
- audio
- fleurs
- few-shot
- studio-quality
configs:
- config_name: mn_mn
  data_files:
    - split: train
      path: data/train-*
    - split: validation
      path: data/validation-*
    - split: test
      path: data/test-*
---

# FLEURS — Mongolian (Studio-Quality Cleaned)

A quality-filtered version of the [FLEURS](https://huggingface.co/datasets/google/fleurs) (`mn_mn`) Mongolian benchmark dataset.

## Source

Derived from `google/fleurs` config `mn_mn`. FLEURS is the speech version of the FLoRes machine translation benchmark, covering 2,009 n-way parallel sentences across 102 languages.

## Cleaning Pipeline

Same 6-stage pipeline as `btsee/common-voices-25-mn` (see above).
Ground truth for sentence verification: `raw_transcription` field.

All passing clips were loudness-normalized to **−23 LUFS**.

## Schema

All original FLEURS fields preserved, plus computed quality metrics:

| Field | Type | Description |
|---|---|---|
| `id` | int32 | Sample ID |
| `num_samples` | int32 | Number of audio samples |
| `path` | string | Audio file path |
| `audio` | Audio(16000) | Decoded audio at 16 kHz |
| `raw_transcription` | string | Original (unnormalized) transcription |
| `transcription` | string | Normalized transcription |
| `gender` | int32 | Speaker gender class |
| `lang_id` | int32 | Language class ID |
| `language` | string | Language name |
| `lang_group_id` | int32 | Language group class ID |
| `snr_db` | float32 | SNR in dB |
| `mean_f0_hz` | float32 | Mean F0 (Hz) |
| `pitch_confidence` | float32 | CREPE pitch confidence |
| `dnsmos_sig` | float32 | DNSMOS signal quality |
| `dnsmos_bak` | float32 | DNSMOS background noise |
| `dnsmos_ovr` | float32 | DNSMOS overall MOS |
| `dnsmos_p808` | float32 | DNSMOS P.808 MOS |
| `cer` | float32 | CER vs. raw_transcription |
| `asr_transcript` | string | Whisper large-v3 output |
| `duration_s` | float32 | Duration in seconds |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("btsee/fleurs-mn", "mn_mn")
sample = ds["train"][0]
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
