# oron-cleaner

**Mongolian speech dataset cleaning and HuggingFace upload pipeline for oron-tts.**

Processes three public Mongolian (`mn`) speech corpora through a practical automated quality filter — removing non-speech audio, noisy clips, and transcript mismatches while preserving usable low-resource speech — then publishes the cleaned datasets to HuggingFace.

| Output dataset | Source | License |
| --- | --- | --- |
| [`btsee/common-voices-25-mn`](https://huggingface.co/datasets/btsee/common-voices-25-mn) | `mozilla-foundation/common_voice_17_0` (mn) | CC0 1.0 |
| [`btsee/fleurs-mn`](https://huggingface.co/datasets/btsee/fleurs-mn) | `google/fleurs` (mn\_mn) | CC BY 4.0 |
| [`btsee/worldspeech-mn`](https://huggingface.co/datasets/btsee/worldspeech-mn) | `disco-eth/WorldSpeech` (mn\_mn) | CC BY-NC 4.0 |

---

## Why

Mongolian TTS and ASR models trained on raw crowd-sourced data suffer from background noise, clipped recordings, and speakers who skip or mumble words. This pipeline keeps the checks that protect text/audio alignment while avoiding pitch-detector false negatives that are common on Mongolian FLEURS audio.

---

## Pipeline

```text
Raw clip
  │
  ├─ 1. Format normalisation   mono · 16 kHz · float32
    ├─ 2. Duration gate          1 s – 30 s
    ├─ 3. Silero VAD             ≥ 25 % speech frames
    ├─ 4. RMS-based SNR          ≥ 8 dB
    ├─ 5. CREPE pitch            diagnostic metadata only
    ├─ 6. DNSMOS P.835           OVR ≥ 2.2 · SIG ≥ 2.4 · BAK ≥ 2.0
    └─ 7. Whisper large-v3 CER   CER ≤ 0.35, or ≤ 0.50 when length-ratio is 0.75–1.25
         │
      └─ PASS → peak-normalise and resample to 24 kHz
                   → save + upload to HuggingFace
```

Stage 7 is the most critical: Whisper large-v3 transcribes each clip in Mongolian and the Character Error Rate against the ground-truth sentence confirms the speaker actually read the full sentence.

---

## Repository layout

```text
oron-cleaner/
├── clean_pipeline.py           CLI entry point
├── pipeline/
│   ├── constants.py            All quality thresholds
│   ├── clip_result.py          ClipResult dataclass
│   ├── audio_filter.py         AudioQualityFilter (all 7 stages)
│   ├── stats.py                CleaningStats + RejectionLog
│   ├── checkpoint.py           Save / resume checkpoints
│   ├── processor.py            Shared per-split iteration loop
│   ├── upload.py               HuggingFace upload helpers
│   ├── datasets/
│   │   ├── common_voice.py
│   │   ├── fleurs.py
│   │   └── worldspeech.py
│   └── cards/
│       ├── common_voice.md     HF dataset card
│       ├── fleurs.md
│       └── worldspeech.md
├── pyproject.toml
├── requirements.txt
└── tests/
```

---

## Requirements

| Requirement | Minimum |
| --- | --- |
| Python | **3.14.5+** |
| VRAM (GPU) | 24 GB recommended (Whisper large-v3 + CREPE) |
| VRAM (CPU-only) | works but ~10× slower |
| Disk | ~50 GB for Common Voice download cache |
| ffmpeg | system-wide install required |

### Install ffmpeg

```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

---

## Installation

```bash
git clone https://github.com/BBadral/oron-cleaner.git
cd oron-cleaner

python3.14 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Or install as a package (gives you the `oron-cleaner` CLI):

```bash
pip install .
```

---

## Usage

```bash
# Clean and upload all three datasets
python clean_pipeline.py --hf-token hf_YOUR_TOKEN_HERE

# Only one dataset
python clean_pipeline.py --hf-token hf_... --datasets cv
python clean_pipeline.py --hf-token hf_... --datasets fleurs
python clean_pipeline.py --hf-token hf_... --datasets ws

# Multiple datasets
python clean_pipeline.py --hf-token hf_... --datasets cv,fleurs

# Force CPU (slow but no GPU needed)
python clean_pipeline.py --hf-token hf_... --device cpu

# Start fresh, ignoring any saved checkpoints
python clean_pipeline.py --hf-token hf_... --no-resume
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--hf-token` | required | HuggingFace write token |
| `--datasets` | `all` | `all`, `cv`, `fleurs`, `ws`, or comma-separated combination |
| `--device` | auto | `cuda` or `cpu` |
| `--resume` / `--no-resume` | resume | Continue from last checkpoint |

---

## Checkpointing

Processing 95k+ Common Voice clips takes 24–48 h on a single GPU. The pipeline saves a checkpoint every 500 clips under `output/checkpoints/{dataset}_{split}_{policy_version}/batch_XXXXXX/`. If interrupted, re-run with the same command and policy version — it will skip already-processed batches automatically.

---

## Using the cleaned datasets

```python
from datasets import load_dataset

# Common Voice 25 Mongolian (clean)
cv = load_dataset("btsee/common-voices-25-mn")
sample = cv["train"][0]
audio  = sample["audio"]["array"]   # float32 numpy array, 24 kHz
text   = sample["sentence"]         # ground-truth sentence
cer    = sample["cer"]              # Whisper CER vs. ground truth
snr    = sample["snr_db"]           # measured SNR

# FLEURS Mongolian (clean)
fl = load_dataset("btsee/fleurs-mn")

# WorldSpeech Mongolian (clean)
ws = load_dataset("btsee/worldspeech-mn")
```

---

## Quality thresholds

All thresholds live in [`pipeline/constants.py`](pipeline/constants.py) and can be changed without touching any other file.

| Constant | Default | Meaning |
| --- | --- | --- |
| `MIN_DURATION_S` | 1.0 s | Minimum clip length |
| `FILTER_POLICY_VERSION` | `v3_pitch_diagnostic_cer_rescue` | Checkpoint/log namespace for the active filter policy |
| `MAX_DURATION_S` | 30.0 s | Maximum clip length |
| `VAD_MIN_SPEECH_RATIO` | 0.25 | Min fraction of frames classified as speech |
| `SNR_MIN_DB` | 8.0 dB | Minimum signal-to-noise ratio |
| `PITCH_MIN_CONF` | 0.25 | CREPE voiced-frame threshold for metadata only |
| `DNSMOS_MIN_OVR` | 2.2 | DNSMOS P.835 overall MOS floor |
| `DNSMOS_MIN_SIG` | 2.4 | DNSMOS signal quality floor |
| `DNSMOS_MIN_BAK` | 2.0 | DNSMOS background noise floor |
| `MAX_CER` | 0.35 | Normal maximum character error rate vs. ground truth |
| `MAX_RESCUE_CER` | 0.50 | Rescue CER ceiling when ASR length is close to target |
| `MIN_LEN_RATIO` | 0.40 | Hard minimum ASR length ratio |
| `MIN_RESCUE_LEN_RATIO` | 0.75 | Minimum length ratio for CER rescue |
| `MAX_RESCUE_LEN_RATIO` | 1.25 | Maximum length ratio for CER rescue |

---

## Python 3.14 notes

This project targets **Python 3.14.5+** (3.14.0–3.14.4 had a GC regression that caused memory pressure in long-running ML loops; 3.14.5 reverted to the stable generational collector).

- **PEP 649** — annotations are deferred by default; `from __future__ import annotations` is not used in this codebase.
- **PEP 750** (t-strings) and **PEP 758** (`except` without brackets) are available but not used here.

---

## Output artefacts

After each run the pipeline writes locally:

| File | Content |
| --- | --- |
| `output/cleaning_report_{cv,fleurs,ws}.txt` | Per-stage rejection counts and pass-rate summary |
| `output/logs/rejected_{dataset}_{split}_{policy_version}.csv` | Per-clip rejection log (clip\_id, stage, reason, ground\_truth) |
| `output/checkpoints/` | Pickle snapshots for resume |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE)
