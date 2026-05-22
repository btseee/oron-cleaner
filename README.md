# oron-cleaner

**Studio-quality Mongolian speech dataset cleaning and HuggingFace upload pipeline.**

Processes three public Mongolian (`mn`) speech corpora through a rigorous 6-stage automated quality filter — removing noise, mumbling, truncated readings, and non-speech audio — then publishes the cleaned datasets to HuggingFace.

| Output dataset | Source | License |
| --- | --- | --- |
| [`btsee/common-voices-25-mn`](https://huggingface.co/datasets/btsee/common-voices-25-mn) | `mozilla-foundation/common_voice_17_0` (mn) | CC0 1.0 |
| [`btsee/fleurs-mn`](https://huggingface.co/datasets/btsee/fleurs-mn) | `google/fleurs` (mn\_mn) | CC BY 4.0 |
| [`btsee/worldspeech-mn`](https://huggingface.co/datasets/btsee/worldspeech-mn) | `disco-eth/WorldSpeech` (mn\_mn) | CC BY-NC 4.0 |

---

## Why

Mongolian TTS and ASR models trained on raw crowd-sourced data suffer from background noise, low-confidence pitch, and speakers who skip or mumble words. This pipeline enforces a hard quality floor so every retained clip is suitable for model training without further preprocessing.

---

## Pipeline

```text
Raw clip
  │
  ├─ 1. Format normalisation   mono · 16 kHz · float32
  ├─ 2. Duration gate          1 s – 15 s
  ├─ 3. Silero VAD             ≥ 60 % speech frames
  ├─ 4. RMS-based SNR          ≥ 15 dB
  ├─ 5. CREPE pitch (torchcrepe)  F₀ ≥ 70 Hz · confidence ≥ 0.55
  ├─ 6. DNSMOS P.835           OVR ≥ 2.8 · SIG ≥ 3.0 · BAK ≥ 2.5
  └─ 7. Whisper large-v3 CER   CER ≤ 0.25 · length-ratio ≥ 0.60
         │
         └─ PASS → loudness-normalise to −23 LUFS (ITU-R BS.1770-4)
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

Processing 95k+ Common Voice clips takes 24–48 h on a single GPU. The pipeline saves a pickle checkpoint every 500 clips under `checkpoints/{dataset}_{split}/batch_XXXXXX/records.pkl`. If interrupted, re-run with the same command — it will skip already-processed batches automatically.

---

## Using the cleaned datasets

```python
from datasets import load_dataset

# Common Voice 25 Mongolian (clean)
cv = load_dataset("btsee/common-voices-25-mn")
sample = cv["train"][0]
audio  = sample["audio"]["array"]   # float32 numpy array, 16 kHz
text   = sample["sentence"]         # ground-truth sentence
cer    = sample["cer"]              # Whisper CER vs. ground truth
snr    = sample["snr_db"]           # measured SNR

# FLEURS Mongolian (clean)
fl = load_dataset("btsee/fleurs-mn", "mn_mn")

# WorldSpeech Mongolian (clean)
ws = load_dataset("btsee/worldspeech-mn", "mn_mn")
```

---

## Quality thresholds

All thresholds live in [`pipeline/constants.py`](pipeline/constants.py) and can be changed without touching any other file.

| Constant | Default | Meaning |
| --- | --- | --- |
| `MIN_DURATION_S` | 1.0 s | Minimum clip length |
| `MAX_DURATION_S` | 15.0 s | Maximum clip length |
| `VAD_MIN_SPEECH_RATIO` | 0.60 | Min fraction of frames classified as speech |
| `SNR_MIN_DB` | 15.0 dB | Minimum signal-to-noise ratio |
| `PITCH_MIN_HZ` | 70 Hz | Minimum mean fundamental frequency |
| `PITCH_MIN_CONF` | 0.55 | Minimum CREPE voiced-frame confidence |
| `DNSMOS_MIN_OVR` | 2.8 | DNSMOS P.835 overall MOS floor |
| `DNSMOS_MIN_SIG` | 3.0 | DNSMOS signal quality floor |
| `DNSMOS_MIN_BAK` | 2.5 | DNSMOS background noise floor |
| `MAX_CER` | 0.25 | Maximum character error rate vs. ground truth |
| `MIN_LEN_RATIO` | 0.60 | Min ratio of ASR length to ground-truth length |
| `TARGET_LUFS` | −23.0 | Loudness normalisation target |

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
| `output/logs/rejected_{dataset}_{split}.csv` | Per-clip rejection log (clip\_id, stage, reason, ground\_truth) |
| `output/checkpoints/` | Pickle snapshots for resume |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE)
