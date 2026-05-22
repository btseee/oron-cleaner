from pathlib import Path

SAMPLE_RATE: int = 16_000
OUTPUT_SAMPLE_RATE: int = 24_000

OUTPUT_DIR: Path = Path("output")

# Duration gates (seconds)
# oron-tts inference splits at ~120 chars / ~25 s; very short clips hurt mel alignment
MIN_DURATION_S: float = 1.0
MAX_DURATION_S: float = 25.0

# VAD — Mongolian speech has natural inter-word pauses; 25 % voiced frames is sufficient
VAD_MIN_SPEECH_RATIO: float = 0.25

# SNR — DeepFilterNet denoising happens downstream in oron-tts; 8 dB is the usable floor
SNR_MIN_DB: float = 8.0

# Pitch (CREPE) — covers all adult Mongolian/Kazakh speakers; low confidence on non-English
# is expected; we just want a voiced-speech sanity check, not studio pitch tracking
PITCH_MIN_HZ: float = 50.0
PITCH_MIN_CONF: float = 0.25

# DNSMOS P.835 — community recordings (CV, FLEURS) score ~2.0–3.0;
# oron-tts's DeepFilterNet denoiser handles residual noise
DNSMOS_MIN_OVR: float = 2.2
DNSMOS_MIN_SIG: float = 2.4
DNSMOS_MIN_BAK: float = 2.0

# Whisper CER — Mongolian WER with Whisper large-v3 is typically 25–40 %;
# 0.35 CER keeps good transcripts while tolerating imperfect ASR on MN
# MIN_LEN_RATIO guards against clipped recordings without over-filtering
MAX_CER: float = 0.35
MIN_LEN_RATIO: float = 0.40

