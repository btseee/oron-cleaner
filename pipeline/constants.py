from pathlib import Path

SAMPLE_RATE: int = 16_000
OUTPUT_SAMPLE_RATE: int = 24_000

OUTPUT_DIR: Path = Path("output")
FILTER_POLICY_VERSION: str = "v3_pitch_diagnostic_cer_rescue"

# Duration gates (seconds)
# oron-tts trains with max_duration_s=30.0; very short clips hurt mel alignment
MIN_DURATION_S: float = 1.0
MAX_DURATION_S: float = 30.0

# VAD — Mongolian speech has natural inter-word pauses; 25 % voiced frames is sufficient
VAD_MIN_SPEECH_RATIO: float = 0.25

# SNR — DeepFilterNet denoising happens downstream in oron-tts; 8 dB is the usable floor
SNR_MIN_DB: float = 8.0

# Pitch (CREPE) — diagnostic metadata only. VAD, DNSMOS, and Whisper CER decide pass/fail.
# Low CREPE periodicity is common on this Mongolian FLEURS audio and should not drop clips.
PITCH_MIN_CONF: float = 0.25

# DNSMOS P.835 — community recordings (CV, FLEURS) score ~2.0–3.0;
# oron-tts's DeepFilterNet denoiser handles residual noise
DNSMOS_MIN_OVR: float = 2.2
DNSMOS_MIN_SIG: float = 2.4
DNSMOS_MIN_BAK: float = 2.0

# Whisper CER — Mongolian Whisper output often has high character error even
# when it clearly follows the target sentence. CER <= 0.35 passes normally;
# CER <= 0.50 is rescued only when ASR length is close to the target.
MAX_CER: float = 0.35
MAX_RESCUE_CER: float = 0.50
MIN_LEN_RATIO: float = 0.40
MIN_RESCUE_LEN_RATIO: float = 0.75
MAX_RESCUE_LEN_RATIO: float = 1.25

