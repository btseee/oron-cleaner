from pathlib import Path

SAMPLE_RATE: int = 16_000
OUTPUT_SAMPLE_RATE: int = 24_000

OUTPUT_DIR: Path = Path("output")

# Duration gates (seconds)
MIN_DURATION_S: float = 1.0
MAX_DURATION_S: float = 30.0

# VAD
VAD_MIN_SPEECH_RATIO: float = 0.40

# SNR
SNR_MIN_DB: float = 10.0

# Pitch (CREPE)
PITCH_MIN_HZ: float = 55.0
PITCH_MIN_CONF: float = 0.40

# DNSMOS P.835
DNSMOS_MIN_OVR: float = 2.5
DNSMOS_MIN_SIG: float = 2.7
DNSMOS_MIN_BAK: float = 2.3

# Whisper CER verification
MAX_CER: float = 0.35
MIN_LEN_RATIO: float = 0.50

