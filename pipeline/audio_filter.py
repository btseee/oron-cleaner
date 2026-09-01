"""AudioQualityFilter — quality gate for speech clips destined for TTS training.

Stages, in order:
  1. Load and normalise format   (mono, 16 kHz, float32)
  2. Duration
  3. Clipping and DC offset
  4. Voice activity              (Silero VAD, edge-trim only)
  5. SNR                         (speech regions vs true non-speech regions)
  6. Bandwidth                   (lowpass shelf detection)
  7. Pitch metadata              (librosa pyin; diagnostic, never rejects)
  8. AI MOS score                (DNSMOS P.835)
  9. Transcript agreement        (wav2vec2-xlsr-mongolian + CER)
 10. Output preparation          (resample to 24 kHz, peak-normalise)

Two properties matter more than any individual threshold:

* **The published audio is edge-trimmed, never spliced.** The previous
  implementation concatenated Silero's speech segments, which deleted every
  interior pause and butt-joined the pieces. For a TTS corpus that destroys
  prosodic pausing and injects a discontinuity at each join -- and it corrupted
  every downstream measurement, since SNR, DNSMOS and the ASR all ran on the
  spliced signal.

* **Measurements are taken on the signal they describe.** SNR needs the real
  noise regions, so it runs on the untrimmed audio using the VAD boundaries;
  duration describes the audio actually shipped, not the pre-trim input.
"""

import logging

import jiwer
import librosa
import numpy as np
import torch
import torchaudio
from oron_tts.text import MongolianNormalizer
from silero_vad import get_speech_timestamps, load_silero_vad
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

from .clip_result import ClipResult
from .constants import (
    DNSMOS_MIN_BAK,
    DNSMOS_MIN_OVR,
    DNSMOS_MIN_SIG,
    MAX_CLIPPED_RATIO,
    MAX_DC_OFFSET,
    MAX_DURATION_S,
    MIN_BANDWIDTH_HZ,
    MIN_DURATION_S,
    OUTPUT_SAMPLE_RATE,
    SAMPLE_RATE,
    SNR_MIN_DB,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_SPEECH_RATIO,
    VAD_SPEECH_PAD_MS,
    VAD_SPEECH_THRESHOLD,
)
from .dsp import (
    clipped_ratio,
    dc_offset,
    edge_trim_bounds,
    estimate_snr,
    for_comparison,
    measure_bandwidth,
    median_f0,
    reading_passes,
)

log = logging.getLogger(__name__)

# whisper-large-v3 has a CER floor of 0.311 on clean, correctly-transcribed
# Mongolian; this model measures 0.123 median at a fifth of the parameters.
ASR_MODEL = "bayartsogt/wav2vec2-large-xlsr-mongolian"


class AudioQualityFilter:
    """Load models once, then call process_clip() per clip.

    Not thread-safe -- use one instance per process.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._normalizer = MongolianNormalizer()

        log.info("Loading Silero VAD …")
        self._vad_model = load_silero_vad()

        log.info("Loading %s …", ASR_MODEL)
        from transformers import AutoModelForCTC, AutoProcessor

        self._asr_processor = AutoProcessor.from_pretrained(ASR_MODEL)
        self._asr = AutoModelForCTC.from_pretrained(ASR_MODEL).to(device).eval()

        log.info("Loading DNSMOS …")
        self._dnsmos = DeepNoiseSuppressionMeanOpinionScore(
            fs=SAMPLE_RATE, personalized=False
        ).to(device)

        log.info("All models loaded.")

    # ── Stage 1 ── Format normalisation ───────────────────────────────────

    def _load_audio(self, audio_input) -> tuple[np.ndarray | None, str]:
        """Accept a HuggingFace Audio dict, a torchcodec decoder, or a path."""
        try:
            if isinstance(audio_input, dict):
                arr = np.array(audio_input["array"], dtype=np.float32)
                sr = int(audio_input["sampling_rate"])
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
            elif hasattr(audio_input, "get_all_samples"):
                samples = audio_input.get_all_samples()
                arr = samples.data.float().mean(0).cpu().numpy()
                sr = int(samples.sample_rate)
            else:
                # torchaudio handles MP3/WAV/FLAC without audioread
                waveform, sr = torchaudio.load(str(audio_input))
                arr = waveform.mean(0).numpy()
            if sr != SAMPLE_RATE:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
            return arr.astype(np.float32), ""
        except Exception as exc:
            return None, str(exc)

    # ── Stage 4 ── Voice activity, edge-trim only ─────────────────────────

    def _run_vad(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray | None, list[dict] | None, str]:
        """Trim leading and trailing silence, preserving interior pauses.

        Returns (trimmed audio, speech timestamps on the ORIGINAL audio, reason).
        The timestamps are returned so SNR can find the real noise regions.
        """
        tensor = torch.from_numpy(audio).float()
        try:
            timestamps = get_speech_timestamps(
                tensor,
                self._vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=VAD_SPEECH_THRESHOLD,
                min_speech_duration_ms=VAD_MIN_SPEECH_MS,
                min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                speech_pad_ms=VAD_SPEECH_PAD_MS,
                return_seconds=False,
            )
        except Exception as exc:
            return None, None, f"vad_error:{exc}"

        if not timestamps:
            return None, None, "vad_no_speech"

        speech_samples = sum(t["end"] - t["start"] for t in timestamps)
        speech_ratio = speech_samples / max(len(audio), 1)
        if speech_ratio < VAD_MIN_SPEECH_RATIO:
            return None, timestamps, f"speech_ratio_{speech_ratio:.2f}"

        # Edge-trim: keep everything between the first and last speech segment,
        # interior pauses included. Splicing the segments together would delete
        # natural pausing and leave a click at every join.
        start, end = edge_trim_bounds(timestamps)
        return audio[start:end], timestamps, ""

    # ── Stage 8 ── DNSMOS ─────────────────────────────────────────────────

    def _score_dnsmos(self, audio: np.ndarray) -> tuple[bool, dict[str, float], str]:
        try:
            tensor = torch.tensor(audio, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                scores = self._dnsmos(tensor).cpu().numpy().flatten()
            sig, bak, ovr, p808 = (float(s) for s in scores[:4])
        except Exception as exc:
            return False, {}, f"dnsmos_error:{exc}"

        d = {"dnsmos_sig": sig, "dnsmos_bak": bak, "dnsmos_ovr": ovr, "dnsmos_p808": p808}
        if ovr < DNSMOS_MIN_OVR:
            return False, d, f"dnsmos_ovr_{ovr:.2f}"
        if sig < DNSMOS_MIN_SIG:
            return False, d, f"dnsmos_sig_{sig:.2f}"
        if bak < DNSMOS_MIN_BAK:
            return False, d, f"dnsmos_bak_{bak:.2f}"
        return True, d, ""

    # ── Stage 9 ── Transcript agreement ───────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        inputs = self._asr_processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self._asr(**inputs).logits
        return self._asr_processor.batch_decode(logits.argmax(-1))[0]

    def _verify_reading(
        self, audio: np.ndarray, ground_truth: str
    ) -> tuple[bool, float, float, str, str]:
        """Returns (passed, cer, length_ratio, asr_text, reject_reason)."""
        # Normalise the ground truth exactly as training will see it, so digits
        # and abbreviations cannot inflate CER. Previously "1990 онд" was scored
        # against "мянга есөн зуун ерэн онд" and rejected as a mismatch.
        try:
            norm_gt = self._normalizer.normalize(ground_truth, strict=False)
        except Exception as exc:
            return False, 1.0, 0.0, "", f"normalize_error:{exc}"
        if not norm_gt.strip():
            return False, 1.0, 0.0, "", "empty_ground_truth"

        try:
            asr_text = self._transcribe(audio)
        except Exception as exc:
            return False, 1.0, 0.0, "", f"asr_error:{exc}"

        gt_cmp = for_comparison(norm_gt)
        asr_cmp = for_comparison(asr_text)
        try:
            cer_val = float(jiwer.cer(gt_cmp, asr_cmp))
        except Exception:
            cer_val = 1.0

        len_ratio = len(asr_cmp) / max(len(gt_cmp), 1)
        passed, reason = reading_passes(cer=cer_val, length_ratio=len_ratio)
        return passed, cer_val, len_ratio, asr_text, reason

    # ── Stage 10 ── Output preparation ────────────────────────────────────

    def _prepare_output_audio(self, audio: np.ndarray) -> np.ndarray:
        resampled = librosa.resample(
            audio, orig_sr=SAMPLE_RATE, target_sr=OUTPUT_SAMPLE_RATE
        )
        peak = float(np.abs(resampled).max())
        if peak < 1e-8:
            return resampled.astype(np.float32)
        target_peak = 10 ** (-1.0 / 20.0)
        return np.clip(
            resampled / (peak + 1e-7) * target_peak, -target_peak, target_peak
        ).astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────

    def process_clip(self, audio_input, ground_truth_text: str) -> ClipResult:
        audio, err = self._load_audio(audio_input)
        if audio is None:
            return ClipResult(passed=False, reject_stage="load", reject_reason=err)

        raw_duration = len(audio) / SAMPLE_RATE
        if raw_duration < MIN_DURATION_S:
            return ClipResult(passed=False, reject_stage="duration",
                              reject_reason=f"too_short_{raw_duration:.2f}s")
        if raw_duration > MAX_DURATION_S:
            return ClipResult(passed=False, reject_stage="duration",
                              reject_reason=f"too_long_{raw_duration:.2f}s")

        clipped = clipped_ratio(audio)
        if clipped > MAX_CLIPPED_RATIO:
            return ClipResult(passed=False, reject_stage="clipping",
                              reject_reason=f"clipped_{clipped:.4f}")
        dc = dc_offset(audio)
        if dc > MAX_DC_OFFSET:
            return ClipResult(passed=False, reject_stage="clipping",
                              reject_reason=f"dc_offset_{dc:.4f}")

        trimmed, timestamps, reason = self._run_vad(audio)
        if trimmed is None:
            return ClipResult(passed=False, reject_stage="vad", reject_reason=reason)

        # The duration that describes the audio actually shipped. The previous
        # implementation reported the pre-VAD length, inflating both this column
        # and every "total hours" figure in the reports.
        duration_s = len(trimmed) / SAMPLE_RATE
        if duration_s < MIN_DURATION_S:
            return ClipResult(passed=False, reject_stage="vad",
                              reject_reason=f"trimmed_too_short_{duration_s:.2f}s")

        # Untrimmed audio, deliberately: the noise regions are the measurement.
        snr = estimate_snr(audio, timestamps)
        if np.isnan(snr):
            return ClipResult(passed=False, reject_stage="snr",
                              reject_reason="no_silence_to_measure_noise_floor",
                              duration_s=duration_s)
        if snr < SNR_MIN_DB:
            return ClipResult(passed=False, reject_stage="snr",
                              reject_reason=f"snr_{snr:.1f}dB",
                              snr_db=snr, duration_s=duration_s)

        bandwidth = measure_bandwidth(trimmed)
        if bandwidth < MIN_BANDWIDTH_HZ:
            return ClipResult(passed=False, reject_stage="bandwidth",
                              reject_reason=f"bandwidth_{bandwidth:.0f}Hz",
                              snr_db=snr, bandwidth_hz=bandwidth,
                              duration_s=duration_s)

        mean_f0, voiced_frac = median_f0(trimmed)

        dnsmos_ok, dnsmos, reason = self._score_dnsmos(trimmed)
        if not dnsmos_ok:
            return ClipResult(passed=False, reject_stage="dnsmos", reject_reason=reason,
                              snr_db=snr, bandwidth_hz=bandwidth, mean_f0_hz=mean_f0,
                              pitch_confidence=voiced_frac, duration_s=duration_s,
                              **dnsmos)

        reading_ok, cer_val, len_ratio, asr_text, reason = self._verify_reading(
            trimmed, ground_truth_text
        )
        if not reading_ok:
            return ClipResult(passed=False, reject_stage="cer", reject_reason=reason,
                              snr_db=snr, bandwidth_hz=bandwidth, mean_f0_hz=mean_f0,
                              pitch_confidence=voiced_frac, cer=cer_val,
                              len_ratio=len_ratio, asr_transcript=asr_text,
                              duration_s=duration_s, **dnsmos)

        return ClipResult(
            passed=True,
            snr_db=snr,
            bandwidth_hz=bandwidth,
            mean_f0_hz=mean_f0,
            pitch_confidence=voiced_frac,
            cer=cer_val,
            len_ratio=len_ratio,
            asr_transcript=asr_text,
            duration_s=duration_s,
            audio_normalized=self._prepare_output_audio(trimmed),
            **dnsmos,
        )
