"""
AudioQualityFilter — six-stage quality gate for speech clips.

Stages:
  1. Format normalisation  (mono, 16 kHz, float32)
  2. Voice activity        (Silero VAD)
  3. SNR                   (RMS-based)
    4. Pitch metadata        (CREPE F0)
  5. AI MOS score          (DNSMOS P.835)
  6. Full-sentence reading (Whisper large-v3 + CER)
  7. Output preparation    (resample to 24 kHz, peak-normalise)
"""

import logging
import re
import unicodedata

import jiwer
import torchcrepe
import librosa
import numpy as np
import torch
import torchaudio
import whisper
from silero_vad import load_silero_vad, get_speech_timestamps
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

from .clip_result import ClipResult
from .constants import (
    DNSMOS_MIN_BAK,
    DNSMOS_MIN_OVR,
    DNSMOS_MIN_SIG,
    MAX_CER,
    MAX_RESCUE_CER,
    MAX_RESCUE_LEN_RATIO,
    MIN_DURATION_S,
    MAX_DURATION_S,
    MIN_LEN_RATIO,
    MIN_RESCUE_LEN_RATIO,
    OUTPUT_SAMPLE_RATE,
    PITCH_MIN_CONF,
    SAMPLE_RATE,
    SNR_MIN_DB,
    VAD_MIN_SPEECH_RATIO,
)

log = logging.getLogger(__name__)


class AudioQualityFilter:
    """
    Load models once, then call process_clip() for each audio clip.
    Not thread-safe — use one instance per process.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

        log.info("Loading Silero VAD …")
        self._vad_model = load_silero_vad()

        log.info("Loading Whisper large-v3 …")
        self._whisper = whisper.load_model("large-v3", device=device)

        log.info("Loading DNSMOS …")
        self._dnsmos = DeepNoiseSuppressionMeanOpinionScore(
            fs=SAMPLE_RATE, personalized=False
        ).to(device)

        log.info("All models loaded.")

    # ── Stage 1 ── Format normalisation ───────────────────────────────────

    def _load_audio(self, audio_input) -> tuple[np.ndarray | None, str]:
        """
        Accept a HuggingFace Audio dict, a torchcodec AudioDecoder object,
        or a file path (str / Path).
        Returns (float32 array at SAMPLE_RATE, error_message).
        """
        try:
            if isinstance(audio_input, dict):
                arr = np.array(audio_input["array"], dtype=np.float32)
                sr = int(audio_input["sampling_rate"])
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                if sr != SAMPLE_RATE:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
            elif hasattr(audio_input, "get_all_samples"):
                samples = audio_input.get_all_samples()
                arr = samples.data.float().mean(0).cpu().numpy()
                sr = int(samples.sample_rate)
                if sr != SAMPLE_RATE:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
            else:
                # File path — torchaudio handles MP3/WAV/FLAC without audioread
                waveform, sr = torchaudio.load(str(audio_input))
                arr = waveform.mean(0).numpy()
                if sr != SAMPLE_RATE:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
            return arr.astype(np.float32), ""
        except Exception as exc:
            return None, str(exc)

    # ── Stage 2 ── Voice activity detection ───────────────────────────────

    def _run_vad(self, audio: np.ndarray) -> tuple[np.ndarray | None, str]:
        """
        Returns (speech-only audio concatenated, reject_reason).
        reject_reason is empty when the clip passes.
        """
        tensor = torch.from_numpy(audio).float()
        try:
            timestamps = get_speech_timestamps(
                tensor, self._vad_model, sampling_rate=SAMPLE_RATE, return_seconds=False
            )
        except Exception as exc:
            return None, f"vad_error:{exc}"

        if not timestamps:
            return None, "vad_no_speech"

        speech_samples = sum(t["end"] - t["start"] for t in timestamps)
        speech_ratio = speech_samples / max(len(audio), 1)
        if speech_ratio < VAD_MIN_SPEECH_RATIO:
            return None, f"speech_ratio_{speech_ratio:.2f}"

        trimmed = np.concatenate([audio[t["start"]: t["end"]] for t in timestamps])
        return trimmed, ""

    # ── Stage 3 ── SNR ─────────────────────────────────────────────────────

    @staticmethod
    def _estimate_snr(audio: np.ndarray) -> float:
        frame_size = int(0.02 * SAMPLE_RATE)
        energies = sorted(
            np.sqrt(np.mean(f ** 2))
            for i in range(0, len(audio) - frame_size, frame_size)
            if len(f := audio[i: i + frame_size]) == frame_size
        )
        if not energies:
            return 0.0
        n_noise = max(1, len(energies) // 10)
        noise_floor = float(np.mean(energies[:n_noise]))
        signal = float(np.mean(energies[len(energies) // 2:]))
        if noise_floor < 1e-10:
            return 40.0
        return 20.0 * np.log10(signal / noise_floor + 1e-10)

    # ── Stage 4 ── Pitch metadata ──────────────────────────────────────────

    def _check_pitch(self, audio: np.ndarray) -> tuple[bool, float, float, str]:
        """Returns pitch diagnostics without rejecting otherwise valid speech."""
        try:
            # torchcrepe expects (1, time) float32 tensor
            audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            hop = int(0.01 * SAMPLE_RATE)  # 10 ms steps
            frequency, periodicity = torchcrepe.predict(
                audio_t,
                SAMPLE_RATE,
                hop_length=hop,
                fmin=50.0,
                fmax=2000.0,
                model="full",
                return_periodicity=True,
                batch_size=512,
                device=self.device,
                decoder=torchcrepe.decode.viterbi,
            )
            frequency   = frequency.squeeze(0).cpu().numpy()    # (frames,)
            periodicity = periodicity.squeeze(0).cpu().numpy()  # (frames,) voiced confidence
        except Exception as exc:
            log.warning("CREPE pitch diagnostics failed: %s", exc)
            return True, 0.0, 0.0, ""

        voiced = periodicity > PITCH_MIN_CONF
        voiced_freq = frequency[voiced]
        voiced_conf = periodicity[voiced]

        if len(voiced_freq) < 10:
            return True, 0.0, 0.0, ""

        mean_f0   = float(np.mean(voiced_freq))
        mean_conf = float(np.mean(voiced_conf))

        return True, mean_f0, mean_conf, ""

    # ── Stage 5 ── DNSMOS ──────────────────────────────────────────────────

    def _score_dnsmos(self, audio: np.ndarray) -> tuple[bool, dict[str, float], str]:
        """Returns (passed, scores, reject_reason)."""
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

    # ── Stage 6 ── Full-sentence reading verification ──────────────────────

    @staticmethod
    def _normalise_mongolian(text: str) -> str:
        text = unicodedata.normalize("NFC", text.lower().strip())
        text = re.sub(r"[^᠀-᢯Ѐ-ӿ\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _verify_reading(
        self, audio: np.ndarray, ground_truth: str
    ) -> tuple[bool, float, float, str, str]:
        """Returns (passed, cer, length_ratio, asr_text, reject_reason)."""
        try:
            result = self._whisper.transcribe(
                audio,
                language="mn",
                task="transcribe",
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                temperature=0.0,
            )
        except Exception as exc:
            return False, 1.0, 0.0, "", f"whisper_error:{exc}"

        asr_text = result.get("text", "")
        norm_gt = self._normalise_mongolian(ground_truth)
        norm_asr = self._normalise_mongolian(asr_text)

        if not norm_gt:
            return False, 1.0, 0.0, asr_text, "empty_ground_truth"

        try:
            cer_val = float(jiwer.cer(norm_gt, norm_asr))
        except Exception:
            cer_val = 1.0

        len_ratio = len(norm_asr) / max(len(norm_gt), 1)

        passed, reason = self._reading_passes(cer=cer_val, length_ratio=len_ratio)
        if not passed:
            return False, cer_val, len_ratio, asr_text, reason

        return True, cer_val, len_ratio, asr_text, ""

    @staticmethod
    def _reading_passes(cer: float, length_ratio: float) -> tuple[bool, str]:
        if length_ratio < MIN_LEN_RATIO:
            return False, f"truncated_ratio_{length_ratio:.2f}"
        if cer <= MAX_CER:
            return True, ""
        if cer > MAX_RESCUE_CER:
            return False, f"high_cer_{cer:.3f}"
        if not (MIN_RESCUE_LEN_RATIO <= length_ratio <= MAX_RESCUE_LEN_RATIO):
            return False, f"uncertain_reading_cer_{cer:.3f}_ratio_{length_ratio:.2f}"
        return True, ""

    # ── Stage 7 ── Output preparation (resample to 24 kHz + peak-normalise) ──

    def _prepare_output_audio(self, audio: np.ndarray) -> np.ndarray:
        resampled = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=OUTPUT_SAMPLE_RATE)
        peak = float(np.abs(resampled).max())
        if peak < 1e-8:
            return resampled.astype(np.float32)
        target_peak = 10 ** (-1.0 / 20.0)
        return np.clip(resampled / (peak + 1e-7) * target_peak, -target_peak, target_peak).astype(np.float32)

    # ── Public API ──────────────────────────────────────────────────────────

    def process_clip(self, audio_input, ground_truth_text: str) -> ClipResult:
        """
        Run the full pipeline on one clip.
        audio_input: HuggingFace Audio dict or file path.
        """
        result = ClipResult(passed=False)

        audio, err = self._load_audio(audio_input)
        if audio is None:
            return ClipResult(passed=False, reject_stage="load", reject_reason=err)

        result.duration_s = len(audio) / SAMPLE_RATE

        if result.duration_s < MIN_DURATION_S:
            return ClipResult(
                passed=False, reject_stage="duration",
                reject_reason=f"too_short_{result.duration_s:.2f}s",
            )
        if result.duration_s > MAX_DURATION_S:
            return ClipResult(
                passed=False, reject_stage="duration",
                reject_reason=f"too_long_{result.duration_s:.2f}s",
            )

        trimmed, reason = self._run_vad(audio)
        if trimmed is None:
            return ClipResult(passed=False, reject_stage="vad", reject_reason=reason)
        trimmed_s = len(trimmed) / SAMPLE_RATE
        if trimmed_s < MIN_DURATION_S:
            return ClipResult(
                passed=False, reject_stage="vad",
                reject_reason=f"trimmed_too_short_{trimmed_s:.2f}s",
            )

        snr = self._estimate_snr(trimmed)
        if snr < SNR_MIN_DB:
            return ClipResult(
                passed=False, reject_stage="snr",
                reject_reason=f"snr_{snr:.1f}dB",
                snr_db=snr,
            )

        pitch_ok, mean_f0, mean_conf, reason = self._check_pitch(trimmed)
        if not pitch_ok:
            return ClipResult(
                passed=False, reject_stage="pitch", reject_reason=reason,
                snr_db=snr, mean_f0_hz=mean_f0, pitch_confidence=mean_conf,
            )

        dnsmos_ok, dnsmos, reason = self._score_dnsmos(trimmed)
        if not dnsmos_ok:
            return ClipResult(
                passed=False, reject_stage="dnsmos", reject_reason=reason,
                snr_db=snr, mean_f0_hz=mean_f0, pitch_confidence=mean_conf,
                **dnsmos,
            )

        reading_ok, cer_val, _, asr_text, reason = self._verify_reading(
            trimmed, ground_truth_text
        )
        if not reading_ok:
            return ClipResult(
                passed=False, reject_stage="cer", reject_reason=reason,
                snr_db=snr, mean_f0_hz=mean_f0, pitch_confidence=mean_conf,
                cer=cer_val, asr_transcript=asr_text, **dnsmos,
            )

        return ClipResult(
            passed=True,
            snr_db=snr,
            mean_f0_hz=mean_f0,
            pitch_confidence=mean_conf,
            cer=cer_val,
            asr_transcript=asr_text,
            duration_s=result.duration_s,
            audio_normalized=self._prepare_output_audio(trimmed),
            **dnsmos,
        )
