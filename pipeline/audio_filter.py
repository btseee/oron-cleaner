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
  9. Transcript agreement        (MMS_FA forced alignment, then CER)
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

from __future__ import annotations

import logging

import librosa
import numpy as np
from oron_tts.text import MongolianNormalizer

# The model stack is imported where it is used, not here. It costs roughly 3 GB
# and a GPU, while every gate decision in `process_clip` is ordinary Python --
# so importing this module to read or test that logic should not require any of
# it. Same reason `checkpoint.py` defers torch and `upload.py` defers HfApi.
from .alignment import ForcedAligner
from .clip_result import ClipResult
from .constants import (
    DNSMOS_MIN_BAK,
    DNSMOS_MIN_OVR,
    DNSMOS_MIN_SIG,
    MAX_CLIPPED_RATIO,
    MAX_DC_OFFSET,
    MAX_DURATION_S,
    MIN_ALIGN_SCORE,
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
from .provenance import PINNED_REVISIONS
from .trimming import trim_to_audio

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
        # `import silero_vad` calls torch.set_num_threads(1) at module scope, and
        # that clamp is process-wide and permanent -- every torch CPU op for the
        # rest of the run inherits it. Measured on a 48-core node: 48 threads
        # before the import, 1 after, and MBSpeech cleaning fell from ~46 clips
        # a minute to 7.7. VAD itself wants one thread; nothing else does.
        import torch

        threads_before = torch.get_num_threads()
        from silero_vad import load_silero_vad

        self._vad_model = load_silero_vad()
        if torch.get_num_threads() != threads_before:
            log.info("silero_vad set torch threads to %d; restoring %d",
                     torch.get_num_threads(), threads_before)
            torch.set_num_threads(threads_before)

        log.info("Loading %s …", ASR_MODEL)
        from transformers import AutoModelForCTC, AutoProcessor

        # Pinned: `main` moves, and a recogniser that changed under the CER
        # gate silently changes which clips enter the corpus.
        rev = PINNED_REVISIONS[ASR_MODEL]
        self._asr_processor = AutoProcessor.from_pretrained(ASR_MODEL, revision=rev)
        self._asr = (
            AutoModelForCTC.from_pretrained(ASR_MODEL, revision=rev).to(device).eval()
        )

        self._aligner = ForcedAligner(device=device)

        log.info("Loading DNSMOS …")
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

        self._dnsmos = DeepNoiseSuppressionMeanOpinionScore(
            fs=SAMPLE_RATE, personalized=False
        ).to(device)

        log.info("All models loaded.")

    def normalized_text(self, text: str) -> str:
        """The exact string that gets published, scored and trained on.

        Exposed so the corpus writer stores the same text the CER gate compared
        against, rather than re-deriving it and risking drift.

        Raises whatever the normaliser raises. It used to swallow the exception
        and return the raw text, which would have published a transcript with
        unexpanded digits -- "20-иос" rather than "хориос" -- as though it were
        the normalised form, and trained on it. The CER and alignment gates
        already reject a clip the normaliser refuses, so reaching this is a
        disagreement between two paths that must not be resolved by publishing
        the worse string. `process_split` turns it into a rejection.
        """
        return self._normalizer.normalize(text, strict=False)

    # ── Stage 1 ── Format normalisation ───────────────────────────────────

    def _load_audio(self, audio_input) -> tuple[np.ndarray | None, str]:
        """Accept a HuggingFace Audio dict, a torchcodec decoder, or a path."""
        import torchaudio

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
        import torch
        from silero_vad import get_speech_timestamps

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
        import torch

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
        import torch

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
        import jiwer

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

    def process_clip(
        self, audio_input, ground_truth_text: str, *, measure_all: bool = False
    ) -> ClipResult:
        """Filter one clip.

        With `measure_all`, every gate is evaluated and recorded instead of
        returning at the first failure. That is how thresholds get calibrated:
        one pass over a slice yields the real distribution of each metric and
        the true per-gate rejection rate, rather than only the rate of whichever
        gate happens to fire first. It is much slower -- the aligner and the ASR
        run on clips that would otherwise have been discarded early -- so it is
        for calibration runs, not production.
        """
        m: dict = {}
        failed: list[tuple[str, str]] = []

        def done(audio_out=None) -> ClipResult:
            stage, reason = failed[0] if failed else ("", "")
            return ClipResult(
                passed=not failed,
                reject_stage=stage,
                reject_reason=reason,
                failed_gates=[f"{st}:{rs}" for st, rs in failed],
                audio_normalized=(
                    self._prepare_output_audio(audio_out)
                    if audio_out is not None and not failed
                    else np.zeros(1, dtype=np.float32)
                ),
                **m,
            )

        def note(stage: str, reason: str) -> bool:
            """Record a failure. Returns True if processing should continue."""
            failed.append((stage, reason))
            return measure_all

        audio, err = self._load_audio(audio_input)
        if audio is None:
            note("load", err)
            return done()

        raw_duration = len(audio) / SAMPLE_RATE
        if raw_duration < MIN_DURATION_S and not note("duration", f"too_short_{raw_duration:.2f}s"):
            return done()
        if raw_duration > MAX_DURATION_S and not note("duration", f"too_long_{raw_duration:.2f}s"):
            return done()

        clipped = clipped_ratio(audio)
        if clipped > MAX_CLIPPED_RATIO and not note("clipping", f"clipped_{clipped:.4f}"):
            return done()
        dc = dc_offset(audio)
        if dc > MAX_DC_OFFSET and not note("clipping", f"dc_offset_{dc:.4f}"):
            return done()

        trimmed, timestamps, reason = self._run_vad(audio)
        if trimmed is None:
            # Terminal regardless of mode: without speech bounds there is
            # nothing downstream can measure.
            note("vad", reason)
            return done()

        # The duration of the audio actually shipped. The previous
        # implementation reported the pre-VAD length, inflating this column and
        # every "total hours" figure in the reports.
        duration_s = len(trimmed) / SAMPLE_RATE
        m["duration_s"] = duration_s
        if duration_s < MIN_DURATION_S and not note(
            "vad", f"trimmed_too_short_{duration_s:.2f}s"
        ):
            return done()

        # Untrimmed audio, deliberately: the noise regions are the measurement.
        snr = estimate_snr(audio, timestamps)
        if np.isnan(snr):
            # Unmeasurable is not the same as failed. estimate_snr returns NaN
            # when there is under 0.1 s of non-speech to compute a noise floor
            # from, which on continuous narration means the clip is spoken end
            # to end -- a property of the reading, not of the recording.
            # Rejecting it dropped 41% of a 200-clip MBSpeech sample whose
            # DNSMOS-BAK was 3.22 at the 5th percentile against a 2.5 floor,
            # i.e. uniformly clean. It also made the calibration report
            # self-contradictory: NaN clips counted as rejections but carried
            # no value into the percentiles, so the same gate read "rejects
            # 41%" and "keeps 86%".
            # Defer to DNSMOS-BAK, which measures background noise directly.
            m["snr_unmeasurable"] = True
        else:
            m["snr_db"] = snr
            if snr < SNR_MIN_DB and not note("snr", f"snr_{snr:.1f}dB"):
                return done()

        bandwidth = measure_bandwidth(trimmed)
        m["bandwidth_hz"] = bandwidth
        if bandwidth < MIN_BANDWIDTH_HZ and not note(
            "bandwidth", f"bandwidth_{bandwidth:.0f}Hz"
        ):
            return done()

        mean_f0, voiced_frac = median_f0(trimmed)
        m["mean_f0_hz"] = mean_f0
        m["pitch_confidence"] = voiced_frac

        dnsmos_ok, dnsmos, reason = self._score_dnsmos(trimmed)
        m.update(dnsmos)
        if not dnsmos_ok and not note("dnsmos", reason):
            return done()

        # Primary transcript gate, ahead of the ASR because it is the stronger
        # signal: measured separation on real Mongolian audio is 0.722 (worst
        # correct) against 0.547 (worst mismatched), where CER's own floor on
        # correct clips is 0.123.
        try:
            norm_gt = self._normalizer.normalize(ground_truth_text, strict=False)
        except Exception as exc:
            note("alignment", f"normalize_error:{exc}")
            return done()

        # Cut the transcript back to the span the audio supports, BEFORE the
        # gates score it. WorldSpeech segments run past the end of their audio
        # -- 80% of clips that passed had audio shorter than their text and 53%
        # ended mid-word -- and no threshold fixes a transcript that is simply
        # longer than the recording. A clip that already aligns is untouched, so
        # this is safe on every source.
        trim = trim_to_audio(self._aligner, trimmed, norm_gt)
        if trim.trimmed:
            norm_gt = trim.text
            ground_truth_text = trim.text
            m["text_trimmed"] = True
            m["text_discarded"] = trim.discarded
            m["words_removed"] = trim.words_removed

        align = self._aligner.score(trimmed, norm_gt)
        if np.isnan(align):
            if not note("alignment", "alignment_unavailable"):
                return done()
        else:
            m["align_score"] = align
            if align < MIN_ALIGN_SCORE and not note("alignment", f"align_{align:.3f}"):
                return done()

        reading_ok, cer_val, len_ratio, asr_text, reason = self._verify_reading(
            trimmed, ground_truth_text
        )
        m["cer"] = cer_val
        m["len_ratio"] = len_ratio
        m["asr_transcript"] = asr_text
        if not reading_ok:
            note("cer", reason)

        return done(trimmed)
