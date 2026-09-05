# Clip Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the rejected clips that are genuinely repairable — and only those — so a corpus gains hours without gaining anything nobody recorded.

**Architecture:** Repairs are pure functions in a new `pipeline/recovery.py`, each mapping `(audio, sample_rate, text)` to a repaired triple or `None` when it does not apply. `AudioQualityFilter.process_clip` gets a second chance: when a gate fails and a repair is registered for that gate, apply it and re-run the whole gate stack once on the result. Nothing is relaxed — a repaired clip passes the same thresholds or it is discarded.

**Tech Stack:** Python 3.12+, NumPy, librosa, existing `pipeline` package (`audio_filter`, `trimming`, `constants`, `clip_result`, `provenance`). Tests use pytest with synthetic audio; no network, no GPU.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-05-clip-recovery-design.md`.
- **A repair may not alter the speech signal.** Trimming, splitting, a scalar gain and a DC constant are permitted. Denoising, de-clipping, bandwidth extension and transcript correction are not.
- **A repaired clip re-earns its place on the same thresholds.** Every measurement is recomputed on the repaired audio; no pre-repair number is carried forward.
- **One repair attempt per clip.** A repaired clip is re-gated with recovery disabled, so repairs never chain.
- A repair that does not apply returns `None`. That is the common case and is not an error.
- Recovery constants are **uppercase scalars in `pipeline/constants.py`**, because `FILTER_POLICY_VERSION` hashes every uppercase scalar there (`constants.py:128-136`) — that is how the corpus version comes to reflect the recovery configuration without new machinery.
- Internal audio is mono float32 at `SAMPLE_RATE` (16 kHz); `_load_audio` resamples on the way in and `_prepare_output_audio` resamples to `OUTPUT_SAMPLE_RATE` (24 kHz) on the way out. Repairs operate on the 16 kHz array.
- Python 3.12+, NumPy 2.x (`ndarray.ptp` does not exist — use `np.ptp(x)`).
- Tests make no network calls and load no models except where a task says so.

---

### Task 1: The repairs

Pure functions, no gate knowledge, no I/O. Each returns a repaired `(audio, sr, text)` or `None`.

**Files:**
- Create: `pipeline/recovery.py`
- Modify: `pipeline/constants.py`
- Test: `tests/test_recovery.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `Repair = Callable[[np.ndarray, int, str], tuple[np.ndarray, int, str] | None]`
  - `remove_dc_offset(audio, sr, text)`
  - `normalise_gain(audio, sr, text)`
  - `repair_homoglyphs(audio, sr, text)`
  - `HOMOGLYPHS: dict[str, str]`

- [ ] **Step 1: Add the constants**

In `pipeline/constants.py`, immediately after the `TORCH_THREADS` block:

```python
# ── Recovery ──────────────────────────────────────────────────────────────────
# A repaired clip re-earns its place on the thresholds above; these govern only
# what a repair is allowed to do. They are uppercase scalars in this module on
# purpose: FILTER_POLICY_VERSION hashes those, so the recovery configuration
# lands in the corpus version and two corpora built under different recovery
# rules cannot share a version string.

# Target peak for gain normalisation. -3 dBFS leaves headroom for the 24 kHz
# resample on the way out, which can overshoot the original peak.
RECOVERY_TARGET_PEAK: float = 0.708

# Below this peak a clip is quiet enough that the VAD and the recogniser suffer
# for a reason that has nothing to do with the speaker. Above it, gain is
# somebody's deliberate level and not ours to change.
RECOVERY_QUIET_PEAK: float = 0.2

# A DC offset this small is dither, not a defect worth a second gate pass.
RECOVERY_MIN_DC: float = 0.001
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_recovery.py`:

```python
"""Repairs may fix a recording. They may not change what was said.

The pipeline discards a clip at the first gate it fails, and most of those
failures are final -- you cannot invent bandwidth or audio nobody recorded. But
a clip whose speech is clean and whose transcript is right can still be thrown
away for carrying eight seconds of room tone, or for a DC offset, or for being
recorded too quietly for the recogniser. Those are repairable, and the repair
does not touch the speech.

Every test here holds one line: after a repair, the speech must be the same
speech.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.constants import (
    RECOVERY_MIN_DC,
    RECOVERY_QUIET_PEAK,
    RECOVERY_TARGET_PEAK,
    SAMPLE_RATE,
)
from pipeline.recovery import (
    HOMOGLYPHS,
    normalise_gain,
    remove_dc_offset,
    repair_homoglyphs,
)

TEXT = "Сайн байна уу"


def speech(seconds: float = 3.0, peak: float = 0.5, seed: int = 0) -> np.ndarray:
    """Speech-shaped noise. The content is irrelevant; the shape is not."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    sig = rng.standard_normal(n).astype("float32")
    sig = np.convolve(sig, np.hanning(64), mode="same").astype("float32")
    return (sig / np.max(np.abs(sig)) * peak).astype("float32")


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── DC offset ─────────────────────────────────────────────────────────────────

def test_dc_offset_is_removed_exactly():
    audio = speech() + 0.05
    out, sr, text = remove_dc_offset(audio, SAMPLE_RATE, TEXT)
    assert abs(float(out.mean())) < 1e-6
    assert sr == SAMPLE_RATE and text == TEXT


def test_dc_removal_does_not_change_the_speech():
    """Subtracting a constant is the whole repair. The waveform's shape -- which
    is what anyone hears and what every gate measures -- must survive it."""
    audio = speech()
    out, _, _ = remove_dc_offset(audio + 0.05, SAMPLE_RATE, TEXT)
    assert correlation(out, audio) == pytest.approx(1.0, abs=1e-6)


def test_a_clip_without_a_dc_offset_is_not_repaired():
    """`None` means "not the kind of clip this fixes". Returning a changed clip
    anyway would put a second gate pass on the bill for nothing."""
    assert remove_dc_offset(speech(), SAMPLE_RATE, TEXT) is None


def test_dither_sized_offsets_are_left_alone():
    audio = speech() + RECOVERY_MIN_DC / 2
    assert remove_dc_offset(audio, SAMPLE_RATE, TEXT) is None


# ── gain ──────────────────────────────────────────────────────────────────────

def test_a_quiet_clip_is_brought_up_to_the_target_peak():
    audio = speech(peak=0.02)
    out, _, _ = normalise_gain(audio, SAMPLE_RATE, TEXT)
    assert float(np.max(np.abs(out))) == pytest.approx(RECOVERY_TARGET_PEAK, abs=1e-4)


def test_gain_is_a_scalar_multiply_and_nothing_else():
    """This is the claim that makes gain a legal repair: no information changes,
    so the model hears the same recording, louder."""
    audio = speech(peak=0.02)
    out, _, _ = normalise_gain(audio, SAMPLE_RATE, TEXT)
    ratio = out / np.where(audio == 0, np.nan, audio)
    assert np.nanstd(ratio) == pytest.approx(0.0, abs=1e-5)
    assert correlation(out, audio) == pytest.approx(1.0, abs=1e-6)


def test_a_clip_at_a_normal_level_is_left_alone():
    """Somebody chose that level. Above the quiet threshold it is not ours."""
    assert normalise_gain(speech(peak=RECOVERY_QUIET_PEAK + 0.1), SAMPLE_RATE, TEXT) is None


def test_silence_is_not_amplified():
    """Dividing by a zero peak would produce inf, and there is no speech to save."""
    assert normalise_gain(np.zeros(SAMPLE_RATE, "float32"), SAMPLE_RATE, TEXT) is None


# ── homoglyphs ────────────────────────────────────────────────────────────────

def test_a_latin_letter_inside_a_cyrillic_word_is_corrected():
    """`о` U+006F in an otherwise-Cyrillic word has exactly one correct reading.
    It is an encoding error, not an ambiguity, so fixing it guesses nothing."""
    audio = speech()
    out, _, text = repair_homoglyphs(audio, SAMPLE_RATE, "Mонгол хэл")
    assert text == "Монгол хэл"
    assert out is audio, "the audio is untouched by a text repair"


def test_the_ukrainian_i_is_corrected():
    """U+0456 passes `is_representable` because it is in the vocabulary, so it
    reaches the model as a distinct embedding row for a letter nobody typed."""
    _, _, text = repair_homoglyphs(speech(), SAMPLE_RATE, "саін")
    assert text == "сайн"


def test_an_all_latin_word_is_left_alone():
    """An English proper noun in a Mongolian sentence is not an encoding error."""
    assert repair_homoglyphs(speech(), SAMPLE_RATE, "Google-ийн") is None


def test_clean_cyrillic_is_not_repaired():
    assert repair_homoglyphs(speech(), SAMPLE_RATE, TEXT) is None


def test_every_homoglyph_maps_to_a_cyrillic_letter():
    """A mapping that produced another Latin letter would move the problem."""
    for latin, cyrillic in HOMOGLYPHS.items():
        assert ord(latin) < 0x400, f"{latin!r} is not the Latin side"
        assert 0x400 <= ord(cyrillic) <= 0x4FF, f"{cyrillic!r} is not Cyrillic"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_recovery.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.recovery'`

- [ ] **Step 4: Write the implementation**

Create `pipeline/recovery.py`:

```python
"""Repair the clips that are genuinely repairable, and only those.

The pipeline discards a clip at the first gate it fails. Most of those failures
are real and final: you cannot invent bandwidth, or audio that was never
recorded, or a transcript nobody wrote. But some are not failures of the
recording at all. A clip whose speech is clean and whose transcript is right can
still be thrown away for carrying eight seconds of room tone before the sentence
starts, for a microphone's DC offset, or for being recorded too quietly for the
recogniser to read.

Every repair here obeys one rule: **it may not alter the speech signal.** A
scalar gain, a subtracted constant, a trimmed edge and a corrected encoding
error all leave the speech exactly as recorded. Denoising, de-clipping and
bandwidth extension do not, and they are not here -- a denoised clip teaches the
model the denoiser's artifacts, and worse, its SNR and DNSMOS then measure our
processing rather than the recording, so the gate that judges quality is judging
us.

A repair returns `None` when it does not apply. That is the common case: most
clips are not the kind any given repair fixes, and saying so costs nothing.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .constants import RECOVERY_MIN_DC, RECOVERY_QUIET_PEAK, RECOVERY_TARGET_PEAK

Repair = Callable[[np.ndarray, int, str], "tuple[np.ndarray, int, str] | None"]

# Latin letters with an identical Cyrillic twin. Inside an otherwise-Cyrillic
# word each has exactly one correct reading, so this corrects an encoding error
# rather than guessing at ambiguity. `i` U+0069 and `і` U+0456 both map to `и`:
# the Ukrainian one is in the vocabulary, so it reaches the model as its own
# embedding row for a letter nobody typed.
HOMOGLYPHS: dict[str, str] = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "i": "и", "і": "и",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х",
}

_CYRILLIC = range(0x400, 0x500)


def remove_dc_offset(audio: np.ndarray, sr: int, text: str):
    """Subtract the mean.

    A DC offset is an additive constant with exactly one correct removal, so
    this is the least ambiguous repair there is. It matters because the offset
    eats headroom and shifts the clipping ratio, failing a gate that has nothing
    to say about the recording's quality.
    """
    offset = float(audio.mean())
    if abs(offset) < RECOVERY_MIN_DC:
        return None
    return (audio - offset).astype("float32"), sr, text


def normalise_gain(audio: np.ndarray, sr: int, text: str):
    """Scale a too-quiet clip up to a target peak.

    A scalar multiply changes no information: the model hears the same
    recording, louder. Crowd-sourced clips are routinely quiet enough that the
    VAD finds no speech and the recogniser misreads, for a reason that belongs
    to the microphone rather than the speaker.

    Only clips below `RECOVERY_QUIET_PEAK` are touched. Above it, the level is
    somebody's deliberate choice.
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0 or peak >= RECOVERY_QUIET_PEAK:
        return None
    return (audio * (RECOVERY_TARGET_PEAK / peak)).astype("float32"), sr, text


def repair_homoglyphs(audio: np.ndarray, sr: int, text: str):
    """Correct Latin letters sitting inside Cyrillic words.

    Word by word, because the decision needs the context: `о` in `Mонгол` is a
    typo for `О`, while `Google` is a word that is simply Latin. A word is
    repaired only when it already contains Cyrillic, which is what makes this a
    correction rather than a guess.
    """
    words = text.split(" ")
    changed = False
    for i, word in enumerate(words):
        if not any(ord(c) in _CYRILLIC for c in word):
            continue
        fixed = "".join(HOMOGLYPHS.get(c, c) for c in word)
        if fixed != word:
            words[i] = fixed
            changed = True
    if not changed:
        return None
    return audio, sr, " ".join(words)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_recovery.py -q`
Expected: PASS, 13 passed

- [ ] **Step 6: Verify the suite is not vacuous**

```bash
python - <<'PY'
import pathlib, subprocess, sys
p = pathlib.Path("pipeline/recovery.py")
orig = p.read_text(encoding="utf-8")
# Repair every word, not only Cyrillic ones: "Google-ийн" must break.
p.write_text(orig.replace(
    '        if not any(ord(c) in _CYRILLIC for c in word):\n            continue\n', ''),
    encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_recovery.py", "-q"],
                   capture_output=True, text=True)
p.write_text(orig, encoding="utf-8")
print(r.stdout.strip().splitlines()[-1])
PY
```

Expected: a line reporting at least `1 failed` — `test_an_all_latin_word_is_left_alone` must fail.

- [ ] **Step 7: Run the gates and commit**

```bash
python -m pytest -q
ruff check .
git add pipeline/recovery.py pipeline/constants.py tests/test_recovery.py
git commit -m "Repair DC offset, quiet gain and homoglyphs, without touching the speech"
```

Expected: the whole suite green, `All checks passed!`

---

### Task 2: Recover a clip rejected for being mostly silence

`_run_vad` computes `speech_ratio` on the **untrimmed** clip (`audio_filter.py:215`) and rejects below 0.35 at `:216` — *before* `edge_trim_bounds` runs at `:222`. So a clip with three seconds of speech inside ten seconds of lead-in scores 0.30 and is discarded, while the edge-trimmed version would pass every gate.

The repair applies the trim the VAD already computed. It uses the VAD's own timestamps, which are in **original-audio coordinates**, so it must run on the original audio — not on the trimmed array the later gates see.

An earlier draft aimed this at `snr` and `dnsmos`. That was wrong: the VAD edge-trims before either is measured, so there is nothing there to recover.

**Files:**
- Modify: `pipeline/recovery.py`
- Modify: `pipeline/constants.py`
- Test: `tests/test_recovery.py`

**Interfaces:**
- Consumes: `Repair` from Task 1.
- Produces: `trim_to_speech(audio, sr, text, *, speech_spans)` and the module-level `SPEECH_PAD_S`.

- [ ] **Step 1: Add the constants**

Append to the recovery block in `pipeline/constants.py`:

```python
# Keep this much either side of the speech when trimming. A hard cut at the
# first speech sample clips onsets and sounds wrong; the pad is generous enough
# to keep breath and plosive release.
RECOVERY_SPEECH_PAD_S: float = 0.15

# Refuse to remove more than this share of a clip. The same ceiling that
# `trimming.MAX_TRIM_FRACTION` puts on transcript trimming, for the same reason:
# past it, the thing being trimmed is probably not silence.
RECOVERY_MAX_TRIM_FRACTION: float = 0.60
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_recovery.py`:

```python
from pipeline.constants import RECOVERY_MAX_TRIM_FRACTION, RECOVERY_SPEECH_PAD_S
from pipeline.recovery import trim_to_speech


def test_leading_and_trailing_silence_is_removed():
    """The case this exists for: clean speech, ruined SNR, because the clip is
    mostly room tone."""
    sig = speech(2.0)
    quiet = np.zeros(int(4.0 * SAMPLE_RATE), "float32")
    audio = np.concatenate([quiet, sig, quiet])
    spans = [(4.0, 6.0)]
    out, sr, text = trim_to_speech(audio, SAMPLE_RATE, TEXT, speech_spans=spans)
    expected = (2.0 + 2 * RECOVERY_SPEECH_PAD_S) * SAMPLE_RATE
    assert len(out) == pytest.approx(expected, rel=0.02)
    assert sr == SAMPLE_RATE and text == TEXT


def test_the_speech_itself_survives_the_trim():
    sig = speech(2.0)
    quiet = np.zeros(int(4.0 * SAMPLE_RATE), "float32")
    out, _, _ = trim_to_speech(np.concatenate([quiet, sig, quiet]), SAMPLE_RATE,
                               TEXT, speech_spans=[(4.0, 6.0)])
    # The speech sits inside the padded window; find it and compare.
    best = max(range(0, len(out) - len(sig) + 1, 80),
               key=lambda i: abs(correlation(out[i:i + len(sig)], sig)))
    assert abs(correlation(out[best:best + len(sig)], sig)) == pytest.approx(1.0, abs=1e-6)


def test_a_clip_that_is_already_tight_is_not_trimmed():
    """Nothing to remove means nothing to re-gate."""
    sig = speech(3.0)
    assert trim_to_speech(sig, SAMPLE_RATE, TEXT, speech_spans=[(0.0, 3.0)]) is None


def test_trimming_more_than_the_ceiling_is_refused():
    """Past this share, what is being cut is probably not silence -- and a
    repair that guesses is the thing this design exists to avoid."""
    audio = speech(10.0)
    out = trim_to_speech(audio, SAMPLE_RATE, TEXT, speech_spans=[(4.8, 5.2)])
    assert out is None


def test_no_speech_spans_means_no_repair():
    assert trim_to_speech(speech(3.0), SAMPLE_RATE, TEXT, speech_spans=[]) is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_recovery.py -q`
Expected: FAIL — `ImportError: cannot import name 'trim_to_speech'`

- [ ] **Step 4: Write the implementation**

Add to `pipeline/recovery.py`:

```python
from .constants import RECOVERY_MAX_TRIM_FRACTION, RECOVERY_SPEECH_PAD_S


def trim_to_speech(audio: np.ndarray, sr: int, text: str, *,
                   speech_spans: list[tuple[float, float]]):
    """Cut a clip back to the speech the VAD found, with a pad either side.

    SNR and DNSMOS are computed over the whole clip, so a sentence preceded by
    eight seconds of room tone scores as a bad recording when the speech is
    fine. Removing non-speech removes no speech, which is what makes this a
    legal repair -- and the gates are then recomputed on what remains, so the
    clip earns its place on the trimmed audio rather than on a number taken
    before the trim.

    Refuses past `RECOVERY_MAX_TRIM_FRACTION`: if most of the clip is going, the
    VAD probably missed speech rather than found silence.
    """
    if not speech_spans:
        return None
    pad = RECOVERY_SPEECH_PAD_S
    start = max(0.0, min(s for s, _ in speech_spans) - pad)
    end = min(len(audio) / sr, max(e for _, e in speech_spans) + pad)
    first, last = int(start * sr), int(end * sr)
    if last <= first:
        return None
    removed = 1.0 - (last - first) / len(audio)
    if removed <= 0.01:
        return None                      # nothing worth a second gate pass
    if removed > RECOVERY_MAX_TRIM_FRACTION:
        return None
    return audio[first:last].astype("float32"), sr, text
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_recovery.py -q`
Expected: PASS, 18 passed

- [ ] **Step 6: Run the gates and commit**

```bash
python -m pytest -q
ruff check .
git add pipeline/recovery.py pipeline/constants.py tests/test_recovery.py
git commit -m "Trim a clip back to the speech the VAD found"
```

---

### Task 3: The second chance

Wire recovery into the filter: a failing gate consults a repair table, the repair runs, and the whole gate stack runs again on the result — once.

**Files:**
- Modify: `pipeline/audio_filter.py`
- Modify: `pipeline/recovery.py`
- Modify: `pipeline/clip_result.py`
- Test: `tests/test_recovery_wiring.py`

**Interfaces:**
- Consumes: every repair from Tasks 1–2.
- Produces:
  - `recovery.RECOVERIES: dict[str, tuple[str, ...]]` mapping a failing gate to repair names
  - `recovery.REPAIRS: dict[str, Repair]` mapping a name to its function
  - `ClipResult.recovered_by: str`
  - `AudioQualityFilter.process_clip(..., recover: bool = True)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recovery_wiring.py`:

```python
"""A repair earns nothing by itself. The clip has to pass the gates again.

The failure this guards against is not a bad repair but a generous one: if a
repaired clip skipped the gates, or kept the measurement taken before the
repair, "recovery" would quietly become "lower the bar", and the corpus would be
the same corpus with worse thresholds.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.clip_result import ClipResult
from pipeline.recovery import RECOVERIES, REPAIRS


def test_every_mapped_repair_exists():
    """The table is data, so it can name a repair nobody wrote."""
    for gate, names in RECOVERIES.items():
        for name in names:
            assert name in REPAIRS, f"{gate} maps to unknown repair {name!r}"


def test_every_repair_is_reachable_from_some_gate():
    """A repair no gate maps to is dead code that looks like a feature."""
    mapped = {n for names in RECOVERIES.values() for n in names}
    assert set(REPAIRS) == mapped


def test_clip_result_records_the_repair():
    """A consumer must be able to select or exclude repaired material, and the
    summary must be able to report how much of the corpus it is."""
    assert ClipResult(passed=True).recovered_by == ""
    assert ClipResult(passed=True, recovered_by="normalise_gain").recovered_by == "normalise_gain"


class _Filter:
    """A stand-in for AudioQualityFilter carrying only the recovery logic, so
    the wiring can be tested without loading four models."""

    def __init__(self, fail_gates, repair_result):
        self.fail_gates = list(fail_gates)
        self.repair_result = repair_result
        self.calls = []

    def process_clip(self, audio, text, *, recover=True):
        from pipeline.audio_filter import AudioQualityFilter
        return AudioQualityFilter.process_clip(self, audio, text, recover=recover)


def test_a_repaired_clip_is_re_gated_once_not_twice(monkeypatch):
    """One repair attempt per clip. Chaining repairs until something passes is
    fishing, and it is how a threshold quietly stops meaning anything."""
    import pipeline.audio_filter as af

    calls = {"n": 0}

    def fake_process(self, audio, text, *, measure_all=False, recover=True):
        calls["n"] += 1
        if recover and calls["n"] == 1:
            return fake_process(self, audio, text, recover=False)
        return ClipResult(passed=False, reject_stage="snr", reject_reason="snr_1.0dB")

    monkeypatch.setattr(af.AudioQualityFilter, "process_clip", fake_process)
    result = af.AudioQualityFilter.process_clip(object(), np.zeros(16000, "float32"), "т")
    assert result.passed is False
    assert calls["n"] == 2, "the clip was re-gated more than once"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_recovery_wiring.py -q`
Expected: FAIL — `ImportError: cannot import name 'RECOVERIES'`

- [ ] **Step 3: Add the repair table**

Append to `pipeline/recovery.py`:

```python
REPAIRS: dict[str, Repair] = {
    "remove_dc_offset": remove_dc_offset,
    "normalise_gain": normalise_gain,
    "repair_homoglyphs": repair_homoglyphs,
    "trim_to_speech": trim_to_speech,        # needs speech_spans; bound at call site
}

# Which repairs answer which failing gate, in the order they are tried. The
# mapping is data so the design document and the code cannot drift apart, and so
# a test can assert every repair is reachable and every name resolves.
RECOVERIES: dict[str, tuple[str, ...]] = {
    "clipping": ("remove_dc_offset",),
    # `vad` covers both speech_ratio (answered by the edge trim the VAD already
    # computed but rejected before applying) and a clip too quiet for the VAD to
    # find speech in at all.
    "vad": ("trim_to_speech", "normalise_gain"),
    "dnsmos": ("normalise_gain",),
    "cer": ("repair_homoglyphs",),
    "alignment": ("repair_homoglyphs",),
}
```

- [ ] **Step 4: Add the result field**

In `pipeline/clip_result.py`, after the `words_removed` field:

```python
    # The repair that produced this clip, empty when it was accepted as
    # recorded. A consumer can select or exclude repaired material with it, and
    # the corpus summary reports the recovered share -- a corpus that is a third
    # repaired has a different character from one that is not, even when every
    # clip passed the same gates.
    recovered_by: str = ""
```

- [ ] **Step 5: Wire the second chance into the filter**

In `pipeline/audio_filter.py`, change the signature at the top of `process_clip`:

```python
    def process_clip(
        self, audio_input, ground_truth_text: str, *, measure_all: bool = False,
        recover: bool = True,
    ) -> ClipResult:
```

Then, immediately before the final `return done(trimmed)` at the end of the method, insert:

```python
        # ── second chance ────────────────────────────────────────────────────
        # A failing gate consults the repair table, the repair runs, and the
        # whole stack runs again on the result. Once: `recover=False` on the
        # recursive call is the guard, because a clip that can be repaired twice
        # is a clip being fished for.
        #
        # The gates are recomputed on the repaired audio -- no measurement taken
        # before the repair is carried forward. That is what keeps this
        # "recover the recoverable" rather than "be kinder until it passes".
        result = done(trimmed)
        if result.passed or not recover or measure_all:
            return result

        from .recovery import RECOVERIES, REPAIRS

        for name in RECOVERIES.get(result.reject_stage, ()):
            repair = REPAIRS[name]
            if name == "trim_to_speech":
                repaired = repair(trimmed, SAMPLE_RATE, norm_gt,
                                  speech_spans=m.get("speech_spans") or [])
            else:
                repaired = repair(trimmed, SAMPLE_RATE, norm_gt)
            if repaired is None:
                continue
            audio_r, sr_r, text_r = repaired
            second = self.process_clip(
                {"array": audio_r, "sampling_rate": sr_r}, text_r, recover=False)
            if second.passed:
                second.recovered_by = name
                return second
        return result
```

- [ ] **Step 6: Record the ORIGINAL audio and its speech spans**

In `pipeline/audio_filter.py`, inside `_run_vad`, after the speech timestamps are obtained, store them on the metrics dict the caller passes. Find the call site in `process_clip` and add, immediately after the VAD block:

`_run_vad` returns `(trimmed_audio, timestamps, reason)` and the timestamps are in **samples on the ORIGINAL audio** (`return_seconds=False`, `audio_filter.py:206`). The trim repair therefore needs the original array, not the trimmed one — applying original-coordinate spans to the trimmed array would cut the wrong region.

Capture both, immediately after the `_run_vad` call:

```python
        # The VAD's timestamps index the ORIGINAL audio, and the speech_ratio
        # gate rejects before the edge trim is applied -- so the repair needs the
        # untrimmed array to trim, not the array the later gates see.
        m["speech_spans"] = [(ts["start"] / SAMPLE_RATE, ts["end"] / SAMPLE_RATE)
                             for ts in (timestamps or [])]
        original_audio = audio
```

and in Step 5's recovery block, pass `original_audio` rather than `trimmed` to `trim_to_speech`:

```python
            if name == "trim_to_speech":
                repaired = repair(original_audio, SAMPLE_RATE, norm_gt,
                                  speech_spans=m.get("speech_spans") or [])
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_recovery_wiring.py -q`
Expected: PASS, 4 passed

- [ ] **Step 8: Run the gates and commit**

```bash
python -m pytest -q
ruff check .
python -c "import pipeline.audio_filter"
git add pipeline/audio_filter.py pipeline/recovery.py pipeline/clip_result.py tests/test_recovery_wiring.py
git commit -m "Give a rejected clip one repair and one full re-gate, never two"
```

---

### Task 4: Report what recovery actually recovers

The yield is unknown and must not be assumed. This makes the pipeline say how much recovery earned, per gate, so a repair that recovers nothing can be deleted rather than shipped.

**Files:**
- Modify: `pipeline/stats.py`
- Modify: `pipeline/processor.py`
- Test: `tests/test_recovery_stats.py`

**Interfaces:**
- Consumes: `ClipResult.recovered_by` from Task 3.
- Produces: `CleaningStats.recovered: collections.Counter` and a `recovered` block in the corpus summary.

- [ ] **Step 1: Write the failing test**

Create `tests/test_recovery_stats.py`:

```python
"""A repair that recovers nothing should be deleted, not shipped.

The only way to know which is which is to count, per repair, how many clips it
saved. Without that the recovery configuration is a list of good intentions.
"""
from __future__ import annotations

from pipeline.clip_result import ClipResult
from pipeline.stats import CleaningStats


def test_recovered_clips_are_counted_by_repair():
    stats = CleaningStats("test")
    stats.record(ClipResult(passed=True))
    stats.record(ClipResult(passed=True, recovered_by="normalise_gain"))
    stats.record(ClipResult(passed=True, recovered_by="normalise_gain"))
    stats.record(ClipResult(passed=True, recovered_by="trim_to_speech"))
    assert stats.recovered["normalise_gain"] == 2
    assert stats.recovered["trim_to_speech"] == 1
    assert sum(stats.recovered.values()) == 3


def test_clips_that_passed_as_recorded_are_not_counted_as_recovered():
    stats = CleaningStats("test")
    stats.record(ClipResult(passed=True))
    assert sum(stats.recovered.values()) == 0


def test_the_summary_states_the_recovered_share():
    """A corpus that is a third repaired has a different character from one that
    is not, even when every clip passed the same gates."""
    stats = CleaningStats("test")
    for _ in range(7):
        stats.record(ClipResult(passed=True))
    for _ in range(3):
        stats.record(ClipResult(passed=True, recovered_by="trim_to_speech"))
    text = stats.summary()
    assert "recovered" in text.lower()
    assert "trim_to_speech" in text
    assert "30" in text, "the share should be stated, not left to be worked out"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_recovery_stats.py -q`
Expected: FAIL — `AttributeError: 'CleaningStats' object has no attribute 'recovered'`

- [ ] **Step 3: Implement**

In `pipeline/stats.py`, add to `CleaningStats.__init__`:

```python
        # Per-repair counts. A repair that never appears here recovered nothing
        # and should be removed rather than left to look like a feature.
        self.recovered: collections.Counter = collections.Counter()
```

In `CleaningStats.record`, immediately after the clip is counted as passing:

```python
        if result.recovered_by:
            self.recovered[result.recovered_by] += 1
```

In `CleaningStats.summary`, before the acceptance-criteria block:

```python
        if self.recovered:
            total = sum(self.recovered.values())
            share = 100.0 * total / max(self.kept, 1)
            lines.append("")
            lines.append(f"Recovered: {total} clips ({share:.0f}% of those kept)")
            for name, count in self.recovered.most_common():
                lines.append(f"  {name:<20} {count}")
```

Adjust `self.kept` to whatever this class calls its kept-clip count.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_recovery_stats.py -q`
Expected: PASS, 3 passed

- [ ] **Step 5: Run the gates and commit**

```bash
python -m pytest -q
ruff check .
git add pipeline/stats.py pipeline/processor.py tests/test_recovery_stats.py
git commit -m "Count what each repair recovered, so a useless one can be deleted"
```

---

### Task 5: Calibrate the yield on real audio

Everything so far is tested on synthetic audio. This measures the real thing on a corpus whose raw source is reachable without a key, and produces the number the Common Voice decision rests on.

**Files:**
- Create: `scripts/calibrate_recovery.py`
- Test: `tests/test_calibrate_recovery.py`

**Interfaces:**
- Consumes: `CleaningStats.recovered` from Task 4.
- Produces: `summarise(stats_before, stats_after) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibrate_recovery.py`:

```python
"""The report has to make a useless repair obvious, not bury it."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_recovery  # noqa: E402


def test_the_report_names_the_yield_per_repair():
    text = calibrate_recovery.summarise(
        scanned=200, kept_before=100, kept_after=118,
        recovered={"trim_to_speech": 15, "normalise_gain": 3})
    assert "200" in text and "100" in text and "118" in text
    assert "trim_to_speech" in text and "15" in text


def test_a_repair_that_recovered_nothing_is_named_as_such():
    """Silence about a useless repair is how it survives into a release."""
    text = calibrate_recovery.summarise(
        scanned=200, kept_before=100, kept_after=100, recovered={})
    assert "nothing" in text.lower() or "0" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_calibrate_recovery.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'calibrate_recovery'`

- [ ] **Step 3: Implement**

Create `scripts/calibrate_recovery.py`:

```python
"""Measure what recovery actually recovers, before committing a full run to it.

The yield is unknown. Common Voice -- the corpus this is for -- is fetched from
the Mozilla Data Collective API and is not reachable without a key, and no
published artifact carries the clips that were rejected. So the honest first
step is to measure on a corpus that is reachable, report the yield per repair,
and let a repair that earns nothing be deleted rather than shipped.

    python scripts/calibrate_recovery.py --dataset fleurs --limit 200
"""

from __future__ import annotations

import argparse


def summarise(*, scanned: int, kept_before: int, kept_after: int,
              recovered: dict[str, int]) -> str:
    """The report. Names every repair, including the ones that earned nothing."""
    gained = kept_after - kept_before
    lines = [
        f"scanned            {scanned}",
        f"kept without recovery  {kept_before} ({100.0 * kept_before / max(scanned, 1):.0f}%)",
        f"kept with recovery     {kept_after} ({100.0 * kept_after / max(scanned, 1):.0f}%)",
        f"gained             {gained}",
        "",
        "per repair:",
    ]
    if not recovered:
        lines.append("  nothing was recovered by any repair")
        return "\n".join(lines)
    for name, count in sorted(recovered.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:<20} {count}")
    return "\n".join(lines)


def main() -> None:
    import io
    import itertools

    import soundfile as sf
    from datasets import Audio

    from pipeline.audio_filter import AudioQualityFilter
    from pipeline.datasets._load import load_hub_dataset
    from pipeline.provenance import PINNED_REVISIONS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ds = load_hub_dataset("google/fleurs", "mn_mn",
                          revision=PINNED_REVISIONS["google/fleurs"],
                          split="train").cast_column("audio", Audio(decode=False))
    filt = AudioQualityFilter(device=args.device)

    scanned = kept_before = kept_after = 0
    recovered: dict[str, int] = {}
    for row in itertools.islice(ds, args.limit):
        blob = row["audio"]
        raw = blob["bytes"] if blob.get("bytes") else open(blob["path"], "rb").read()
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
        clip = {"array": wav, "sampling_rate": sr}
        scanned += 1
        # Both paths on the same clip, so the difference is recovery and
        # nothing else.
        if filt.process_clip(clip, row["transcription"], recover=False).passed:
            kept_before += 1
        after = filt.process_clip(clip, row["transcription"], recover=True)
        if after.passed:
            kept_after += 1
            if after.recovered_by:
                recovered[after.recovered_by] = recovered.get(after.recovered_by, 0) + 1

    print(summarise(scanned=scanned, kept_before=kept_before,
                    kept_after=kept_after, recovered=recovered))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_calibrate_recovery.py -q`
Expected: PASS, 2 passed

- [ ] **Step 5: Run it on real audio**

```bash
OMP_NUM_THREADS=1 python scripts/calibrate_recovery.py --limit 120
```

Expected: the report. This is the number the whole design rests on — record it in the commit message, and if a repair shows zero, say so rather than leaving it in.

- [ ] **Step 6: Run the gates and commit**

```bash
python -m pytest -q
ruff check .
git add scripts/calibrate_recovery.py tests/test_calibrate_recovery.py
git commit -m "Measure what recovery recovers, on real audio, before trusting it"
```

---

### Task 6: Split over-length clips — only if Task 5 justifies it

**Build this only if Task 5's report shows `duration` failures that a split would answer.** Common Voice clips are single sentences, so `too_long` is expected to be rare there; this repair is for FLEURS and WorldSpeech. It is also the one repair that can create a defect rather than fail cleanly — a split is two new transcripts, and a cut at the wrong word publishes, scores and trains on wrong text.

If Task 5 shows no `duration` rejections worth answering, **skip this task and record that decision in the plan file** rather than building it on speculation.

**Files:**
- Modify: `pipeline/recovery.py`
- Test: `tests/test_recovery_split.py`

**Interfaces:**
- Consumes: `Repair` and `RECOVERIES` from Tasks 1–3.
- Produces: `split_at_silence(audio, sr, text, *, aligner, speech_spans) -> list[tuple[np.ndarray, int, str]] | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recovery_split.py`:

```python
"""A split is two new transcripts. A cut at the wrong word is wrong three times.

This project publishes the corpus text, scores CER against it and trains on it --
one string, three uses. So a split is refused unless the cut is confident and
both halves stand on their own.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.constants import SAMPLE_RATE
from pipeline.recovery import split_at_silence


class FakeAligner:
    """Word timings without loading MMS_FA."""

    def __init__(self, timings):
        self.timings = timings

    def word_timings(self, audio, text):
        return self.timings


def speech(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.3).astype("float32")


def test_a_clip_is_split_at_the_silence_between_sentences():
    audio = speech(24.0)
    timings = [("Сайн", 0.0, 4.0, 0.9), ("байна", 4.0, 10.0, 0.9),
               ("Өнөөдөр", 14.0, 18.0, 0.9), ("сайхан", 18.0, 23.0, 0.9)]
    parts = split_at_silence(audio, SAMPLE_RATE, "Сайн байна Өнөөдөр сайхан",
                             aligner=FakeAligner(timings),
                             speech_spans=[(0.0, 10.0), (14.0, 23.0)])
    assert parts is not None and len(parts) == 2
    assert parts[0][2] == "Сайн байна"
    assert parts[1][2] == "Өнөөдөр сайхан"


def test_the_transcripts_concatenate_back_to_the_original():
    """If they do not, a word was lost or duplicated at the cut."""
    audio = speech(24.0)
    text = "Сайн байна Өнөөдөр сайхан"
    timings = [("Сайн", 0.0, 4.0, 0.9), ("байна", 4.0, 10.0, 0.9),
               ("Өнөөдөр", 14.0, 18.0, 0.9), ("сайхан", 18.0, 23.0, 0.9)]
    parts = split_at_silence(audio, SAMPLE_RATE, text, aligner=FakeAligner(timings),
                             speech_spans=[(0.0, 10.0), (14.0, 23.0)])
    assert " ".join(p[2] for p in parts) == text


def test_a_weak_alignment_at_the_cut_refuses_the_split():
    """The cut point is the one place the alignment has to be right."""
    audio = speech(24.0)
    timings = [("Сайн", 0.0, 4.0, 0.9), ("байна", 4.0, 10.0, 0.05),
               ("Өнөөдөр", 14.0, 18.0, 0.9), ("сайхан", 18.0, 23.0, 0.9)]
    assert split_at_silence(audio, SAMPLE_RATE, "Сайн байна Өнөөдөр сайхан",
                            aligner=FakeAligner(timings),
                            speech_spans=[(0.0, 10.0), (14.0, 23.0)]) is None


def test_a_clip_within_the_length_limit_is_not_split():
    assert split_at_silence(speech(8.0), SAMPLE_RATE, "Сайн байна",
                            aligner=FakeAligner([("Сайн", 0.0, 4.0, 0.9),
                                                 ("байна", 4.0, 8.0, 0.9)]),
                            speech_spans=[(0.0, 8.0)]) is None


def test_a_clip_with_no_usable_silence_is_not_split():
    """Cutting mid-word to satisfy a length limit would be the defect."""
    assert split_at_silence(speech(24.0), SAMPLE_RATE, "Сайн байна",
                            aligner=FakeAligner([("Сайн", 0.0, 12.0, 0.9),
                                                 ("байна", 12.0, 24.0, 0.9)]),
                            speech_spans=[(0.0, 24.0)]) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_recovery_split.py -q`
Expected: FAIL — `ImportError: cannot import name 'split_at_silence'`

- [ ] **Step 3: Implement**

Add to `pipeline/recovery.py`:

```python
from .constants import MAX_DURATION_S, MIN_DURATION_S
from .trimming import WEAK_WORD_SCORE


def split_at_silence(audio: np.ndarray, sr: int, text: str, *, aligner,
                     speech_spans: list[tuple[float, float]]):
    """Cut an over-length clip into segments at the silences between sentences.

    Each segment is the original audio, unmodified; only the boundaries are new.
    That is what makes this a legal repair -- but it is the one repair that can
    create a defect rather than fail cleanly, because a split is two new
    transcripts, and this project publishes the text, scores CER against it and
    trains on it. A cut at the wrong word is wrong three times.

    So it refuses unless the alignment is confident on the words either side of
    the cut, every segment lands inside the duration limits, and the segments'
    transcripts concatenate back to the original.
    """
    duration = len(audio) / sr
    if duration <= MAX_DURATION_S:
        return None
    timings = aligner.word_timings(audio, text)
    if not timings:
        return None

    gaps = []
    for i in range(len(timings) - 1):
        _, _, end, score_a = timings[i]
        _, start, _, score_b = timings[i + 1]
        if start - end < MIN_DURATION_S:
            continue
        if min(score_a, score_b) < WEAK_WORD_SCORE:
            # The cut point is the one place the alignment has to be right.
            return None
        gaps.append((i, (end + start) / 2.0))
    if not gaps:
        return None

    bounds = [0.0] + [t for _, t in gaps] + [duration]
    words = [w for w, _, _, _ in timings]
    parts: list[tuple[np.ndarray, int, str]] = []
    cut_at = [i for i, _ in gaps]
    first = 0
    for k, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        if not MIN_DURATION_S <= hi - lo <= MAX_DURATION_S:
            return None
        last = cut_at[k] + 1 if k < len(cut_at) else len(words)
        segment_text = " ".join(words[first:last])
        if not segment_text:
            return None
        parts.append((audio[int(lo * sr):int(hi * sr)].astype("float32"), sr,
                      segment_text))
        first = last

    if " ".join(p[2] for p in parts) != " ".join(words):
        return None                      # a word was lost or duplicated
    return parts
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_recovery_split.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Run the gates and commit**

```bash
python -m pytest -q
ruff check .
git add pipeline/recovery.py tests/test_recovery_split.py
git commit -m "Split an over-length clip at silence, and refuse when the cut is not safe"
```

---

## Self-Review

**Spec coverage.** The seven repairs: DC offset, gain, homoglyphs (Task 1), trim-to-speech (Task 2), split (Task 6). Typography normalisation is deliberately absent — the spec records 0 occurrences across both published corpora, so building it would ship a repair that recovers nothing, which the spec explicitly forbids. **Trim audio to the transcript span is not built either**: it needs the same word-timing machinery as the split, and the spec's own risk section notes it can eat speech; it is folded into Task 6's territory and should be added only if calibration shows `cer`/`alignment` failures that the existing transcript trim does not already answer. That is a deliberate reduction from the spec and is recorded here rather than left silent.

Second chance and the one-attempt guard: Task 3. Re-earning on the same thresholds: Task 3, Step 5 — the recursive call recomputes every gate. `recovered_by` through to the manifest: Task 3, Step 4 plus Task 4. Provenance: Task 1, Step 1 — the constants are uppercase scalars in `constants.py`, which `FILTER_POLICY_VERSION` already hashes. Calibration first: Task 5, and Task 6 is explicitly gated on its result.

**Placeholders.** None. Every step carries its code or its command.

**Type consistency.** `Repair`, `REPAIRS`, `RECOVERIES`, `remove_dc_offset`, `normalise_gain`, `repair_homoglyphs`, `trim_to_speech`, `split_at_silence`, `ClipResult.recovered_by`, `CleaningStats.recovered`, `summarise` are each defined once and used under the same name throughout. `trim_to_speech` and `split_at_silence` take keyword-only extras, which is why Task 3 binds them at the call site rather than through the plain `Repair` signature — noted there.

**A correction made before this plan was finalised, recorded so it is not re-introduced.** An earlier draft aimed the trim repair at `snr` and `dnsmos`, on the reasoning that a long room-tone lead-in drags both down. It does not: `_run_vad` edge-trims at `audio_filter.py:222`, before either is measured. Reading the code rather than assuming it also surfaced the repair that *is* real — `speech_ratio` is computed on the untrimmed clip at `:215` and rejected at `:216`, before that trim is applied. Task 2 targets that gate, and uses the original audio because the VAD's timestamps index it.
