# Clip recovery

Repair the clips that are genuinely repairable, and only those.

The pipeline discards a clip at the first gate it fails. Most of those failures
are real and final — you cannot invent bandwidth, or audio that was never
recorded, or a transcript nobody wrote. But some are not failures of the
recording at all. A clip whose speech is clean and whose transcript is right can
still be thrown away because it carries eight seconds of room tone before the
sentence starts, or because the microphone had a DC offset, or because it is one
second over a length limit that a silence in its middle would satisfy.

This adds a repair step for exactly those, and nothing else.

Written 2026-09-05, after the model shipped. `docs/superpowers/specs/` in
`oron-tts` holds the training-side specs; this one is corpus-side and lives with
the pipeline it changes.

## The rule that decides everything

**A repair may not alter the speech signal, and a repaired clip re-earns its
place on the same thresholds.**

Both halves matter.

The first half rules out the tempting things — denoising a noisy clip, de-clipping
a hot one, extending bandwidth, correcting a transcript that does not match what
was said. Each would recover more hours. Each also puts something into the corpus
that nobody recorded. A denoised clip teaches the model the denoiser's artifacts,
and worse, its SNR and DNSMOS then measure the denoiser rather than the
recording, so the gate that was supposed to judge quality is judging our own
processing. Transcript repair is the sharpest edge: this project publishes the
corpus text, scores CER against it, and trains on it — one string, three uses —
so a wrong repair is wrong three times.

The second half rules out the subtler failure: recovering clips by being kinder
to them. Every repair is followed by a **complete re-run of the gate stack**, and
every measurement is recomputed on the repaired audio. Trim a clip to raise its
SNR and the SNR is measured again on what remains; the pre-repair number is never
carried forward. One repair attempt per failing gate, one re-gate, and then the
answer stands — no iterating until something passes, which is fishing.

## What is repairable

Measured or reasoned, per gate:

| gate | repair | why it is information-preserving |
| --- | --- | --- |
| `snr`, `dnsmos` | trim to the aligned speech span | Both are computed over the whole clip, so a long room-tone lead-in drags them down while the speech is untouched. Removing non-speech removes no speech. |
| `duration too_long` | split at internal silence | Each segment is the original audio, unmodified; only the boundaries are new. |
| `clipping dc_offset` | subtract the mean | Exact. A DC offset is an additive constant with one correct removal. |
| `dnsmos`, `cer`, `vad` | normalise gain to a target peak | A scalar multiply changes no information. Crowd-sourced clips are often too quiet for the VAD to find speech or the recogniser to read it. |
| `cer`, `alignment` | trim audio to the transcript span | The mirror of the transcript trimming already in `pipeline/trimming.py`: when the reader says something after the sentence, cut the audio. Edges only, never the middle. |
| `vocab`, `cer` | homoglyph repair | A Latin `o` inside an otherwise-Cyrillic word has exactly one correct reading. It is an encoding error, not an ambiguity. |

Measured in the published corpora: **3 Latin homoglyphs inside Cyrillic words in
`common-voice-26-mn`'s 15,092 kept clips, 19 in `fleurs-mn`'s 1,908.** Small, but
they are in the shipped corpus, each takes its own embedding row, and the
adversarial review found `і` U+0456 passing `is_representable` for this reason.

Also measured, and the reason one candidate repair is not in the table:
**0 non-NFC transcripts, 0 non-breaking spaces, soft hyphens, zero-width
characters, curly quotes or dashes** across both corpora. The normaliser already
handles typography. A repair for it would recover nothing; it stays out.

## What is not repairable

Recorded so the question stays settled:

* `duration too_short` — the audio is not there.
* `vad` with no speech found — likewise.
* `bandwidth` — extension is synthesis.
* `clipping` by sample ratio — de-clipping invents the clipped peaks.
* `cer` from a genuine misreading — the speaker said something else.
* numeral refusals — the normaliser refuses case suffixes it cannot expand
  without guessing, and that needs a native speaker, not code.

## Design

### Where it goes

Recovery is a **second chance after rejection**, not a transform applied up
front. `AudioQualityFilter.process_clip` already knows which gate failed and
why. When a gate fails and a repair is registered for it, apply the repair and
re-run `process_clip` from the top on the result.

That ordering earns three things. Repairs run only on clips that need them, so
nothing that already passes is touched. The repaired clip goes through every
gate, not just the one that failed, so a trim that fixes SNR cannot smuggle
through a clip that now fails duration. And the code reads as what it is: a
rejection, then one attempt to answer it.

### Components

**`pipeline/recovery.py`** — the repairs, each a pure function from
`(audio, sample_rate, text)` to a repaired triple or `None` when it does not
apply. No gate knowledge, no I/O. `None` means "this clip is not the kind this
repair fixes", which is the common case and not an error.

**`RECOVERIES: dict[str, tuple[Repair, ...]]`** — which repairs to try for which
failing gate, in order. The mapping is data, so the table above and the code
cannot drift apart, and a test can assert every entry is reachable.

**`AudioQualityFilter.process_clip`** gains a `recover: bool = True` parameter
and a single recursion guard: a repaired clip is re-gated with `recover=False`,
so one repair attempt per clip, never a chain.

**`ClipResult`** gains `recovered_by: str` — the repair that produced this clip,
empty when it was accepted as recorded. Splitting yields more than one clip, so
`process_clip` returns a list where it returned one result; every caller updates.

### Calibration first

The yield is unknown and cannot be assumed. `common_voice` is fetched from the
Mozilla Data Collective API, so its rejected clips are not reachable until the
leaked key is rotated, and no published artifact carries the rejected material.

So the first deliverable is a **recovery calibration pass**: over a sample of
rejected clips, report per gate how many had a repair available, how many the
repair changed, and how many then passed every gate. That is the number the
decision rests on, and it costs one sample rather than one full run. It mirrors
`--calibrate`, which exists for exactly this reason.

A recovery that turns out to recover nothing gets deleted, not shipped.

### Provenance

`FILTER_POLICY_VERSION` hashes the thresholds. Recovery changes the corpus
without changing a threshold, so the recovery configuration — which repairs are
enabled and their constants — goes into the version. Otherwise two corpora with
the same version string differ, which is the failure the version exists to
prevent.

Every recovered clip carries `recovered_by` into `manifest.jsonl` and the
published parquet, so a consumer can select or exclude repaired material, and so
the corpus summary can report how much of it is recovered.

## Risks

**A split is two new transcripts.** If the alignment cuts at the wrong word,
both halves are published, scored and trained on with wrong text. So a split is
refused unless the alignment is confident at the cut point and both halves
independently pass alignment and CER. This is the one repair that can create a
defect rather than fail cleanly, and it is gated hardest.

**Trimming can eat speech.** `pipeline/trimming.py` already refuses to remove
more than `MAX_TRIM_FRACTION` of a transcript; audio trimming needs the same
ceiling, for the same reason.

**Recovery can flatter the corpus statistics.** A corpus that is 30% recovered
clips has a different character from one that is not, even when every clip
passed the same gates. The summary reports the recovered fraction so the
difference is visible rather than absorbed.

## Testing

* Each repair, on audio constructed to need it and audio constructed not to:
  the second must return `None` rather than a changed clip.
* DC removal and gain normalisation assert the speech is unchanged — correlation
  with the original at 1.0 to floating-point tolerance — because "does not alter
  the speech" is the specification, not an aspiration.
* The re-gate is not skipped: a repaired clip that still fails is rejected. Test
  by repairing a clip whose repair cannot save it.
* One repair attempt only: a clip that fails again after repair is not repaired
  again. Assert the recursion guard by counting calls.
* Homoglyph repair on a word that is legitimately Latin — an English proper noun
  in a Mongolian sentence — must leave it alone.
* Splitting: refuse when the cut point's alignment is weak; refuse when either
  half fails; assert the two transcripts concatenate back to the original.

## Out of scope

* Denoising, de-clipping, bandwidth extension, transcript correction.
* Re-cleaning MBSpeech: its raw mirror was deleted and only the cleaned corpus
  survives.
* WorldSpeech: it is CC-BY-NC and the model does not train on it, so recovery
  there improves a dataset nobody trains on.
* Rotating the Mozilla key. That is a human action and it blocks the Common
  Voice run, not this work.
