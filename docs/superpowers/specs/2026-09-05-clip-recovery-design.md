# Clip recovery

Repair the clips that are genuinely repairable, and only those.

The pipeline discards a clip at the first gate it fails. Most of those failures
are real and final — you cannot invent bandwidth, or audio that was never
recorded, or a transcript nobody wrote. But some are not failures of the
recording at all. A clip whose speech is clean and whose transcript is right can
still be thrown away for running past a length limit that exists to suit a batch
sampler, when a silence in its middle would satisfy that limit twice over.

This adds a repair step for exactly that, and nothing else. Six repairs were
considered and measured; **one shipped.** The measurements that killed the other
five are kept below, because "why not also trim the edges?" is a question worth
answering once rather than every time it is asked.

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

## Measured, before building anything

Per-gate failure rates, `measure_all` so every gate is scored rather than only
the first to fire. FLEURS 120 clips, Common Voice 80, WorldSpeech 80.

| gate | FLEURS | Common Voice | WorldSpeech |
| --- | --- | --- | --- |
| kept | 42% | **74%** | 35% |
| `cer` | 22% | 0% | 51% |
| `dnsmos` | 22% | 15% | 20% |
| `snr` | 26% | 0% | 14% |
| `bandwidth` | 16% | 11% | 14% |
| `alignment` | 5% | 0% | 12% |
| `duration` | 8% | 0% | 8% |
| `vad` | 2% | 0% | 9% |
| `clipping` | 0% | 0% | 0% |

**This design is not worth building for Common Voice.** Only two gates fire
there, and neither is repairable under the rule above: `bandwidth` cannot be
extended without synthesis, and the failing `dnsmos` values (2.08, 2.18, 2.25,
2.79 against a threshold near 3.0) are far below what a gain change could lift
— DNSMOS normalises level internally. Every repair below targets a gate that
never fired on Common Voice.

Where recovery would pay is **WorldSpeech**: 35% kept, with `duration` 8%,
`vad` 9% and `alignment` 12% all in repairable territory. That is the corpus
excluded from the curriculum for being CC-BY-NC.

Three earlier claims in this document were wrong and are corrected here rather
than quietly edited away:

* Common Voice does **not** reject ~48%. That conflated gate failures with the
  per-speaker cap and split filtering; the gate rate is about 26%.
* Trimming to the speech span is **not** the biggest Common Voice lever. It
  recovers nothing — the VAD already edge-trims, and Common Voice has no `vad`
  failures at all.
* Splitting was deprioritised for Common Voice for the wrong reason. It is
  correctly zero there, but it is 8% on both FLEURS and WorldSpeech.

### Corrected with the real Common Voice 26

The CV 17 proxy was wrong, and wrong in the direction that mattered: it used
`train.tsv`, which is curated, while the pipeline reads `validated.tsv`, which is
not. Measured on the real CV 26 archive, `validated.tsv`, 100 clips:

| gate | FLEURS | CV 17 (proxy) | **CV 26 (real)** | WorldSpeech |
| --- | --- | --- | --- | --- |
| kept | 42% | 74% | **42%** | 35% |
| `dnsmos` | 22% | 15% | **34%** | 20% |
| `cer` | 22% | 0% | **32%** | 51% |
| `snr` | 26% | 0% | **18%** | 14% |
| `alignment` | 5% | 0% | **15%** | 12% |
| `bandwidth` | 16% | 11% | **13%** | 14% |
| `vad` | 2% | 0% | **3%** | 9% |
| `duration` | 8% | 0% | **0%** | 8% |
| `clipping` | 0% | 0% | **0%** | 0% |

So Common Voice 26 rejects 58%, not 26% — there is plenty of rejected material.
**None of it is recoverable under this design's rule.** What it rejects is noise
(`dnsmos` 2.20, `snr` 12.2 dB), misreadings (`cer` 0.347, 0.500, 1.136), genuine
transcript mismatch (`align_0.212`) and bandwidth — every one of which needs the
speech altered or the text guessed. Its 3% `vad` failures are all
`vad_no_speech`, not the `speech_ratio` case this design targets, and `duration`
and `clipping` are flat zero, so splitting and DC removal have nothing to act on.

WorldSpeech's `vad` failures are `trimmed_too_short_0.99s` and `0.95s` — clips
that fall just under the 1.0 s floor after edge trimming. Not repairable either:
the audio really is 0.99 s. Its one recoverable gate is `duration`, and measured
directly on 300 clips, **5% run over the 20 s limit** (median 2.8 s, max 30 s).

**Conclusion: splitting over-length clips is the only repair with measurable
yield anywhere — about 5% on WorldSpeech and 8% on FLEURS, and zero on Common
Voice.** The repairs already built (DC offset, gain, homoglyphs) recover
essentially nothing on any of the three. By this document's own rule — *a
recovery that turns out to recover nothing gets deleted, not shipped* — the work
stops here rather than continuing through the remaining tasks.

The homoglyph repair is worth keeping for a different reason: it is a
correctness fix, not a recovery one. It removes letters that reach the model as
their own embedding rows for characters nobody typed.

Caveats: the CV 17 figures above were measured from an ungated mirror, using
`train.tsv`, because CV 26 needs the Mozilla Data Collective key that is pending
rotation and `validated.tsv` is what the pipeline actually reads. Same corpus
family, different release, more curated split. 80 clips is a sample, not a
census.

## What was considered repairable

Measured or reasoned, per gate, *before* the yield of any of it was known. Only
the `duration too_long` row shipped: the measurements below the table found the
others recover approximately nothing on all three corpora. Homoglyph repair
survives, but not as a recovery — it is a text normalisation applied to every
transcript (`audio_filter.py:124`, called from `normalized_text`), not a second
chance offered to a clip that was already rejected.

| gate | repair | why it is information-preserving |
| --- | --- | --- |
| `vad speech_ratio` | apply the edge trim the VAD already computed | The ratio is measured on the **untrimmed** clip (`audio_filter.py:215`) and rejected at `audio_filter.py:216`, *before* `edge_trim_bounds` runs at `:222`. So a clip with three seconds of speech inside ten seconds of lead-in scores 0.30, fails the 0.35 gate, and is discarded — while the edge-trimmed version would pass everything. The trim keeps interior pauses and removes only non-speech at the edges. |
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

* `snr` and `dnsmos` by edge trimming — **already done.** `_run_vad` edge-trims
  before either is measured (`audio_filter.py:222`), so a repair here would
  recover nothing. This was in an earlier draft of this design and was removed
  after reading the code rather than assuming it.
* `duration too_short` — the audio is not there.
* `vad` with no speech found — likewise.
* `bandwidth` — extension is synthesis.
* `clipping` by sample ratio — de-clipping invents the clipped peaks.
* `cer` from a genuine misreading — the speaker said something else.
* numeral refusals — the normaliser refuses case suffixes it cannot expand
  without guessing, and that needs a native speaker, not code.

## Design

### Where it goes

Recovery is a **second chance after rejection**, and it lives in the loop, not
inside the filter. `process_split` (`pipeline/processor.py`) already holds
everything the decision needs: the `ClipResult` that says which gate failed and
why, the source item, the writer, the rejection log and the stats.

So when — and only when — a clip is rejected with `reject_stage == "duration"`
and a reason beginning `too_long`, `_split_clip` is offered the clip. Every
other rejection stage is final and nothing is attempted for it.

An earlier draft of this document put recovery inside
`AudioQualityFilter.process_clip`, behind a `recover: bool` parameter and a
recursion guard. That is not what shipped, and the difference is not cosmetic.
`process_clip` returns one judgement about one clip; a split returns **two or
more clips**, each needing its own id, its own manifest row, its own rejection
log entry and its own place in the stats. Fitting that inside `process_clip`
would have pushed corpus-writing concerns down into the filter and made every
caller unpack a list to get one result. Keeping the filter a pure judgement and
letting the loop decide what to do with a rejection puts the only stateful
decision where the state already is.

The ordering still earns what it was meant to earn. A repair runs only on a clip
that needs it, so nothing that already passes is touched. Every segment goes
through the whole gate stack via `_finalize_clip`, exactly as any other clip
does, so a split cannot smuggle a clip past a gate it did not face. And a
segment is never itself offered a split, because only the source clip ever
reaches that branch.

### Components

**`pipeline/recovery.py`** — one function.
`split_at_silence(audio, sr, text, *, aligner, speech_spans)` returns a list of
`(audio, sample_rate, text)` segments, or `None` when it will not cut. `None` is
the common case and not an error. There is no `RECOVERIES` mapping and no repair
registry: gate-to-repairs is the right shape for six repairs and pure overhead
for one, and this one has a signature no other repair would have shared — it
needs an aligner and VAD spans, which a `(audio, sr, text)` repair does not.

**`pipeline/processor.py::_split_clip`** — the adapter between the loop and the
repair. It decodes the audio again (`process_clip` discarded its copy on the way
to rejecting the clip), normalises the transcript, re-runs the VAD for its
speech spans, and converts those spans from sample indices to seconds, which is
what `split_at_silence` compares word timings against. It returns `None` for
every refusal, including a transcript the normaliser will not publish: a clip
that cannot be published is not one to spend a split on.

**Normalisation happens before the split, not after.** The aligner is given the
transcript that will be published, for two reasons that point the same way.
`word_timings` refuses when romanisation does not emit one token per source
word, and a digit romanises to nothing — so on raw text every clip carrying a
date or a number refuses, which is disproportionately the long ones this repair
exists for. And because a segment's transcript is a slice of whatever went in,
normalising first is what makes "the segments concatenate back to the original"
an invariant about the *published* corpus rather than about an intermediate
string.

**`ClipResult.recovered_by: str`** — the repair that produced this clip, empty
for a clip accepted as recorded. `process_clip` still returns one result for one
clip; the loop sets this field on each segment's result.

**Constants** — `RECOVERY_MIN_SILENCE_S` and `RECOVERY_MIN_CUT_SCORE` in
`pipeline/constants.py`, as uppercase scalars, because `_policy_version()`
hashes that module's uppercase scalars and nothing else. Tuning either changes
which clips split, which changes the corpus, so it has to change
`FILTER_POLICY_VERSION` with it.

### Calibration first

This is what happened, and it is the section that decided the shape of
everything above. The yield was measured on a sample before the code was
written, and the measurement killed five of the six repairs — see the tables
above. A recovery that turns out to recover nothing gets deleted rather than
shipped, and five were.

### Provenance

`FILTER_POLICY_VERSION` hashes the thresholds. Recovery changes the corpus
without changing a gate, so the recovery constants sit in
`pipeline/constants.py` beside them and are hashed in. Two corpora built under
different recovery rules cannot share a version string.

Every recovered clip carries `recovered_by` into `manifest.jsonl` and the
published parquet, so a consumer can select or exclude repaired material.
`CleaningStats` counts kept clips by the repair that produced them and prints
the count with its share of kept clips; `corpus.summarise` prints the same from
the manifest. Both print when the count is zero, so "nothing was recovered" is a
measurement rather than a line that happened not to appear.

Resume needs a third piece. A source consumed by splitting is never written
under its own id — the corpus gets `{clip_id}_p0`, `_p1` — and every segment has
to re-earn its place, so a run where segment 0 was rejected, or where every
segment was, leaves nothing on disk named after the source at all. The ids of
consumed sources are therefore recorded in a sidecar beside the stats
checkpoint, namespaced by `FILTER_POLICY_VERSION` for the same reason the stats
are. Without it every restart re-decoded the source, re-split it, re-ran the
model stack on each segment and appended a second copy of every count.

## Risks

**A split is two new transcripts.** If the alignment cuts at the wrong word,
both halves are published, scored and trained on with wrong text. So a cut is
refused unless the alignment scores at least `RECOVERY_MIN_CUT_SCORE` on the
words either side of it, the gap between them is at least
`RECOVERY_MIN_SILENCE_S`, the VAD independently agrees there is silence at that
point, every segment lands inside the duration limits, and the segments'
transcripts concatenate back to the original word for word. Any one of those
failing refuses the whole split, not just the one cut. This is the one repair
that can create a defect rather than fail cleanly, and it is gated hardest.

**A failing segment does not veto its siblings.** An earlier draft of this
document said a split is refused unless *both halves independently pass
alignment and CER*. That is not what shipped, and the shipped rule is the better
one. Requiring every segment to pass or none conflates two different questions:
"is this cut in the right place?", which the refusals above answer for the split
as a whole, and "is this clip good enough?", which the gates answer for every
clip in the corpus one at a time. A segment that is genuinely noisier than its
sibling has failed on its own merits, and discarding the clean sibling with it
buys nothing. This branch's rule is that a repaired clip re-earns *its* place on
the same thresholds; earning it jointly would be a different and weaker rule.

**Recovery can flatter the corpus statistics.** A corpus that is 30% recovered
clips has a different character from one that is not, even when every clip
passed the same gates. The cleaning report and the corpus summary both print the
recovered count and its share of kept clips, so the difference is visible rather
than absorbed.

**The repair runs inside a 24–48 h loop.** `_split_clip` decodes audio, runs a
VAD and reaches a forced aligner, all on arbitrary corpus text; each segment then
goes through the entire model stack. Neither call is internally total, so both
are wrapped the way the source `process_clip` call is — log, count the clip as a
crash rejection, continue. Unwrapped, an exception from either ended the pass.

## Testing

* `tests/test_recovery_split.py` — `split_at_silence` as a pure function: it
  cuts a two-sentence clip at its silence, and refuses when the cut point's
  alignment is weak, when the gap is too short, when the VAD hears no silence
  there, when a segment would fall outside the duration limits, and when the
  segment transcripts do not concatenate back to the original.
* `tests/test_processor.py` — the wiring, with the filter stubbed and
  `split_at_silence` monkeypatched: only a `too_long` rejection is offered a
  split; a refused split leaves the clip exactly as rejected; the corpus gets
  the segments and never the source; a failing segment does not block a passing
  sibling; a segment is never split again; provenance survives the
  normalisation-refusal path and reaches the manifest; segments are measured on
  every gate during calibration; a split source is not re-split on restart under
  any combination of segment outcomes; the recovered count survives a restart;
  and a crash in either the split or a segment costs one clip rather than the
  run. Nothing heavier than `soundfile` may be imported at module scope here —
  a module-scope `pytest.importorskip` aborts collection of the whole file, and
  once did.
* `tests/test_processor_real_split.py` — `_split_clip` driving the **real**
  `split_at_silence` through a real `ForcedAligner`, with only the 1.18 GB
  acoustic model stubbed. Its own module because it needs uroman and torchaudio.
  These exist because every wiring test faked `speech_spans`, which is the one
  argument production got wrong: `_split_clip` passed `[]`, so nothing ever
  split while the suite stayed green.
* The homoglyph tests live with the text path in `tests/test_audio_filter.py`,
  where the repair now is: an English proper noun inside a Mongolian sentence is
  left alone, every mapping lands on a Cyrillic letter, and the published text
  is the repaired one.

## Out of scope

* Denoising, de-clipping, bandwidth extension, transcript correction.
* Re-cleaning MBSpeech: its raw mirror was deleted and only the cleaned corpus
  survives.
* WorldSpeech: it is CC-BY-NC and the model does not train on it, so recovery
  there improves a dataset nobody trains on.
* Rotating the Mozilla key. That is a human action and it blocks the Common
  Voice run, not this work.
