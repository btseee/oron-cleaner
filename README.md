# oron-cleaner

Builds the strict Mongolian speech corpus that [oron-tts](../oron-tts) finetunes
`F5TTS_v1_Base` on.

Sources are merged into **one** corpus rather than published separately, because
the consumer is a single training run and the per-speaker cap and the
speaker-disjoint split only make sense across the whole thing.

| source | licence | raw hours | notes |
|---|---|---|---|
| Common Voice 26.0 `mn` | CC0-1.0 | ~40 h | `validated` only, after the down-vote gate |
| FLEURS `mn_mn` | CC-BY-4.0 | ~13 h | 16 kHz native; **no speaker column**, so train-only |
| MBSpeech `mn` | MIT | ~6 h | single male narrator, 16 kHz native |

FLEURS' schema is `id, num_samples, path, audio, transcription,
raw_transcription, gender, lang_id, language, lang_group_id` — there is no
speaker field, and `id` indexes the *sentence*. Its clips therefore go wholly to
training: a split cannot be shown disjoint from a voice it cannot name. Its
`gender` is a `ClassLabel`, so a row yields `0`/`1`/`2` and needs decoding before
it means anything.

All commercially usable.

`disco-eth/WorldSpeech` `mn_mn` is by far the largest Mongolian corpus — ~221 h,
**24 kHz native**, the only source with real full-band content — but it is
**CC-BY-NC-4.0**. It is excluded from `all` and needs an explicit double opt-in,
because including it makes the trained model non-commercial:

```bash
python clean_pipeline.py --datasets cv,ws --allow-non-commercial
```

## Run it

```bash
pip install -e .
python clean_pipeline.py                       # all sources, then upload
python clean_pipeline.py --datasets cv --no-upload
python clean_pipeline.py --finalize-only       # re-split without refiltering
```

`API_KEY` (Mozilla Data Collective) and `HF_TOKEN` load from `.env`.

> **The Common Voice dataset id is release-specific and goes stale.** Each
> language of each release is its own dataset, and the API has no list or search
> endpoint, so when `_DATASET_ID` stops resolving take the current one from the
> dataset page URL. Downloads also require the account to have accepted that
> dataset's terms in the web UI, and are capped at 30/day per organisation.

## Calibrate before the full run

Every threshold below was set from published figures, small samples, or
reasoning about what a strict corpus needs — **none against this corpus's own
distribution**. A threshold 10% too strict silently discards hours of usable
audio while looking like it worked, and the full pass is 24–48 h.

```bash
python clean_pipeline.py --datasets cv --calibrate --limit 500 --no-upload
```

Calibration mode scores **every** gate instead of stopping at the first failure.
That matters: in a normal run a clip rejected for SNR is never scored for
DNSMOS, so the DNSMOS rate is only the rate among clips that already passed SNR,
and the gates cannot be compared. The report gives independent per-gate
rejection rates, the full distribution of each metric, what the current
threshold keeps, and the threshold that would hit a target yield.

Then edit `pipeline/constants.py` and re-run without `--calibrate` — the policy
hash invalidates cached work automatically.

## Output

```
output/oron_mn_strict/
  wavs/<clip_id>.wav        24 kHz mono, edge-trimmed, peak-normalised
  metadata.csv              audio_file|text — F5-TTS prepare_csv_wavs.py contract
  metadata_validation.csv   likewise
  metadata_test.csv         likewise
  manifest.parquet          per-clip metrics and speaker metadata
  manifest.jsonl            the same rows, written as clips pass
  corpus_summary.txt        hours, speakers, and the acceptance criteria
```

## Gates

| gate | threshold | why |
|---|---|---|
| duration | 1–20 s | F5-TTS silently drops clips longer than the frame budget |
| clipping / DC | ≤0.1% full-scale, ≤0.01 offset | survives every other gate; teaches the model to reproduce distortion |
| voice activity | ≥35% speech, **edge-trimmed** | interior pauses preserved |
| SNR | ≥15 dB | speech regions vs **true** non-speech regions |
| bandwidth | ≥6 kHz lowpass shelf | recorded per clip; no Mongolian source is full-band |
| DNSMOS P.835 | OVR ≥2.8 · SIG ≥3.0 · BAK ≥2.5 | ~2.0 is "poor" on a 1–5 scale |
| **forced alignment** | **≥0.65** | primary transcript gate |
| CER | ≤0.20 | secondary, on clips that already aligned |
| per-speaker cap | 0.6 h (8 h for a single-narrator source) | top 10 of 511 speakers held 45.7% of clips; a count would track clip length instead of speech |

Thresholds live in `pipeline/constants.py` and are hashed into
`FILTER_POLICY_VERSION`, so changing one invalidates cached work instead of
silently mixing policies.

### Why forced alignment rather than ASR

Free-running ASR is the wrong instrument in Mongolian. The best available model
has a CER floor of **0.123** on clean, correctly-transcribed speech
(whisper-large-v3 is **0.311**), so an absolute CER threshold sits near the
recogniser's own error — it rejects good clips and admits bad ones.

Forced alignment is constrained to the transcript it is given, so a low score is
evidence the audio does not contain those words. Measured on real Mongolian
audio, scoring each clip against its own transcript and against another's:

| corpus | correct (min) | mismatched (max) |
|---|---|---|
| FLEURS | 0.829 | 0.443 |
| Common Voice | 0.722 | 0.547 |

Clean separation on both. The threshold sits at 0.65 and is biased toward
rejecting: a mismatched clip teaches a wrong text-to-audio mapping, a rejected
good clip only costs data.

## Gender

F5-TTS takes voice identity from a reference clip, not a token, so the corpus
needs enough clean speech of each gender and a way to find the best candidate.

- Common Voice ≥v17 emits `male_masculine` / `female_feminine`.
- 39.7% of clips carry no label, but there are only 520 speakers — so a label on
  any one of a speaker's clips settles all of them.
- Where a speaker declared nothing anywhere, gender is inferred from median F0,
  calibrated on 80 self-declared clips: male 70–150 Hz, female 173–306 Hz, **zero
  overlap**. The 155–170 Hz dead band is left unknown rather than guessed.

Values describing identity rather than vocal tract (`non_binary`, `intersex`,
`transgender`, `do_not_wish_to_say`) map to unknown and are never inferred.

## Architecture

`pipeline/dsp.py` and `pipeline/speakers.py` hold pure functions over arrays and
record dicts — no models, no IO — so the logic that decides what enters the
corpus is testable without a GPU or a 3 GB download. `pipeline/audio_filter.py`
is the only module that loads models.

Clips are written to disk the moment they pass. They used to accumulate in a
list with their decoded audio and then be copied again, which is roughly 14 GB
at Common Voice scale. Resume is keyed by clip id, so a restart re-does no work
regardless of dataset ordering.

## Tests

```bash
pytest                       # no models required
RUN_SLOW_TESTS=1 pytest      # adds the alignment test (downloads 1.18 GB)
```
