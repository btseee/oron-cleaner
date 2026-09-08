# Working in oron-cleaner

Builds the strict Mongolian speech corpora that `oron-tts` trains on. Input is
Common Voice, FLEURS and MBSpeech — all three commercially usable; output is a
corpus directory plus a manifest, published to HuggingFace under `btsee/`.

**WorldSpeech is not an ordinary fourth input.** It is CC-BY-NC-4.0, excluded
from `--datasets all`, and requires both `--datasets ws` and
`--allow-non-commercial` to include. This has already shipped wrong once: a
published model card carried `license: cc-by-nc-4.0` inherited from
WorldSpeech even though WorldSpeech had failed the pass-rate gate and was never
trained on.

The consumer is a separate repository, `oron-tts`. Read its `AGENTS.md` too if
you are touching anything that reaches the model.

## The one rule that explains most of the design

**The published corpus text, the text CER is scored against, and the text
training reads are one string.** That is why the normaliser refuses
constructions it cannot expand rather than guessing — a wrong expansion is
published, scored, and learned, and the CER gate cannot catch it because the
reference *is* the corrupted string. The normaliser itself lives in the sibling
repo: `pipeline/audio_filter.py` imports `MongolianNormalizer` from
`oron_tts.text`.

## Four things that fail silently

**1. `bandwidth_hz` is meaningless without `native_sr`.** A bandwidth
measurement can never exceed its decode rate's Nyquist. Before filter policy v4
the pipeline decoded everything to 16 kHz *before measuring*, so the column was
pinned below 8 kHz for every corpus and looked like a measurement. Both fields
are now written per row. If `native_sr` is absent, the corpus predates the fix
and its bandwidth column is the truncation.

**2. `FILTER_POLICY_VERSION` hashes every uppercase scalar in
`pipeline/constants.py`, except `OUTPUT_DIR` and the version itself.** Change a
threshold and the version changes, which is what stops two corpora built under
different rules from being pooled. It is recorded once per corpus in
`provenance.json` — *not* on manifest rows. oron-tts's `build_f5_dataset.py`
reads it from there and refuses a mismatch — but only when it can import
`pipeline.constants` at all. On a training pod with `oron-tts` but not
`oron-cleaner` installed, that check is a silent no-op, not a refusal.

**3. The pipeline is single-threaded on purpose.** One MMS_FA alignment of a
6 s clip costs 0.07 s on **one thread** and 0.84 s on forty-eight — short
sequences, so synchronisation dominates. `TORCH_THREADS = 1`
(`pipeline/constants.py`) is deliberate; "fixing" it made a pass twelve times
slower. The consequence is that a full Common Voice pass (~30k clips) runs for
days on one core. Sharding clips across worker processes is the prerequisite
for any full corpus rebuild.

**4. `pip install -e .` does not work on a clean machine.** `pyproject.toml`
lists `oron-tts` as a dependency and oron-tts is not on PyPI — it is installed
from its own checkout — so the editable install tries to resolve it from an
index and fails. Install from `requirements.txt`, which omits `oron-tts`,
mirrors the rest of pyproject, and is asserted not to drift behind it. That file
was empty once, which made `pip install -r` a silent no-op and cost three pod
runs, each dying on a different missing import.

There is a third file, `requirements.lock`: the pinned transitive closure of
`pyproject.toml`'s dependencies, gated by `scripts/check_lockfile.py`. Every
dependency in `pyproject.toml` was declared `>=` with no ceiling, so an
upstream `transformers`, `torch` or `datasets` release could silently change
what the filter gates measure with no signal. Adding a dependency correctly
means touching all three files: `pyproject.toml`, `requirements.txt`, and
`requirements.lock`.

## Order of operations that matters

- **VAD edge-trims before SNR and DNSMOS are measured**, but `speech_ratio` is
  computed on the *untrimmed* clip and rejects before the trim. Measurements run
  on the signal they describe.
- **Edge-trim, never splice.** Concatenating the VAD's speech segments deletes
  every interior pause and butt-joins the pieces — it destroys prosody and
  corrupts every downstream measurement.
- **Forced alignment gates the transcript before the ASR does.** Measured
  separation on real Mongolian audio is 0.722 worst-correct against 0.547
  worst-mismatched, where CER's own floor on correct clips is 0.123.

## How to run things

```bash
ruff check .                                    # CI gate
python scripts/check_ci_imports.py              # CI gate: no test may reach torch at import time
python scripts/check_lockfile.py                # CI gate: requirements.lock covers pyproject.toml
python -m pytest tests/ -q                      # 290 passing, 2 skipped
python clean_pipeline.py --datasets cv --calibrate --limit 300 \
    --corpus-dir <dir> --no-upload             # tune thresholds first
python clean_pipeline.py --datasets cv --corpus-dir <dir> --no-upload
python clean_pipeline.py --finalize-only --corpus-dir <dir>   # splits, gender, holdout
```

The two skipped tests are legitimate: one needs a 1.18 GB download behind
`RUN_SLOW_TESTS=1`, the other guards a POSIX-only failure that cannot occur on
Windows. A test that skips silently is not a test — this project has shipped
two defects behind that pattern, including 26 hidden by a module-scope
`importorskip`.

## Secrets

`API_KEY` (Mozilla Data Collective, for Common Voice downloads) and `HF_TOKEN`
live in `.env`, which is gitignored and must never be tracked. On a remote
machine write them over stdin with `chmod 600` — never as environment variables
on the pod, which the provider's API returns in plain text.
