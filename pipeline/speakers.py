"""Speaker and gender handling.

This is what makes a male and a female voice deliverable. F5-TTS takes voice
identity from a reference clip, not from a token, so the corpus needs enough
clean speech of each gender and a way to find the best candidate clip in each.

Three problems the raw metadata has:

* **39.7% of Common Voice clips carry no gender label** (13,237 of 33,258
  validated), but there are only 520 speakers -- so a label on any one of a
  speaker's clips settles all of them.
* **Common Voice ≥ v17 emits `male_masculine` / `female_feminine`**, which the
  downstream mapping did not recognise, so every row silently produced no
  gender at all.
* **The top 10 speakers hold 45.7% of validated clips**, the largest with 1,956.
  Without a cap the model collapses toward a handful of voices.

Pure functions over record dicts -- no models, no IO.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

from .constants import MAX_NARRATOR_HOURS, MAX_SPEAKER_HOURS

log = logging.getLogger(__name__)

MALE = "male"
FEMALE = "female"
UNKNOWN = ""

# Common Voice has used several vocabularies across releases. Anything not
# listed maps to UNKNOWN rather than being guessed at.
_GENDER_ALIASES: dict[str, str] = {
    "male": MALE, "male_masculine": MALE, "m": MALE, "man": MALE,
    "female": FEMALE, "female_feminine": FEMALE, "f": FEMALE, "woman": FEMALE,
    # Explicitly not inferred: these describe identity, not vocal tract, and
    # guessing would be both wrong and disrespectful.
    "other": UNKNOWN, "non_binary": UNKNOWN, "intersex": UNKNOWN,
    "transgender": UNKNOWN, "do_not_wish_to_say": UNKNOWN, "": UNKNOWN,
}

# Calibrated on 80 Common Voice clips with self-declared labels (40 per gender):
#     male    min 70.1   median 115.2   max 150.3
#     female  min 172.7  median 241.4   max 305.9
# Zero overlap, with a 150.3..172.7 gap. The dead band between these bounds is
# left UNKNOWN rather than guessed.
MALE_F0_MAX_HZ = 155.0
FEMALE_F0_MIN_HZ = 170.0

# An evaluation split with one voice measures that voice. Three is the smallest
# number that lets a per-speaker outlier be seen as one.
MIN_SPEAKERS_PER_EVAL_SPLIT = 3

# Sentences withheld from training so the CER target text is genuinely unseen.
# eval_mn.py reports over 200 by default; the margin covers sentences later lost
# to the audio gates.
EVAL_HELD_OUT_SENTENCES = 400


def normalize_gender(value: Any) -> str:
    """Map any release's gender vocabulary onto male/female/unknown."""
    if value is None:
        return UNKNOWN
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _GENDER_ALIASES.get(key, UNKNOWN)


def gender_from_f0(median_f0_hz: float) -> str:
    """Infer gender from median F0, refusing to guess inside the dead band."""
    if not median_f0_hz or median_f0_hz <= 0:
        return UNKNOWN
    if median_f0_hz <= MALE_F0_MAX_HZ:
        return MALE
    if median_f0_hz >= FEMALE_F0_MIN_HZ:
        return FEMALE
    return UNKNOWN


def propagate_gender(
    records: list[dict], *, speaker_key: str = "client_id", gender_key: str = "gender"
) -> tuple[list[dict], dict[str, int]]:
    """Fill missing gender from a speaker's other clips, then from F0.

    Declared labels always win over inference, and a speaker whose declared
    labels disagree is left unknown rather than resolved by majority -- that
    usually means the speaker id is shared, which is itself a problem.
    """
    declared: dict[str, set[str]] = defaultdict(set)
    for r in records:
        g = normalize_gender(r.get(gender_key))
        if g:
            declared[str(r.get(speaker_key) or "")].add(g)

    resolved: dict[str, str] = {}
    conflicted = 0
    for spk, genders in declared.items():
        if len(genders) == 1:
            resolved[spk] = next(iter(genders))
        else:
            conflicted += 1
            log.warning("Speaker %s has conflicting gender labels %s", spk, genders)

    counts = {"declared": 0, "propagated": 0, "from_f0": 0, "unknown": 0,
              "conflicting_speakers": conflicted}
    for r in records:
        spk = str(r.get(speaker_key) or "")
        own = normalize_gender(r.get(gender_key))
        if own:
            r["gender_resolved"], r["gender_source"] = own, "declared"
            counts["declared"] += 1
        elif spk and spk in resolved:
            r["gender_resolved"], r["gender_source"] = resolved[spk], "speaker"
            counts["propagated"] += 1
        else:
            inferred = gender_from_f0(float(r.get("mean_f0_hz") or 0.0))
            r["gender_resolved"] = inferred
            r["gender_source"] = "f0" if inferred else "unknown"
            counts["from_f0" if inferred else "unknown"] += 1
    return records, counts


def _quality(record: dict) -> float:
    """Rank clips within a speaker. Alignment first: it is the strongest signal."""
    return (
        float(record.get("align_score") or 0.0) * 2.0
        + float(record.get("dnsmos_ovr") or 0.0) / 5.0
        + float(record.get("bandwidth_hz") or 0.0) / 20000.0
    )


def cap_per_speaker(
    records: list[dict],
    *,
    speaker_key: str = "client_id",
    max_hours: float = MAX_SPEAKER_HOURS,
    narrator_hours: float = MAX_NARRATOR_HOURS,
) -> list[dict]:
    """Keep each speaker's best clips up to an hours budget, in input order.

    Without a cap the corpus is dominated by a few prolific contributors and the
    model collapses toward their voices -- the top 10 of 511 Common Voice
    speakers hold 45.7% of validated clips.

    Two things this gets right that a clip-count cap did not.

    **The budget is hours.** A count silently tracks clip length, so the same
    number meant a different amount of speech for every source.

    **A single-narrator source gets its own budget.** Capping one narrator
    cannot increase voice diversity, because there is no second voice in that
    source to make room for -- it can only delete audio. The clip cap took
    MBSpeech from 6.3 h to 0.66 h: 5.64 h of the cleanest male speech in the
    corpus, discarded against an acceptance criterion measured in male hours.
    Sources declare `single_narrator: True`; the exemption is not inferred from
    a speaker simply being large.
    """
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_speaker[str(r.get(speaker_key) or f"__anon_{i}")].append(i)

    keep: set[int] = set()
    for indices in by_speaker.values():
        budget = (
            narrator_hours
            if any(records[i].get("single_narrator") for i in indices)
            else max_hours
        )
        # Best first, so the budget buys the best available speech.
        ranked = sorted(indices, key=lambda i: _quality(records[i]), reverse=True)
        spent = 0.0
        for i in ranked:
            duration = float(records[i].get("duration_s") or 0.0) / 3600.0
            if spent + duration > budget and spent > 0.0:
                continue
            keep.add(i)
            spent += duration
    return [r for i, r in enumerate(records) if i in keep]


def _hours(rs: list[dict]) -> float:
    return sum(float(x.get("duration_s") or 0.0) for x in rs) / 3600.0


def has_known_speaker(record: dict, speaker_key: str = "client_id") -> bool:
    """Whether this clip's voice can be told apart from another's.

    A source that supplies no speaker column sets `speaker_known: False`
    explicitly. Absent that flag, a non-empty id is taken at face value.
    """
    if record.get("speaker_known") is False:
        return False
    return bool(str(record.get(speaker_key) or "").strip())


def speaker_disjoint_split(
    records: list[dict],
    *,
    speaker_key: str = "client_id",
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    min_speakers: int = MIN_SPEAKERS_PER_EVAL_SPLIT,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split by speaker, so no speaker appears in more than one split.

    Common Voice's own train/dev/test are not speaker-disjoint, and neither is a
    row-level random split. Evaluating a voice-cloning model on speakers it
    trained on measures memorisation.

    Two properties this has to hold that the first version did not.

    **Clips whose speaker is unknown go wholly to training.** FLEURS supplies no
    speaker column at all -- the fallback used to invent `__anon_{i}` per clip,
    so for that ~13 h block the "speaker-disjoint" split degenerated to a
    row-level random one and FLEURS' ~100 real speakers appeared on both sides.
    Sending the whole block to training cannot leak: a split it never enters
    cannot share a voice with it.

    **Evaluation splits are filled smallest-first, alternating gender.**
    Assigning largest-first to the emptiest split put the single most prolific
    contributor in test and the next in validation -- one voice each, and the
    two biggest speakers removed from training. Small speakers reach the same
    5% target while leaving the large ones where the model needs them, and
    alternating genders keeps a male reference available in every split.
    """
    identified = [r for r in records if has_known_speaker(r, speaker_key)]
    anonymous = [r for r in records if not has_known_speaker(r, speaker_key)]
    if anonymous:
        log.warning(
            "%d clips (%.1f h) have no speaker id; assigning all to train. "
            "They cannot be used for evaluation -- a split cannot be shown "
            "disjoint from a voice it cannot name.",
            len(anonymous), _hours(anonymous),
        )

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in identified:
        by_speaker[str(r[speaker_key])].append(r)

    # Targets are fractions of the whole corpus, so the anonymous block does not
    # shrink the evaluation splits -- it only limits which clips can fill them.
    total = _hours(records)
    targets = {"test": total * test_fraction, "validation": total * val_fraction}

    def gender_of(clips: list[dict]) -> str:
        # A speaker has one gender; take the first resolved one seen.
        for c in clips:
            if c.get("gender_resolved"):
                return str(c["gender_resolved"])
        return UNKNOWN

    # Shuffle first so equal-duration speakers tie-break deterministically but
    # not by dictionary order, then sort smallest-first.
    speakers = list(by_speaker.items())
    random.Random(seed).shuffle(speakers)
    speakers.sort(key=lambda kv: _hours(kv[1]))

    # Smallest-first queue per gender. Popping from the front takes the smallest
    # remaining speaker of that gender.
    pools: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for spk, clips in speakers:
        pools[gender_of(clips)].append((spk, clips))
    # Unknown gender last: a clip that cannot be attributed to a voice type is
    # the least useful thing to spend a small evaluation split on.
    gender_order = sorted(pools, key=lambda g: (g == UNKNOWN, g))

    out: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    taken: set[str] = set()

    # No evaluation split may claim more than a third of the speakers, however
    # short of its hours target it is. Without the cap a corpus with three
    # speakers gives test all three and leaves train empty -- the minimum is a
    # goal, not a licence to consume the corpus.
    budget = max(1, len(speakers) // 3)

    for split in ("test", "validation"):
        chosen: list[str] = []
        filled = 0.0
        by_gender = dict.fromkeys(gender_order, 0.0)
        while len(chosen) < budget and (filled < targets[split] or len(chosen) < min_speakers):
            # Take from whichever gender this split has least of, so neither
            # split ends up single-gender and unable to supply a male prompt.
            available = [g for g in gender_order if pools[g]]
            if not available:
                break
            gender = min(available, key=lambda g: (by_gender[g], gender_order.index(g)))
            spk, clips = pools[gender].pop(0)
            out[split].extend(clips)
            taken.add(spk)
            chosen.append(spk)
            filled += _hours(clips)
            by_gender[gender] += _hours(clips)
        if len(chosen) < min_speakers:
            log.warning(
                "%s has only %d speaker(s); %d were asked for. A split with one "
                "voice measures that voice, not the model.",
                split, len(chosen), min_speakers,
            )

    # Original record order, so the manifest is stable across runs.
    out["train"] = [
        r for r in records
        if not has_known_speaker(r, speaker_key) or str(r[speaker_key]) not in taken
    ]

    for name, rs in out.items():
        log.info("  %-10s %5d clips  %5.1f h  %3d speakers",
                 name, len(rs), _hours(rs),
                 len({str(r.get(speaker_key)) for r in rs if has_known_speaker(r, speaker_key)}))
    return out


def text_key(text: str) -> str:
    """Compare sentences by content, not by incidental spacing or case."""
    return " ".join(str(text or "").split()).casefold()


def reserve_eval_sentences(
    records: list[dict],
    *,
    n_sentences: int = EVAL_HELD_OUT_SENTENCES,
    max_fraction: float = 0.2,
    text_field: str = "text",
    seed: int = 42,
) -> set[str]:
    """Choose sentences to withhold from training, rarest first.

    A speaker-disjoint split is not a text-disjoint one. Common Voice mn has
    28,858 usable clips over **6,062 distinct sentences** -- 4.76x repetition --
    so a sentence read by an evaluation speaker was almost certainly read by a
    training speaker too. Measured under the previous split, **99.6%** of test
    clips (1,705 of 1,712) had their text in train, which makes CER over them a
    measure of recall rather than of intelligibility.

    Rarest first is what makes this affordable. Withholding a sentence costs
    training every clip that carries it, and the sentence distribution is
    heavy-tailed: on the measured shape, 400 sentences taken from the tail cost
    about 400 clips (~1.5% of training), where 400 taken at random would cost
    nearly five times that.
    """
    by_text: dict[str, int] = defaultdict(int)
    for r in records:
        by_text[text_key(r.get(text_field, ""))] += 1
    by_text.pop("", None)

    keys = list(by_text)
    random.Random(seed).shuffle(keys)          # deterministic tie-break
    keys.sort(key=lambda k: by_text[k])        # then rarest first

    # Never withhold most of a small corpus. The absolute count is sized for
    # Common Voice's 6,062 sentences; on anything smaller the fraction binds.
    limit = min(n_sentences, int(len(keys) * max_fraction))
    if limit < n_sentences:
        log.info("Corpus has %d distinct sentences; withholding %d, not %d",
                 len(keys), limit, n_sentences)
    return set(keys[:limit])


def withhold_eval_sentences(
    splits: dict[str, list[dict]],
    reserved: set[str],
    *,
    speaker_key: str = "client_id",
    text_field: str = "text",
) -> dict[str, list[dict]]:
    """Remove every reserved sentence from training, so it is genuinely unseen.

    The reference prompt and the target text are two different objects with two
    different requirements, and conflating them was the error in an earlier
    attempt at this:

      * the **prompt** must come from an unseen *speaker* -- that is the
        zero-shot condition, and its text is handed to the model anyway;
      * the **target text** must be unseen *text* -- that is the intelligibility
        condition, and which voice once read it does not matter.

    Requiring both of the same clip intersects a 10% speaker holdout with a 10%
    text holdout: on the measured corpus shape that left **47 clips** in test,
    too few to supply even one reference prompt per gender. Holding the two
    apart keeps the speaker-disjoint test split whole for prompts and the
    ground-truth topline, and yields a full unseen sentence list for CER.

    A reserved sentence read by a *training* speaker is usable nowhere: in train
    it would void the text holdout, and in an evaluation split it would void
    speaker-disjointness. Those clips move to a `withheld` split rather than
    being deleted -- the audio still exists, and a manifest that silently loses
    rows is worse than one that says why it kept them out.
    """
    out = dict(splits)
    train = out.get("train", [])
    withheld = [r for r in train if text_key(r.get(text_field, "")) in reserved]
    if withheld:
        log.info(
            "Withheld %d sentences from training: %d clips (%.2f h, %.1f%%) moved "
            "to the 'withheld' split so the evaluation text is genuinely unseen",
            len(reserved), len(withheld), _hours(withheld),
            100.0 * len(withheld) / max(1, len(train)),
        )
    out["train"] = [r for r in train if text_key(r.get(text_field, "")) not in reserved]
    out["withheld"] = out.get("withheld", []) + withheld
    return out
