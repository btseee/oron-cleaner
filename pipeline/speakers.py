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

from .constants import MAX_CLIPS_PER_SPEAKER

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
    max_clips: int = MAX_CLIPS_PER_SPEAKER,
) -> list[dict]:
    """Keep each speaker's best `max_clips` clips, preserving input order.

    Without this the corpus is dominated by a few prolific contributors and the
    model collapses toward their voices.
    """
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_speaker[str(r.get(speaker_key) or f"__anon_{i}")].append(i)

    keep: set[int] = set()
    for indices in by_speaker.values():
        if len(indices) <= max_clips:
            keep.update(indices)
        else:
            ranked = sorted(indices, key=lambda i: _quality(records[i]), reverse=True)
            keep.update(ranked[:max_clips])
    return [r for i, r in enumerate(records) if i in keep]


def speaker_disjoint_split(
    records: list[dict],
    *,
    speaker_key: str = "client_id",
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split by speaker, so no speaker appears in more than one split.

    Common Voice's own train/dev/test are not speaker-disjoint, and neither is a
    row-level random split. Evaluating a voice-cloning model on speakers it
    trained on measures memorisation.

    Speakers are assigned largest-first to whichever split is furthest below its
    target duration, which keeps small splits from being dominated by one
    prolific speaker.
    """
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for i, r in enumerate(records):
        by_speaker[str(r.get(speaker_key) or f"__anon_{i}")].append(r)

    def hours(rs: list[dict]) -> float:
        return sum(float(x.get("duration_s") or 0.0) for x in rs) / 3600.0

    total = hours(records)
    targets = {
        "validation": total * val_fraction,
        "test": total * test_fraction,
        "train": total * (1.0 - val_fraction - test_fraction),
    }
    out: dict[str, list[dict]] = {k: [] for k in targets}
    filled = dict.fromkeys(targets, 0.0)

    # Shuffle first so equal-duration speakers tie-break deterministically but
    # not by dictionary order, then sort largest-first.
    speakers = list(by_speaker.items())
    random.Random(seed).shuffle(speakers)
    speakers.sort(key=lambda kv: hours(kv[1]), reverse=True)

    for _spk, clips in speakers:
        deficit = max(targets, key=lambda k: targets[k] - filled[k])
        out[deficit].extend(clips)
        filled[deficit] += hours(clips)

    for name, rs in out.items():
        log.info("  %-10s %5d clips  %5.1f h  %3d speakers",
                 name, len(rs), hours(rs),
                 len({str(r.get(speaker_key)) for r in rs}))
    return out
