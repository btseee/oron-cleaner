"""Speaker and gender resolution.

These functions decide which voices the model can produce, so the failure modes
matter more than the happy paths: a mislabelled speaker, a speaker leaking
across the train/test boundary, or one prolific contributor dominating.
"""

import pytest

from pipeline.speakers import (
    FEMALE,
    MALE,
    UNKNOWN,
    cap_per_speaker,
    gender_from_f0,
    normalize_gender,
    propagate_gender,
    speaker_disjoint_split,
)


def clip(spk, gender="", f0=0.0, dur=5.0, align=0.9, **kw):
    return {"client_id": spk, "gender": gender, "mean_f0_hz": f0,
            "duration_s": dur, "align_score": align, **kw}


# ── gender vocabulary ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["male_masculine", "Male", "MALE", "m", "man"])
def test_male_aliases(raw):
    assert normalize_gender(raw) == MALE


@pytest.mark.parametrize("raw", ["female_feminine", "Female", "f", "woman"])
def test_female_aliases(raw):
    assert normalize_gender(raw) == FEMALE


def test_common_voice_v17_vocabulary_is_recognised():
    """The exact strings Common Voice emits today.

    These previously mapped to nothing, so --gender-column silently produced no
    gender for every row of the dataset it was documented against.
    """
    assert normalize_gender("male_masculine") == MALE
    assert normalize_gender("female_feminine") == FEMALE


@pytest.mark.parametrize(
    "raw", ["", None, "other", "non_binary", "intersex", "transgender",
            "do_not_wish_to_say", "unknown-value"],
)
def test_unresolvable_gender_is_not_guessed(raw):
    assert normalize_gender(raw) == UNKNOWN


# ── F0 inference ──────────────────────────────────────────────────────────────

def test_f0_bands_match_the_calibration():
    """Calibrated on 80 self-declared clips: male max 150.3, female min 172.7."""
    assert gender_from_f0(115.2) == MALE      # measured male median
    assert gender_from_f0(241.4) == FEMALE    # measured female median
    assert gender_from_f0(150.0) == MALE      # measured male maximum
    assert gender_from_f0(173.0) == FEMALE    # measured female minimum


def test_dead_band_refuses_to_guess():
    """Between the measured distributions there is a gap; do not guess in it."""
    assert gender_from_f0(160.0) == UNKNOWN
    assert gender_from_f0(165.0) == UNKNOWN


def test_missing_f0_is_unknown():
    assert gender_from_f0(0.0) == UNKNOWN
    assert gender_from_f0(-1.0) == UNKNOWN


# ── propagation ───────────────────────────────────────────────────────────────

def test_gender_propagates_across_a_speakers_clips():
    """39.7% of Common Voice clips are unlabelled, across only 520 speakers."""
    records = [clip("s1", "male_masculine"), clip("s1"), clip("s1")]
    out, counts = propagate_gender(records)
    assert [r["gender_resolved"] for r in out] == [MALE, MALE, MALE]
    assert counts["declared"] == 1 and counts["propagated"] == 2


def test_declared_label_beats_f0_inference():
    """A speaker's own declaration outranks acoustics."""
    out, _ = propagate_gender([clip("s1", "female_feminine", f0=120.0)])
    assert out[0]["gender_resolved"] == FEMALE
    assert out[0]["gender_source"] == "declared"


def test_f0_fills_in_only_when_nothing_is_declared():
    out, counts = propagate_gender([clip("s9", "", f0=115.0)])
    assert out[0]["gender_resolved"] == MALE
    assert out[0]["gender_source"] == "f0"
    assert counts["from_f0"] == 1


def test_conflicting_labels_leave_the_speaker_unknown():
    """Disagreement usually means a shared speaker id, not a mislabel."""
    records = [clip("s1", "male_masculine"), clip("s1", "female_feminine"), clip("s1")]
    out, counts = propagate_gender(records)
    assert counts["conflicting_speakers"] == 1
    assert out[2]["gender_resolved"] == UNKNOWN


# ── per-speaker cap ───────────────────────────────────────────────────────────

def test_prolific_speaker_is_capped():
    """The largest Common Voice contributor has 1,956 of 33,258 clips."""
    records = [clip("loud", align=0.9) for _ in range(50)] + [clip("quiet")]
    out = cap_per_speaker(records, max_clips=10)
    assert sum(1 for r in out if r["client_id"] == "loud") == 10
    assert sum(1 for r in out if r["client_id"] == "quiet") == 1


def test_cap_keeps_the_best_clips():
    records = [clip("s", align=0.5), clip("s", align=0.95), clip("s", align=0.7)]
    out = cap_per_speaker(records, max_clips=1)
    assert out[0]["align_score"] == 0.95


def test_cap_preserves_input_order():
    records = [clip("s", align=a) for a in (0.95, 0.5, 0.9)]
    out = cap_per_speaker(records, max_clips=2)
    assert [r["align_score"] for r in out] == [0.95, 0.9]


# ── speaker-disjoint splits ───────────────────────────────────────────────────

def test_splits_share_no_speaker():
    """Common Voice's own splits are not speaker-disjoint, nor is a row-level
    random split. Evaluating a cloning model on speakers it trained on measures
    memorisation."""
    records = [clip(f"s{i}", dur=10.0) for i in range(40) for _ in range(5)]
    splits = speaker_disjoint_split(records)
    seen = [{r["client_id"] for r in rs} for rs in splits.values()]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert not (seen[i] & seen[j])


def test_every_clip_lands_in_exactly_one_split():
    records = [clip(f"s{i}", dur=10.0) for i in range(30) for _ in range(4)]
    splits = speaker_disjoint_split(records)
    assert sum(len(v) for v in splits.values()) == len(records)


def test_train_receives_the_bulk_of_the_audio():
    records = [clip(f"s{i}", dur=10.0) for i in range(60) for _ in range(5)]
    splits = speaker_disjoint_split(records)
    assert len(splits["train"]) > len(splits["validation"]) + len(splits["test"])


def test_small_splits_are_not_starved():
    """Every split must receive clips even when speakers are few and large.

    Ranking by absolute deficit sends every speaker to train until train alone
    is satisfied, so with 12 equal speakers validation got one and test got
    none. Fractional deficit fills the small splits first.
    """
    records = [clip(f"s{i}", dur=120.0) for i in range(12) for _ in range(30)]
    splits = speaker_disjoint_split(records)
    for name, rs in splits.items():
        assert rs, f"{name} is empty"


def test_every_split_is_populated_at_the_minimum_speaker_count():
    """Three speakers is the fewest that can fill three splits."""
    records = [clip(f"s{i}", dur=60.0) for i in range(3) for _ in range(5)]
    splits = speaker_disjoint_split(records)
    assert all(len(rs) > 0 for rs in splits.values())


def test_split_is_deterministic():
    records = [clip(f"s{i}", dur=7.0) for i in range(25) for _ in range(3)]
    a = speaker_disjoint_split(records, seed=1)
    b = speaker_disjoint_split(records, seed=1)
    assert {k: [r["client_id"] for r in v] for k, v in a.items()} == \
           {k: [r["client_id"] for r in v] for k, v in b.items()}
