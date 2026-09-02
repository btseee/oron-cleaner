"""Speaker and gender resolution.

These functions decide which voices the model can produce, so the failure modes
matter more than the happy paths: a mislabelled speaker, a speaker leaking
across the train/test boundary, or one prolific contributor dominating.
"""

import pytest

from pipeline.speakers import (
    FEMALE,
    MALE,
    MAX_NARRATOR_HOURS,
    MAX_SPEAKER_HOURS,
    UNKNOWN,
    cap_per_speaker,
    gender_from_f0,
    normalize_gender,
    propagate_gender,
    reserve_eval_sentences,
    speaker_disjoint_split,
    text_key,
    withhold_eval_sentences,
)


def clip(spk, gender="", f0=0.0, dur=5.0, align=0.9, text="", **kw):
    return {"client_id": spk, "gender": gender, "mean_f0_hz": f0,
            "duration_s": dur, "align_score": align, "text": text, **kw}


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
    records = [clip("loud", align=0.9, dur=360.0) for _ in range(50)] + [clip("quiet")]
    out = cap_per_speaker(records, max_hours=1.0)
    assert sum(1 for r in out if r["client_id"] == "loud") == 10
    assert sum(1 for r in out if r["client_id"] == "quiet") == 1


def test_cap_keeps_the_best_clips():
    records = [clip("s", align=a, dur=3600.0) for a in (0.5, 0.95, 0.7)]
    out = cap_per_speaker(records, max_hours=1.0)
    assert [r["align_score"] for r in out] == [0.95]


def test_cap_preserves_input_order():
    records = [clip("s", align=a, dur=1800.0) for a in (0.95, 0.5, 0.9)]
    out = cap_per_speaker(records, max_hours=1.0)
    assert [r["align_score"] for r in out] == [0.95, 0.9]


def test_the_budget_is_hours_not_clips():
    """A clip count silently tracks clip length: 400 clips is 0.56 h of Common
    Voice at its 5.07 s mean and something else entirely for any other source."""
    short = [clip("a", dur=2.0) for _ in range(3000)]     # 1.67 h
    long = [clip("b", dur=20.0) for _ in range(300)]       # 1.67 h, 10x fewer clips
    out = cap_per_speaker(short + long, max_hours=1.0)
    a = sum(r["duration_s"] for r in out if r["client_id"] == "a") / 3600
    b = sum(r["duration_s"] for r in out if r["client_id"] == "b") / 3600
    assert a == pytest.approx(1.0, abs=0.01)
    assert b == pytest.approx(1.0, abs=0.01)


def test_a_single_narrator_source_gets_its_own_budget():
    """M14: at 400 clips the cap kept 0.66 h of MBSpeech and deleted 5.64 h --
    the cleanest male audio in a corpus gated on male hours."""
    mb = [clip("narrator", dur=5.9, single_narrator=True) for _ in range(3846)]
    kept = sum(r["duration_s"] for r in cap_per_speaker(mb)) / 3600
    assert kept == pytest.approx(6.3, abs=0.1)


def test_the_narrator_exemption_is_declared_not_inferred():
    """A merely prolific crowd contributor must not acquire the exemption by
    being large."""
    loud = [clip("loud", dur=5.9) for _ in range(3846)]
    kept = sum(r["duration_s"] for r in cap_per_speaker(loud)) / 3600
    assert kept == pytest.approx(MAX_SPEAKER_HOURS, abs=0.01)


def test_even_an_exempt_narrator_is_bounded():
    mb = [clip("narrator", dur=60.0, single_narrator=True) for _ in range(2000)]
    kept = sum(r["duration_s"] for r in cap_per_speaker(mb)) / 3600
    assert kept == pytest.approx(MAX_NARRATOR_HOURS, abs=0.05)


def test_a_speaker_under_budget_is_untouched():
    records = [clip("s", dur=5.0) for _ in range(10)]
    assert len(cap_per_speaker(records)) == 10


def test_one_clip_longer_than_the_budget_is_still_kept():
    """Dropping it would silently delete a speaker entirely."""
    assert len(cap_per_speaker([clip("s", dur=7200.0)], max_hours=1.0)) == 1


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


# ── split composition ─────────────────────────────────────────────────────────

def test_clips_without_a_speaker_id_all_go_to_train():
    """FLEURS supplies no speaker column at all.

    The `__anon_{i}` fallback gave every such clip its own pseudo-speaker, so
    an anonymous clip could be assigned to test while other recordings of the
    same unnameable voice sat in train. The speaker distribution is heavy-tailed
    on purpose -- that is what the real corpus looks like, and it is the shape
    under which the leak appears at all.
    """
    known = [clip(f"s{s}", gender="male" if s % 2 else "female", dur=5.0)
             for s in range(40)
             for _ in range(max(1, int(120 / (s + 1) ** 0.85)))]
    anon = [clip("", dur=9.8, speaker_known=False) for _ in range(500)]
    splits = speaker_disjoint_split(propagate_gender(known + anon)[0])
    assert sum(1 for r in splits["train"] if r.get("speaker_known") is False) == len(anon)
    for name in ("validation", "test"):
        assert not any(r.get("speaker_known") is False for r in splits[name])


def test_an_empty_client_id_counts_as_unknown():
    records = [clip("", dur=60.0) for _ in range(20)] + \
              [clip(f"s{i}", dur=60.0) for i in range(9) for _ in range(5)]
    splits = speaker_disjoint_split(records)
    assert all(r["client_id"] for r in splits["test"])
    assert all(r["client_id"] for r in splits["validation"])


def test_evaluation_splits_carry_several_speakers():
    """F3: the largest-first assignment put one speaker in test and one in
    validation, so the whole evaluation rested on two voices."""
    records = [clip(f"s{i}", gender="male" if i % 2 else "female", dur=30.0)
               for i in range(30) for _ in range(20)]
    splits = speaker_disjoint_split(propagate_gender(records)[0])
    for name in ("validation", "test"):
        assert len({r["client_id"] for r in splits[name]}) >= 3, name


def test_evaluation_splits_carry_both_genders():
    """A split with no male clip cannot supply a male reference prompt, and the
    male voice is the corpus's binding constraint."""
    records = [clip(f"s{i}", gender="male" if i % 2 else "female", dur=30.0)
               for i in range(30) for _ in range(20)]
    splits = speaker_disjoint_split(propagate_gender(records)[0])
    for name in ("validation", "test"):
        assert {r["gender_resolved"] for r in splits[name]} == {MALE, FEMALE}, name


def test_the_largest_speakers_stay_in_training():
    """Removing the most prolific contributors costs training far more than it
    buys a 5% split, which smaller speakers can fill just as well."""
    records = [clip(f"small{i}", gender="male" if i % 2 else "female", dur=5.0)
               for i in range(40) for _ in range(5)]
    records += [clip("whale", gender="female", dur=5.0) for _ in range(2000)]
    splits = speaker_disjoint_split(propagate_gender(records)[0])
    assert all(r["client_id"] == "whale" for r in splits["train"] if r["client_id"] == "whale")
    assert "whale" not in {r["client_id"] for r in splits["test"]}
    assert "whale" not in {r["client_id"] for r in splits["validation"]}


def test_no_speaker_appears_in_two_splits():
    records = [clip(f"s{i}", gender="male" if i % 2 else "female", dur=20.0)
               for i in range(30) for _ in range(8)]
    splits = speaker_disjoint_split(propagate_gender(records)[0])
    seen = [{r["client_id"] for r in rs} for rs in splits.values()]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])


def test_no_clip_is_lost_or_duplicated():
    records = [clip(f"s{i}", dur=9.0, tag=f"{i}_{j}") for i in range(20) for j in range(6)]
    splits = speaker_disjoint_split(records)
    tags = [r["tag"] for rs in splits.values() for r in rs]
    assert sorted(tags) == sorted(r["tag"] for r in records)


# ── text holdout ──────────────────────────────────────────────────────────────

def test_reserved_sentences_are_the_rarest_ones():
    """Withholding a sentence costs training every clip carrying it, and the
    sentence distribution is heavy-tailed -- so take from the tail."""
    records = ([clip("a", text="common") for _ in range(100)]
               + [clip("a", text=f"rare {i}") for i in range(10)])
    reserved = reserve_eval_sentences(records, n_sentences=5, max_fraction=1.0)
    assert "common" not in reserved
    assert len(reserved) == 5


def test_reserving_never_eats_a_small_corpus():
    """The absolute count is sized for Common Voice's 6,062 sentences."""
    records = [clip("a", text=f"s{i}") for i in range(10)]
    assert len(reserve_eval_sentences(records, n_sentences=400)) == 2


def test_reserved_sentences_leave_training_entirely():
    """F2: 99.6% of test clips had their text in train, so CER measured recall."""
    records = [clip(f"s{i}", text=f"sentence {j}", dur=20.0)
               for i in range(20) for j in range(30)]
    reserved = reserve_eval_sentences(records, n_sentences=5, max_fraction=1.0)
    splits = withhold_eval_sentences(speaker_disjoint_split(records), reserved)
    train_texts = {text_key(r["text"]) for r in splits["train"]}
    assert not (reserved & train_texts)


def test_withheld_clips_are_moved_not_deleted():
    """A manifest that silently loses rows is worse than one that says why."""
    records = [clip(f"s{i}", text=f"sentence {j}", dur=20.0)
               for i in range(20) for j in range(30)]
    reserved = reserve_eval_sentences(records, n_sentences=5, max_fraction=1.0)
    splits = withhold_eval_sentences(speaker_disjoint_split(records), reserved)
    assert sum(len(v) for v in splits.values()) == len(records)
    assert splits["withheld"]


def test_the_text_holdout_does_not_shrink_the_audio_test_split():
    """The error worth not repeating.

    Requiring one clip to be both speaker-unseen and text-unseen intersects two
    10% holdouts: on the measured corpus shape that left 47 clips in test, too
    few for even one reference prompt per gender. The prompt needs an unseen
    speaker; the target text needs unseen text; they are different objects.
    """
    records = [clip(f"s{i}", gender="male" if i % 2 else "female",
                    text=f"sentence {j}", dur=20.0)
               for i in range(30) for j in range(40)]
    records = propagate_gender(records)[0]
    before = speaker_disjoint_split(records)
    after = withhold_eval_sentences(before, reserve_eval_sentences(records,
                                                                  n_sentences=20,
                                                                  max_fraction=1.0))
    assert len(after["test"]) == len(before["test"])


def test_text_key_ignores_spacing_and_case():
    assert text_key("  Сайн   байна\tуу ") == text_key("сайн байна уу")


def test_training_keeps_the_majority_of_a_small_corpus():
    """Found by running the pipeline end to end, not by reasoning about it.

    At a third of the speakers *each*, a 12-speaker corpus gave test 4 and
    validation 4 and left training 4 -- 39% of the clips. The cap is a third
    across both evaluation splits together. Invisible at Common Voice scale,
    where the hours target binds hundreds of speakers before the cap does.
    """
    records = propagate_gender(
        [clip(f"s{s}", gender="male" if s % 2 else "female", dur=6.0, text=f"t{s}{c}")
         for s in range(12) for c in range(20)]
    )[0]
    splits = speaker_disjoint_split(records)
    total = sum(len(v) for v in splits.values())
    assert len(splits["train"]) / total > 0.6


def test_evaluation_never_takes_more_than_a_third_of_the_speakers():
    for n_speakers in (6, 12, 30, 120):
        records = propagate_gender(
            [clip(f"s{s}", gender="male" if s % 2 else "female", dur=6.0, text=f"t{s}{c}")
             for s in range(n_speakers) for c in range(10)]
        )[0]
        splits = speaker_disjoint_split(records)
        held = len({r["client_id"] for r in splits["validation"]}) + \
               len({r["client_id"] for r in splits["test"]})
        assert held <= max(2, n_speakers // 3), f"{n_speakers} speakers: {held} held out"


def test_train_is_never_empty_however_few_speakers():
    for n_speakers in (1, 2, 3, 5):
        records = [clip(f"s{s}", dur=60.0) for s in range(n_speakers) for _ in range(5)]
        assert speaker_disjoint_split(records)["train"], f"{n_speakers} speakers"
