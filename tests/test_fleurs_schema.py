"""What FLEURS' schema forces the loader to do.

Verified against `load_dataset_builder("google/fleurs", "mn_mn").info.features`:

    gender   ClassLabel(names=['male', 'female', 'other'])
    id       Value('int32')          # the *sentence* index, not a speaker

There is no speaker column in the schema at all. These tests pin both
consequences without needing the 13 h download.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.datasets.fleurs import _Decoded  # noqa: E402
from pipeline.speakers import FEMALE, MALE, UNKNOWN, normalize_gender  # noqa: E402


class _ClassLabel:
    """The one method of datasets.ClassLabel this loader uses."""

    def __init__(self, names):
        self.names = names

    def int2str(self, i):
        return self.names[i]


class _FakeSplit:
    def __init__(self, rows, gender_feature):
        self._rows = rows
        self.features = {"gender": gender_feature}

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return dict(self._rows[idx])


LABEL = _ClassLabel(["male", "female", "other"])


def test_raw_classlabel_values_resolve_to_nothing():
    """The bug, stated as a test.

    `datasets` returns the int for a ClassLabel column. normalize_gender
    stringifies it to "0", which matches no alias -- so before the decode every
    FLEURS gender label was discarded, the same failure the module docstring
    records for Common Voice's `male_masculine` one file over.
    """
    assert normalize_gender(0) == UNKNOWN
    assert normalize_gender(1) == UNKNOWN


def test_the_wrapper_maps_the_label_index_to_its_name():
    split = _Decoded(_FakeSplit([{"gender": 0}, {"gender": 1}], LABEL))
    assert normalize_gender(split[0]["gender"]) == MALE
    assert normalize_gender(split[1]["gender"]) == FEMALE


def test_other_still_resolves_to_unknown():
    """'other' describes identity, not a vocal tract; guessing would be wrong."""
    split = _Decoded(_FakeSplit([{"gender": 2}], LABEL))
    assert normalize_gender(split[0]["gender"]) == UNKNOWN


def test_a_release_that_ships_plain_strings_is_left_alone():
    """Guarding on the method rather than the type keeps this working if the
    schema changes under us."""
    split = _Decoded(_FakeSplit([{"gender": "female"}], gender_feature=None))
    assert normalize_gender(split[0]["gender"]) == FEMALE


def test_every_clip_declares_its_speaker_unknown():
    """FLEURS has no speaker column, so the split must not invent one."""
    split = _Decoded(_FakeSplit([{"gender": 0}, {"gender": 1}], LABEL))
    assert all(split[i]["speaker_known"] is False for i in range(len(split)))


def test_other_columns_survive_the_wrapper():
    rows = [{"gender": 0, "id": 42, "raw_transcription": "Сайн байна уу"}]
    item = _Decoded(_FakeSplit(rows, LABEL))[0]
    assert item["id"] == 42
    assert item["raw_transcription"] == "Сайн байна уу"


def test_the_wrapper_does_not_mutate_the_underlying_split():
    inner = _FakeSplit([{"gender": 0}], LABEL)
    _Decoded(inner)[0]
    assert inner[0]["gender"] == 0
