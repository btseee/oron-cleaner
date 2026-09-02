"""What each loader attaches that its source does not carry.

Every one of these wrappers exists because a source is missing a field the
split or the cap depends on, and every one of those gaps was silent before it
was found: FLEURS' ClassLabel gender resolved to nothing, MBSpeech's absent
speaker id made one narrator look like 3,846 speakers.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.datasets.mbspeech import SPEAKER_ID, _Wrapped  # noqa: E402
from pipeline.speakers import MALE, has_known_speaker, normalize_gender  # noqa: E402


class _FakeSplit:
    def __init__(self, rows):
        self._rows = rows
        self.features = {}

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return dict(self._rows[idx])


# ── MBSpeech ──────────────────────────────────────────────────────────────────

def test_mbspeech_supplies_a_constant_speaker_id():
    """One narrator with no speaker column. Without an id every clip becomes
    its own speaker, so both the per-speaker cap and the speaker-disjoint split
    silently do nothing for the 6.3 h block."""
    split = _Wrapped(_FakeSplit([{"sentence_orig": "a"}, {"sentence_orig": "b"}]))
    assert {split[i]["client_id"] for i in range(len(split))} == {SPEAKER_ID}
    assert all(has_known_speaker(split[i]) for i in range(len(split)))


def test_mbspeech_declares_its_gender():
    """Measured: median F0 140.7 Hz across 10 clips, against a calibration of
    115.1 male and 239.3 female."""
    split = _Wrapped(_FakeSplit([{"sentence_orig": "a"}]))
    assert normalize_gender(split[0]["gender"]) == MALE


def test_mbspeech_declares_itself_a_single_narrator_source():
    """Which exempts it from the per-speaker cap. At 400 clips the cap kept
    0.66 h of 6.3 h -- the cleanest male audio in a corpus gated on male
    hours."""
    split = _Wrapped(_FakeSplit([{"sentence_orig": "a"}]))
    assert split[0]["single_narrator"] is True


def test_mbspeech_supplies_a_stable_clip_path():
    """`_clip_id` derives resume identity from `path`; without one it falls back
    to the enumeration index, and a restart re-processes everything."""
    split = _Wrapped(_FakeSplit([{"sentence_orig": "a"}, {"sentence_orig": "b"}]))
    paths = [split[i]["path"] for i in range(len(split))]
    assert len(set(paths)) == 2
    assert all(p for p in paths)


def test_mbspeech_keeps_an_existing_path():
    """A source that does have one must not be overwritten."""
    split = _Wrapped(_FakeSplit([{"sentence_orig": "a", "path": "real.wav"}]))
    assert split[0]["path"] == "real.wav"


def test_mbspeech_does_not_mutate_the_underlying_split():
    inner = _FakeSplit([{"sentence_orig": "a"}])
    _Wrapped(inner)[0]
    assert "client_id" not in inner[0]


# ── WorldSpeech ───────────────────────────────────────────────────────────────

def test_worldspeech_needs_a_double_opt_in():
    """CC-BY-NC-4.0: including it makes the trained model non-commercial, so
    naming it is not enough."""
    from pipeline.datasets.worldspeech import process_worldspeech

    with pytest.raises(SystemExit) as exc:
        process_worldspeech(None, None, allow_non_commercial=False)
    assert "non-commercial" in str(exc.value).lower()
