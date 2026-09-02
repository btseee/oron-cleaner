"""Reading Common Voice's own TSVs.

The largest source in the corpus, and the one whose selection logic already
shipped a defect: an earlier version chose the split file with
`name.endswith("validated.tsv")`, which is also true of `invalidated.tsv` --
the voter-*rejected* clips -- and matched it first, yielding zero usable rows.

Everything here is pure file reading, so none of it needs the network or the
API key.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.datasets.common_voice import (  # noqa: E402
    _SPLITS,
    _load_clip_durations,
    _load_split,
)


def _lang_dir(tmp_path: Path, rows: list[dict], durations: dict[str, int] | None = None,
              split: str = "validated") -> Path:
    """A minimal Common Voice language directory."""
    lang = tmp_path / "mn"
    (lang / "clips").mkdir(parents=True, exist_ok=True)
    header = ["client_id", "path", "sentence", "up_votes", "down_votes",
              "age", "gender", "accents", "variant", "segment", "locale"]
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join(str(r.get(k, "")) for k in header))
        (lang / "clips" / r["path"]).write_bytes(b"")
    (lang / f"{split}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if durations is not None:
        d = ["clip\tduration[ms]"] + [f"{k}\t{v}" for k, v in durations.items()]
        (lang / "clip_durations.tsv").write_text("\n".join(d) + "\n", encoding="utf-8")
    return lang


def _row(path: str, **kw) -> dict:
    base = {"client_id": "spk1", "path": path, "sentence": "Сайн байна уу",
            "up_votes": 2, "down_votes": 0, "gender": "male_masculine",
            "locale": "mn"}
    return {**base, **kw}


# ── the split that is read ────────────────────────────────────────────────────

def test_only_validated_is_read():
    """`other` is not human-confirmed and `invalidated` was actively rejected by
    voters. Blending them is the "volume over purity" choice this corpus exists
    not to make."""
    assert _SPLITS == ["validated"]


def test_invalidated_is_not_mistaken_for_validated(tmp_path):
    """The shipped defect: "invalidated.tsv".endswith("validated.tsv") is True.

    Both files exist in a real download, so a suffix match found the rejected
    clips first.
    """
    lang = _lang_dir(tmp_path, [_row("good.mp3")])
    _lang_dir(tmp_path, [_row("bad.mp3")], split="invalidated")
    split = _load_split(lang, "validated")
    assert split is not None
    assert [r["path"] for r in split] == ["good.mp3"]


def test_a_missing_split_returns_none(tmp_path):
    lang = _lang_dir(tmp_path, [_row("a.mp3")])
    assert _load_split(lang, "test") is None


# ── the down-vote gate ────────────────────────────────────────────────────────

def test_a_down_voted_clip_is_dropped(tmp_path):
    """A listener judged it wrong, which is stronger evidence than anything the
    audio filters can infer. 13.2% of validated clips carry one."""
    lang = _lang_dir(tmp_path, [
        _row("keep.mp3", down_votes=0),
        _row("drop.mp3", down_votes=1),
        _row("drop2.mp3", down_votes=3),
    ])
    assert [r["path"] for r in _load_split(lang, "validated")] == ["keep.mp3"]


def test_a_blank_down_vote_field_is_not_a_down_vote(tmp_path):
    """Common Voice leaves it empty rather than zero in places, and int("")
    raises -- so an unguarded read would drop every such clip."""
    lang = _lang_dir(tmp_path, [_row("a.mp3", down_votes="")])
    assert len(_load_split(lang, "validated")) == 1


# ── clips that are listed but absent ──────────────────────────────────────────

def test_a_row_without_its_audio_is_skipped(tmp_path):
    """The TSV lists more than the archive always contains."""
    lang = _lang_dir(tmp_path, [_row("present.mp3")])
    with open(lang / "validated.tsv", "a", encoding="utf-8") as f:
        f.write("spk1\tmissing.mp3\tтекст\t1\t0\t\t\t\t\t\tmn\n")
    assert [r["path"] for r in _load_split(lang, "validated")] == ["present.mp3"]


# ── durations ─────────────────────────────────────────────────────────────────

def test_durations_come_from_the_shipped_table(tmp_path):
    """Free, and it lets a split be sized before any audio is decoded."""
    lang = _lang_dir(tmp_path, [_row("a.mp3"), _row("b.mp3")],
                     durations={"a.mp3": 4500, "b.mp3": 6000})
    rows = list(_load_split(lang, "validated"))
    assert [r["duration_tsv"] for r in rows] == [4.5, 6.0]


def test_a_missing_duration_table_is_not_fatal(tmp_path):
    lang = _lang_dir(tmp_path, [_row("a.mp3")])
    assert _load_clip_durations(lang) == {}
    assert _load_split(lang, "validated")[0]["duration_tsv"] == 0.0


def test_a_malformed_duration_row_is_skipped_not_fatal(tmp_path):
    lang = _lang_dir(tmp_path, [_row("a.mp3")], durations={"a.mp3": 4000})
    with open(lang / "clip_durations.tsv", "a", encoding="utf-8") as f:
        f.write("b.mp3\tnot-a-number\n")
    durations = _load_clip_durations(lang)
    assert durations == {"a.mp3": 4.0}


# ── the fields the rest of the pipeline needs ────────────────────────────────

def test_the_speaker_and_gender_columns_survive(tmp_path):
    """Both are what the split and the voice selection are built on -- their
    absence is what F0b was."""
    lang = _lang_dir(tmp_path, [_row("a.mp3", client_id="abc", gender="female_feminine")])
    row = _load_split(lang, "validated")[0]
    assert row["client_id"] == "abc"
    assert row["gender"] == "female_feminine"
    assert row["audio"].endswith("a.mp3")


def test_the_sentence_is_carried_verbatim(tmp_path):
    """Normalisation happens later, once, so this must not touch it."""
    lang = _lang_dir(tmp_path, [_row("a.mp3", sentence="1990 онд 25 хүн")])
    assert _load_split(lang, "validated")[0]["sentence"] == "1990 онд 25 хүн"


def test_an_empty_split_is_none_rather_than_an_empty_wrapper(tmp_path):
    """So the caller reports "no clips" rather than iterating nothing."""
    lang = _lang_dir(tmp_path, [_row("a.mp3", down_votes=5)])
    assert _load_split(lang, "validated") is None
