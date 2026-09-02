"""On-disk corpus writing and the F5-TTS output contract.

`prepare_csv_wavs.py` validates its input strictly -- exact header, "|"
delimiter, absolute paths -- and raises on anything else, so these are the tests
that catch a corpus that will not load.
"""

import csv

import numpy as np
import pytest

from pipeline.constants import OUTPUT_SAMPLE_RATE
from pipeline.corpus import (
    CorpusWriter,
    read_manifest,
    summarise,
    write_f5_metadata,
)

sf = pytest.importorskip("soundfile")


def _audio(seconds=1.0):
    t = np.arange(int(seconds * OUTPUT_SAMPLE_RATE)) / OUTPUT_SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _corpus(tmp_path, n=3):
    with CorpusWriter(tmp_path) as w:
        for i in range(n):
            w.add(f"clip{i}", _audio(), f"өгүүлбэр {i}",
                  {"client_id": f"s{i}", "duration_s": 1.0, "gender_resolved": "male"})
    return tmp_path


# ── writing ───────────────────────────────────────────────────────────────────

def test_audio_is_written_at_the_output_sample_rate(tmp_path):
    _corpus(tmp_path, n=1)
    info = sf.info(tmp_path / "wavs" / "clip0.wav")
    assert info.samplerate == OUTPUT_SAMPLE_RATE
    assert info.channels == 1


def test_manifest_row_per_clip(tmp_path):
    rows = read_manifest(_corpus(tmp_path, n=3))
    assert len(rows) == 3
    assert {r["clip_id"] for r in rows} == {"clip0", "clip1", "clip2"}


def test_manifest_paths_are_relative_so_the_corpus_can_move(tmp_path):
    rows = read_manifest(_corpus(tmp_path, n=1))
    assert rows[0]["audio_path"] == "wavs/clip0.wav"


def test_no_audio_is_retained_in_memory(tmp_path):
    """The manifest holds metadata only.

    process_split used to accumulate every decoded clip -- roughly 14 GB at
    Common Voice scale -- and Dataset.from_list then doubled it.
    """
    rows = read_manifest(_corpus(tmp_path, n=2))
    for r in rows:
        assert not any(isinstance(v, (list, bytes)) for v in r.values())


def test_mongolian_text_survives_the_round_trip(tmp_path):
    with CorpusWriter(tmp_path) as w:
        w.add("c", _audio(), "Өнөөдөр үүлшинэ", {"duration_s": 1.0})
    assert read_manifest(tmp_path)[0]["text"] == "Өнөөдөр үүлшинэ"


# ── resume ────────────────────────────────────────────────────────────────────

def test_resume_skips_clips_already_written(tmp_path):
    _corpus(tmp_path, n=2)
    with CorpusWriter(tmp_path, resume=True) as w:
        assert "clip0" in w
        w.add("clip0", _audio(), "different text", {"duration_s": 1.0})
        w.add("new", _audio(), "шинэ", {"duration_s": 1.0})
    rows = read_manifest(tmp_path)
    assert len(rows) == 3
    assert rows[0]["text"] == "өгүүлбэр 0"      # not overwritten


def test_resume_false_starts_clean(tmp_path):
    _corpus(tmp_path, n=2)
    with CorpusWriter(tmp_path, resume=False) as w:
        w.add("only", _audio(), "ганц", {"duration_s": 1.0})
    assert [r["clip_id"] for r in read_manifest(tmp_path)] == ["only"]


# ── the F5-TTS contract ───────────────────────────────────────────────────────

def test_metadata_csv_uses_the_exact_header_f5_validates(tmp_path):
    """prepare_csv_wavs.py raises unless the header is exactly audio_file|text."""
    write_f5_metadata(tmp_path, read_manifest(_corpus(tmp_path, n=2)))
    with open(tmp_path / "metadata.csv", encoding="utf-8-sig") as f:
        header = next(csv.reader(f, delimiter="|"))
    assert header == ["audio_file", "text"]


def test_metadata_csv_paths_are_absolute(tmp_path):
    """prepare_csv_wavs.py raises on any relative path."""
    from pathlib import Path

    write_f5_metadata(tmp_path, read_manifest(_corpus(tmp_path, n=2)))
    with open(tmp_path / "metadata.csv", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter="|"))[1:]
    for path, _text in rows:
        assert Path(path).is_absolute()
        assert Path(path).exists()


def test_metadata_csv_text_matches_the_manifest(tmp_path):
    """Published text, CER-scored text and training text must be one string."""
    records = read_manifest(_corpus(tmp_path, n=3))
    write_f5_metadata(tmp_path, records)
    with open(tmp_path / "metadata.csv", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter="|"))[1:]
    assert [t for _p, t in rows] == [r["text"] for r in records]


def test_non_train_splits_get_their_own_file(tmp_path):
    records = read_manifest(_corpus(tmp_path, n=1))
    assert write_f5_metadata(tmp_path, records, split="test").name == "metadata_test.csv"


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_reports_acceptance_criteria():
    splits = {
        "train": [{"client_id": f"m{i}", "duration_s": 3600.0, "gender_resolved": "male"}
                  for i in range(6)]
                 + [{"client_id": "f1", "duration_s": 72000.0, "gender_resolved": "female"}],
    }
    text = summarise(splits)
    assert "PASS" in text
    assert "FAIL" not in text


def test_summary_fails_a_thin_male_corpus():
    """The male floor is the criterion most at risk, so it must be visible."""
    splits = {
        "train": [{"client_id": "m1", "duration_s": 3600.0, "gender_resolved": "male"},
                  {"client_id": "f1", "duration_s": 90000.0, "gender_resolved": "female"}],
    }
    text = summarise(splits)
    assert "male  >= 5 h             FAIL" in text
    assert "male speakers >= 3       FAIL" in text


# ── text diversity ────────────────────────────────────────────────────────────

def test_diversity_reports_sentence_repetition():
    """Audio hours say nothing about this: Common Voice mn is 28,858 clips over
    6,062 distinct sentences, so a 40 h corpus can still show the model a narrow
    slice of the orthography."""
    from pipeline.corpus import text_diversity

    splits = {"train": [{"text": "нэг өгүүлбэр"}] * 4 + [{"text": "өөр өгүүлбэр"}]}
    out = text_diversity(splits)
    assert "distinct sentences    2" in out
    assert "2.50x repetition" in out


def test_diversity_names_letters_that_never_appear():
    """A letter absent from training cannot be pronounced, and the model has
    only a barely-trained embedding row for it."""
    from pipeline.corpus import text_diversity

    out = text_diversity({"train": [{"text": "аб"}]})
    assert "NEVER APPEARS" in out
    assert "ө" in out.split("NEVER APPEARS")[1]


def test_diversity_counts_the_two_letters_a_naive_range_would_miss():
    """A [а-я] range is U+0410-U+044F and excludes ө U+04E9 and ү U+04AF."""
    from pipeline.corpus import text_diversity

    out = text_diversity({"train": [{"text": "өү"}]})
    assert "letters covered       2/35" in out


def test_diversity_ignores_case_and_punctuation():
    from pipeline.corpus import text_diversity

    a = text_diversity({"train": [{"text": "Сайн."}]})
    b = text_diversity({"train": [{"text": "сайн"}]})
    assert a.split("letters covered")[1] == b.split("letters covered")[1]


def test_diversity_on_an_empty_corpus_says_nothing_rather_than_crashing():
    from pipeline.corpus import text_diversity

    assert text_diversity({"train": []}) == ""
