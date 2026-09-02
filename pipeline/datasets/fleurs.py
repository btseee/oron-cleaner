"""FLEURS Mongolian.

CC-BY-4.0, ~13 h. 16 kHz native, so its bandwidth is capped at ~7.7 kHz -- it
adds clean read speech and both genders, but cannot supply a bright reference
clip.

Two things the schema forces, both verified against the dataset builder:

* **`gender` is a `ClassLabel(names=['male', 'female', 'other'])`**, so a row
  yields `0`/`1`/`2`, not a string. `normalize_gender(0)` stringifies to `"0"`,
  misses every alias, and returns unknown -- so before this wrapper *every*
  FLEURS gender label was silently discarded. Same bug class as the
  `male_masculine` one the speakers module documents for Common Voice v17.
* **There is no speaker column at all.** id, num_samples, path, audio,
  transcription, raw_transcription, gender, lang_id, language, lang_group_id --
  that is the whole schema. `id` is the *sentence* index, shared by every
  recording of that sentence, so it identifies text rather than a voice.

The second cannot be fixed here, only declared: each clip carries
`speaker_known: False` so `speaker_disjoint_split` routes the whole block to
training instead of inventing one pseudo-speaker per clip and degenerating to a
row-level random split for ~20% of the corpus.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from datasets import load_dataset

from ..corpus import CorpusWriter
from ..processor import process_split
from ..provenance import PINNED_REVISIONS
from ..stats import CleaningStats

# AudioQualityFilter is only referenced as a type here. Importing it at
# runtime would drag Silero VAD, transformers, torchmetrics and the MMS_FA
# aligner into any process that merely wants to inspect a loader.
if TYPE_CHECKING:
    from ..audio_filter import AudioQualityFilter
    from ..calibrate import Calibration

log = logging.getLogger(__name__)

_EXTRA_FIELDS = [
    "id", "num_samples", "path", "raw_transcription", "transcription",
    "gender", "lang_id", "language", "lang_group_id", "speaker_known",
]


class _Decoded:
    """Turn the ClassLabel gender into its name and declare speakers unknown."""

    def __init__(self, split) -> None:
        self._split = split
        feature = split.features.get("gender")
        # int2str exists only on ClassLabel. Guarding on the method rather than
        # the type keeps this working if a future release ships plain strings.
        self._int2str = getattr(feature, "int2str", None)

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self._split[idx])
        raw = item.get("gender")
        if self._int2str is not None and isinstance(raw, int):
            item["gender"] = self._int2str(raw)
        # 'other' is in the label set and maps to unknown, as it should: it
        # describes identity, not a vocal tract.
        item["speaker_known"] = False
        return item


def process_fleurs(
    filt: AudioQualityFilter, writer: CorpusWriter, *, resume: bool = True,
    limit: int | None = None,
    calibration: Calibration | None = None,
) -> CleaningStats:
    log.info("Loading FLEURS Mongolian …")
    fleurs = load_dataset("google/fleurs", "mn_mn", revision=PINNED_REVISIONS["google/fleurs"])

    all_stats = CleaningStats("fleurs_mn")
    for split_name in fleurs:
        all_stats.merge(process_split(
            _Decoded(fleurs[split_name]),
            filt,
            writer,
            audio_field="audio",
            # raw_transcription is the human-written text; `transcription` is a
            # normalised derivative, and scoring against it would compare the
            # audio to something the speaker did not read.
            text_field="raw_transcription",
            dataset_name="fleurs",
            split_name=split_name,
            extra_fields=_EXTRA_FIELDS,
            resume=resume,
            limit=limit,
            calibration=calibration,
        ))
    return all_stats
