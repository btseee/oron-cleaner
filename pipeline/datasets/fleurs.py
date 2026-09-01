"""FLEURS Mongolian.

CC-BY-4.0, ~13 h. 16 kHz native, so its bandwidth is capped at ~7.7 kHz -- it
adds clean read speech and both genders, but cannot supply a bright reference
clip.
"""

import logging

from datasets import load_dataset

from ..audio_filter import AudioQualityFilter
from ..corpus import CorpusWriter
from ..processor import process_split
from ..stats import CleaningStats

log = logging.getLogger(__name__)

_EXTRA_FIELDS = [
    "id", "num_samples", "path", "raw_transcription", "transcription",
    "gender", "lang_id", "language", "lang_group_id",
]


def process_fleurs(
    filt: AudioQualityFilter, writer: CorpusWriter, *, resume: bool = True
) -> CleaningStats:
    log.info("Loading FLEURS Mongolian …")
    fleurs = load_dataset("google/fleurs", "mn_mn")

    all_stats = CleaningStats("fleurs_mn")
    for split_name in fleurs:
        all_stats.merge(process_split(
            fleurs[split_name],
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
        ))
    return all_stats
