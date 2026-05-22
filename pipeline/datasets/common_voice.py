from __future__ import annotations

import logging

from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset

from ..audio_filter import AudioQualityFilter
from ..constants import SAMPLE_RATE
from ..processor import process_split
from ..stats import CleaningStats

log = logging.getLogger(__name__)

_EXTRA_FIELDS = [
    "client_id", "path", "sentence", "up_votes", "down_votes",
    "age", "gender", "accents", "variant", "segment", "locale",
]

_FEATURES = Features({
    "client_id":        Value("string"),
    "path":             Value("string"),
    "audio":            Audio(sampling_rate=SAMPLE_RATE),
    "sentence":         Value("string"),
    "up_votes":         Value("int32"),
    "down_votes":       Value("int32"),
    "age":              Value("string"),
    "gender":           Value("string"),
    "accents":          Value("string"),
    "variant":          Value("string"),
    "segment":          Value("string"),
    "locale":           Value("string"),
    "snr_db":           Value("float32"),
    "mean_f0_hz":       Value("float32"),
    "pitch_confidence": Value("float32"),
    "dnsmos_sig":       Value("float32"),
    "dnsmos_bak":       Value("float32"),
    "dnsmos_ovr":       Value("float32"),
    "dnsmos_p808":      Value("float32"),
    "cer":              Value("float32"),
    "asr_transcript":   Value("string"),
    "duration_s":       Value("float32"),
})


def process_common_voice(
    filt: AudioQualityFilter, *, resume: bool = True
) -> tuple[DatasetDict, CleaningStats]:
    log.info("Loading Common Voice 25 Mongolian …")
    cv = load_dataset("mozilla-foundation/common_voice_17_0", "mn", trust_remote_code=True)

    all_stats = CleaningStats("common_voice_25_mn")
    split_map: dict[str, Dataset] = {}

    for split_name in cv.keys():
        passing, stats = process_split(
            cv[split_name],
            filt,
            audio_field="audio",
            text_field="sentence",
            dataset_name="cv",
            split_name=split_name,
            extra_fields=_EXTRA_FIELDS,
            resume=resume,
        )
        all_stats.merge(stats)

        if passing:
            out_name = "other_clean" if split_name == "other" else split_name
            split_map[out_name] = Dataset.from_list(passing, features=_FEATURES)
            log.info("  %s → %d clips passed", out_name, len(passing))

    return DatasetDict(split_map), all_stats
