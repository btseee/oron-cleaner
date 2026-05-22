import logging

from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset

from ..audio_filter import AudioQualityFilter
from ..constants import SAMPLE_RATE
from ..processor import process_split
from ..stats import CleaningStats

log = logging.getLogger(__name__)

_EXTRA_FIELDS = [
    "id", "num_samples", "path", "raw_transcription", "transcription",
    "gender", "lang_id", "language", "lang_group_id",
]

_FEATURES = Features({
    "id":                Value("int32"),
    "num_samples":       Value("int32"),
    "path":              Value("string"),
    "audio":             Audio(sampling_rate=SAMPLE_RATE),
    "raw_transcription": Value("string"),
    "transcription":     Value("string"),
    "gender":            Value("int32"),
    "lang_id":           Value("int32"),
    "language":          Value("string"),
    "lang_group_id":     Value("int32"),
    "snr_db":            Value("float32"),
    "mean_f0_hz":        Value("float32"),
    "pitch_confidence":  Value("float32"),
    "dnsmos_sig":        Value("float32"),
    "dnsmos_bak":        Value("float32"),
    "dnsmos_ovr":        Value("float32"),
    "dnsmos_p808":       Value("float32"),
    "cer":               Value("float32"),
    "asr_transcript":    Value("string"),
    "duration_s":        Value("float32"),
})


def process_fleurs(
    filt: AudioQualityFilter, *, resume: bool = True
) -> tuple[DatasetDict, CleaningStats]:
    log.info("Loading FLEURS Mongolian …")
    fleurs = load_dataset("google/fleurs", "mn_mn")

    all_stats = CleaningStats("fleurs_mn")
    split_map: dict[str, Dataset] = {}

    for split_name in fleurs.keys():
        passing, stats = process_split(
            fleurs[split_name],
            filt,
            audio_field="audio",
            text_field="raw_transcription",
            dataset_name="fleurs",
            split_name=split_name,
            extra_fields=_EXTRA_FIELDS,
            resume=resume,
        )
        all_stats.merge(stats)

        if passing:
            split_map[split_name] = Dataset.from_list(passing, features=_FEATURES)
            log.info("  %s → %d clips passed", split_name, len(passing))

    return DatasetDict(split_map), all_stats
