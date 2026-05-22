import logging

from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset

from ..audio_filter import AudioQualityFilter
from ..constants import OUTPUT_SAMPLE_RATE, SAMPLE_RATE
from ..processor import process_split
from ..stats import CleaningStats

log = logging.getLogger(__name__)

_FEATURES = Features({
    "audio":                    Audio(sampling_rate=OUTPUT_SAMPLE_RATE),
    "sentence":                 Value("string"),
    "human_transcript":         Value("string"),
    "original_asr_transcript":  Value("string"),
    "original_cer":             Value("float32"),
    "snr":                      Value("float32"),
    "dnsmos_sig":               Value("float32"),
    "dnsmos_bak":               Value("float32"),
    "dnsmos_ovr":               Value("float32"),
    "dnsmos_p808":              Value("float32"),
    "duration":                 Value("float32"),
    "source":                   Value("string"),
    "source_url":               Value("string"),
    "source_start_s":           Value("float32"),
    "source_end_s":             Value("float32"),
    "session_date":             Value("string"),
    "segment_id":               Value("string"),
    "language":                 Value("string"),
    "country":                  Value("string"),
    "clean_snr_db":             Value("float32"),
    "clean_mean_f0_hz":         Value("float32"),
    "clean_pitch_confidence":   Value("float32"),
    "clean_dnsmos_sig":         Value("float32"),
    "clean_dnsmos_bak":         Value("float32"),
    "clean_dnsmos_ovr":         Value("float32"),
    "clean_dnsmos_p808":        Value("float32"),
    "clean_cer":                Value("float32"),
    "clean_asr_transcript":     Value("string"),
})

# Extra fields copied verbatim from each original item.
# Fields conflicting with freshly-computed metrics are renamed via _FIELD_RENAMES
# so they don't overwrite the pipeline's own output.
_EXTRA_FIELDS = [
    "human_transcript", "snr", "duration", "source", "source_url",
    "source_start_s", "source_end_s", "session_date", "segment_id",
    "language", "country",
    "asr_transcript",
    "cer",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "dnsmos_p808",
]

_FIELD_RENAMES = {
    "asr_transcript": "original_asr_transcript",
    "cer":            "original_cer",
    "dnsmos_sig":     "_ws_dnsmos_sig",
    "dnsmos_bak":     "_ws_dnsmos_bak",
    "dnsmos_ovr":     "_ws_dnsmos_ovr",
    "dnsmos_p808":    "_ws_dnsmos_p808",
}


def process_worldspeech(
    filt: AudioQualityFilter, *, resume: bool = True
) -> tuple[DatasetDict, CleaningStats]:
    log.info("Loading WorldSpeech Mongolian …")
    ws = load_dataset("disco-eth/WorldSpeech", "mn_mn")

    all_stats = CleaningStats("worldspeech_mn")
    split_map: dict[str, Dataset] = {}

    for split_name in ws.keys():
        split = _prefilter_by_snr(ws[split_name], split_name)

        passing_raw, stats = process_split(
            split,
            filt,
            audio_field="audio",
            text_field="human_transcript",
            dataset_name="ws",
            split_name=split_name,
            extra_fields=_EXTRA_FIELDS,
            field_renames=_FIELD_RENAMES,
            resume=resume,
        )
        all_stats.merge(stats)

        if passing_raw:
            passing = [_promote_ws_fields(rec) for rec in passing_raw]
            split_map[split_name] = Dataset.from_list(passing, features=_FEATURES)
            log.info("  %s → %d clips passed", split_name, len(passing))

    return DatasetDict(split_map), all_stats


def _prefilter_by_snr(split, split_name: str):
    log.info("  Pre-filtering %s by original SNR < 10 …", split_name)
    filtered = split.filter(lambda x: (x.get("snr") or 0.0) >= 10.0, desc="snr_prefilter")
    log.info("  %d clips remain after SNR pre-filter.", len(filtered))
    return filtered


def _promote_ws_fields(rec: dict) -> dict:
    """
    Rename computed metrics to clean_* and promote the temporarily-prefixed
    original WorldSpeech dnsmos values to their final unprefixed names.
    """
    out = dict(rec)
    out["clean_snr_db"]          = out.pop("snr_db",           0.0)
    out["clean_mean_f0_hz"]      = out.pop("mean_f0_hz",        0.0)
    out["clean_pitch_confidence"]= out.pop("pitch_confidence",  0.0)
    out["clean_dnsmos_sig"]      = out.pop("dnsmos_sig",        0.0)
    out["clean_dnsmos_bak"]      = out.pop("dnsmos_bak",        0.0)
    out["clean_dnsmos_ovr"]      = out.pop("dnsmos_ovr",        0.0)
    out["clean_dnsmos_p808"]     = out.pop("dnsmos_p808",       0.0)
    out["clean_cer"]             = out.pop("cer",               0.0)
    out["clean_asr_transcript"]  = out.pop("asr_transcript",    "")
    out["dnsmos_sig"]   = out.pop("_ws_dnsmos_sig",  0.0)
    out["dnsmos_bak"]   = out.pop("_ws_dnsmos_bak",  0.0)
    out["dnsmos_ovr"]   = out.pop("_ws_dnsmos_ovr",  0.0)
    out["dnsmos_p808"]  = out.pop("_ws_dnsmos_p808", 0.0)
    out["sentence"] = out.get("human_transcript", "")
    return out
