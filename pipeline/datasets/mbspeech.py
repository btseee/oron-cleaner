"""MBSpeech Mongolian.

MIT, ~6.3 h of biblical narration. Confirmed single male speaker: median F0
140.7 Hz across 10 clips (spread 45 Hz), against a calibration of 115.1 Hz male
and 239.3 Hz female from self-declared Common Voice labels.

That makes it the cleanest male audio available, and Common Voice only has
10.6 h of labelled male speech, so it matters disproportionately. But it is
16 kHz native, capped at ~7.7 kHz, so it cannot supply a bright reference clip.

A loader was missing entirely even though `btsee/mbspeech_mn` was oron-tts's
default training dataset.

**The raw mirror no longer exists.** `btsee/mbspeech_mn` was deleted once the
cleaned corpus was published, so this loader cannot run and there is nothing to
re-clean from. `btsee/mbspeech-mn` holds the cleaned result -- 2,995 clips,
5.40 h -- and is restored directly rather than rebuilt. The loader is kept
because it documents how that corpus was produced, and because a future raw
source can be pointed at `RAW_REPO`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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

# `datasets` is imported inside the loader, not here. This module's own
# comment below says a process that merely wants to inspect a loader should
# not pull the model stack -- and importing `datasets` at module scope broke
# exactly that, taking four test files down with it in an environment that
# has no model stack installed.

log = logging.getLogger(__name__)

# The raw upstream. None because btsee/mbspeech_mn was deleted; set it to a raw
# mirror to make this loader runnable again.
RAW_REPO: str | None = None

# The cleaned corpus this loader produced, published so it never has to be
# produced twice.
CLEAN_REPO = "btsee/mbspeech-mn"

# One narrator, but there is no speaker column, so a constant id is supplied.
# Without it every clip looks like its own speaker and both the per-speaker cap
# and the speaker-disjoint split silently do nothing.
SPEAKER_ID = "mbspeech_narrator_01"

_EXTRA_FIELDS = ["sentence_orig", "sentence_norm", "client_id", "gender",
                 "single_narrator"]


class _Wrapped:
    """Attach the constant speaker id and gender the source does not carry."""

    def __init__(self, split) -> None:
        self._split = split

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self._split[idx])
        item["client_id"] = SPEAKER_ID
        item["gender"] = "male"
        # Exempts this source from the per-speaker cap. A cap buys voice
        # diversity; one narrator has none to buy, so capping here would only
        # delete the cleanest male audio in the corpus -- 5.64 h of 6.3 h under
        # the previous 400-clip rule.
        item["single_narrator"] = True
        item.setdefault("path", f"mbspeech_{idx:06d}")
        return item


def process_mbspeech(
    filt: AudioQualityFilter, writer: CorpusWriter, *, resume: bool = True,
    limit: int | None = None,
    calibration: Calibration | None = None,
) -> CleaningStats:
    from ._load import load_hub_dataset

    if RAW_REPO is None:
        raise SystemExit(
            "MBSpeech has no raw source any more: btsee/mbspeech_mn was deleted "
            "after the cleaned corpus was published. Restore the cleaned corpus "
            "from btsee/mbspeech-mn instead of re-cleaning, or set RAW_REPO to a "
            "raw mirror. Re-cleaning the *cleaned* dataset would gate already "
            "gated audio and shrink the corpus for no reason.")

    log.info("Loading MBSpeech Mongolian from %s …", RAW_REPO)
    ds = load_hub_dataset(RAW_REPO, revision=PINNED_REVISIONS.get(RAW_REPO))

    all_stats = CleaningStats("mbspeech_mn")
    for split_name in ds:
        all_stats.merge(process_split(
            _Wrapped(ds[split_name]),
            filt,
            writer,
            audio_field="audio",
            # sentence_orig is what the narrator read; sentence_norm is a
            # derivative, and scoring against it would compare the audio to
            # something that was never spoken.
            text_field="sentence_orig",
            dataset_name="mbspeech",
            split_name=split_name,
            extra_fields=_EXTRA_FIELDS,
            resume=resume,
            limit=limit,
            calibration=calibration,
        ))
    return all_stats
