"""
Mongolian speech dataset cleaning & HuggingFace upload pipeline.

Targets:
  btsee/common-voices-25-mn  ← mozilla-foundation/common_voice_17_0 (mn)
  btsee/fleurs-mn            ← google/fleurs (mn_mn)
  btsee/worldspeech-mn       ← disco-eth/WorldSpeech (mn_mn)

Usage:
  python clean_pipeline.py --hf-token <TOKEN>
  python clean_pipeline.py --hf-token <TOKEN> --datasets cv,fleurs
  python clean_pipeline.py --hf-token <TOKEN> --device cpu --no-resume
"""

from __future__ import annotations

import argparse
import logging
import warnings

warnings.quality_filtererwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-token", required=True, help="HuggingFace write token")
    p.add_argument("--datasets", default="all", help="Comma-separated: all | cv | fleurs | ws")
    p.add_argument("--device", default="", help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore checkpoints, start fresh")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    import torch
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_datasets(raw: str) -> list[str]:
    keys = [s.strip().lower() for s in raw.split(",")]
    return ["cv", "fleurs", "ws"] if "all" in keys else keys


def main() -> None:
    args = parse_args()

    from huggingface_hub import login
    login(token=args.hf_token)
    log.info("HuggingFace login successful.")

    device = resolve_device(args.device)
    log.info("Using device: %s", device)

    selected = resolve_datasets(args.datasets)
    log.info("Datasets to process: %s", selected)

    from pipeline.audio_quality_filterer import AudioQualityFilter
    from pipeline.datasets.common_voice import process_common_voice
    from pipeline.datasets.fleurs import process_fleurs
    from pipeline.datasets.worldspeech import process_worldspeech
    from pipeline.upload import upload_dataset

    quality_filter = AudioQualityFilter(device=device)

    if "cv" in selected:
        log.info("=" * 60)
        log.info("Common Voice 25 Mongolian")
        ds, stats = process_common_voice(quality_filter, resume=args.resume)
        stats.save(path=__import__("pathlib").Path("cleaning_report_cv.txt"))
        upload_dataset("cv", ds, stats)

    if "fleurs" in selected:
        log.info("=" * 60)
        log.info("FLEURS Mongolian")
        ds, stats = process_fleurs(quality_filter, resume=args.resume)
        stats.save(path=__import__("pathlib").Path("cleaning_report_fleurs.txt"))
        upload_dataset("fleurs", ds, stats)

    if "ws" in selected:
        log.info("=" * 60)
        log.info("WorldSpeech Mongolian")
        ds, stats = process_worldspeech(quality_filter, resume=args.resume)
        stats.save(path=__import__("pathlib").Path("cleaning_report_ws.txt"))
        upload_dataset("ws", ds, stats)

    log.info("All done.")


if __name__ == "__main__":
    main()
