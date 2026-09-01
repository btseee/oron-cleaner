"""Build the strict Mongolian TTS corpus for oron-tts.

Sources are merged into one corpus rather than published separately, because the
consumer is a single training run and the speaker cap and speaker-disjoint split
only make sense across the whole thing:

  Common Voice 26.0 mn   CC0        ~40 h after the down-vote gate
  FLEURS mn_mn           CC-BY-4.0  ~13 h
  MBSpeech mn            MIT        ~6 h, single male narrator

All three are commercially usable. WorldSpeech is deliberately absent: it is by
far the largest Mongolian corpus (~221 h, 24 kHz native) but CC-BY-NC-4.0.

Output is wavs plus a manifest, and a `metadata.csv` in the exact
`audio_file|text` form F5-TTS's prepare_csv_wavs.py requires.

Usage:
  python clean_pipeline.py --hf-token <TOKEN>
  python clean_pipeline.py --datasets cv,fleurs --no-upload
  python clean_pipeline.py --finalize-only        # re-split without refiltering

API keys load from .env automatically (API_KEY, HF_TOKEN).
"""

import argparse
import logging
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Populate os.environ from .env before argparse computes its defaults."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALL_SOURCES = ["cv", "fleurs", "mbspeech"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                   help="HuggingFace write token (or set HF_TOKEN in .env)")
    p.add_argument("--cv-api-key", default=os.environ.get("API_KEY", ""),
                   help="Mozilla Data Collective API key (or set API_KEY in .env)")
    p.add_argument("--datasets", default="all",
                   help=f"Comma-separated: all | {' | '.join(ALL_SOURCES)}")
    p.add_argument("--corpus-dir", type=Path, default=Path("output/oron_mn_strict"))
    p.add_argument("--device", default="", help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--finalize-only", action="store_true",
                   help="Re-run splitting and export from the existing manifest")
    p.add_argument("--no-upload", action="store_true", help="Skip the HuggingFace push")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    import torch

    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def resolve_sources(raw: str) -> list[str]:
    keys = [s.strip().lower() for s in raw.split(",") if s.strip()]
    if "all" in keys:
        return list(ALL_SOURCES)
    unknown = [k for k in keys if k not in ALL_SOURCES]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Choose from {ALL_SOURCES}.")
    return keys


def finalize(corpus_dir: Path) -> dict:
    """Resolve gender, cap speakers, split, and export. No audio is touched."""
    from pipeline.corpus import (
        read_manifest,
        summarise,
        write_f5_metadata,
        write_parquet_manifest,
    )
    from pipeline.speakers import cap_per_speaker, propagate_gender, speaker_disjoint_split

    records = read_manifest(corpus_dir)
    if not records:
        raise SystemExit(f"No manifest at {corpus_dir}. Run the filtering pass first.")
    log.info("Finalising %d clips", len(records))

    records, counts = propagate_gender(records)
    log.info("Gender: %d declared, %d propagated by speaker, %d from F0, %d unknown",
             counts["declared"], counts["propagated"], counts["from_f0"], counts["unknown"])
    if counts["conflicting_speakers"]:
        log.warning("%d speakers had conflicting gender labels and were left unknown",
                    counts["conflicting_speakers"])

    before = len(records)
    records = cap_per_speaker(records)
    if len(records) < before:
        log.info("Per-speaker cap removed %d clips", before - len(records))

    splits = speaker_disjoint_split(records)
    for name, rs in splits.items():
        write_f5_metadata(corpus_dir, rs, split=name)
    write_parquet_manifest(corpus_dir, splits)

    report = summarise(splits)
    (corpus_dir / "corpus_summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report + "\n")
    return splits


def main() -> None:
    args = parse_args()
    from pipeline.constants import OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        finalize(args.corpus_dir)
        return

    sources = resolve_sources(args.datasets)
    log.info("Sources: %s", sources)

    # Check credentials before loading ~3 GB of models, not after.
    if "cv" in sources and not args.cv_api_key:
        raise SystemExit(
            "Common Voice needs --cv-api-key (or API_KEY in .env). "
            "The dataset id is release-specific; take the current one from the "
            "dataset page URL on mozilladatacollective.com."
        )
    if not args.no_upload and not args.hf_token:
        raise SystemExit("Uploading needs --hf-token (or HF_TOKEN in .env); or pass --no-upload.")

    device = resolve_device(args.device)
    log.info("Device: %s", device)

    from pipeline.audio_filter import AudioQualityFilter
    from pipeline.corpus import CorpusWriter
    from pipeline.datasets.common_voice import process_common_voice
    from pipeline.datasets.fleurs import process_fleurs
    from pipeline.datasets.mbspeech import process_mbspeech

    quality_filter = AudioQualityFilter(device=device)

    with CorpusWriter(args.corpus_dir, resume=args.resume) as writer:
        if "cv" in sources:
            log.info("=" * 60)
            process_common_voice(
                quality_filter, writer, api_key=args.cv_api_key, resume=args.resume
            ).save(OUTPUT_DIR / "cleaning_report_cv.txt")
        if "fleurs" in sources:
            log.info("=" * 60)
            process_fleurs(quality_filter, writer, resume=args.resume).save(
                OUTPUT_DIR / "cleaning_report_fleurs.txt"
            )
        if "mbspeech" in sources:
            log.info("=" * 60)
            process_mbspeech(quality_filter, writer, resume=args.resume).save(
                OUTPUT_DIR / "cleaning_report_mbspeech.txt"
            )

    log.info("=" * 60)
    finalize(args.corpus_dir)

    if not args.no_upload:
        from huggingface_hub import login

        from pipeline.upload import upload_corpus

        login(token=args.hf_token)
        upload_corpus(args.corpus_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
