"""Command-line interface for the offline corpus build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .build import CorpusBuildError, build_corpus
from .dates import DateConfig
from .embeddings import DeterministicTestEncoder, Siglip2LocalEncoder, build_embedding_index
from .met import MET_API_BASE, build_met_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Build reproducible Mnemosyne corpus and date artifacts from a local clean CSV.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build an immutable corpus artifact directory")
    build.add_argument("--input", type=Path, required=True, help="local ArtiFact_clean-style CSV")
    build.add_argument("--output", type=Path, required=True, help="new or empty artifact directory")
    build.add_argument("--corpus-version", required=True, help="immutable product corpus version")
    build.add_argument("--source-revision", required=True, help="pinned upstream commit or release")
    build.add_argument(
        "--source-url",
        default="https://huggingface.co/datasets/deem-data/ArtiFact",
    )

    build_met = subparsers.add_parser(
        "build-met", help="build a public-domain, image-backed Met corpus from MetObjects.csv"
    )
    build_met.add_argument("--input", type=Path, required=True, help="official MetObjects.csv")
    build_met.add_argument("--output", type=Path, required=True)
    build_met.add_argument("--corpus-version", required=True)
    build_met.add_argument("--source-revision", required=True, help="pinned metmuseum/openaccess commit")
    build_met.add_argument(
        "--image-ids",
        type=Path,
        help="saved Met hasImages search JSON; fetched and snapshotted when omitted",
    )
    build_met.add_argument("--met-api-base", default=MET_API_BASE)
    build_met.add_argument("--retrieved-at")
    build_met.add_argument("--bin-size", type=int, default=10)
    build_met.add_argument("--circa-years", type=int, default=5)
    build_met.add_argument("--open-range-years", type=int, default=25)
    build_met.add_argument("--min-year", type=int)
    build_met.add_argument("--max-year", type=int)
    build_met.add_argument("--require-parquet", action="store_true")
    build.add_argument(
        "--retrieved-at",
        help="ISO-8601 retrieval timestamp; omit for deterministic 'not-recorded'",
    )
    build.add_argument("--metadata-license", default="")
    build.add_argument("--bin-size", type=int, default=10)
    build.add_argument("--circa-years", type=int, default=5)
    build.add_argument("--open-range-years", type=int, default=25)
    build.add_argument("--min-year", type=int)
    build.add_argument("--max-year", type=int)
    build.add_argument(
        "--require-parquet",
        action="store_true",
        help="fail rather than emit the always-available CSV fallback when PyArrow is absent",
    )

    embed = subparsers.add_parser("embed", help="build normalized image embeddings and an exact index")
    embed.add_argument("--corpus-dir", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--image-root", type=Path)
    embed.add_argument("--encoder", choices=("deterministic", "siglip2"), default="siglip2")
    embed.add_argument("--model", default="google/siglip2-base-patch16-224")
    embed.add_argument("--model-revision")
    embed.add_argument("--dimension", type=int, default=32, help="deterministic encoder only")
    embed.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    embed.add_argument("--batch-size", type=int, default=16)
    embed.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    embed.add_argument("--allow-model-download", action="store_true")
    embed.add_argument("--allow-unreviewed-images", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_corpus(
                args.input,
                args.output,
                corpus_version=args.corpus_version,
                source_revision=args.source_revision,
                source_url=args.source_url,
                retrieved_at=args.retrieved_at,
                metadata_license=args.metadata_license,
                date_config=DateConfig(
                    bin_size=args.bin_size,
                    circa_years=args.circa_years,
                    open_range_years=args.open_range_years,
                    min_year=args.min_year,
                    max_year=args.max_year,
                ),
                require_parquet=args.require_parquet,
            )
        elif args.command == "build-met":
            manifest = build_met_corpus(
                args.input,
                args.output,
                corpus_version=args.corpus_version,
                source_revision=args.source_revision,
                image_ids_path=args.image_ids,
                api_base=args.met_api_base,
                retrieved_at=args.retrieved_at,
                date_config=DateConfig(
                    bin_size=args.bin_size,
                    circa_years=args.circa_years,
                    open_range_years=args.open_range_years,
                    min_year=args.min_year,
                    max_year=args.max_year,
                ),
                require_parquet=args.require_parquet,
            )
        elif args.command == "embed":
            if args.encoder == "deterministic":
                encoder = DeterministicTestEncoder(args.dimension)
            else:
                encoder = Siglip2LocalEncoder(
                    args.model,
                    args.model_revision or "",
                    allow_download=args.allow_model_download,
                    device=args.device,
                )
            manifest = build_embedding_index(
                args.corpus_dir,
                args.output,
                encoder,
                image_root=args.image_root,
                dtype=args.dtype,
                batch_size=args.batch_size,
                allow_unreviewed_images=args.allow_unreviewed_images,
            )
        else:
            return 2
    except (CorpusBuildError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
