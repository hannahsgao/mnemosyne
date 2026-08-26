"""Command-line interface for the offline corpus build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .build import CorpusBuildError, SUPPORTED_COUNTING_UNITS, build_corpus
from .dates import DateConfig
from .embedded_corpus import derive_embedded_corpus
from .embeddings import DeterministicTestEncoder, Siglip2LocalEncoder, build_embedding_index
from .met import MET_API_BASE, build_met_corpus
from .met_visual import DEFAULT_MAX_IMAGE_BYTES, prepare_met_visual_subset
from .merge import merge_embedding_bundles
from .nga_visual import (
    DEFAULT_MAX_DIMENSION as NGA_DEFAULT_MAX_DIMENSION,
    DEFAULT_MIN_SHORT_SIDE as NGA_DEFAULT_MIN_SHORT_SIDE,
    prepare_nga_visual_subset,
)
from .repack import repack_embedded_bundle
from .export_concepts import add_arguments as add_export_concept_arguments


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
    build.add_argument(
        "--source-kind",
        default="artifact-clean-local-csv",
        help="source adapter/provenance kind recorded in the build manifest",
    )
    build.add_argument(
        "--counting-unit",
        default="physical-object",
        choices=sorted(SUPPORTED_COUNTING_UNITS),
        help="unit represented by one canonical row (for example catalog-record)",
    )

    build_met = subparsers.add_parser(
        "build-met", help="build a dateable Met corpus with a local SQLite FTS5 index"
    )
    build_met.add_argument("--input", type=Path, required=True, help="official MetObjects.csv")
    build_met.add_argument("--output", type=Path, required=True)
    build_met.add_argument("--corpus-version", required=True)
    build_met.add_argument("--source-revision", required=True, help="pinned metmuseum/openaccess commit")
    build_met.add_argument(
        "--image-ids",
        type=Path,
        help="optional saved Met hasImages search JSON used to mark image availability",
    )
    build_met.add_argument(
        "--fetch-image-ids",
        action="store_true",
        help="fetch and snapshot image-bearing IDs once during the build",
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
    build.add_argument(
        "--source-payload",
        type=Path,
        action="append",
        dest="source_payloads",
        help="additional source/provenance file to checksum and bundle; repeat as needed",
    )
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
    embed.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="parallel remote image fetches per streamed embedding batch",
    )
    embed.add_argument("--image-fetch-retries", type=int, default=2)
    embed.add_argument("--image-request-timeout", type=float, default=30)
    embed.add_argument("--max-image-pixels", type=int, default=100_000_000)
    embed.add_argument(
        "--image-host",
        action="append",
        dest="image_hosts",
        help="allowed HTTPS image host; repeat to allow multiple (default: images.metmuseum.org)",
    )
    embed.add_argument(
        "--no-build-faiss",
        action="store_true",
        help="emit only the exact NumPy index (recommended on macOS)",
    )
    embed.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="resumable work directory; defaults to a hidden sibling of --output",
    )
    embed.add_argument("--allow-model-download", action="store_true")
    embed.add_argument("--allow-unreviewed-images", action="store_true")

    prepare_visual = subparsers.add_parser(
        "prepare-met-visual",
        help="prepare a deterministic public-domain Met image subset for embedding",
    )
    prepare_visual.add_argument("--met-corpus", type=Path, required=True)
    prepare_visual.add_argument("--artifact-csv", type=Path, required=True)
    prepare_visual.add_argument("--output-csv", type=Path, required=True)
    prepare_visual.add_argument(
        "--image-dir",
        type=Path,
        help="optional durable image cache; omit to stream pixels only while embedding",
    )
    prepare_visual.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="deterministic sample size; 0 indexes every eligible artwork",
    )
    prepare_visual.add_argument("--source-revision", required=True)
    prepare_visual.add_argument("--seed", default="met-public-domain-visual-v1")
    prepare_visual.add_argument("--workers", type=int, default=16)
    prepare_visual.add_argument("--max-dimension", type=int, default=512)
    prepare_visual.add_argument("--max-image-bytes", type=int, default=DEFAULT_MAX_IMAGE_BYTES)
    prepare_visual.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip the resumable header-only URL availability check",
    )

    prepare_nga = subparsers.add_parser(
        "prepare-nga-visual",
        help="prepare a deterministic CC0 NGA primary-image subset for embedding",
    )
    prepare_nga.add_argument(
        "--objects", type=Path, required=True, help="official NGA objects.csv"
    )
    prepare_nga.add_argument(
        "--published-images",
        type=Path,
        required=True,
        help="official NGA published_images.csv",
    )
    prepare_nga.add_argument(
        "--object-associations",
        type=Path,
        required=True,
        help="official NGA object_associations.csv used for inseparable-object grouping",
    )
    prepare_nga.add_argument("--output-csv", type=Path, required=True)
    prepare_nga.add_argument(
        "--source-revision",
        required=True,
        help="pinned 40-character NationalGalleryOfArt/opendata commit",
    )
    prepare_nga.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="deterministic sample size; 0 prepares every eligible artwork",
    )
    prepare_nga.add_argument(
        "--seed", default="nga-open-access-primary-visual-v1"
    )
    prepare_nga.add_argument("--workers", type=int, default=16)
    prepare_nga.add_argument(
        "--max-dimension", type=int, default=NGA_DEFAULT_MAX_DIMENSION
    )
    prepare_nga.add_argument(
        "--min-short-side", type=int, default=NGA_DEFAULT_MIN_SHORT_SIDE
    )
    prepare_nga.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip the resumable header-only IIIF availability check",
    )
    prepare_nga.add_argument(
        "--include-undated",
        action="store_true",
        help=(
            "include image-eligible objects without trustworthy timeline bounds; "
            "excluded by default because NGA numeric dates can be artist lifespans"
        ),
    )

    derive = subparsers.add_parser(
        "derive-embedded-corpus",
        help="derive a hash-identified canonical CSV from an embedding bundle",
    )
    derive.add_argument("--bundle", type=Path, required=True)
    derive.add_argument(
        "--visual-manifest",
        type=Path,
        required=True,
        help="preflight manifest containing the source selection and image rights gate",
    )
    derive.add_argument("--output", type=Path, required=True)

    repack = subparsers.add_parser(
        "repack-embedded-bundle",
        help="pair existing float32 vectors with hash-reconciled corpus metadata",
    )
    repack.add_argument("--embedding-bundle", type=Path, required=True)
    repack.add_argument("--corpus-dir", type=Path, required=True)
    repack.add_argument("--output", type=Path, required=True)
    repack.add_argument(
        "--copy-faiss",
        action="store_true",
        help="copy an aligned existing exact FAISS index after validating it",
    )

    merge = subparsers.add_parser(
        "merge-embedded-bundles",
        help="merge compatible completed embedding bundles without re-embedding",
    )
    merge.add_argument(
        "--bundle",
        type=Path,
        action="append",
        required=True,
        dest="bundles",
        help="completed embedding bundle; repeat at least twice",
    )
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--corpus-version", required=True)
    merge.add_argument(
        "--corpus-label",
        help="optional user-facing label recorded in both merged manifests",
    )
    export_concepts = subparsers.add_parser(
        "export-concepts",
        help="export compact static concept timelines from completed embeddings",
    )
    add_export_concept_arguments(export_concepts)
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
                source_kind=args.source_kind,
                counting_unit=args.counting_unit,
                date_config=DateConfig(
                    bin_size=args.bin_size,
                    circa_years=args.circa_years,
                    open_range_years=args.open_range_years,
                    min_year=args.min_year,
                    max_year=args.max_year,
                ),
                require_parquet=args.require_parquet,
                source_payloads=[args.input, *(args.source_payloads or [])],
            )
        elif args.command == "build-met":
            manifest = build_met_corpus(
                args.input,
                args.output,
                corpus_version=args.corpus_version,
                source_revision=args.source_revision,
                image_ids_path=args.image_ids,
                api_base=args.met_api_base,
                fetch_image_ids=args.fetch_image_ids,
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
                    download_workers=args.download_workers,
                    request_timeout=args.image_request_timeout,
                    fetch_retries=args.image_fetch_retries,
                    max_image_pixels=args.max_image_pixels,
                    allowed_image_hosts=args.image_hosts or ("images.metmuseum.org",),
                    checkpoint_dir=(
                        args.checkpoint_dir
                        or args.output.with_name(f".{args.output.name}.checkpoint")
                    ),
                    progress=lambda completed, total: print(
                        f"embedded={completed}/{total}", file=sys.stderr, flush=True
                    ),
                )
            manifest = build_embedding_index(
                args.corpus_dir,
                args.output,
                encoder,
                image_root=args.image_root,
                dtype=args.dtype,
                batch_size=args.batch_size,
                allow_unreviewed_images=args.allow_unreviewed_images,
                write_faiss=not args.no_build_faiss,
            )
        elif args.command == "prepare-met-visual":
            manifest = prepare_met_visual_subset(
                args.met_corpus,
                args.artifact_csv,
                args.output_csv,
                args.image_dir,
                sample_size=args.sample_size,
                source_revision=args.source_revision,
                seed=args.seed,
                workers=args.workers,
                max_dimension=args.max_dimension,
                max_image_bytes=args.max_image_bytes,
                preflight=not args.no_preflight,
                progress=lambda examined, prepared, total: print(
                    f"examined={examined} prepared={prepared}/"
                    f"{args.sample_size or total} "
                    f"eligible={total}",
                    file=sys.stderr,
                    flush=True,
                ),
            )
        elif args.command == "prepare-nga-visual":
            manifest = prepare_nga_visual_subset(
                args.objects,
                args.published_images,
                args.output_csv,
                object_associations_csv=args.object_associations,
                source_revision=args.source_revision,
                sample_size=args.sample_size,
                seed=args.seed,
                workers=args.workers,
                max_dimension=args.max_dimension,
                min_short_side=args.min_short_side,
                preflight=not args.no_preflight,
                include_undated=args.include_undated,
                progress=lambda examined, prepared, total: print(
                    f"examined={examined} prepared={prepared}/"
                    f"{args.sample_size or total} "
                    f"eligible={total}",
                    file=sys.stderr,
                    flush=True,
                ),
            )
        elif args.command == "derive-embedded-corpus":
            manifest = derive_embedded_corpus(
                args.bundle,
                args.visual_manifest,
                args.output,
            )
        elif args.command == "repack-embedded-bundle":
            manifest = repack_embedded_bundle(
                args.embedding_bundle,
                args.corpus_dir,
                args.output,
                copy_faiss=args.copy_faiss,
            )
        elif args.command == "merge-embedded-bundles":
            manifest = merge_embedding_bundles(
                args.bundles,
                args.output,
                corpus_version=args.corpus_version,
                corpus_label=args.corpus_label,
            )
        elif args.command == "export-concepts":
            from .export_concepts import run as run_export_concepts

            return_code_payload = run_export_concepts(args)
            print(
                json.dumps(
                    return_code_payload, ensure_ascii=False, indent=2, sort_keys=True
                )
            )
            return 1 if return_code_payload["run"]["failures"] else 0
        else:
            return 2
    except (CorpusBuildError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
