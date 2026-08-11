"""CLI for exporting a static visual-concept catalog from completed vectors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--concept-id",
        action="append",
        dest="concept_ids",
        help="stable concept ID to export; repeat to select several",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".local-data/concept-export-state"),
        help="private local checksum-verification state (never published)",
    )


def _progress(event) -> None:
    fields = " ".join(f"{key}={value}" for key, value in event.items())
    print(fields, file=sys.stderr, flush=True)


def run(args: argparse.Namespace) -> dict:
    try:
        from mnemosyne_search.encoders import Siglip2TextEncoder
        from mnemosyne_search.exporter import (
            artifact_manifest_sha256,
            export_catalog,
            load_artifacts_for_export,
            load_concept_source,
        )
    except ImportError as error:
        raise RuntimeError(
            "install the local search package first: python -m pip install -e './service[siglip2]'"
        ) from error

    source = load_concept_source(args.concepts)
    artifacts, used_verification_cache = load_artifacts_for_export(
        args.artifacts,
        state_dir=args.state_dir,
        resume=args.resume,
    )
    encoder = Siglip2TextEncoder(
        artifacts.model_id,
        revision=artifacts.model_version,
        device=args.device,
        local_files_only=not args.allow_model_download,
    )
    manifest, stats = export_catalog(
        artifacts,
        source,
        args.output,
        encoder,
        artifact_manifest_hash=artifact_manifest_sha256(args.artifacts),
        selected_concept_ids=args.concept_ids,
        limit=args.limit,
        batch_size=args.batch_size,
        resume=args.resume,
        progress=_progress,
    )
    return {
        "manifest": manifest,
        "run": {
            **asdict(stats),
            "usedVerificationCache": used_verification_cache,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.export_concepts",
        description=(
            "Export compact, resumable visual-concept timelines from an existing "
            "Mnemosyne embedding artifact bundle."
        ),
    )
    add_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if result["run"]["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
