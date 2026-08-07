"""Command line entrypoint for the persistent query service."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import ArtifactBundle
from .encoders import FixtureTextEncoder, Siglip2TextEncoder
from .http import serve
from .met_artifacts import MetKeywordArtifacts
from .met_client import SEARCH_MODES, SqliteMetClient
from .met_service import MetKeywordConfig, MetKeywordSearchService
from .prompting import PromptEnsemble
from .service import SearchService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-faiss", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--siglip2", action="store_true", help="load the local SigLIP 2 text tower")
    mode.add_argument("--fixture-vectors", type=Path, help="deterministic fixture vector JSON")
    mode.add_argument(
        "--met-keyword",
        action="store_true",
        help="use the local Met SQLite FTS5 index and metadata-frequency aggregation",
    )
    parser.add_argument("--met-search-mode", choices=sorted(SEARCH_MODES), default="broad")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="allow one-time model provisioning when the pinned revision is not cached",
    )
    parser.add_argument(
        "--prompt-template",
        action="append",
        help="versioned prompt template containing {query}; repeat for an ensemble",
    )
    parser.add_argument("--prompt-version", default="art-concept-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.met_keyword:
        met_artifacts = MetKeywordArtifacts.load(args.artifacts)
        service = MetKeywordSearchService(
            met_artifacts,
            SqliteMetClient(met_artifacts.keyword_index_path),
            config=MetKeywordConfig(search_mode=args.met_search_mode),
        )
    else:
        artifacts = ArtifactBundle.load(args.artifacts)
        if args.fixture_vectors:
            text_encoder = FixtureTextEncoder.from_json(args.fixture_vectors)
        else:
            text_encoder = Siglip2TextEncoder(
                artifacts.model_id,
                revision=artifacts.model_version,
                local_files_only=not args.allow_model_download,
            )
        prompts = PromptEnsemble(
            version=args.prompt_version,
            templates=tuple(args.prompt_template) if args.prompt_template else PromptEnsemble().templates,
        )
        service = SearchService(
            artifacts,
            text_encoder,
            prompt_ensemble=prompts,
            prefer_faiss=not args.no_faiss,
        )
    try:
        serve(service, args.host, args.port)
    except KeyboardInterrupt:
        return 130
    finally:
        close = getattr(service, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
