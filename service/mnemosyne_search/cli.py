"""Command line entrypoint for the persistent query service."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .artifacts import ArtifactBundle
from .encoders import FixtureTextEncoder, Siglip2TextEncoder
from .http import serve
from .met_artifacts import MetKeywordArtifacts
from .met_client import HttpMetClient, SEARCH_MODES, SqliteMetClient
from .met_service import MetKeywordConfig, MetKeywordSearchService
from .prompting import PromptEnsemble
from .service import SearchConfig, SearchService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    index = parser.add_mutually_exclusive_group()
    index.add_argument("--no-faiss", action="store_true")
    index.add_argument(
        "--force-faiss",
        action="store_true",
        help="force FAISS on macOS after verifying wheel OpenMP compatibility",
    )
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
        "--met-offline-evidence",
        action="store_true",
        help="do not fetch image metadata for selected Met evidence cards",
    )
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
    parser.add_argument("--prompt-version", default="art-concept-fixed64-v2")
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.01,
        help="retrieval candidate fraction evaluated before the evidence cutoff (default: 0.01)",
    )
    parser.add_argument(
        "--evidence-percentile",
        type=float,
        default=0.001,
        help="stricter corpus fraction eligible for evidence cards (default: 0.001)",
    )
    parser.add_argument(
        "--minimum-evidence-score",
        type=float,
        default=0.125,
        help="minimum cosine similarity for visible embedding evidence (default: 0.125)",
    )
    parser.add_argument(
        "--minimum-evidence-clusters",
        type=int,
        default=3,
        help="preferred distinct visual clusters for the automatically selected period",
    )
    parser.add_argument(
        "--minimum-bin-evidence-clusters",
        type=int,
        default=2,
        help="minimum distinct visual clusters required to plot a period",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="embedding text-tower device; auto prefers CUDA, then Apple MPS",
    )
    return parser


def _prefer_faiss(args: argparse.Namespace, platform: str = sys.platform) -> bool:
    if args.force_faiss:
        return True
    if args.no_faiss:
        return False
    # Common pip wheels for PyTorch and FAISS load duplicate libomp runtimes on
    # macOS and can terminate the first query. NumPy/BLAS remains exact and was
    # faster than accepting that unsafe runtime configuration on Apple Silicon.
    return not (platform == "darwin" and args.siglip2)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.met_keyword:
        met_artifacts = MetKeywordArtifacts.load(args.artifacts)
        search_client = SqliteMetClient(met_artifacts.keyword_index_path)
        service = MetKeywordSearchService(
            met_artifacts,
            search_client,
            evidence_client=search_client if args.met_offline_evidence else HttpMetClient(),
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
                device=args.device,
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
            config=SearchConfig(
                percentile=args.percentile,
                evidence_percentile=args.evidence_percentile,
                minimum_evidence_score=args.minimum_evidence_score,
                minimum_evidence_clusters=args.minimum_evidence_clusters,
                minimum_bin_evidence_clusters=args.minimum_bin_evidence_clusters,
            ),
            prefer_faiss=_prefer_faiss(args),
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
