"""Exact retrieval, concentration-lift aggregation, diagnostics, and evidence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

import numpy as np

from .artifacts import ArtifactBundle, ResolvedCorpus
from .cache import InMemorySeriesCache, series_cache_key
from .encoders import TextEncoder
from .index import ExactIndex, create_exact_index
from .models import (
    BinJSON,
    DiagnosticsJSON,
    EvidenceCardJSON,
    EvidenceSlicesJSON,
    PointJSON,
    QueryTerm,
    SearchRequest,
    SearchResponse,
    SeriesJSON,
)
from .parsing import parse_query
from .prompting import EncodedSeries, PromptEnsemble


SCHEMA_VERSION = "mnemosyne.search.v1"
METRIC_ID = "global-top-percentile-concentration-lift"


@dataclass(frozen=True)
class SearchConfig:
    percentile: float = 0.01
    metric_version: str = "v1-p01"
    minimum_denominator: float = 20.0
    minimum_standardized_separation: float = 1.0
    minimum_prompt_jaccard: float = 0.25
    control_sample_size: int = 128
    evidence_per_slice: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.percentile <= 1:
            raise ValueError("percentile must be in (0, 1]")
        if self.control_sample_size < 1 or self.evidence_per_slice < 1:
            raise ValueError("sample sizes must be positive")


@dataclass(frozen=True)
class SeriesComputation:
    cache_key: str
    query_vector: np.ndarray
    hit_indices: np.ndarray
    hit_scores: np.ndarray
    k: int
    threshold: float
    low_signal: bool
    diagnostics: DiagnosticsJSON
    points: tuple[PointJSON, ...]


def _stable_sample(rows: Sequence[int] | np.ndarray, count: int, salt: str) -> np.ndarray:
    ranked = sorted(
        (int(row) for row in rows),
        key=lambda row: hashlib.sha256(f"{salt}:{row}".encode("utf-8")).digest(),
    )
    return np.asarray(ranked[:count], dtype=np.int64)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set, right_set = set(map(int, left)), set(map(int, right))
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


class SearchService:
    def __init__(
        self,
        artifacts: ArtifactBundle,
        encoder: TextEncoder,
        *,
        prompt_ensemble: PromptEnsemble | None = None,
        config: SearchConfig | None = None,
        index: ExactIndex | None = None,
        cache: InMemorySeriesCache[SeriesComputation] | None = None,
        prefer_faiss: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.encoder = encoder
        self.prompt_ensemble = prompt_ensemble or PromptEnsemble()
        self.config = config or SearchConfig()
        self.index = index or create_exact_index(
            artifacts.embeddings,
            prefer_faiss=prefer_faiss,
            faiss_index_path=artifacts.faiss_index_path,
        )
        self.cache = cache or InMemorySeriesCache()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if encoder.model_id != artifacts.model_id or encoder.model_version != artifacts.model_version:
            raise ValueError(
                "text encoder model id/version must match the offline image-embedding artifact"
            )

    def search(self, request: SearchRequest) -> SearchResponse:
        terms = parse_query(request.query)
        corpus = self.artifacts.resolve_corpus(request.corpus_view, request.filters)
        computations = self._series(terms, corpus)

        selected_index = 0
        if request.selected_query_id is not None:
            matching = [i for i, term in enumerate(terms) if term.id == request.selected_query_id]
            if not matching:
                raise ValueError("selectedQueryId does not name a parsed query series")
            selected_index = matching[0]
        selected_term = terms[selected_index]
        selected_computation = computations[selected_index]
        bin_index = self._selected_bin_index(
            request.selected_bin_key, selected_computation, corpus
        )

        bins = self._bin_payload(corpus)
        series = [self._series_payload(term, result) for term, result in zip(terms, computations)]
        evidence = self._evidence_payload(
            selected_term, selected_computation, corpus, bin_index
        )
        warnings: list[str] = []
        for term, result in zip(terms, computations):
            if result.low_signal:
                warnings.append(f"Low-signal result for {term.label!r}; inspect evidence before interpreting it.")
        sparse_labels = [
            self.artifacts.bins[i].label
            for i, denominator in enumerate(corpus.denominators)
            if denominator < self.config.minimum_denominator
        ]
        if sparse_labels:
            warnings.append(
                f"{len(sparse_labels)} bin(s) are below the minimum denominator of "
                f"{self.config.minimum_denominator:g}."
            )

        return {
            "schemaVersion": SCHEMA_VERSION,
            "queries": [
                {"id": term.id, "label": term.label, "normalized": term.normalized}
                for term in terms
            ],
            "corpus": {
                "id": self.artifacts.corpus_id,
                "version": self.artifacts.corpus_version,
                "label": self.artifacts.corpus_label,
                "count": len(corpus.row_ids),
                "countingUnit": self.artifacts.counting_unit,
                "view": corpus.view,
                "filters": {key: list(values) for key, values in corpus.filters.items()},
            },
            "model": {
                "id": self.artifacts.model_id,
                "version": self.artifacts.model_version,
                "promptTemplateVersion": self.prompt_ensemble.version,
            },
            "metric": {
                "id": METRIC_ID,
                "version": self.config.metric_version,
                "label": f"Global top-{self.config.percentile * 100:g}% concentration lift",
                "percentile": self.config.percentile,
                "unit": "lift",
            },
            "bins": bins,
            "series": series,
            "selectedEvidence": evidence,
            "warnings": warnings,
            "generatedAt": self.clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _series(
        self, terms: Sequence[QueryTerm], corpus: ResolvedCorpus
    ) -> list[SeriesComputation]:
        keys = [self._cache_key(term, corpus) for term in terms]
        results: list[SeriesComputation | None] = [self.cache.get(key) for key in keys]
        missing_positions = [index for index, result in enumerate(results) if result is None]
        if missing_positions:
            missing_terms = [terms[index] for index in missing_positions]
            encoded = self.prompt_ensemble.encode(missing_terms, self.encoder)
            vectors = np.stack([item.combined for item in encoded]).astype(np.float32)
            k = max(1, math.ceil(self.config.percentile * len(corpus.row_ids)))
            hits = self.index.search(vectors, k, eligible_indices=self._index_filter(corpus))
            for batch_index, original_position in enumerate(missing_positions):
                result = self._aggregate_one(
                    missing_terms[batch_index],
                    encoded[batch_index],
                    hits.indices[batch_index],
                    hits.scores[batch_index],
                    corpus,
                    keys[original_position],
                )
                self.cache.put(keys[original_position], result)
                results[original_position] = result
        return [result for result in results if result is not None]

    def _cache_key(self, term: QueryTerm, corpus: ResolvedCorpus) -> str:
        return series_cache_key(
            {
                "query": term.normalized,
                "corpusId": self.artifacts.corpus_id,
                "corpusVersion": self.artifacts.corpus_version,
                "view": corpus.view,
                "filters": corpus.filters,
                "modelId": self.artifacts.model_id,
                "modelVersion": self.artifacts.model_version,
                "promptTemplateVersion": self.prompt_ensemble.version,
                "metricId": METRIC_ID,
                "metricVersion": self.config.metric_version,
                "percentile": self.config.percentile,
                "countingUnit": self.artifacts.counting_unit,
            }
        )

    def _aggregate_one(
        self,
        term: QueryTerm,
        encoded: EncodedSeries,
        hit_indices: np.ndarray,
        hit_scores: np.ndarray,
        corpus: ResolvedCorpus,
        cache_key: str,
    ) -> SeriesComputation:
        hit_mass = self.artifacts.date_weights.aggregate(hit_indices)
        points: list[PointJSON] = []
        hit_set = set(map(int, hit_indices))
        for bin_index, bin_item in enumerate(self.artifacts.bins):
            denominator = float(corpus.denominators[bin_index])
            share = float(hit_mass[bin_index] / denominator) if denominator else 0.0
            lift = share / self.config.percentile if denominator else 0.0
            contributors = [
                row
                for row in hit_set
                if self.artifacts.date_weights.weight(row, bin_index) > 0
            ]
            clusters = {
                self.artifacts.metadata[row].get("visualClusterId") or f"row-{row}"
                for row in contributors
            }
            points.append(
                {
                    "binKey": bin_item.key,
                    "value": lift,
                    "share": share,
                    "lift": lift,
                    "hitMass": float(hit_mass[bin_index]),
                    "objectCount": len(contributors),
                    "clusterCount": len(clusters),
                }
            )

        diagnostics, low_signal = self._diagnostics(
            term, encoded, hit_indices, hit_scores, corpus
        )
        return SeriesComputation(
            cache_key=cache_key,
            query_vector=encoded.combined,
            hit_indices=np.asarray(hit_indices, dtype=np.int64),
            hit_scores=np.asarray(hit_scores, dtype=np.float32),
            k=len(hit_indices),
            threshold=float(hit_scores[-1]),
            low_signal=low_signal,
            diagnostics=diagnostics,
            points=tuple(points),
        )

    def _diagnostics(
        self,
        term: QueryTerm,
        encoded: EncodedSeries,
        hit_indices: np.ndarray,
        hit_scores: np.ndarray,
        corpus: ResolvedCorpus,
    ) -> tuple[DiagnosticsJSON, bool]:
        hit_set = set(map(int, hit_indices))
        non_hits = np.asarray(
            [int(row) for row in corpus.row_ids if int(row) not in hit_set], dtype=np.int64
        )
        controls = _stable_sample(
            non_hits,
            min(self.config.control_sample_size, len(non_hits)),
            f"controls:{self.artifacts.corpus_version}:{term.normalized}",
        )
        if len(controls):
            control_scores = self.index.score(encoded.combined, controls)
            control_mean = float(control_scores.mean())
            control_std = float(control_scores.std())
            gap = float(hit_scores.mean()) - control_mean
            separation = gap / control_std if control_std > 1e-8 else (999.0 if gap > 0 else 0.0)
        else:
            control_mean = float(hit_scores.mean())
            control_std = 0.0
            separation = 0.0

        prompt_jaccards: list[float] = []
        for prompt_vector in encoded.prompt_vectors:
            prompt_hits = self.index.search(
                prompt_vector.reshape(1, -1),
                len(hit_indices),
                eligible_indices=self._index_filter(corpus),
            ).indices[0]
            prompt_jaccards.append(_jaccard(hit_indices, prompt_hits))
        prompt_jaccard = float(np.mean(prompt_jaccards)) if prompt_jaccards else 1.0
        reasons: list[str] = []
        if separation < self.config.minimum_standardized_separation:
            reasons.append("top matches are not well separated from deterministic controls")
        if prompt_jaccard < self.config.minimum_prompt_jaccard:
            reasons.append("prompt variants produce unstable top-match sets")
        return (
            {
                "standardizedSeparation": float(separation),
                "controlMean": control_mean,
                "controlStdDev": control_std,
                "promptTopKJaccard": prompt_jaccard,
                "reasons": reasons,
            },
            bool(reasons),
        )

    def _index_filter(self, corpus: ResolvedCorpus) -> np.ndarray | None:
        all_rows = np.arange(self.artifacts.embeddings.shape[0], dtype=np.int64)
        return None if np.array_equal(corpus.row_ids, all_rows) else corpus.row_ids

    def _bin_payload(self, corpus: ResolvedCorpus) -> list[BinJSON]:
        payload: list[BinJSON] = []
        for bin_index, item in enumerate(self.artifacts.bins):
            rows, _ = self.artifacts.date_weights.rows_for_bin(corpus.row_ids, bin_index)
            clusters = {
                self.artifacts.metadata[int(row)].get("visualClusterId") or f"row-{row}"
                for row in rows
            }
            denominator = float(corpus.denominators[bin_index])
            payload.append(
                {
                    "key": item.key,
                    "label": item.label,
                    "start": item.start,
                    "end": item.end,
                    "denominator": denominator,
                    "objectCount": len(rows),
                    "clusterCount": len(clusters),
                    "belowMinimumDenominator": denominator < self.config.minimum_denominator,
                }
            )
        return payload

    @staticmethod
    def _series_payload(term: QueryTerm, result: SeriesComputation) -> SeriesJSON:
        return {
            "queryId": term.id,
            "k": result.k,
            "threshold": result.threshold,
            "lowSignal": result.low_signal,
            "diagnostics": result.diagnostics,
            "points": list(result.points),
            "cacheKey": result.cache_key,
        }

    def _selected_bin_index(
        self,
        selected_bin_key: str | None,
        computation: SeriesComputation,
        corpus: ResolvedCorpus,
    ) -> int:
        if selected_bin_key is not None:
            matches = [i for i, item in enumerate(self.artifacts.bins) if item.key == selected_bin_key]
            if not matches:
                raise ValueError("selectedBinKey does not name a timeline bin")
            return matches[0]
        eligible = [
            (float(point["lift"]), -index, index)
            for index, point in enumerate(computation.points)
            if corpus.denominators[index] >= self.config.minimum_denominator
        ]
        if not eligible:
            eligible = [
                (float(point["lift"]), -index, index)
                for index, point in enumerate(computation.points)
            ]
        return max(eligible)[2]

    def _evidence_payload(
        self,
        term: QueryTerm,
        computation: SeriesComputation,
        corpus: ResolvedCorpus,
        bin_index: int,
    ) -> dict:
        weights = self.artifacts.date_weights
        period_rows, _ = weights.rows_for_bin(corpus.row_ids, bin_index)
        scores = self.index.score(computation.query_vector, period_rows)
        score_by_row = {int(row): float(score) for row, score in zip(period_rows, scores)}
        hit_set = set(map(int, computation.hit_indices))
        contributors = np.asarray(
            [int(row) for row in period_rows if int(row) in hit_set], dtype=np.int64
        )
        non_contributors = np.asarray(
            [int(row) for row in period_rows if int(row) not in hit_set], dtype=np.int64
        )
        count = self.config.evidence_per_slice

        strongest = sorted(contributors, key=lambda row: (-score_by_row[int(row)], int(row)))[:count]
        contributor_by_score = sorted(contributors, key=lambda row: (score_by_row[int(row)], int(row)))
        if contributor_by_score:
            middle = (len(contributor_by_score) - 1) / 2
            representative = sorted(
                contributor_by_score,
                key=lambda row: (
                    abs(contributor_by_score.index(row) - middle),
                    int(row),
                ),
            )[:count]
        else:
            representative = []
        borderline = sorted(
            period_rows,
            key=lambda row: (
                abs(score_by_row[int(row)] - computation.threshold),
                int(row),
            ),
        )[:count]
        random_contributors = _stable_sample(
            contributors,
            min(count, len(contributors)),
            f"contributor:{term.normalized}:{self.artifacts.bins[bin_index].key}",
        )
        best_non_contributors = sorted(
            non_contributors, key=lambda row: (-score_by_row[int(row)], int(row))
        )[:count]
        random_denominator = _stable_sample(
            period_rows,
            min(count, len(period_rows)),
            f"denominator:{term.normalized}:{self.artifacts.bins[bin_index].key}",
        )

        def cards(rows: Sequence[int] | np.ndarray) -> list[EvidenceCardJSON]:
            return [
                self._evidence_card(
                    int(row),
                    score_by_row[int(row)],
                    weights.weight(int(row), bin_index),
                    int(row) in hit_set,
                )
                for row in rows
            ]

        slices: EvidenceSlicesJSON = {
            "strongest": cards(strongest),
            "representative": cards(representative),
            "borderline": cards(borderline),
            "randomContributors": cards(random_contributors),
            "bestNonContributors": cards(best_non_contributors),
            "randomDenominator": cards(random_denominator),
        }
        return {
            "queryId": term.id,
            "binKey": self.artifacts.bins[bin_index].key,
            "slices": slices,
        }

    def _evidence_card(
        self, row: int, score: float, weight: float, contributor: bool
    ) -> EvidenceCardJSON:
        item = self.artifacts.metadata[row]
        return {
            "artworkId": str(item["artworkId"]),
            "physicalObjectId": str(item.get("physicalObjectId", item["artworkId"])),
            "visualClusterId": str(item.get("visualClusterId", item["artworkId"])),
            "title": str(item.get("title", "Untitled")),
            "artist": str(item.get("artist", "Unknown artist")),
            "institution": str(item.get("institution", "Unknown institution")),
            "sourceRecordUrl": str(item.get("sourceRecordUrl", "")),
            "imageUrl": str(item.get("imageUrl", "")),
            "dateDisplay": str(item.get("dateDisplay", "Unknown date")),
            "dateStart": item.get("dateStart"),
            "dateEnd": item.get("dateEnd"),
            "dateQualifier": str(item.get("dateQualifier", "exact")),
            "rawScore": float(score),
            "contributionWeight": float(weight),
            "contributor": contributor,
            "metadataLicense": str(item.get("metadataLicense", "")),
            "imageRightsUri": str(item.get("imageRightsUri", "")),
            "creditLine": str(item.get("creditLine", "")),
            "publicDomain": bool(item.get("publicDomain", False)),
        }
