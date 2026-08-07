"""Met catalogue keyword retrieval with local date-frequency aggregation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

import numpy as np

from .artifacts import ResolvedCorpus
from .cache import InMemorySeriesCache, series_cache_key
from .met_artifacts import MetKeywordArtifacts
from .met_client import MetClient, SEARCH_MODES
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


SCHEMA_VERSION = "mnemosyne.search.v1"
METRIC_ID = "met-metadata-frequency"
METADATA_WARNING = (
    "Met metadata frequency measures catalogue keyword matches among eligible Met objects, "
    "not visual prevalence in the images."
)


@dataclass(frozen=True)
class MetKeywordConfig:
    search_mode: str = "broad"
    metric_version: str = "v1"
    minimum_denominator: float = 20.0
    evidence_count: int = 5

    def __post_init__(self) -> None:
        if self.search_mode not in SEARCH_MODES:
            raise ValueError(f"unsupported Met search mode: {self.search_mode}")
        if self.minimum_denominator < 0 or self.evidence_count < 1:
            raise ValueError("Met denominator threshold must be non-negative and evidence count positive")


@dataclass(frozen=True)
class MetSeriesComputation:
    cache_key: str
    match_rows: np.ndarray
    total_matches: int
    points: tuple[PointJSON, ...]


def _stable_sample(rows: Sequence[int] | np.ndarray, count: int, salt: str) -> list[int]:
    candidates = np.asarray(rows, dtype=np.int64)
    if count >= len(candidates):
        return candidates.tolist()
    seed = int.from_bytes(hashlib.sha256(salt.encode("utf-8")).digest()[:16], "big")
    positions = np.random.default_rng(seed).choice(len(candidates), size=count, replace=False)
    return candidates[positions].tolist()


class MetKeywordSearchService:
    def __init__(
        self,
        artifacts: MetKeywordArtifacts,
        client: MetClient,
        *,
        evidence_client: MetClient | None = None,
        config: MetKeywordConfig | None = None,
        cache: InMemorySeriesCache[MetSeriesComputation] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.client = client
        self.evidence_client = evidence_client or client
        self.config = config or MetKeywordConfig()
        self.cache = cache or InMemorySeriesCache()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def close(self) -> None:
        closed: set[int] = set()
        for client in (self.client, self.evidence_client):
            if id(client) in closed:
                continue
            closed.add(id(client))
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def health(self) -> dict[str, object]:
        backend = str(getattr(self.client, "backend", "met-keyword-provider"))
        return {
            "status": "ok",
            "mode": "met-keyword",
            "corpusVersion": self.artifacts.corpus_version,
            "modelVersion": "met-local-fts5-v1",
            "indexBackend": f"{backend}+local-sparse-date-weights",
            "searchMode": self.config.search_mode,
        }

    def search(self, request: SearchRequest) -> SearchResponse:
        terms = parse_query(request.query)
        corpus = self.artifacts.resolve_corpus(request.corpus_view, request.filters)
        eligible_rows = frozenset(map(int, corpus.row_ids))
        computations = [self._series(term, corpus, eligible_rows) for term in terms]

        selected_index = 0
        if request.selected_query_id is not None:
            matches = [index for index, term in enumerate(terms) if term.id == request.selected_query_id]
            if not matches:
                raise ValueError("selectedQueryId does not name a parsed query series")
            selected_index = matches[0]
        selected_bin_index = self._selected_bin_index(
            request.selected_bin_key, computations[selected_index], corpus
        )

        warnings = [METADATA_WARNING]
        sparse_count = sum(
            denominator < self.config.minimum_denominator for denominator in corpus.denominators
        )
        if sparse_count:
            warnings.append(
                f"{sparse_count} bin(s) are below the minimum denominator of "
                f"{self.config.minimum_denominator:g}."
            )
        for term, computation in zip(terms, computations):
            if len(computation.match_rows) == 0:
                warnings.append(f"No eligible corpus matches were found for {term.label!r}.")

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
                "id": "met-local-fts5-keyword",
                "version": "v1",
                "promptTemplateVersion": "none",
            },
            "metric": {
                "id": METRIC_ID,
                "version": self.config.metric_version,
                "label": "Met metadata frequency",
                "percentile": None,
                "unit": "frequency",
                "description": "Date-weighted matching objects divided by all eligible objects in each bin.",
            },
            "bins": self._bin_payload(corpus),
            "series": [
                self._series_payload(term, computation)
                for term, computation in zip(terms, computations)
            ],
            "selectedEvidence": self._evidence_payload(
                terms[selected_index], computations[selected_index], selected_bin_index
            ),
            "warnings": warnings,
            "generatedAt": self.clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _series(
        self,
        term: QueryTerm,
        corpus: ResolvedCorpus,
        eligible_rows: frozenset[int],
    ) -> MetSeriesComputation:
        key = series_cache_key(
            {
                "query": term.normalized,
                "corpusId": self.artifacts.corpus_id,
                "corpusVersion": self.artifacts.corpus_version,
                "view": corpus.view,
                "filters": corpus.filters,
                "searchMode": self.config.search_mode,
                "metricId": METRIC_ID,
                "metricVersion": self.config.metric_version,
                "countingUnit": self.artifacts.counting_unit,
            }
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        object_ids = self.client.search(term.normalized, self.config.search_mode)
        rows = np.asarray(
            sorted(
                {
                    row
                    for object_id in object_ids
                    if (row := self.artifacts.source_id_to_row.get(object_id)) is not None
                    and row in eligible_rows
                }
            ),
            dtype=np.int64,
        )
        hit_mass = self.artifacts.date_weights.aggregate(rows)
        object_counts, cluster_counts = self.artifacts.date_weights.membership_counts(
            rows, self.artifacts.cluster_ids
        )
        points: list[PointJSON] = []
        for bin_index, bin_item in enumerate(self.artifacts.bins):
            denominator = float(corpus.denominators[bin_index])
            share = float(hit_mass[bin_index] / denominator) if denominator else 0.0
            points.append(
                {
                    "binKey": bin_item.key,
                    "value": share,
                    "share": share,
                    "lift": None,
                    "hitMass": float(hit_mass[bin_index]),
                    "objectCount": int(object_counts[bin_index]),
                    "clusterCount": int(cluster_counts[bin_index]),
                }
            )
        computation = MetSeriesComputation(
            cache_key=key,
            match_rows=rows,
            total_matches=len(object_ids),
            points=tuple(points),
        )
        self.cache.put(key, computation)
        return computation

    def _bin_payload(self, corpus: ResolvedCorpus) -> list[BinJSON]:
        payload: list[BinJSON] = []
        for index, item in enumerate(self.artifacts.bins):
            denominator = float(corpus.denominators[index])
            payload.append(
                {
                    "key": item.key,
                    "label": item.label,
                    "start": item.start,
                    "end": item.end,
                    "denominator": denominator,
                    "objectCount": int(corpus.object_counts[index]),
                    "clusterCount": int(corpus.cluster_counts[index]),
                    "belowMinimumDenominator": denominator < self.config.minimum_denominator,
                }
            )
        return payload

    @staticmethod
    def _series_payload(term: QueryTerm, result: MetSeriesComputation) -> SeriesJSON:
        diagnostics: DiagnosticsJSON = {
            "standardizedSeparation": None,
            "controlMean": None,
            "controlStdDev": None,
            "promptTopKJaccard": None,
            "reasons": [],
        }
        return {
            "queryId": term.id,
            "k": len(result.match_rows),
            "threshold": None,
            "lowSignal": None,
            "diagnostics": diagnostics,
            "points": list(result.points),
            "cacheKey": result.cache_key,
            "totalMatches": result.total_matches,
        }

    def _selected_bin_index(
        self,
        selected_bin_key: str | None,
        computation: MetSeriesComputation,
        corpus: ResolvedCorpus,
    ) -> int:
        if selected_bin_key is not None:
            matches = [
                index for index, item in enumerate(self.artifacts.bins) if item.key == selected_bin_key
            ]
            if not matches:
                raise ValueError("selectedBinKey does not name a timeline bin")
            return matches[0]
        eligible = [
            (float(point["value"]), -index, index)
            for index, point in enumerate(computation.points)
            if corpus.denominators[index] >= self.config.minimum_denominator
        ]
        if not eligible:
            eligible = [
                (float(point["value"]), -index, index)
                for index, point in enumerate(computation.points)
            ]
        return max(eligible)[2]

    def _evidence_payload(
        self,
        term: QueryTerm,
        computation: MetSeriesComputation,
        bin_index: int,
    ) -> dict:
        contributors, _ = self.artifacts.date_weights.rows_for_bin(
            computation.match_rows, bin_index
        )
        image_contributors = np.asarray(
            [
                int(row)
                for row in contributors
                if self.artifacts.metadata[int(row)].get("publicDomain", False)
                and self.artifacts.metadata[int(row)].get("imageAvailable", False)
            ],
            dtype=np.int64,
        )
        public_contributors = np.asarray(
            [
                int(row)
                for row in contributors
                if self.artifacts.metadata[int(row)].get("publicDomain", False)
            ],
            dtype=np.int64,
        )
        evidence_pool = (
            image_contributors
            if len(image_contributors)
            else public_contributors
            if len(public_contributors)
            else contributors
        )
        selected = _stable_sample(
            evidence_pool,
            min(self.config.evidence_count, len(evidence_pool)),
            f"met-evidence:{term.normalized}:{self.artifacts.bins[bin_index].key}",
        )
        cards = [self._evidence_card(row, bin_index) for row in selected]
        slices: EvidenceSlicesJSON = {
            "strongest": [],
            "representative": [],
            "borderline": [],
            "randomContributors": cards,
            "bestNonContributors": [],
            "randomDenominator": [],
        }
        return {
            "queryId": term.id,
            "binKey": self.artifacts.bins[bin_index].key,
            "slices": slices,
        }

    def _evidence_card(self, row: int, bin_index: int) -> EvidenceCardJSON:
        item = self.artifacts.metadata[row]
        object_id = int(item["sourceId"])
        try:
            detail = self.evidence_client.object(object_id) or {}
        except RuntimeError:
            detail = {}
        public_domain = bool(item.get("publicDomain", False)) and bool(
            detail.get("isPublicDomain", True)
        )
        return {
            "artworkId": str(item["artworkId"]),
            "physicalObjectId": str(item.get("physicalObjectId", item["artworkId"])),
            "visualClusterId": str(item.get("visualClusterId", item["artworkId"])),
            "title": str(detail.get("title") or item.get("title") or "Untitled"),
            "artist": str(detail.get("artistDisplayName") or item.get("artist") or "Unknown artist"),
            "institution": "The Metropolitan Museum of Art",
            "sourceRecordUrl": str(detail.get("objectURL") or item.get("sourceRecordUrl", "")),
            "imageUrl": str(detail.get("primaryImageSmall") or "") if public_domain else "",
            "dateDisplay": str(detail.get("objectDate") or item.get("dateDisplay") or "Unknown date"),
            "dateStart": item.get("dateStart"),
            "dateEnd": item.get("dateEnd"),
            "dateQualifier": str(item.get("dateQualifier", "range")),
            "rawScore": None,
            "contributionWeight": self.artifacts.date_weights.weight(row, bin_index),
            "contributor": True,
            "metadataLicense": str(item.get("metadataLicense", "")),
            "imageRightsUri": str(item.get("imageRightsUri", "")),
            "creditLine": str(detail.get("creditLine") or item.get("creditLine", "")),
            "publicDomain": public_domain,
        }
