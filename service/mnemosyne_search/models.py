"""Typed request and versioned JSON response contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NotRequired, TypedDict


@dataclass(frozen=True)
class QueryTerm:
    id: str
    label: str
    normalized: str


@dataclass(frozen=True)
class SearchRequest:
    query: str
    selected_query_id: str | None = None
    selected_bin_key: str | None = None
    corpus_view: str = "all"
    filters: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


class QueryJSON(TypedDict):
    id: str
    label: str
    normalized: str


class CorpusJSON(TypedDict):
    id: str
    version: str
    label: str
    count: int
    countingUnit: str
    view: str
    filters: dict[str, list[str]]


class ModelJSON(TypedDict):
    id: str
    version: str
    promptTemplateVersion: str


class MetricJSON(TypedDict):
    id: str
    version: str
    label: str
    percentile: float | None
    unit: str
    description: NotRequired[str]


class BinJSON(TypedDict):
    key: str
    label: str
    start: int
    end: int
    denominator: float
    objectCount: int
    clusterCount: int
    belowMinimumDenominator: bool


class PointJSON(TypedDict):
    binKey: str
    value: float
    share: float | None
    lift: float | None
    hitMass: float
    objectCount: int
    clusterCount: int


class DiagnosticsJSON(TypedDict):
    standardizedSeparation: float | None
    controlMean: float | None
    controlStdDev: float | None
    promptTopKJaccard: float | None
    reasons: list[str]


class SeriesJSON(TypedDict):
    queryId: str
    k: int
    threshold: float | None
    candidateK: NotRequired[int]
    candidateThreshold: NotRequired[float | None]
    lowSignal: bool | None
    diagnostics: DiagnosticsJSON
    points: list[PointJSON]
    suppressedBinKeys: NotRequired[list[str]]
    cacheKey: str
    totalMatches: NotRequired[int]


class EvidenceCardJSON(TypedDict):
    artworkId: str
    physicalObjectId: str
    visualClusterId: str
    title: str
    artist: str
    institution: str
    sourceRecordUrl: str
    imageUrl: str
    dateDisplay: str
    dateStart: NotRequired[int | None]
    dateEnd: NotRequired[int | None]
    dateQualifier: str
    rawScore: float | None
    contributionWeight: float
    contributor: bool
    metadataLicense: str
    imageRightsUri: str
    creditLine: str
    publicDomain: bool


class EvidenceSlicesJSON(TypedDict):
    strongest: list[EvidenceCardJSON]
    representative: list[EvidenceCardJSON]
    borderline: list[EvidenceCardJSON]
    randomContributors: list[EvidenceCardJSON]
    bestNonContributors: list[EvidenceCardJSON]
    randomDenominator: list[EvidenceCardJSON]


class SelectedEvidenceJSON(TypedDict):
    queryId: str
    binKey: str
    percentile: NotRequired[float | None]
    threshold: NotRequired[float | None]
    contributorCount: NotRequired[int]
    slices: EvidenceSlicesJSON


class SearchResponse(TypedDict):
    schemaVersion: str
    queries: list[QueryJSON]
    corpus: CorpusJSON
    model: ModelJSON
    metric: MetricJSON
    bins: list[BinJSON]
    series: list[SeriesJSON]
    selectedEvidence: SelectedEvidenceJSON | None
    warnings: list[str]
    generatedAt: str


class EvidenceResponse(TypedDict):
    schemaVersion: str
    selectedEvidence: SelectedEvidenceJSON | None
    generatedAt: str


JSONValue = str | int | float | bool | None | list[Any] | dict[str, Any]
