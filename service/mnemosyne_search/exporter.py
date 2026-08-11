"""Deterministic static concept-catalog export from an existing embedding bundle.

This module never imports the offline image pipeline.  It encodes text with the
manifest-pinned text tower, searches the completed image matrix, and publishes
only compact series/evidence JSON suitable for immutable CDN delivery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifacts import ArtifactBundle, ResolvedCorpus
from .cache import InMemorySeriesCache
from .encoders import TextEncoder
from .models import QueryTerm
from .parsing import normalize_term
from .prompting import PromptEnsemble
from .service import METRIC_ID, SearchConfig, SearchService, SeriesComputation


SOURCE_SCHEMA_VERSION = "mnemosyne.concepts.source.v1"
EXPORT_SCHEMA_VERSION = "mnemosyne.concept-catalog.v1"
POINTER_SCHEMA_VERSION = "mnemosyne.concept-catalog-pointer.v1"
SERIES_SCHEMA_VERSION = "mnemosyne.concept-series.v1"
EVIDENCE_SCHEMA_VERSION = "mnemosyne.concept-evidence.v1"
BINS_SCHEMA_VERSION = "mnemosyne.concept-bins.v1"
CONCEPTS_SCHEMA_VERSION = "mnemosyne.concepts.v1"
CONCEPT_EXPORT_FAILURE_CODE = "concept-export-failed"
_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    aliases: tuple[str, ...] = ()
    category: str | None = None
    description: str | None = None

    @property
    def normalized(self) -> str:
        return normalize_term(self.label)

    def public_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "normalized": self.normalized,
            "aliases": list(self.aliases),
        }
        if self.category:
            payload["category"] = self.category
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class ConceptSource:
    version: str
    concepts: tuple[Concept, ...]


@dataclass(frozen=True)
class ExportStats:
    completed: int
    skipped: int
    failures: tuple[dict[str, str], ...]
    elapsed_seconds: float
    concepts_per_second: float


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, payload: Any) -> int:
    encoded = _canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return len(encoded)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_concept_source(path: str | Path) -> ConceptSource:
    payload = _read_json(Path(path))
    if payload.get("schemaVersion") != SOURCE_SCHEMA_VERSION:
        raise ValueError("concept source has an unsupported schemaVersion")
    version = str(payload.get("version", "")).strip()
    if not version:
        raise ValueError("concept source must declare a version")
    raw_concepts = payload.get("concepts")
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise ValueError("concept source must contain a non-empty concepts array")

    concepts: list[Concept] = []
    ids: set[str] = set()
    resolved_names: dict[str, str] = {}
    for raw in raw_concepts:
        if not isinstance(raw, dict):
            raise ValueError("every concept must be an object")
        concept_id = str(raw.get("id", "")).strip()
        label = str(raw.get("label", "")).strip()
        if not _ID_RE.fullmatch(concept_id):
            raise ValueError(f"invalid stable concept id: {concept_id!r}")
        if concept_id in ids:
            raise ValueError(f"duplicate concept id: {concept_id}")
        if not label:
            raise ValueError(f"concept {concept_id!r} has no label")
        raw_aliases = raw.get("aliases", [])
        if not isinstance(raw_aliases, list) or not all(
            isinstance(alias, str) for alias in raw_aliases
        ):
            raise ValueError(f"concept {concept_id!r} aliases must be strings")
        aliases = tuple(
            dict.fromkeys(
                alias.strip()
                for alias in raw_aliases
                if alias.strip() and normalize_term(alias) != normalize_term(label)
            )
        )
        concept = Concept(
            id=concept_id,
            label=label,
            aliases=aliases,
            category=str(raw.get("category", "")).strip() or None,
            description=str(raw.get("description", "")).strip() or None,
        )
        for name in (concept.label, *concept.aliases):
            normalized = normalize_term(name)
            owner = resolved_names.get(normalized)
            if owner is not None and owner != concept_id:
                raise ValueError(
                    f"concept name or alias {name!r} resolves to both {owner!r} "
                    f"and {concept_id!r}"
                )
            resolved_names[normalized] = concept_id
        ids.add(concept_id)
        concepts.append(concept)
    return ConceptSource(version=version, concepts=tuple(concepts))


def _manifest_path(root: Path) -> Path:
    model = root / "model-manifest.json"
    return model if model.exists() else root / "build-manifest.json"


def artifact_manifest_sha256(root: str | Path) -> str:
    return _sha256_bytes(_manifest_path(Path(root)).read_bytes())


def artifact_stat_fingerprint(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    manifest_path = _manifest_path(base)
    manifest = _read_json(manifest_path)
    entries: list[dict[str, Any]] = []
    for entry in manifest.get("artifacts", []):
        path = base / str(entry["path"])
        stat = path.stat()
        entries.append(
            {
                "path": str(entry["path"]),
                "bytes": stat.st_size,
                "mtimeNs": stat.st_mtime_ns,
                "declaredSha256": str(entry["sha256"]),
            }
        )
    return {
        "artifactManifestSha256": _sha256_bytes(manifest_path.read_bytes()),
        "entries": entries,
    }


def load_artifacts_for_export(
    root: str | Path,
    *,
    state_dir: str | Path,
    resume: bool,
) -> tuple[ArtifactBundle, bool]:
    """Load artifacts, reusing a local verification stamp when files are unchanged."""

    fingerprint = artifact_stat_fingerprint(root)
    state_path = Path(state_dir) / (
        f"artifact-{fingerprint['artifactManifestSha256']}.json"
    )
    verified_from_cache = False
    if resume and state_path.is_file():
        try:
            verified_from_cache = _read_json(state_path) == fingerprint
        except (OSError, ValueError, json.JSONDecodeError):
            verified_from_cache = False
    bundle = ArtifactBundle.load(root, verify_checksums=not verified_from_cache)
    if not verified_from_cache:
        _atomic_json(state_path, fingerprint)
    return bundle, verified_from_cache


def _source_fingerprint(
    artifacts: ArtifactBundle,
    artifact_manifest_hash: str,
    prompt_ensemble: PromptEnsemble,
    config: SearchConfig,
    encoder: TextEncoder,
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "exportSchemaVersion": EXPORT_SCHEMA_VERSION,
                "artifactManifestSha256": artifact_manifest_hash,
                "corpusId": artifacts.corpus_id,
                "corpusVersion": artifacts.corpus_version,
                "modelId": artifacts.model_id,
                "modelRevision": artifacts.model_version,
                "promptTemplateVersion": prompt_ensemble.version,
                "promptTemplates": list(prompt_ensemble.templates),
                "metricId": METRIC_ID,
                "searchConfig": asdict(config),
                "encoderRuntime": {
                    "implementation": (
                        f"{type(encoder).__module__}.{type(encoder).__qualname__}"
                    ),
                    "device": getattr(encoder, "device", "unspecified"),
                    "versions": getattr(encoder, "runtime_versions", {}),
                },
            }
        )
    )


def _concept_fingerprint(concept: Concept, source_fingerprint: str) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "sourceFingerprint": source_fingerprint,
                "concept": concept.public_payload(),
            }
        )
    )


def _term(concept: Concept) -> QueryTerm:
    return QueryTerm(
        id=f"concept:{concept.id}",
        label=concept.label,
        normalized=concept.normalized,
    )


def _bins_payload(
    service: SearchService, corpus: ResolvedCorpus, source_fingerprint: str
) -> dict[str, Any]:
    bins = service._bin_payload(corpus)
    return {
        "schemaVersion": BINS_SCHEMA_VERSION,
        "sourceFingerprint": source_fingerprint,
        "keys": [item["key"] for item in bins],
        "labels": [item["label"] for item in bins],
        "starts": [item["start"] for item in bins],
        "ends": [item["end"] for item in bins],
        "denominators": [item["denominator"] for item in bins],
        "objectCounts": [item["objectCount"] for item in bins],
        "clusterCounts": [item["clusterCount"] for item in bins],
        "unreliableIndices": [
            index
            for index, item in enumerate(bins)
            if item["belowMinimumDenominator"]
        ],
    }


def _series_payload(
    service: SearchService,
    term: QueryTerm,
    concept: Concept,
    computation: SeriesComputation,
    corpus: ResolvedCorpus,
    source_fingerprint: str,
    concept_fingerprint: str,
) -> dict[str, Any]:
    expanded = service._series_payload(term, computation, corpus)
    bin_index = {item.key: index for index, item in enumerate(service.artifacts.bins)}
    points = expanded["points"]
    point_indices = [bin_index[point["binKey"]] for point in points]
    default_index = service._selected_bin_index(None, computation, corpus)
    return {
        "schemaVersion": SERIES_SCHEMA_VERSION,
        "sourceFingerprint": source_fingerprint,
        "conceptFingerprint": concept_fingerprint,
        "conceptId": concept.id,
        "label": concept.label,
        "normalized": concept.normalized,
        "k": expanded["k"],
        "threshold": expanded["threshold"],
        "candidateK": expanded.get("candidateK"),
        "candidateThreshold": expanded.get("candidateThreshold"),
        "lowSignal": expanded["lowSignal"],
        "diagnostics": expanded["diagnostics"],
        "pointIndices": point_indices,
        "values": [point["value"] for point in points],
        "shares": [point["share"] for point in points],
        "hitMasses": [point["hitMass"] for point in points],
        "objectCounts": [point["objectCount"] for point in points],
        "clusterCounts": [point["clusterCount"] for point in points],
        "suppressedBinIndices": [
            bin_index[key] for key in expanded.get("suppressedBinKeys", [])
        ],
        "evidenceBinIndices": point_indices,
        "defaultEvidenceBinIndex": default_index,
    }


def _evidence_payload(
    service: SearchService,
    term: QueryTerm,
    concept: Concept,
    computation: SeriesComputation,
    corpus: ResolvedCorpus,
    bin_index: int,
    source_fingerprint: str,
    concept_fingerprint: str,
) -> dict[str, Any]:
    del corpus  # The qualified set already belongs to the resolved corpus.
    weights = service.artifacts.date_weights
    qualified_rows = computation.hit_indices[: computation.evidence_k]
    qualified_scores = computation.hit_scores[: computation.evidence_k]
    contributor_count = 0
    strongest_rows: list[tuple[int, float, float]] = []
    seen_clusters: set[str] = set()
    for raw_row, raw_score in zip(qualified_rows, qualified_scores):
        row = int(raw_row)
        weight = weights.weight(row, bin_index)
        if weight <= 0:
            continue
        contributor_count += 1
        cluster = str(
            service.artifacts.metadata[row].get("visualClusterId") or f"row-{row}"
        )
        if (
            cluster in seen_clusters
            or len(strongest_rows) >= service.config.evidence_per_slice
        ):
            continue
        seen_clusters.add(cluster)
        strongest_rows.append((row, float(raw_score), weight))
    cards = [
        service._evidence_card(row, score, weight, True)
        for row, score, weight in strongest_rows
    ]
    return {
        "schemaVersion": EVIDENCE_SCHEMA_VERSION,
        "sourceFingerprint": source_fingerprint,
        "conceptFingerprint": concept_fingerprint,
        "conceptId": concept.id,
        "label": concept.label,
        "binIndex": bin_index,
        "binKey": service.artifacts.bins[bin_index].key,
        "percentile": service.config.evidence_percentile,
        "threshold": computation.evidence_threshold,
        "contributorCount": contributor_count,
        # The shipped UI displays strongest contributors.  Other diagnostic
        # slices remain available from the live exact service and are omitted
        # here to avoid multiplying static payload bytes.
        "cards": cards,
    }


def _evidence_bundle_payload(
    service: SearchService,
    term: QueryTerm,
    concept: Concept,
    computation: SeriesComputation,
    corpus: ResolvedCorpus,
    bin_indices: Sequence[int],
    source_fingerprint: str,
    concept_fingerprint: str,
) -> dict[str, Any]:
    artworks: dict[str, dict[str, Any]] = {}
    periods: list[dict[str, Any]] = []
    for bin_index in bin_indices:
        evidence = _evidence_payload(
            service,
            term,
            concept,
            computation,
            corpus,
            int(bin_index),
            source_fingerprint,
            concept_fingerprint,
        )
        artwork_ids: list[str] = []
        contribution_weights: list[float] = []
        for card in evidence["cards"]:
            artwork_id = str(card["artworkId"])
            artwork_ids.append(artwork_id)
            contribution_weights.append(float(card["contributionWeight"]))
            if artwork_id not in artworks:
                artworks[artwork_id] = {
                    key: value
                    for key, value in card.items()
                    if key != "contributionWeight"
                }
        periods.append(
            {
                "binIndex": int(bin_index),
                "contributorCount": evidence["contributorCount"],
                "artworkIds": artwork_ids,
                "contributionWeights": contribution_weights,
            }
        )
    return {
        "schemaVersion": EVIDENCE_SCHEMA_VERSION,
        "sourceFingerprint": source_fingerprint,
        "conceptFingerprint": concept_fingerprint,
        "conceptId": concept.id,
        "label": concept.label,
        "percentile": service.config.evidence_percentile,
        "threshold": computation.evidence_threshold,
        "artworks": artworks,
        "periods": periods,
    }


def _series_is_complete(
    output: Path,
    concept: Concept,
    source_fingerprint: str,
    concept_fingerprint: str,
) -> bool:
    series_path = output / "series" / f"{concept.id}.json"
    if not series_path.is_file():
        return False
    try:
        payload = _read_json(series_path)
        if (
            payload.get("schemaVersion") != SERIES_SCHEMA_VERSION
            or payload.get("sourceFingerprint") != source_fingerprint
            or payload.get("conceptFingerprint") != concept_fingerprint
        ):
            return False
        evidence_path = output / "evidence" / f"{concept.id}.json"
        if not evidence_path.is_file():
            return False
        evidence = _read_json(evidence_path)
        if (
            evidence.get("schemaVersion") != EVIDENCE_SCHEMA_VERSION
            or evidence.get("sourceFingerprint") != source_fingerprint
            or evidence.get("conceptFingerprint") != concept_fingerprint
            or [period.get("binIndex") for period in evidence.get("periods", [])]
            != payload.get("evidenceBinIndices", [])
        ):
            return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _chunks(values: Sequence[Concept], size: int) -> Iterable[Sequence[Concept]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def export_catalog(
    artifacts: ArtifactBundle,
    concept_source: ConceptSource,
    output: str | Path,
    encoder: TextEncoder,
    *,
    prompt_ensemble: PromptEnsemble | None = None,
    config: SearchConfig | None = None,
    artifact_manifest_hash: str,
    selected_concept_ids: Sequence[str] | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    resume: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ExportStats]:
    if batch_size < 1:
        raise ValueError("concept batch size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("concept limit must be positive")
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)

    prompt_policy = prompt_ensemble or PromptEnsemble()
    search_config = config or SearchConfig()
    source_fingerprint = _source_fingerprint(
        artifacts, artifact_manifest_hash, prompt_policy, search_config, encoder
    )
    concepts = list(concept_source.concepts)
    if selected_concept_ids:
        requested = list(dict.fromkeys(selected_concept_ids))
        by_id = {concept.id: concept for concept in concepts}
        unknown = [concept_id for concept_id in requested if concept_id not in by_id]
        if unknown:
            raise ValueError(f"unknown concept ids: {', '.join(unknown)}")
        concepts = [by_id[concept_id] for concept_id in requested]
    if limit is not None:
        concepts = concepts[:limit]

    release_fingerprint = _sha256_bytes(
        _canonical_json(
            {
                "sourceFingerprint": source_fingerprint,
                "catalogVersion": concept_source.version,
                "concepts": [concept.public_payload() for concept in concepts],
            }
        )
    )
    release_name = release_fingerprint[:24]
    destination = output_root / "releases" / release_name
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise ValueError(
            "this immutable release already exists; use --resume or change the "
            "catalog/version"
        )
    destination.mkdir(parents=True, exist_ok=True)

    corpus = artifacts.resolve_corpus("all", {})
    service = SearchService(
        artifacts,
        encoder,
        prompt_ensemble=prompt_policy,
        config=search_config,
        prefer_faiss=False,
        cache=InMemorySeriesCache(max_entries=max(batch_size, 8)),
    )

    bins_payload = _bins_payload(service, corpus, source_fingerprint)
    _atomic_json(destination / "bins.json", bins_payload)

    started = time.perf_counter()
    skipped = 0
    completed: list[Concept] = []
    # Detailed errors are returned in private run stats and progress events.
    # The release manifest receives only stable, sanitized failure codes.
    failures: list[dict[str, str]] = []
    pending: list[Concept] = []
    for concept in concepts:
        fingerprint = _concept_fingerprint(concept, source_fingerprint)
        if resume and _series_is_complete(
            destination, concept, source_fingerprint, fingerprint
        ):
            completed.append(concept)
            skipped += 1
            if progress:
                progress({"status": "skipped", "conceptId": concept.id})
        else:
            pending.append(concept)

    def export_batch(batch: Sequence[Concept]) -> None:
        terms = [_term(concept) for concept in batch]
        computations = service._series(terms, corpus)
        if len(computations) != len(batch):
            raise RuntimeError("search service returned an incomplete concept batch")
        for concept, term, computation in zip(batch, terms, computations):
            concept_fingerprint = _concept_fingerprint(
                concept, source_fingerprint
            )
            series = _series_payload(
                service,
                term,
                concept,
                computation,
                corpus,
                source_fingerprint,
                concept_fingerprint,
            )
            evidence = _evidence_bundle_payload(
                service,
                term,
                concept,
                computation,
                corpus,
                series["evidenceBinIndices"],
                source_fingerprint,
                concept_fingerprint,
            )
            _atomic_json(
                destination / "evidence" / f"{concept.id}.json", evidence
            )
            _atomic_json(destination / "series" / f"{concept.id}.json", series)
            completed.append(concept)
            if progress:
                progress(
                    {
                        "status": "completed",
                        "conceptId": concept.id,
                        "completed": len(completed),
                        "selected": len(concepts),
                        "evidencePeriods": len(series["evidenceBinIndices"]),
                    }
                )

    for batch in _chunks(pending, batch_size):
        try:
            export_batch(batch)
        except Exception as batch_error:
            # A single bad label or transient encoding problem should not erase
            # progress for unrelated catalog entries. Retry each entry so the
            # failure report names only concepts that genuinely failed.
            if len(batch) == 1:
                failures.append(
                    {"conceptId": batch[0].id, "error": str(batch_error)}
                )
                if progress:
                    progress(
                        {
                            "status": "failed",
                            "conceptId": batch[0].id,
                            "error": str(batch_error),
                        }
                    )
                continue
            for concept in batch:
                try:
                    export_batch([concept])
                except Exception as concept_error:
                    failures.append(
                        {"conceptId": concept.id, "error": str(concept_error)}
                    )
                    if progress:
                        progress(
                            {
                                "status": "failed",
                                "conceptId": concept.id,
                                "error": str(concept_error),
                            }
                        )

    completed_ids = {concept.id for concept in completed}
    ordered_completed = [
        concept for concept in concepts if concept.id in completed_ids
    ]
    concepts_payload = {
        "schemaVersion": CONCEPTS_SCHEMA_VERSION,
        "sourceFingerprint": source_fingerprint,
        "catalogVersion": concept_source.version,
        "concepts": [concept.public_payload() for concept in ordered_completed],
    }
    _atomic_json(destination / "concepts.json", concepts_payload)

    public_failures = [
        {
            "conceptId": failure["conceptId"],
            "code": CONCEPT_EXPORT_FAILURE_CODE,
        }
        for failure in failures
    ]
    manifest: dict[str, Any] = {
        "schemaVersion": EXPORT_SCHEMA_VERSION,
        "releaseFingerprint": release_fingerprint,
        "sourceFingerprint": source_fingerprint,
        "catalogVersion": concept_source.version,
        "complete": not failures and len(ordered_completed) == len(concepts),
        "fullCatalog": (
            not failures
            and len(concepts) == len(concept_source.concepts)
            and len(ordered_completed) == len(concept_source.concepts)
        ),
        "selection": (
            "full"
            if len(concepts) == len(concept_source.concepts)
            else "subset"
        ),
        "conceptCount": len(ordered_completed),
        "selectedConceptCount": len(concepts),
        "sourceConceptCount": len(concept_source.concepts),
        "corpus": {
            "id": artifacts.corpus_id,
            "version": artifacts.corpus_version,
            "label": artifacts.corpus_label,
            "count": len(corpus.row_ids),
            "countingUnit": artifacts.counting_unit,
        },
        "model": {
            "id": artifacts.model_id,
            "revision": artifacts.model_version,
            "promptTemplateVersion": prompt_policy.version,
            "promptTemplates": list(prompt_policy.templates),
            "encoderDevice": getattr(encoder, "device", "unspecified"),
            "runtimeVersions": getattr(encoder, "runtime_versions", {}),
        },
        "metric": {
            "id": METRIC_ID,
            "version": search_config.metric_version,
            "candidateFraction": search_config.percentile,
            "qualifiedFraction": search_config.evidence_percentile,
            "minimumEvidenceScore": search_config.minimum_evidence_score,
            "minimumDenominator": search_config.minimum_denominator,
            "minimumBinEvidenceClusters": search_config.minimum_bin_evidence_clusters,
        },
        "artifactManifestSha256": artifact_manifest_hash,
        "files": {
            "bins": "bins.json",
            "concepts": "concepts.json",
            "seriesTemplate": "series/{conceptId}.json",
            "evidenceTemplate": "evidence/{conceptId}.json",
        },
        "failures": public_failures,
    }
    _atomic_json(destination / "manifest.json", manifest)
    release_path = f"releases/{release_name}"
    published = bool(manifest["complete"] and manifest["fullCatalog"])
    if published:
        pointer = {
            "schemaVersion": POINTER_SCHEMA_VERSION,
            "catalogVersion": concept_source.version,
            "releaseFingerprint": release_fingerprint,
            "release": release_path,
            "conceptCount": len(ordered_completed),
            "complete": True,
            "fullCatalog": True,
            "corpusVersion": artifacts.corpus_version,
            "modelRevision": artifacts.model_version,
        }
        _atomic_json(output_root / "manifest.json", pointer)
    if progress:
        progress(
            {
                "status": "published" if published else "release-written",
                "release": release_path,
                "published": published,
                "complete": manifest["complete"],
                "fullCatalog": manifest["fullCatalog"],
            }
        )
    elapsed = time.perf_counter() - started
    newly_completed = max(0, len(ordered_completed) - skipped)
    stats = ExportStats(
        completed=len(ordered_completed),
        skipped=skipped,
        failures=tuple(failures),
        elapsed_seconds=elapsed,
        concepts_per_second=(newly_completed / elapsed if elapsed > 0 else 0.0),
    )
    return manifest, stats
