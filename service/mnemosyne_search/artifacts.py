"""Load and validate an immutable retrieval artifact bundle."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .encoders import l2_normalize


ARTIFACT_SCHEMA_VERSION = "mnemosyne.artifacts.v1"
PIPELINE_ARTIFACT_SCHEMA_VERSION = "mnemosyne-embedding-build/v1"


def _camel_case(value: str) -> str:
    return re.sub(r"_([a-z])", lambda match: match.group(1).upper(), value)


def load_metadata(path: Path) -> tuple[dict[str, Any], ...]:
    if path.suffix == ".json":
        return tuple(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix != ".csv":
        raise ValueError("corpus metadata must use .csv or fixture-only .json format")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            record: dict[str, Any] = {}
            for key, value in raw.items():
                name = _camel_case(key)
                if name in {"dateStart", "dateEnd", "imageWidth", "imageHeight", "embeddingOffset"}:
                    record[name] = int(value) if value else None
                elif name in {"publicDomain", "imageAvailable"}:
                    record[name] = value.casefold() in {"1", "true", "yes"}
                else:
                    record[name] = value
            records.append(record)
    return tuple(records)


def load_denominators(path: Path) -> tuple[np.ndarray, tuple[Bin, ...] | None]:
    if path.suffix == ".json":
        return (
            np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64),
            None,
        )
    if path.suffix != ".csv":
        raise ValueError("bin denominators must use .csv or fixture-only .json format")
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["bin_index"]))
    denominators = np.asarray([float(row["eligible_weight"]) for row in rows], dtype=np.float64)
    bins = tuple(
        Bin(
            key=row["bin_key"],
            label=row["bin_label"],
            start=int(row["bin_start"]),
            end=int(row["bin_end"]),
        )
        for row in rows
    )
    return denominators, bins


@dataclass(frozen=True)
class Bin:
    key: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class SparseDateWeights:
    """Minimal CSR implementation with artwork rows and time-bin columns."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]

    @classmethod
    def from_json(cls, path: Path) -> "SparseDateWeights":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            indptr=np.asarray(payload["indptr"], dtype=np.int64),
            indices=np.asarray(payload["indices"], dtype=np.int64),
            data=np.asarray(payload["data"], dtype=np.float64),
            shape=(int(payload["shape"][0]), int(payload["shape"][1])),
        ).validated()

    @classmethod
    def from_npz(cls, path: Path) -> "SparseDateWeights":
        with np.load(path) as payload:
            return cls(
                indptr=np.asarray(payload["indptr"], dtype=np.int64),
                indices=np.asarray(payload["indices"], dtype=np.int64),
                data=np.asarray(payload["data"], dtype=np.float64),
                shape=(int(payload["shape"][0]), int(payload["shape"][1])),
            ).validated()

    def validated(self) -> "SparseDateWeights":
        rows, columns = self.shape
        if self.indptr.shape != (rows + 1,):
            raise ValueError("date-weight indptr length does not match shape")
        if self.indptr[0] != 0 or self.indptr[-1] != len(self.data):
            raise ValueError("date-weight indptr bounds are invalid")
        if len(self.indices) != len(self.data):
            raise ValueError("date-weight index and value lengths differ")
        if np.any(self.indices < 0) or np.any(self.indices >= columns):
            raise ValueError("date-weight column index is out of range")
        if np.any(self.data < 0) or not np.all(np.isfinite(self.data)):
            raise ValueError("date weights must be finite and non-negative")
        for row in range(rows):
            total = self.data[self.indptr[row] : self.indptr[row + 1]].sum()
            if total and not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"dated artwork row {row} weights sum to {total}, not one")
        return self

    def dated_rows(self) -> np.ndarray:
        return np.flatnonzero(np.diff(self.indptr) > 0).astype(np.int64)

    def aggregate(self, rows: np.ndarray) -> np.ndarray:
        totals = np.zeros(self.shape[1], dtype=np.float64)
        for row in np.asarray(rows, dtype=np.int64):
            start, end = self.indptr[row], self.indptr[row + 1]
            np.add.at(totals, self.indices[start:end], self.data[start:end])
        return totals

    def weight(self, row: int, bin_index: int) -> float:
        start, end = self.indptr[row], self.indptr[row + 1]
        columns = self.indices[start:end]
        positions = np.flatnonzero(columns == bin_index)
        return float(self.data[start + positions[0]]) if len(positions) else 0.0

    def rows_for_bin(self, rows: np.ndarray, bin_index: int) -> tuple[np.ndarray, np.ndarray]:
        found_rows: list[int] = []
        weights: list[float] = []
        for row in np.asarray(rows, dtype=np.int64):
            weight = self.weight(int(row), bin_index)
            if weight > 0:
                found_rows.append(int(row))
                weights.append(weight)
        return np.asarray(found_rows, dtype=np.int64), np.asarray(weights, dtype=np.float64)


@dataclass(frozen=True)
class ResolvedCorpus:
    view: str
    filters: dict[str, tuple[str, ...]]
    row_ids: np.ndarray
    denominators: np.ndarray


@dataclass(frozen=True)
class ArtifactBundle:
    root: Path
    corpus_id: str
    corpus_version: str
    corpus_label: str
    counting_unit: str
    model_id: str
    model_version: str
    embeddings: np.ndarray
    faiss_index_path: Path | None
    metadata: tuple[dict[str, Any], ...]
    bins: tuple[Bin, ...]
    date_weights: SparseDateWeights
    default_denominators: np.ndarray
    views: Mapping[str, np.ndarray]
    allowed_filter_fields: frozenset[str]

    @classmethod
    def load(cls, root: str | Path) -> "ArtifactBundle":
        base = Path(root)
        manifest_path = (
            base / "model-manifest.json"
            if (base / "model-manifest.json").exists()
            else base / "build-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_version = manifest.get("artifactSchemaVersion", manifest.get("schema_version"))
        if schema_version not in {ARTIFACT_SCHEMA_VERSION, PIPELINE_ARTIFACT_SCHEMA_VERSION}:
            raise ValueError("unsupported or missing artifactSchemaVersion")

        files = manifest["files"]
        metadata = load_metadata(base / files["metadata"])
        embedding_path = base / files["embeddings"]
        if embedding_path.suffix == ".npy":
            raw_embeddings = np.load(embedding_path, mmap_mode="r")
        elif embedding_path.suffix == ".json":
            raw_embeddings = np.asarray(
                json.loads(embedding_path.read_text(encoding="utf-8")), dtype=np.float32
            )
        else:
            raise ValueError("embeddings must use .npy or fixture-only .json format")
        raw_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        raw_norms = np.linalg.norm(raw_embeddings, axis=1)
        if np.any(raw_norms == 0) or not np.allclose(raw_norms, 1.0, rtol=0, atol=5e-3):
            raise ValueError("offline artwork embeddings must be L2-normalized")
        embeddings = l2_normalize(raw_embeddings)

        weight_path = base / files["dateWeights"]
        weights = (
            SparseDateWeights.from_npz(weight_path)
            if weight_path.suffix == ".npz"
            else SparseDateWeights.from_json(weight_path)
        )
        denominators, bins_from_file = load_denominators(base / files["binDenominators"])
        bins = (
            tuple(
                Bin(
                    key=item["key"],
                    label=item["label"],
                    start=int(item["start"]),
                    end=int(item["end"]),
                )
                for item in manifest["bins"]
            )
            if "bins" in manifest
            else bins_from_file
        )
        if bins is None:
            raise ValueError("timeline bins must be present in the manifest or denominator CSV")

        if len(metadata) != embeddings.shape[0] or weights.shape[0] != embeddings.shape[0]:
            raise ValueError("metadata, embeddings, and date weights must share corpus row order")
        offsets = [record.get("embeddingOffset") for record in metadata]
        if any(offset is not None for offset in offsets) and offsets != list(range(len(metadata))):
            raise ValueError("corpus metadata must be in contiguous embedding_offset order")
        if weights.shape[1] != len(bins) or denominators.shape != (len(bins),):
            raise ValueError("bin artifacts have inconsistent dimensions")
        calculated = weights.aggregate(weights.dated_rows())
        if not np.allclose(calculated, denominators, rtol=0, atol=1e-6):
            raise ValueError("precomputed bin denominators do not match date weights")
        expected_dimension = int(
            manifest["model"].get(
                "embeddingDimension", manifest.get("matrix", {}).get("dimensions", 0)
            )
        )
        if embeddings.shape[1] != expected_dimension:
            raise ValueError("embedding dimension does not match build manifest")

        views: dict[str, np.ndarray] = {"all": weights.dated_rows()}
        for name, definition in manifest.get("views", {}).items():
            row_ids = definition.get("rowIds")
            if row_ids is not None:
                candidate = np.unique(np.asarray(row_ids, dtype=np.int64))
                if np.any(candidate < 0) or np.any(candidate >= len(metadata)):
                    raise ValueError(f"view {name!r} contains an invalid row")
                views[name] = np.intersect1d(candidate, weights.dated_rows(), assume_unique=True)

        corpus = manifest["corpus"]
        model = manifest["model"]
        faiss_index_name = files.get("faissIndex")
        faiss_index_path = base / faiss_index_name if faiss_index_name else None
        if faiss_index_path is not None and not faiss_index_path.exists():
            raise ValueError("manifest-declared FAISS index does not exist")
        return cls(
            root=base,
            corpus_id=corpus["id"],
            corpus_version=corpus["version"],
            corpus_label=corpus.get("label", corpus["id"]),
            counting_unit=corpus.get("countingUnit", "physical-object"),
            model_id=model["id"],
            model_version=model.get("version", model.get("revision", "")),
            embeddings=embeddings,
            faiss_index_path=faiss_index_path,
            metadata=metadata,
            bins=bins,
            date_weights=weights,
            default_denominators=denominators,
            views=views,
            allowed_filter_fields=frozenset(
                manifest.get(
                    "allowedFilterFields",
                    ["institution", "objectType", "medium", "publicDomain"],
                )
            ),
        )

    def resolve_corpus(
        self, view: str, filters: Mapping[str, tuple[str, ...]]
    ) -> ResolvedCorpus:
        if view not in self.views:
            raise ValueError(f"unknown corpus view: {view}")
        normalized_filters = {
            key: tuple(sorted({value.casefold() for value in values if value.strip()}))
            for key, values in sorted(filters.items())
        }
        unknown_fields = set(normalized_filters) - self.allowed_filter_fields
        if unknown_fields:
            raise ValueError(f"unsupported filter fields: {', '.join(sorted(unknown_fields))}")

        row_ids = self.views[view]
        if normalized_filters:
            selected: list[int] = []
            for row in row_ids:
                record = self.metadata[int(row)]
                matches = True
                for field, accepted in normalized_filters.items():
                    raw_value = record.get(field)
                    values = raw_value if isinstance(raw_value, list) else [raw_value]
                    normalized_values = {str(value).casefold() for value in values if value is not None}
                    if not normalized_values.intersection(accepted):
                        matches = False
                        break
                if matches:
                    selected.append(int(row))
            row_ids = np.asarray(selected, dtype=np.int64)
        if len(row_ids) == 0:
            raise ValueError("corpus view and filters select no dated artworks")
        denominators = (
            self.default_denominators.copy()
            if view == "all" and not normalized_filters
            else self.date_weights.aggregate(row_ids)
        )
        return ResolvedCorpus(
            view=view,
            filters=normalized_filters,
            row_ids=np.asarray(row_ids, dtype=np.int64),
            denominators=denominators,
        )
