"""Load and validate an immutable retrieval artifact bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ARTIFACT_SCHEMA_VERSION = "mnemosyne.artifacts.v1"
PIPELINE_ARTIFACT_SCHEMA_VERSION = "mnemosyne-embedding-build/v1"

# Fields required to validate row order, build filters, and construct the
# public evidence-card contract.  Large provenance-only columns remain in the
# immutable CSV and are not expanded into per-row Python dictionaries at query
# or export time.
RUNTIME_METADATA_FIELDS = frozenset(
    {
        "artworkId",
        "physicalObjectId",
        "visualClusterId",
        "institution",
        "sourceId",
        "sourceRecordUrl",
        "title",
        "artist",
        "dateDisplay",
        "dateStart",
        "dateEnd",
        "dateQualifier",
        "metadataLicense",
        "imageRightsUri",
        "creditLine",
        "publicDomain",
        "imageAvailable",
        "imageUrl",
        "embeddingOffset",
    }
)


def _camel_case(value: str) -> str:
    return re.sub(r"_([a-z])", lambda match: match.group(1).upper(), value)


def load_metadata(
    path: Path, *, fields: frozenset[str] | None = None
) -> tuple[dict[str, Any], ...]:
    if path.suffix == ".json":
        records = tuple(json.loads(path.read_text(encoding="utf-8")))
        if fields is None:
            return records
        return tuple(
            {key: value for key, value in record.items() if key in fields}
            for record in records
        )
    if path.suffix != ".csv":
        raise ValueError("corpus metadata must use .csv or fixture-only .json format")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        # Resolve column names once.  The production corpus has hundreds of
        # thousands of rows; recomputing the snake-to-camel mapping for every
        # cell adds millions of regex calls during startup.
        names = {key: _camel_case(key) for key in (reader.fieldnames or ())}
        for raw in reader:
            record: dict[str, Any] = {}
            for key, value in raw.items():
                name = names[key]
                if fields is not None and name not in fields:
                    continue
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


def load_bin_counts(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load build-time object and visual-cluster counts from denominator CSV."""

    if path.suffix != ".csv":
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["bin_index"]))
    required = {"physical_object_count", "visual_cluster_count"}
    if not rows or not required.issubset(rows[0]):
        return None
    return (
        np.asarray([int(row["physical_object_count"]) for row in rows], dtype=np.int64),
        np.asarray([int(row["visual_cluster_count"]) for row in rows], dtype=np.int64),
    )


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
    column_indptr: np.ndarray | None = None
    column_rows: np.ndarray | None = None
    column_data: np.ndarray | None = None

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
        row_for_value = np.repeat(np.arange(rows, dtype=np.int64), np.diff(self.indptr))
        row_totals = np.bincount(row_for_value, weights=self.data, minlength=rows)
        invalid_rows = np.flatnonzero((row_totals != 0) & ~np.isclose(row_totals, 1.0, atol=1e-6))
        if len(invalid_rows):
            row = int(invalid_rows[0])
            raise ValueError(f"dated artwork row {row} weights sum to {row_totals[row]}, not one")

        # CSR is ideal for aggregating selected artworks. This companion CSC-like
        # index makes per-bin evidence and denominator reads proportional to the
        # size of that bin instead of rescanning the entire corpus for every bin.
        order = np.argsort(self.indices, kind="stable")
        sorted_columns = self.indices[order]
        column_indptr = np.searchsorted(
            sorted_columns, np.arange(columns + 1, dtype=np.int64)
        ).astype(np.int64)
        return SparseDateWeights(
            indptr=self.indptr,
            indices=self.indices,
            data=self.data,
            shape=self.shape,
            column_indptr=column_indptr,
            column_rows=row_for_value[order],
            column_data=self.data[order],
        )

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
        if not 0 <= bin_index < self.shape[1]:
            raise IndexError("date-weight bin index is out of range")
        assert self.column_indptr is not None
        assert self.column_rows is not None
        assert self.column_data is not None
        start, end = self.column_indptr[bin_index : bin_index + 2]
        bin_rows = self.column_rows[start:end]
        bin_weights = self.column_data[start:end]
        requested = np.asarray(rows, dtype=np.int64)
        if len(requested) == self.shape[0] and np.array_equal(
            requested, np.arange(self.shape[0], dtype=np.int64)
        ):
            return bin_rows, bin_weights
        _, bin_positions, _ = np.intersect1d(
            bin_rows, requested, assume_unique=True, return_indices=True
        )
        return bin_rows[bin_positions], bin_weights[bin_positions]

    def membership_counts(
        self, rows: np.ndarray, group_ids: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Count row and distinct-group bin membership in one CSR pass."""

        if len(group_ids) != self.shape[0]:
            raise ValueError("group IDs must align with date-weight rows")
        object_counts = np.zeros(self.shape[1], dtype=np.int64)
        groups: list[set[str]] = [set() for _ in range(self.shape[1])]
        for raw_row in np.asarray(rows, dtype=np.int64):
            row = int(raw_row)
            start, end = self.indptr[row], self.indptr[row + 1]
            columns = self.indices[start:end]
            np.add.at(object_counts, columns, 1)
            for column in columns:
                groups[int(column)].add(group_ids[row])
        return object_counts, np.asarray([len(group) for group in groups], dtype=np.int64)


@dataclass(frozen=True)
class ResolvedCorpus:
    view: str
    filters: dict[str, tuple[str, ...]]
    row_ids: np.ndarray
    denominators: np.ndarray
    object_counts: np.ndarray
    cluster_counts: np.ndarray
    covers_index: bool


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
    default_object_counts: np.ndarray
    default_cluster_counts: np.ndarray
    cluster_ids: tuple[str, ...]
    views: Mapping[str, np.ndarray]
    allowed_filter_fields: frozenset[str]
    image_paths: Mapping[str, Path]

    @staticmethod
    def _verify_artifacts(
        base: Path, manifest: Mapping[str, Any], *, verify_checksums: bool = True
    ) -> None:
        resolved_base = base.resolve()
        for entry in manifest.get("artifacts", ()):
            path = (base / str(entry["path"])).resolve()
            try:
                path.relative_to(resolved_base)
            except ValueError as error:
                raise ValueError("artifact manifest path escapes the bundle root") from error
            if not path.is_file():
                raise ValueError(f"manifest-declared artifact does not exist: {entry['path']}")
            if path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"artifact byte count does not match manifest: {entry['path']}")
            if not verify_checksums:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != str(entry["sha256"]).lower():
                raise ValueError(f"artifact checksum does not match manifest: {entry['path']}")

    @classmethod
    def load(
        cls, root: str | Path, *, verify_checksums: bool = True
    ) -> "ArtifactBundle":
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
        cls._verify_artifacts(base, manifest, verify_checksums=verify_checksums)

        files = manifest["files"]
        allowed_filter_fields = frozenset(
            manifest.get(
                "allowedFilterFields",
                ["institution", "objectType", "medium", "publicDomain"],
            )
        )
        metadata = load_metadata(
            base / files["metadata"],
            fields=RUNTIME_METADATA_FIELDS | allowed_filter_fields,
        )
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
        # Validate a bounded view at a time.  A single full-matrix norm call can
        # allocate another matrix-sized temporary when NumPy squares float32
        # values before reducing them.
        norm_block_size = 8_192
        for start in range(0, raw_embeddings.shape[0], norm_block_size):
            block = raw_embeddings[start : start + norm_block_size]
            raw_norms = np.linalg.norm(block, axis=1)
            if np.any(raw_norms == 0) or not np.allclose(
                raw_norms, 1.0, rtol=0, atol=5e-3
            ):
                raise ValueError("offline artwork embeddings must be L2-normalized")
        # Keep the memory-mapped array intact. Offline builds are already
        # normalized and copying it here used to double resident memory.
        embeddings = raw_embeddings

        weight_path = base / files["dateWeights"]
        weights = (
            SparseDateWeights.from_npz(weight_path)
            if weight_path.suffix == ".npz"
            else SparseDateWeights.from_json(weight_path)
        )
        denominator_path = base / files["binDenominators"]
        denominators, bins_from_file = load_denominators(denominator_path)
        bin_counts = load_bin_counts(denominator_path)
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
        cluster_ids = tuple(
            str(record.get("visualClusterId") or f"row-{index}")
            for index, record in enumerate(metadata)
        )
        default_object_counts, default_cluster_counts = (
            bin_counts
            if bin_counts is not None
            else weights.membership_counts(weights.dated_rows(), cluster_ids)
        )
        if (
            default_object_counts.shape != (len(bins),)
            or default_cluster_counts.shape != (len(bins),)
        ):
            raise ValueError("precomputed bin counts have inconsistent dimensions")

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

        image_paths: dict[str, Path] = {}
        embedded_images_name = files.get("embeddedImages")
        if embedded_images_name:
            with (base / embedded_images_name).open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_path = row.get("image_path", "").strip()
                    if not raw_path:
                        continue
                    candidate = Path(raw_path)
                    resolved = candidate if candidate.is_absolute() else base / candidate
                    if resolved.is_file():
                        image_paths[str(row["artwork_id"])] = resolved.resolve()
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
            default_object_counts=default_object_counts,
            default_cluster_counts=default_cluster_counts,
            cluster_ids=cluster_ids,
            views=views,
            allowed_filter_fields=allowed_filter_fields,
            image_paths=image_paths,
        )

    def image_path_for(self, artwork_id: str) -> Path | None:
        return self.image_paths.get(artwork_id)

    def resolve_corpus(
        self, view: str, filters: Mapping[str, tuple[str, ...]]
    ) -> ResolvedCorpus:
        if view not in self.views:
            raise ValueError(f"unknown corpus view: {view}")
        normalized_filters = {
            key: tuple(
                sorted({value.strip().casefold() for value in values if value.strip()})
            )
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
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if view == "all" and not normalized_filters:
            object_counts = self.default_object_counts.copy()
            cluster_counts = self.default_cluster_counts.copy()
        else:
            object_counts, cluster_counts = self.date_weights.membership_counts(
                row_ids, self.cluster_ids
            )
        return ResolvedCorpus(
            view=view,
            filters=normalized_filters,
            row_ids=row_ids,
            denominators=denominators,
            object_counts=object_counts,
            cluster_counts=cluster_counts,
            covers_index=len(row_ids) == len(self.metadata)
            and np.array_equal(row_ids, np.arange(len(self.metadata), dtype=np.int64)),
        )
