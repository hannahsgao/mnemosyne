"""Load the embedding-free corpus artifacts used by Met keyword search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import (
    Bin,
    ResolvedCorpus,
    SparseDateWeights,
    load_bin_counts,
    load_denominators,
    load_metadata,
)


MET_CORPUS_SCHEMA_VERSION = "mnemosyne-corpus-build/v1"
MET_SOURCE_KIND = "met-open-access-csv-with-local-fts"


@dataclass(frozen=True)
class MetKeywordArtifacts:
    root: Path
    corpus_id: str
    corpus_version: str
    corpus_label: str
    counting_unit: str
    keyword_index_path: Path
    metadata: tuple[dict[str, Any], ...]
    bins: tuple[Bin, ...]
    date_weights: SparseDateWeights
    default_denominators: np.ndarray
    default_object_counts: np.ndarray
    default_cluster_counts: np.ndarray
    cluster_ids: tuple[str, ...]
    source_id_to_row: Mapping[int, int]
    allowed_filter_fields: frozenset[str]

    @classmethod
    def load(cls, root: str | Path) -> "MetKeywordArtifacts":
        base = Path(root)
        manifest_path = base / "build-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MET_CORPUS_SCHEMA_VERSION:
            raise ValueError("Met keyword search requires a corpus build-manifest.json")
        if manifest.get("source", {}).get("kind") != MET_SOURCE_KIND:
            raise ValueError("artifact directory is not a Met Open Access corpus build")

        files = manifest["files"]
        if files.get("embeddings") is not None:
            raise ValueError("Met keyword artifacts must use the embedding-free corpus build")
        keyword_index_name = files.get("keywordIndex")
        if not keyword_index_name:
            raise ValueError("Met keyword artifacts are missing the local FTS index")
        keyword_index_path = (base / str(keyword_index_name)).resolve()
        if not keyword_index_path.is_file():
            raise ValueError("manifest-declared Met FTS index does not exist")
        metadata = load_metadata(base / files["metadata"])
        weight_path = base / files["dateWeights"]
        weights = (
            SparseDateWeights.from_npz(weight_path)
            if weight_path.suffix == ".npz"
            else SparseDateWeights.from_json(weight_path)
        )
        denominator_path = base / files["binDenominators"]
        denominators, bins_from_file = load_denominators(denominator_path)
        bin_counts = load_bin_counts(denominator_path)
        bins = tuple(
            Bin(
                key=item["key"],
                label=item["label"],
                start=int(item["start"]),
                end=int(item["end"]),
            )
            for item in manifest.get("bins", ())
        ) or bins_from_file
        if bins is None:
            raise ValueError("Met timeline bins are missing")
        if len(metadata) != weights.shape[0]:
            raise ValueError("Met metadata and date weights must share corpus row order")
        if weights.shape[1] != len(bins) or denominators.shape != (len(bins),):
            raise ValueError("Met bin artifacts have inconsistent dimensions")
        calculated = weights.aggregate(weights.dated_rows())
        if not np.allclose(calculated, denominators, rtol=0, atol=1e-6):
            raise ValueError("Met denominators do not match date weights")
        if len(weights.dated_rows()) != len(metadata):
            raise ValueError("Met keyword corpus must contain only dated artworks")
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
            raise ValueError("Met precomputed bin counts have inconsistent dimensions")

        source_id_to_row: dict[int, int] = {}
        for index, record in enumerate(metadata):
            try:
                source_id = int(record["sourceId"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Met corpus row {index} has an invalid source_id") from exc
            if source_id in source_id_to_row:
                raise ValueError(f"Met corpus contains duplicate source_id {source_id}")
            source_id_to_row[source_id] = index

        corpus = manifest["corpus"]
        search_index = manifest.get("search_index", {})
        if (
            search_index.get("backend") != "sqlite-fts5"
            or int(search_index.get("rows", -1)) != len(metadata)
        ):
            raise ValueError("Met FTS manifest metadata is inconsistent with the corpus")
        return cls(
            root=base,
            corpus_id=str(corpus["id"]),
            corpus_version=str(corpus["version"]),
            corpus_label=str(corpus.get("label", "The Met Open Access collection")),
            counting_unit=str(corpus.get("countingUnit", "physical-object")),
            keyword_index_path=keyword_index_path,
            metadata=metadata,
            bins=tuple(bins),
            date_weights=weights,
            default_denominators=denominators,
            default_object_counts=default_object_counts,
            default_cluster_counts=default_cluster_counts,
            cluster_ids=cluster_ids,
            source_id_to_row=source_id_to_row,
            allowed_filter_fields=frozenset(
                manifest.get(
                    "allowedFilterFields",
                    [
                        "department",
                        "objectType",
                        "medium",
                        "culture",
                        "classification",
                        "period",
                        "dynasty",
                        "publicDomain",
                    ],
                )
            ),
        )

    def resolve_corpus(
        self, view: str, filters: Mapping[str, tuple[str, ...]]
    ) -> ResolvedCorpus:
        if view != "all":
            raise ValueError("Met keyword artifacts currently expose only the 'all' corpus view")
        normalized_filters = {
            key: tuple(sorted({value.casefold() for value in values if value.strip()}))
            for key, values in sorted(filters.items())
        }
        unknown_fields = set(normalized_filters) - self.allowed_filter_fields
        if unknown_fields:
            raise ValueError(f"unsupported filter fields: {', '.join(sorted(unknown_fields))}")

        row_ids = self.date_weights.dated_rows()
        if normalized_filters:
            selected: list[int] = []
            for raw_row in row_ids:
                row = int(raw_row)
                record = self.metadata[row]
                if all(
                    str(record.get(field, "")).casefold() in accepted
                    for field, accepted in normalized_filters.items()
                ):
                    selected.append(row)
            row_ids = np.asarray(selected, dtype=np.int64)
        if len(row_ids) == 0:
            raise ValueError("corpus view and filters select no dated artworks")
        denominators = (
            self.default_denominators.copy()
            if not normalized_filters
            else self.date_weights.aggregate(row_ids)
        )
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if not normalized_filters:
            object_counts = self.default_object_counts.copy()
            cluster_counts = self.default_cluster_counts.copy()
        else:
            object_counts, cluster_counts = self.date_weights.membership_counts(
                row_ids, self.cluster_ids
            )
        return ResolvedCorpus(
            view="all",
            filters=normalized_filters,
            row_ids=row_ids,
            denominators=denominators,
            object_counts=object_counts,
            cluster_counts=cluster_counts,
            covers_index=len(row_ids) == len(self.metadata),
        )
