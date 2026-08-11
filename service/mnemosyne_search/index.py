"""Exact normalized inner-product search with FAISS and NumPy paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .encoders import l2_normalize


@dataclass(frozen=True)
class SearchHits:
    indices: np.ndarray
    scores: np.ndarray


class ExactIndex(Protocol):
    backend: str

    def search(
        self, queries: np.ndarray, k: int, *, eligible_indices: np.ndarray | None = None
    ) -> SearchHits: ...

    def score(self, query: np.ndarray, indices: np.ndarray) -> np.ndarray: ...


class NumpyFlatIPIndex:
    backend = "numpy-flat-ip"
    block_size = 65_536

    def __init__(self, embeddings: np.ndarray) -> None:
        # Artifact loading validates normalization. Keeping this as a view
        # preserves an on-disk memmap instead of duplicating the full matrix.
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a matrix")

    @staticmethod
    def _top_k(
        candidate_indices: np.ndarray, candidate_scores: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select exact top-k rows without fully sorting every score block."""

        # The first row block can contain fewer than the final requested k.
        # Retain every candidate until enough blocks have accumulated, then
        # keep exactly k for the remaining passes.
        retained_count = min(k, candidate_scores.shape[1])
        output_indices = np.empty(
            (len(candidate_scores), retained_count), dtype=np.int64
        )
        output_scores = np.empty(
            (len(candidate_scores), retained_count), dtype=np.float32
        )
        for query_index, scores in enumerate(candidate_scores):
            indices = candidate_indices[query_index]
            if len(scores) <= retained_count:
                selected = np.arange(len(scores), dtype=np.int64)
            else:
                partition = np.argpartition(-scores, retained_count - 1)[
                    :retained_count
                ]
                cutoff = scores[partition].min()
                better = np.flatnonzero(scores > cutoff)
                equal = np.flatnonzero(scores == cutoff)
                equal_order = np.argsort(indices[equal], kind="stable")
                selected = np.concatenate(
                    (
                        better,
                        equal[equal_order[: retained_count - len(better)]],
                    )
                )
            order = np.lexsort((indices[selected], -scores[selected]))
            selected = selected[order]
            output_indices[query_index] = indices[selected]
            output_scores[query_index] = scores[selected]
        return output_indices, output_scores

    def search(
        self, queries: np.ndarray, k: int, *, eligible_indices: np.ndarray | None = None
    ) -> SearchHits:
        query_rows = l2_normalize(queries)
        candidate_count = (
            self.embeddings.shape[0] if eligible_indices is None else len(eligible_indices)
        )
        if not 1 <= k <= candidate_count:
            raise ValueError("k must be between one and the eligible corpus size")
        eligible = None if eligible_indices is None else np.asarray(eligible_indices, dtype=np.int64)
        if eligible is not None and (
            eligible.ndim != 1
            or np.any(eligible < 0)
            or np.any(eligible >= self.embeddings.shape[0])
            or len(np.unique(eligible)) != len(eligible)
        ):
            raise ValueError("eligible indices must be unique, one-dimensional corpus rows")
        best_indices = np.empty((len(query_rows), 0), dtype=np.int64)
        best_scores = np.empty((len(query_rows), 0), dtype=np.float32)
        for start in range(0, candidate_count, self.block_size):
            end = min(candidate_count, start + self.block_size)
            if eligible is None:
                block_indices = np.arange(start, end, dtype=np.int64)
                block_embeddings = self.embeddings[start:end]
            else:
                block_indices = eligible[start:end]
                # Advanced indexing is bounded to one block instead of copying
                # the entire filtered corpus into memory.
                block_embeddings = self.embeddings[block_indices]
            block_scores = (query_rows @ block_embeddings.T).astype(np.float32, copy=False)
            combined_indices = np.concatenate(
                (best_indices, np.broadcast_to(block_indices, block_scores.shape)), axis=1
            )
            combined_scores = np.concatenate((best_scores, block_scores), axis=1)
            best_indices, best_scores = self._top_k(combined_indices, combined_scores, k)
        return SearchHits(indices=best_indices, scores=best_scores)

    def score(self, query: np.ndarray, indices: np.ndarray) -> np.ndarray:
        vector = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        return (self.embeddings[np.asarray(indices, dtype=np.int64)] @ vector).astype(np.float32)


class FaissFlatIPIndex:
    backend = "faiss-index-flat-ip"

    def __init__(self, embeddings: np.ndarray, *, index_path: str | Path | None = None) -> None:
        try:
            import faiss
        except ImportError as error:  # pragma: no cover - optional production dependency
            raise RuntimeError("FAISS is not installed") from error
        self._faiss = faiss
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a matrix")
        if index_path is None:
            self._index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self._index.add(np.ascontiguousarray(self.embeddings))
        else:
            self._index = faiss.read_index(str(index_path))
            if (
                self._index.d != self.embeddings.shape[1]
                or self._index.ntotal != self.embeddings.shape[0]
                or self._index.metric_type != faiss.METRIC_INNER_PRODUCT
                or "IndexFlat" not in type(self._index).__name__
            ):
                raise ValueError("prebuilt FAISS index is not the expected exact IndexFlatIP")
        self._numpy_fallback = NumpyFlatIPIndex(self.embeddings)

    def search(
        self, queries: np.ndarray, k: int, *, eligible_indices: np.ndarray | None = None
    ) -> SearchHits:
        query_rows = np.ascontiguousarray(l2_normalize(queries), dtype=np.float32)
        candidates = (
            np.arange(self.embeddings.shape[0], dtype=np.int64)
            if eligible_indices is None
            else np.asarray(eligible_indices, dtype=np.int64)
        )
        if not 1 <= k <= len(candidates):
            raise ValueError("k must be between one and the corpus size")
        if eligible_indices is None:
            scores, indices = self._index.search(query_rows, k)
        else:
            # Building an IndexFlatIP for every filtered view copies the entire
            # selected matrix into FAISS and a former 16-entry cache could retain
            # several gigabytes.  The blocked NumPy path remains exact and keeps
            # temporary memory bounded to one row tile.
            return self._numpy_fallback.search(
                query_rows, k, eligible_indices=candidates
            )
        # Normalize FAISS ordering for deterministic evidence output.
        for row in range(indices.shape[0]):
            order = np.lexsort((indices[row], -scores[row]))
            indices[row] = indices[row, order]
            scores[row] = scores[row, order]
        return SearchHits(indices=indices.astype(np.int64), scores=scores.astype(np.float32))

    def score(self, query: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return self._numpy_fallback.score(query, indices)


def create_exact_index(
    embeddings: np.ndarray,
    *,
    prefer_faiss: bool = False,
    faiss_index_path: str | Path | None = None,
) -> ExactIndex:
    if prefer_faiss:
        try:
            return FaissFlatIPIndex(embeddings, index_path=faiss_index_path)
        except RuntimeError:
            pass
    return NumpyFlatIPIndex(embeddings)
