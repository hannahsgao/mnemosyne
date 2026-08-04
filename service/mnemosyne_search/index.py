"""Exact normalized inner-product search with FAISS and NumPy paths."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
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

    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))

    def search(
        self, queries: np.ndarray, k: int, *, eligible_indices: np.ndarray | None = None
    ) -> SearchHits:
        query_rows = l2_normalize(queries)
        candidates = (
            np.arange(self.embeddings.shape[0], dtype=np.int64)
            if eligible_indices is None
            else np.asarray(eligible_indices, dtype=np.int64)
        )
        if not 1 <= k <= len(candidates):
            raise ValueError("k must be between one and the eligible corpus size")
        scores = query_rows @ self.embeddings[candidates].T
        # Stable sort preserves ascending corpus row as the deterministic tie-break.
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return SearchHits(
            indices=candidates[order],
            scores=np.take_along_axis(scores, order, axis=1).astype(np.float32),
        )

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
        self.embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
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
        self._subset_indices: OrderedDict[str, tuple[object, np.ndarray]] = OrderedDict()
        self._subset_lock = RLock()

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
            subset = self._subset_index(candidates)
            scores, local_indices = subset.search(query_rows, k)
            indices = candidates[local_indices]
        # Normalize FAISS ordering for deterministic evidence output.
        for row in range(indices.shape[0]):
            order = np.lexsort((indices[row], -scores[row]))
            indices[row] = indices[row, order]
            scores[row] = scores[row, order]
        return SearchHits(indices=indices.astype(np.int64), scores=scores.astype(np.float32))

    def _subset_index(self, candidates: np.ndarray):
        """Cache exact IndexFlatIP instances for named/filtered corpus views."""

        digest = hashlib.sha256(np.ascontiguousarray(candidates).tobytes()).hexdigest()
        with self._subset_lock:
            cached = self._subset_indices.get(digest)
            if cached is not None and np.array_equal(cached[1], candidates):
                self._subset_indices.move_to_end(digest)
                return cached[0]
            index = self._faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(np.ascontiguousarray(self.embeddings[candidates], dtype=np.float32))
            self._subset_indices[digest] = (index, candidates.copy())
            self._subset_indices.move_to_end(digest)
            while len(self._subset_indices) > 16:
                self._subset_indices.popitem(last=False)
            return index

    def score(self, query: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return self._numpy_fallback.score(query, indices)


def create_exact_index(
    embeddings: np.ndarray,
    *,
    prefer_faiss: bool = True,
    faiss_index_path: str | Path | None = None,
) -> ExactIndex:
    if prefer_faiss:
        try:
            return FaissFlatIPIndex(embeddings, index_path=faiss_index_path)
        except RuntimeError:
            pass
    return NumpyFlatIPIndex(embeddings)
