from __future__ import annotations

import unittest

import numpy as np

from mnemosyne_search.index import NumpyFlatIPIndex, create_exact_index


class ExactIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = np.asarray(
            [[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=np.float32
        )

    def test_numpy_search_is_exact_and_stably_ordered(self) -> None:
        index = NumpyFlatIPIndex(self.embeddings)
        hits = index.search(np.asarray([[1, 0]], dtype=np.float32), 3)
        self.assertEqual(hits.indices.tolist(), [[0, 1, 2]])
        np.testing.assert_allclose(hits.scores, [[1, 0.8, 0]], atol=1e-6)

    def test_filtered_search_scores_subset_instead_of_post_filtering(self) -> None:
        index = NumpyFlatIPIndex(self.embeddings)
        hits = index.search(
            np.asarray([[1, 0]], dtype=np.float32),
            2,
            eligible_indices=np.asarray([1, 2, 3]),
        )
        self.assertEqual(hits.indices.tolist(), [[1, 2]])

    def test_blocked_top_k_preserves_corpus_order_for_boundary_ties(self) -> None:
        index = NumpyFlatIPIndex(
            np.asarray([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
        )
        index.block_size = 2
        hits = index.search(np.asarray([[1, 0]], dtype=np.float32), 2)
        self.assertEqual(hits.indices.tolist(), [[0, 1]])

    def test_blocked_top_k_accumulates_when_k_exceeds_first_block(self) -> None:
        index = NumpyFlatIPIndex(self.embeddings)
        index.block_size = 2

        hits = index.search(np.asarray([[1, 0]], dtype=np.float32), 3)

        self.assertEqual(hits.indices.tolist(), [[0, 1, 2]])
        np.testing.assert_allclose(hits.scores, [[1, 0.8, 0]], atol=1e-6)

    def test_factory_has_working_numpy_fallback(self) -> None:
        index = create_exact_index(self.embeddings, prefer_faiss=False)
        self.assertEqual(index.backend, "numpy-flat-ip")
        self.assertEqual(index.search(np.asarray([[0, 1]], dtype=np.float32), 1).indices[0, 0], 2)


if __name__ == "__main__":
    unittest.main()
