from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mnemosyne_search.artifacts import ArtifactBundle, SparseDateWeights


class PipelineArtifactContractTests(unittest.TestCase):
    def test_sparse_membership_counts_rows_and_distinct_clusters_in_one_pass(self) -> None:
        weights = SparseDateWeights(
            indptr=np.asarray([0, 2, 3], dtype=np.int64),
            indices=np.asarray([0, 1, 1], dtype=np.int64),
            data=np.asarray([0.5, 0.5, 1.0], dtype=np.float64),
            shape=(2, 2),
        ).validated()

        objects, clusters = weights.membership_counts(
            np.asarray([0, 1], dtype=np.int64), ("shared-cluster", "shared-cluster")
        )

        np.testing.assert_array_equal(objects, [1, 2])
        np.testing.assert_array_equal(clusters, [1, 1])

    def test_loads_pipeline_model_manifest_csv_and_scipy_csr_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "corpus.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "artwork_id",
                        "embedding_offset",
                        "physical_object_id",
                        "visual_cluster_id",
                        "institution",
                        "public_domain",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "artwork_id": "a",
                        "embedding_offset": "0",
                        "physical_object_id": "object-a",
                        "visual_cluster_id": "cluster-a",
                        "institution": "AIC",
                        "public_domain": "true",
                    }
                )
                writer.writerow(
                    {
                        "artwork_id": "b",
                        "embedding_offset": "1",
                        "physical_object_id": "object-b",
                        "visual_cluster_id": "cluster-b",
                        "institution": "The Met",
                        "public_domain": "false",
                    }
                )
            np.save(root / "embeddings.npy", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
            # This is the array layout written by scipy.sparse.save_npz for CSR.
            np.savez(
                root / "date-weights.npz",
                indices=np.asarray([0, 1], dtype=np.int32),
                indptr=np.asarray([0, 1, 2], dtype=np.int32),
                data=np.asarray([1.0, 1.0], dtype=np.float64),
                shape=np.asarray([2, 2], dtype=np.int64),
                format=np.asarray(b"csr"),
            )
            with (root / "bin-denominators.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "bin_index",
                        "bin_key",
                        "bin_start",
                        "bin_end",
                        "bin_label",
                        "eligible_weight",
                        "physical_object_count",
                        "visual_cluster_count",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "bin_index": 0,
                        "bin_key": "1800-1849",
                        "bin_start": 1800,
                        "bin_end": 1849,
                        "bin_label": "1800–1849",
                        "eligible_weight": 1,
                        "physical_object_count": 1,
                        "visual_cluster_count": 1,
                    }
                )
                writer.writerow(
                    {
                        "bin_index": 1,
                        "bin_key": "1850-1899",
                        "bin_start": 1850,
                        "bin_end": 1899,
                        "bin_label": "1850–1899",
                        "eligible_weight": 1,
                        "physical_object_count": 1,
                        "visual_cluster_count": 1,
                    }
                )
            manifest = {
                "schema_version": "mnemosyne-embedding-build/v1",
                "corpus": {
                    "id": "proof-v1",
                    "version": "proof-v1",
                    "count": 2,
                    "countingUnit": "physical-object",
                },
                "model": {"id": "model", "revision": "pinned-sha"},
                "matrix": {"rows": 2, "dimensions": 2, "l2_normalized": True},
                "files": {
                    "metadata": "corpus.csv",
                    "embeddings": "embeddings.npy",
                    "dateWeights": "date-weights.npz",
                    "binDenominators": "bin-denominators.csv",
                },
                "bins": [
                    {"index": 0, "key": "1800-1849", "start": 1800, "end": 1849, "label": "1800–1849"},
                    {"index": 1, "key": "1850-1899", "start": 1850, "end": 1899, "label": "1850–1899"},
                ],
            }
            (root / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            bundle = ArtifactBundle.load(root)
            self.assertEqual(bundle.corpus_version, "proof-v1")
            self.assertEqual(bundle.model_version, "pinned-sha")
            self.assertEqual(bundle.metadata[1]["physicalObjectId"], "object-b")
            np.testing.assert_array_equal(bundle.default_denominators, [1, 1])


if __name__ == "__main__":
    unittest.main()
