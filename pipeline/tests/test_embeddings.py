from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.build import build_corpus
from pipeline.embeddings import DeterministicTestEncoder, build_embedding_index


class EmbeddingBuildTests(unittest.TestCase):
    def _build_corpus(self, root: Path) -> Path:
        source = root / "ArtiFact_clean.csv"
        fields = (
            "object_ID",
            "title",
            "date_begin",
            "date_end",
            "date_begin_bce",
            "date_end_bce",
            "image_url",
            "public_domain",
        )
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "object_ID": "MET_20",
                        "title": "Twenty",
                        "date_begin": "1900",
                        "date_end": "1900",
                        "date_begin_bce": "false",
                        "date_end_bce": "false",
                        "image_url": "https://example.test/20.jpg",
                        "public_domain": "true",
                    },
                    {
                        "object_ID": "AIC_10",
                        "title": "Ten",
                        "date_begin": "1800",
                        "date_end": "1800",
                        "date_begin_bce": "false",
                        "date_end_bce": "false",
                        "image_url": "https://example.test/10.jpg",
                        "public_domain": "true",
                    },
                ]
            )
        corpus_dir = root / "corpus"
        build_corpus(
            source,
            corpus_dir,
            corpus_version="fixture-v1",
            source_revision="deadbeef",
            retrieved_at="2026-08-03T00:00:00Z",
        )
        return corpus_dir

    def test_deterministic_encoder_is_normalized_and_in_corpus_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._build_corpus(root)
            output = root / "embeddings"
            manifest = build_embedding_index(
                corpus_dir,
                output,
                DeterministicTestEncoder(16),
            )
            matrix = np.load(output / "embeddings.npy", allow_pickle=False)
            self.assertEqual(matrix.shape, (2, 16))
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(2), rtol=1e-6)
            fallback = np.load(output / "index-flat-ip.npz", allow_pickle=False)
            self.assertEqual(fallback["artwork_ids"].tolist(), ["AIC_10", "MET_20"])
            self.assertEqual(manifest["corpus"]["id"], "fixture-v1")
            self.assertEqual(manifest["index"]["metric"], "inner-product-on-l2-normalized-vectors")
            self.assertEqual(manifest["files"]["metadata"], "corpus.csv")
            self.assertEqual(manifest["files"]["embeddings"], "embeddings.npy")
            self.assertTrue((output / "embedded-images.manifest.csv").is_file())

    def test_embedding_manifest_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._build_corpus(root)
            manifests = []
            for suffix in ("one", "two"):
                output = root / suffix
                build_embedding_index(corpus_dir, output, DeterministicTestEncoder(8))
                manifests.append((output / "model-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifests[0], manifests[1])
            parsed = json.loads(manifests[0])
            self.assertTrue(parsed["model"]["settings"]["fixture_only"])


if __name__ == "__main__":
    unittest.main()
