from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.build import CorpusBuildError, build_corpus, sha256_file
from pipeline.embeddings import DeterministicTestEncoder, build_embedding_index
from pipeline.repack import REPACK_SCHEMA_VERSION, repack_embedded_bundle


class EmbeddedBundleRepackTests(unittest.TestCase):
    def _input_csv(
        self,
        root: Path,
        name: str,
        artwork_ids: tuple[str, ...] = ("MET_1", "MET_2"),
        *,
        stale_hash: bool = False,
    ) -> Path:
        image_root = root / "images"
        image_root.mkdir(exist_ok=True)
        fields = (
            "artwork_id",
            "physical_object_id",
            "visual_cluster_id",
            "title",
            "date_start",
            "date_end",
            "public_domain",
            "image_available",
            "image_path",
            "image_url",
            "image_sha256",
            "image_use_permitted",
        )
        rows = []
        for index, artwork_id in enumerate(artwork_ids):
            image = image_root / f"{artwork_id}.jpg"
            if not image.exists():
                image.write_bytes(f"fixture-image-{artwork_id}".encode("utf-8"))
            digest = sha256_file(image)
            declared = "f" * 64 if stale_hash and index == 0 else digest
            rows.append(
                {
                    "artwork_id": artwork_id,
                    "physical_object_id": artwork_id,
                    "visual_cluster_id": f"sha256:{declared}",
                    "title": artwork_id,
                    "date_start": str(1900 + index),
                    "date_end": str(1900 + index),
                    "public_domain": "true",
                    "image_available": "true",
                    "image_path": str(image.relative_to(root)),
                    "image_url": f"https://images.metmuseum.org/original/{artwork_id}.jpg",
                    "image_sha256": declared,
                    "image_use_permitted": "true",
                }
            )
        path = root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _build_fixture(self, root: Path) -> tuple[Path, Path]:
        source_csv = self._input_csv(root, "source.csv")
        source_corpus = root / "source-corpus"
        build_corpus(
            source_csv,
            source_corpus,
            corpus_version="source-v1",
            source_revision="source-revision",
        )
        source_bundle = root / "source-bundle"
        build_embedding_index(
            source_corpus,
            source_bundle,
            DeterministicTestEncoder(dimension=8),
            image_root=root,
            write_faiss=False,
        )

        provenance = root / "reconciliation.json"
        provenance.write_text('{"rights_gate":"fixture"}\n', encoding="utf-8")
        rebuilt_csv = self._input_csv(root, "rebuilt.csv")
        rebuilt_corpus = root / "rebuilt-corpus"
        build_corpus(
            rebuilt_csv,
            rebuilt_corpus,
            corpus_version="reconciled-v2",
            source_revision="reconciled-revision",
            source_payloads=(rebuilt_csv, provenance),
        )
        return source_bundle, rebuilt_corpus

    def test_repacks_aligned_float32_vectors_with_reconciled_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_bundle, rebuilt_corpus = self._build_fixture(root)
            output = root / "repacked"
            source_matrix_sha = sha256_file(source_bundle / "embeddings.npy")

            manifest = repack_embedded_bundle(source_bundle, rebuilt_corpus, output)

            self.assertEqual(manifest["repack"]["schema_version"], REPACK_SCHEMA_VERSION)
            self.assertEqual(manifest["corpus"]["version"], "reconciled-v2")
            self.assertEqual(manifest["index"]["backend"], "numpy-flat-ip")
            self.assertIsNone(manifest["files"]["faissIndex"])
            self.assertEqual(sha256_file(output / "embeddings.npy"), source_matrix_sha)
            self.assertEqual(
                np.load(output / "embeddings.npy", allow_pickle=False).dtype,
                np.dtype("float32"),
            )
            self.assertTrue((output / "source-provenance" / "reconciliation.json").is_file())
            self.assertEqual(
                json.loads((output / "model-manifest.json").read_text(encoding="utf-8")),
                manifest,
            )
            with (output / "corpus.csv").open(encoding="utf-8", newline="") as handle:
                corpus_rows = list(csv.DictReader(handle))
            with (output / "embedded-images.manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                embedded_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["artwork_id"] for row in corpus_rows],
                [row["artwork_id"] for row in embedded_rows],
            )
            for corpus, embedded in zip(corpus_rows, embedded_rows, strict=True):
                self.assertEqual(corpus["image_sha256"], embedded["input_sha256"])
                self.assertEqual(
                    corpus["visual_cluster_id"], f"sha256:{embedded['input_sha256']}"
                )
            artifact_paths = {entry["path"] for entry in manifest["artifacts"]}
            self.assertIn("embeddings.npy", artifact_paths)
            self.assertIn("source-provenance/reconciliation.json", artifact_paths)
            self.assertTrue(source_bundle.is_dir())

    def test_rejects_different_ordered_artwork_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_bundle, _rebuilt_corpus = self._build_fixture(root)
            mismatch_csv = self._input_csv(root, "mismatch.csv", ("MET_1", "MET_3"))
            mismatch_corpus = root / "mismatch-corpus"
            build_corpus(
                mismatch_csv,
                mismatch_corpus,
                corpus_version="mismatch-v1",
                source_revision="mismatch-revision",
            )

            with self.assertRaisesRegex(CorpusBuildError, "artwork order differ"):
                repack_embedded_bundle(source_bundle, mismatch_corpus, root / "output")
            self.assertFalse((root / "output").exists())

    def test_rejects_metadata_hash_not_reconciled_to_embedding_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_bundle, _rebuilt_corpus = self._build_fixture(root)
            stale_csv = self._input_csv(root, "stale.csv", stale_hash=True)
            stale_corpus = root / "stale-corpus"
            build_corpus(
                stale_csv,
                stale_corpus,
                corpus_version="stale-v1",
                source_revision="stale-revision",
            )

            with self.assertRaisesRegex(CorpusBuildError, "image_sha256 does not match"):
                repack_embedded_bundle(source_bundle, stale_corpus, root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
