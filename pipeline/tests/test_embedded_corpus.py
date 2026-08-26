from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from pipeline.build import (
    BUILD_SCHEMA_VERSION,
    CANONICAL_FIELDS,
    IMAGE_MANIFEST_FIELDS,
    CorpusBuildError,
    sha256_file,
)
from pipeline.embedded_corpus import TRANSACTION_SCHEMA_VERSION, derive_embedded_corpus
from pipeline.embeddings import EMBED_SCHEMA_VERSION


EMBEDDED_FIELDS = (
    "embedding_offset",
    "artwork_id",
    "image_path",
    "image_url",
    "declared_image_sha256",
    "input_sha256",
    "input_kind",
    "input_source",
    "input_width",
    "input_height",
    "input_policy",
    "permission_status",
)


class EmbeddedCorpusDerivationTests(unittest.TestCase):
    @staticmethod
    def _artifact(bundle: Path, path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(bundle).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    def _refresh_bundle_manifests(
        self, bundle: Path, *, update_count: bool = False
    ) -> None:
        corpus_manifest_path = bundle / "corpus-build-manifest.json"
        corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        if update_count:
            with (bundle / "corpus.csv").open(encoding="utf-8", newline="") as handle:
                count = sum(1 for _row in csv.DictReader(handle))
            corpus_manifest["corpus"]["count"] = count
            corpus_manifest["counts"]["canonical_rows"] = count
        for entry in corpus_manifest["artifacts"]:
            path = bundle / entry["path"]
            if path.is_file():
                entry.update(self._artifact(bundle, path))
        corpus_manifest_path.write_text(
            json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        model_manifest_path = bundle / "model-manifest.json"
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        model_manifest["corpus"] = corpus_manifest["corpus"]
        model_manifest["corpus_manifest_sha256"] = sha256_file(corpus_manifest_path)
        matrix = np.load(bundle / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        model_manifest["matrix"].update(
            {
                "rows": int(matrix.shape[0]),
                "dimensions": int(matrix.shape[1]),
                "dtype": str(matrix.dtype),
            }
        )
        for entry in model_manifest["artifacts"]:
            path = bundle / entry["path"]
            entry.update(self._artifact(bundle, path))
        model_manifest_path.write_text(
            json.dumps(model_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_bundle(self, root: Path) -> tuple[Path, Path]:
        bundle = root / "embedded"
        bundle.mkdir()
        hashes = {
            "MET_1": "a" * 64,
            "MET_2": "b" * 64,
            "MET_3": "c" * 64,
            "MET_4": "a" * 64,
        }
        corpus_rows = []
        embedded_rows = []
        for offset, artwork_id in enumerate(hashes):
            corpus_rows.append(
                {
                    **{field: "" for field in CANONICAL_FIELDS},
                    "artwork_id": artwork_id,
                    "physical_object_id": artwork_id,
                    "visual_cluster_id": f"object:{artwork_id}",
                    "institution": "met",
                    "source_id": artwork_id.removeprefix("MET_"),
                    "source_record_url": f"https://www.metmuseum.org/art/{artwork_id}",
                    "source_dataset_version": "met-revision-1",
                    "title": f"Artwork {artwork_id}",
                    "date_display": "ca. 1900",
                    "date_start": "1895",
                    "date_end": "1905",
                    "date_qualifier": "circa",
                    "date_parse_method": "source_range",
                    "metadata_license": "CC0",
                    "image_rights_uri": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "credit_line": "The Met",
                    "public_domain": "True",
                    "image_available": "True",
                    "image_url": f"https://images.metmuseum.org/original/{artwork_id}.jpg",
                    "embedding_offset": str(offset),
                }
            )
            embedded_rows.append(
                {
                    "embedding_offset": str(offset),
                    "artwork_id": artwork_id,
                    "image_path": "",
                    "image_url": f"https://images.metmuseum.org/original/{artwork_id}.jpg",
                    "declared_image_sha256": "",
                    "input_sha256": hashes[artwork_id],
                    "input_kind": "remote-stream",
                    "input_source": f"https://images.metmuseum.org/web-large/{artwork_id}.jpg",
                    "input_width": "640",
                    "input_height": "480",
                    "input_policy": "fixture-policy/v1",
                    "permission_status": "public-domain",
                }
            )
        embedded_rows[1]["input_source"] = (
            "https://images.metmuseum.org/CRDImages/ep/web-large/IMAGES-RESTRICTED.JPG?x=1"
        )
        corpus_rows[2]["image_url"] = (
            "https://images.metmuseum.org/CRDImages/ad/original/image-number-only.jpg"
        )

        with (bundle / "corpus.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(corpus_rows)
        with (bundle / "embedded-images.manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=EMBEDDED_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(embedded_rows)

        with (bundle / "images.manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=IMAGE_MANIFEST_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(
                {
                    "artwork_id": row["artwork_id"],
                    "image_available": "True",
                    "image_url": row["image_url"],
                    "image_path": "",
                    "image_sha256": "",
                    "image_rights_uri": (
                        "https://creativecommons.org/publicdomain/zero/1.0/"
                    ),
                    "public_domain": "True",
                    "permission_status": "public-domain",
                }
                for row in corpus_rows
            )
        np.save(
            bundle / "embeddings.npy",
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(hashes), 1)),
            allow_pickle=False,
        )
        sparse.save_npz(
            bundle / "date-weights.npz",
            sparse.csr_matrix(np.ones((len(hashes), 1), dtype=np.float64)),
        )
        with (bundle / "bin-denominators.csv").open(
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
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "bin_index": "0",
                    "bin_key": "1890",
                    "bin_start": "1890",
                    "bin_end": "1899",
                    "bin_label": "1890–1899",
                    "eligible_weight": "4",
                    "physical_object_count": "4",
                    "visual_cluster_count": "4",
                }
            )

        visual_source = root / "met-visual.csv"
        visual_source.write_text("fixture visual input\n", encoding="utf-8")
        corpus_identity = {
            "id": "fixture-visual-v1",
            "version": "fixture-visual-v1",
            "count": len(hashes),
            "countingUnit": "physical-object",
        }
        deployed = {
            "metadata": bundle / "corpus.csv",
            "dateWeights": bundle / "date-weights.npz",
            "binDenominators": bundle / "bin-denominators.csv",
            "imageManifest": bundle / "images.manifest.csv",
        }
        corpus_manifest = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "corpus": corpus_identity,
            "source": {
                "kind": "fixture-met-visual",
                "input_filename": visual_source.name,
                "input_sha256": sha256_file(visual_source),
                "payloads": [
                    {
                        "filename": visual_source.name,
                        "sha256": sha256_file(visual_source),
                        "bytes": visual_source.stat().st_size,
                    }
                ],
            },
            "counts": {"canonical_rows": len(hashes)},
            "files": {**{key: path.name for key, path in deployed.items()}},
            "artifacts": sorted(
                (self._artifact(bundle, path) for path in deployed.values()),
                key=lambda entry: str(entry["path"]),
            ),
        }
        corpus_manifest_path = bundle / "corpus-build-manifest.json"
        corpus_manifest_path.write_text(
            json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        visual_manifest = root / "met-visual-preflight.manifest.json"
        visual_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "fixture-visual/v1",
                    "selection": {
                        "eligible_candidates": 5,
                        "prepared_rows": 4,
                        "requested_rows": 5,
                    },
                    "rights_gate": {
                        "institution": "met",
                        "public_domain_required": True,
                    },
                    "images": {"availability_preflight": True},
                    "output": {
                        "csv": visual_source.name,
                        "sha256": sha256_file(visual_source),
                        "bytes": visual_source.stat().st_size,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        model_files = {
            "metadata": "corpus.csv",
            "embeddings": "embeddings.npy",
            "dateWeights": "date-weights.npz",
            "binDenominators": "bin-denominators.csv",
            "imageManifest": "images.manifest.csv",
            "embeddedImages": "embedded-images.manifest.csv",
            "numpyIndex": "embeddings.npy",
            "faissIndex": None,
            "sourceProvenance": [],
        }
        model_artifact_paths = [
            bundle / "corpus.csv",
            bundle / "embeddings.npy",
            bundle / "date-weights.npz",
            bundle / "bin-denominators.csv",
            bundle / "images.manifest.csv",
            bundle / "embedded-images.manifest.csv",
            corpus_manifest_path,
        ]
        model_manifest = {
            "schema_version": EMBED_SCHEMA_VERSION,
            "corpus": corpus_identity,
            "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
            "model": {
                "id": "fixture/siglip",
                "revision": "pinned-fixture-revision",
                "settings": {"processor_class": "FixtureProcessor"},
            },
            "matrix": {
                "rows": len(hashes),
                "dimensions": 2,
                "dtype": "float32",
                "l2_normalized": True,
                "row_order": "corpus.csv embedding_offset",
            },
            "index": {
                "metric": "inner-product-on-l2-normalized-vectors",
                "backend": "numpy-flat-ip",
                "exact": True,
                "numpy_fallback": "embeddings.npy",
                "faiss": None,
            },
            "files": model_files,
            "artifacts": sorted(
                (self._artifact(bundle, path) for path in model_artifact_paths),
                key=lambda entry: str(entry["path"]),
            ),
        }
        (bundle / "model-manifest.json").write_text(
            json.dumps(model_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return bundle, visual_manifest

    def test_accepts_completed_image_availability_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)

            manifest = derive_embedded_corpus(
                bundle, visual_manifest, root / "preflighted.csv"
            )

            self.assertTrue(
                manifest["visual_preflight"]["images"]["availability_preflight"]
            )

    def test_rejects_visual_manifest_without_image_availability_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            visual_payload = json.loads(visual_manifest.read_text(encoding="utf-8"))
            visual_payload["images"]["availability_preflight"] = False
            visual_manifest.write_text(
                json.dumps(visual_payload, sort_keys=True) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                CorpusBuildError,
                r"images\.availability_preflight=true",
            ):
                derive_embedded_corpus(
                    bundle, visual_manifest, root / "not-preflighted.csv"
                )

    def test_derives_hash_clusters_and_excludes_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            output = root / "clean" / "met-visual-clean.csv"

            manifest = derive_embedded_corpus(bundle, visual_manifest, output)

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["artwork_id"] for row in rows], ["MET_1", "MET_4"])
            self.assertEqual(rows[0]["image_sha256"], "a" * 64)
            self.assertEqual(rows[0]["visual_cluster_id"], f"sha256:{'a' * 64}")
            self.assertEqual(rows[0]["image_path"], "")
            self.assertEqual(rows[0]["embedding_offset"], "")
            self.assertEqual(
                rows[0]["image_input_policy"], rows[0]["input_policy"]
            )
            self.assertEqual(rows[0]["image_width"], "640")
            self.assertEqual(rows[0]["image_height"], "480")
            self.assertEqual(rows[0]["image_use_permitted"], "true")
            self.assertEqual(rows[0]["source_dataset_version"], "met-revision-1")
            self.assertEqual(rows[0]["date_start"], "1895")
            self.assertEqual(rows[0]["image_rights_uri"], "https://creativecommons.org/publicdomain/zero/1.0/")

            self.assertEqual(manifest["counts"]["output_rows"], 2)
            self.assertEqual(manifest["counts"]["upstream_visual_excluded_rows"], 1)
            self.assertEqual(manifest["counts"]["excluded_placeholder_rows"], 2)
            self.assertEqual(manifest["counts"]["total_excluded_from_upstream_request"], 3)
            self.assertEqual(manifest["counts"]["unique_input_sha256"], 1)
            self.assertEqual(manifest["counts"]["duplicate_visual_rows"], 1)
            self.assertEqual(
                manifest["counts"]["excluded_by_placeholder"],
                {"Images-Restricted.jpg": 1, "image-number-only.jpg": 1},
            )
            self.assertEqual(
                manifest["sources"]["visual_preflight_manifest"]["sha256"],
                sha256_file(visual_manifest),
            )
            self.assertEqual(
                manifest["visual_preflight"]["rights_gate"]["institution"], "met"
            )
            sidecar = output.with_suffix(".manifest.json")
            self.assertTrue(sidecar.is_file())
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), manifest)
            self.assertFalse(any(output.parent.glob(".*.derive-*")))

            with self.assertRaisesRegex(CorpusBuildError, "refusing to overwrite"):
                derive_embedded_corpus(bundle, visual_manifest, output)

            transaction = output.with_name(f".{output.name}.derive-transaction.json")
            transaction.write_text(
                json.dumps(
                    {
                        "schema_version": TRANSACTION_SCHEMA_VERSION,
                        "output_csv": output.name,
                        "output_csv_sha256": sha256_file(output),
                        "output_manifest": sidecar.name,
                        "output_manifest_sha256": sha256_file(sidecar),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            output.unlink()
            recovered = derive_embedded_corpus(bundle, visual_manifest, output)
            self.assertEqual(recovered, manifest)
            self.assertTrue(output.is_file())
            self.assertFalse(transaction.exists())

    def test_adapter_can_declare_no_placeholder_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            visual_payload = json.loads(visual_manifest.read_text(encoding="utf-8"))
            visual_payload["placeholder_basenames"] = []
            visual_manifest.write_text(
                json.dumps(visual_payload, sort_keys=True) + "\n", encoding="utf-8"
            )

            output = root / "nga-visual-clean.csv"
            manifest = derive_embedded_corpus(bundle, visual_manifest, output)

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(manifest["counts"]["excluded_placeholder_rows"], 0)
            self.assertEqual(manifest["counts"]["excluded_by_placeholder"], {})
            self.assertEqual(manifest["operation"]["known_placeholder_basenames"], [])
            self.assertEqual(
                manifest["operation"]["placeholder_policy_source"],
                "visual-manifest",
            )

    def test_rejects_invalid_adapter_placeholder_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            visual_payload = json.loads(visual_manifest.read_text(encoding="utf-8"))
            visual_payload["placeholder_basenames"] = ["folder/placeholder.jpg"]
            visual_manifest.write_text(
                json.dumps(visual_payload, sort_keys=True) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(CorpusBuildError, "must be a basename"):
                derive_embedded_corpus(bundle, visual_manifest, root / "clean.csv")

    def test_rejects_tampered_bundle_before_reading_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            corpus_path = bundle / "corpus.csv"
            original = corpus_path.read_bytes()
            tampered = original.replace(b"Artwork", b"Artw0rk", 1)
            self.assertEqual(len(tampered), len(original))
            corpus_path.write_bytes(tampered)

            with self.assertRaisesRegex(
                CorpusBuildError, "artifact integrity check failed: corpus.csv"
            ):
                derive_embedded_corpus(bundle, visual_manifest, root / "clean.csv")

    def test_rejects_same_sized_stale_visual_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            original_size = visual_manifest.stat().st_size
            visual_payload = json.loads(visual_manifest.read_text(encoding="utf-8"))
            visual_payload["output"]["sha256"] = "f" * 64
            visual_manifest.write_text(
                json.dumps(visual_payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            self.assertEqual(visual_manifest.stat().st_size, original_size)

            with self.assertRaisesRegex(
                CorpusBuildError, "visual preflight manifest is not bound"
            ):
                derive_embedded_corpus(bundle, visual_manifest, root / "clean.csv")

    def test_accepts_exact_visual_manifest_declared_as_source_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            corpus_manifest_path = bundle / "corpus-build-manifest.json"
            corpus_manifest = json.loads(
                corpus_manifest_path.read_text(encoding="utf-8")
            )
            corpus_manifest["source"]["input_sha256"] = "f" * 64
            corpus_manifest["source"]["payloads"].append(
                {
                    "filename": visual_manifest.name,
                    "sha256": sha256_file(visual_manifest),
                    "bytes": visual_manifest.stat().st_size,
                }
            )
            corpus_manifest_path.write_text(
                json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._refresh_bundle_manifests(bundle)

            manifest = derive_embedded_corpus(
                bundle, visual_manifest, root / "payload-bound.csv"
            )
            self.assertEqual(manifest["counts"]["joined_rows"], 4)

    def test_rejects_visual_manifest_without_output_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            visual_payload = json.loads(visual_manifest.read_text(encoding="utf-8"))
            del visual_payload["output"]["sha256"]
            visual_manifest.write_text(
                json.dumps(visual_payload, sort_keys=True) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(CorpusBuildError, "output.sha256"):
                derive_embedded_corpus(bundle, visual_manifest, root / "unbound.csv")

    def test_rejects_reordered_embedded_provenance_with_refreshed_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            embedded_path = bundle / "embedded-images.manifest.csv"
            with embedded_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0], rows[1] = rows[1], rows[0]
            for offset, row in enumerate(rows):
                row["embedding_offset"] = str(offset)
            with embedded_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=EMBEDDED_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
            self._refresh_bundle_manifests(bundle)

            with self.assertRaisesRegex(
                CorpusBuildError, "artwork IDs do not match in embedding_offset order"
            ):
                derive_embedded_corpus(bundle, visual_manifest, root / "reordered.csv")

    def test_rejects_noncontiguous_offsets_in_each_join_input(self) -> None:
        for filename, fields in (
            ("corpus.csv", CANONICAL_FIELDS),
            ("embedded-images.manifest.csv", EMBEDDED_FIELDS),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle, visual_manifest = self._write_bundle(root)
                path = bundle / filename
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                rows[1]["embedding_offset"] = "7"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=fields, lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                self._refresh_bundle_manifests(bundle)

                with self.assertRaisesRegex(
                    CorpusBuildError,
                    rf"{filename} embedding_offset values must be contiguous from zero",
                ):
                    derive_embedded_corpus(bundle, visual_manifest, root / "offsets.csv")

    def test_rejects_bundle_with_unmatched_artwork_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            embedded_path = bundle / "embedded-images.manifest.csv"
            with embedded_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with embedded_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=EMBEDDED_FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows[:-1])
            self._refresh_bundle_manifests(bundle)

            with self.assertRaisesRegex(CorpusBuildError, "artwork IDs do not match"):
                derive_embedded_corpus(bundle, visual_manifest, root / "clean.csv")

    def test_rejects_invalid_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            embedded_path = bundle / "embedded-images.manifest.csv"
            with embedded_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["input_sha256"] = "not-a-sha"
            with embedded_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=EMBEDDED_FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            self._refresh_bundle_manifests(bundle)

            with self.assertRaisesRegex(CorpusBuildError, "64-character hex"):
                derive_embedded_corpus(bundle, visual_manifest, root / "clean.csv")

    def test_reconciles_a_previously_filtered_downstream_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            for filename, fields in (
                ("corpus.csv", CANONICAL_FIELDS),
                ("embedded-images.manifest.csv", EMBEDDED_FIELDS),
                ("images.manifest.csv", IMAGE_MANIFEST_FIELDS),
            ):
                path = bundle / filename
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows[:1])
            matrix = np.load(bundle / "embeddings.npy", allow_pickle=False)
            np.save(bundle / "embeddings.npy", matrix[:1], allow_pickle=False)
            sparse.save_npz(
                bundle / "date-weights.npz",
                sparse.csr_matrix(np.ones((1, 1), dtype=np.float64)),
            )
            self._refresh_bundle_manifests(bundle, update_count=True)

            manifest = derive_embedded_corpus(
                bundle, visual_manifest, root / "reconciled.csv"
            )

            self.assertEqual(manifest["counts"]["output_rows"], 1)
            self.assertEqual(manifest["counts"]["prior_downstream_excluded_rows"], 3)
            self.assertEqual(manifest["counts"]["total_excluded_from_upstream_request"], 4)


if __name__ == "__main__":
    unittest.main()
