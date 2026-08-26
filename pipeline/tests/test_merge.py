from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scipy import sparse

from pipeline.build import CorpusBuildError, build_corpus, sha256_file
from pipeline.dates import DateConfig
from pipeline.embeddings import DeterministicTestEncoder, build_embedding_index
from pipeline.merge import MERGE_SCHEMA_VERSION, _load_bundle, merge_embedding_bundles


SOURCE_FIELDS = (
    "artwork_id",
    "physical_object_id",
    "visual_cluster_id",
    "institution",
    "source_id",
    "source_record_url",
    "source_dataset_version",
    "title",
    "artist",
    "object_type",
    "medium",
    "date_display",
    "date_start",
    "date_end",
    "metadata_license",
    "image_rights_uri",
    "credit_line",
    "public_domain",
    "image_available",
    "image_url",
    "image_path",
    "image_sha256",
    "image_use_permitted",
)


class EmbeddingBundleMergeTests(unittest.TestCase):
    def _build_bundle(
        self,
        root: Path,
        name: str,
        institution: str,
        records: list[dict[str, object]],
        *,
        date_config: DateConfig | None = None,
        counting_unit: str = "physical-object",
    ) -> Path:
        image_root = root / "images"
        image_root.mkdir(exist_ok=True)
        source = root / f"{name}-source.csv"
        rows: list[dict[str, object]] = []
        for record in records:
            artwork_id = str(record["artwork_id"])
            image = image_root / f"{name}-{artwork_id}.jpg"
            image.write_bytes(str(record.get("image_payload", artwork_id)).encode("utf-8"))
            digest = sha256_file(image)
            year = int(record["year"])
            rows.append(
                {
                    "artwork_id": artwork_id,
                    "physical_object_id": record.get("physical_object_id", artwork_id),
                    "visual_cluster_id": record.get(
                        "visual_cluster_id", f"sha256:{digest}"
                    ),
                    "institution": institution,
                    "source_id": artwork_id.partition("_")[2],
                    "source_record_url": f"https://example.test/{institution}/{artwork_id}",
                    "source_dataset_version": f"{institution}-data-v1",
                    "title": record.get("title", artwork_id),
                    "artist": f"{institution} artist",
                    "object_type": record.get("object_type", "painting"),
                    "medium": record.get("medium", "oil on canvas"),
                    "date_display": str(year),
                    "date_start": str(year),
                    "date_end": str(year),
                    "metadata_license": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "image_rights_uri": f"https://rights.test/{institution}/cc0",
                    "credit_line": f"Courtesy {institution}",
                    "public_domain": "true",
                    "image_available": "true",
                    "image_url": f"https://images.test/{institution}/{artwork_id}.jpg",
                    "image_path": str(image.relative_to(root)),
                    "image_sha256": digest,
                    "image_use_permitted": "true",
                }
            )
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        audit = root / f"{name}-audit.json"
        audit.write_text(
            json.dumps({"institution": institution, "rights_gate": "fixture"}) + "\n",
            encoding="utf-8",
        )
        corpus = root / f"{name}-corpus"
        build_corpus(
            source,
            corpus,
            corpus_version=f"{name}-corpus-v1",
            source_revision=f"{name}-revision",
            date_config=date_config,
            source_payloads=(source, audit),
            counting_unit=counting_unit,
        )
        bundle = root / f"{name}-bundle"
        build_embedding_index(
            corpus,
            bundle,
            DeterministicTestEncoder(dimension=8),
            image_root=root,
            write_faiss=False,
        )
        return bundle

    def _fixtures(self, root: Path) -> tuple[Path, Path]:
        shared_payload = "same visual bytes"
        first = self._build_bundle(
            root,
            "met",
            "met",
            [
                {"artwork_id": "MET_1", "year": 1805},
                {
                    "artwork_id": "MET_2",
                    "year": 1815,
                    "image_payload": shared_payload,
                },
            ],
        )
        shared_digest = sha256_file(root / "images" / "met-MET_2.jpg")
        second = self._build_bundle(
            root,
            "nga",
            "nga",
            [
                {
                    "artwork_id": "NGA_1",
                    "year": 1815,
                    "image_payload": shared_payload,
                    "visual_cluster_id": f"sha256:{shared_digest}",
                },
                {"artwork_id": "NGA_2", "year": 1905},
            ],
        )
        return first, second

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _write_manifest(path: Path, payload: dict[str, object]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _refresh_model_artifact(self, bundle: Path, relative: str) -> None:
        manifest_path = bundle / "model-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = bundle / relative
        for entry in manifest["artifacts"]:
            if entry["path"] == relative:
                entry["bytes"] = target.stat().st_size
                entry["sha256"] = sha256_file(target)
                break
        else:
            self.fail(f"missing fixture artifact {relative}")
        if relative == "corpus-build-manifest.json":
            manifest["corpus_manifest_sha256"] = sha256_file(target)
        self._write_manifest(manifest_path, manifest)

    def _refresh_corpus_artifacts(
        self, bundle: Path, relatives: tuple[str, ...]
    ) -> None:
        manifest_path = bundle / "corpus-build-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_path = {entry["path"]: entry for entry in manifest["artifacts"]}
        for relative in relatives:
            target = bundle / relative
            entry = by_path[relative]
            entry["bytes"] = target.stat().st_size
            entry["sha256"] = sha256_file(target)
        self._write_manifest(manifest_path, manifest)
        for relative in relatives:
            self._refresh_model_artifact(bundle, relative)
        self._refresh_model_artifact(bundle, "corpus-build-manifest.json")

    def test_merges_rows_vectors_dates_rights_coverage_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, nga = self._fixtures(root)
            output = root / "combined"
            source_matrices = [
                np.load(bundle / "embeddings.npy", allow_pickle=False)
                for bundle in (met, nga)
            ]

            manifest = merge_embedding_bundles(
                (met, nga), output, corpus_version="met-nga-v1"
            )

            self.assertEqual(manifest["merge"]["schema_version"], MERGE_SCHEMA_VERSION)
            self.assertEqual(manifest["corpus"]["count"], 4)
            self.assertEqual(manifest["matrix"]["dtype"], "float32")
            self.assertEqual(manifest["index"]["backend"], "numpy-flat-ip")
            self.assertIsNone(manifest["files"]["faissIndex"])
            self.assertFalse((output / "index.faiss").exists())
            self.assertEqual(
                manifest["model"]["settings"]["merged_from_completed_bundles"], 2
            )

            corpus = self._read_csv(output / "corpus.csv")
            self.assertEqual(
                [row["artwork_id"] for row in corpus],
                ["MET_1", "MET_2", "NGA_1", "NGA_2"],
            )
            self.assertEqual(
                [row["embedding_offset"] for row in corpus], ["0", "1", "2", "3"]
            )
            self.assertEqual(
                [row["institution"] for row in corpus], ["met", "met", "nga", "nga"]
            )
            self.assertEqual(
                [row["image_rights_uri"] for row in corpus],
                [
                    "https://rights.test/met/cc0",
                    "https://rights.test/met/cc0",
                    "https://rights.test/nga/cc0",
                    "https://rights.test/nga/cc0",
                ],
            )
            embedded = self._read_csv(output / "embedded-images.manifest.csv")
            self.assertEqual(
                [row["artwork_id"] for row in embedded],
                [row["artwork_id"] for row in corpus],
            )
            self.assertEqual(
                [row["embedding_offset"] for row in embedded], ["0", "1", "2", "3"]
            )
            np.testing.assert_array_equal(
                np.load(output / "embeddings.npy", allow_pickle=False),
                np.vstack(source_matrices),
            )

            weights = sparse.load_npz(output / "date-weights.npz")
            self.assertEqual(weights.shape, (4, 11))
            bins = manifest["bins"]
            self.assertEqual([item["start"] for item in bins], list(range(1800, 1910, 10)))
            denominators = self._read_csv(output / "bin-denominators.csv")
            by_start = {int(row["bin_start"]): row for row in denominators}
            self.assertEqual(float(by_start[1800]["eligible_weight"]), 1.0)
            self.assertEqual(float(by_start[1810]["eligible_weight"]), 2.0)
            self.assertEqual(by_start[1810]["physical_object_count"], "2")
            self.assertEqual(by_start[1810]["visual_cluster_count"], "1")
            self.assertEqual(float(by_start[1900]["eligible_weight"]), 1.0)

            coverage = self._read_csv(output / "coverage.csv")
            institution_coverage = {
                row["value"]: int(row["row_count"])
                for row in coverage
                if row["dimension"] == "institution"
            }
            self.assertEqual(institution_coverage, {"met": 2, "nga": 2})

            build_manifest = json.loads(
                (output / "corpus-build-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                build_manifest["merge"]["schema_version"], MERGE_SCHEMA_VERSION
            )
            self.assertEqual(build_manifest["files"]["coverage"], "coverage.csv")
            sources = manifest["merge"]["sources"]
            self.assertEqual(
                [source["institutions"] for source in sources], [["met"], ["nga"]]
            )
            for source in sources:
                model_path = output / source["model_manifest"]
                corpus_path = output / source["corpus_build_manifest"]
                self.assertEqual(sha256_file(model_path), source["model_manifest_sha256"])
                self.assertEqual(
                    sha256_file(corpus_path), source["corpus_build_manifest_sha256"]
                )
            provenance = manifest["files"]["sourceProvenance"]
            self.assertTrue(any(path.endswith("met-audit.json") for path in provenance))
            self.assertTrue(any(path.endswith("nga-audit.json") for path in provenance))
            for entry in manifest["artifacts"]:
                path = output / entry["path"]
                self.assertEqual(path.stat().st_size, entry["bytes"])
                self.assertEqual(sha256_file(path), entry["sha256"])
            self.assertEqual(
                json.loads((output / "model-manifest.json").read_text(encoding="utf-8")),
                manifest,
            )

    def test_rejects_incompatible_model_matrix_and_processor_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, nga = self._fixtures(root)

            def change_revision(bundle: Path) -> None:
                path = bundle / "model-manifest.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["model"]["revision"] = "different-revision"
                self._write_manifest(path, payload)

            def change_dimension(bundle: Path) -> None:
                path = bundle / "model-manifest.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["matrix"]["dimensions"] = 9
                self._write_manifest(path, payload)

            def change_dtype(bundle: Path) -> None:
                path = bundle / "model-manifest.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["matrix"]["dtype"] = "float16"
                self._write_manifest(path, payload)

            def change_processor(bundle: Path) -> None:
                path = bundle / "model-manifest.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["model"]["settings"]["algorithm"] = "incompatible-v2"
                self._write_manifest(path, payload)

            def break_normalization(bundle: Path) -> None:
                path = bundle / "embeddings.npy"
                matrix = np.load(path, allow_pickle=False)
                matrix[0] = 0
                np.save(path, matrix, allow_pickle=False)
                self._refresh_model_artifact(bundle, "embeddings.npy")

            scenarios = (
                ("revision", change_revision, "model IDs or revisions"),
                ("dimension", change_dimension, "does not match the float32 matrix"),
                ("dtype", change_dtype, "does not match the float32 matrix"),
                ("processor", change_processor, "processor contracts"),
                ("normalization", break_normalization, "not L2-normalized"),
            )
            for name, mutate, message in scenarios:
                with self.subTest(name=name):
                    candidate = root / f"nga-{name}"
                    shutil.copytree(nga, candidate)
                    mutate(candidate)
                    output = root / f"output-{name}"
                    with self.assertRaisesRegex(CorpusBuildError, message):
                        merge_embedding_bundles(
                            (met, candidate), output, corpus_version=f"invalid-{name}"
                        )
                    self.assertFalse(output.exists())

    def test_allows_source_specific_input_policies_and_preserves_each_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, nga = self._fixtures(root)
            met_policy = "remote-original-met-web-large-min-side-224/v2"
            nga_policy = "nga-iiif-fit-1024-short-side-256/v1"

            for bundle, policy, include_plural in (
                (met, met_policy, False),
                (nga, nga_policy, True),
            ):
                embedded_path = bundle / "embedded-images.manifest.csv"
                with embedded_path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                    fields = tuple(reader.fieldnames or ())
                for row in rows:
                    row["input_policy"] = policy
                with embedded_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
                self._refresh_model_artifact(bundle, "embedded-images.manifest.csv")

                if include_plural:
                    image_path = bundle / "images.manifest.csv"
                    with image_path.open(encoding="utf-8", newline="") as handle:
                        reader = csv.DictReader(handle)
                        image_rows = list(reader)
                        image_fields = (
                            *tuple(reader.fieldnames or ()),
                            "image_input_policy",
                        )
                    for row in image_rows:
                        row["image_input_policy"] = policy
                    with image_path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=image_fields, lineterminator="\n"
                        )
                        writer.writeheader()
                        writer.writerows(image_rows)
                    self._refresh_corpus_artifacts(
                        bundle, ("images.manifest.csv",)
                    )

                model_path = bundle / "model-manifest.json"
                manifest = json.loads(model_path.read_text(encoding="utf-8"))
                manifest["model"]["settings"]["image_input_policy"] = policy
                if include_plural:
                    manifest["model"]["settings"]["image_input_policies"] = [policy]
                self._write_manifest(model_path, manifest)

            output = root / "combined"
            merge_embedding_bundles((met, nga), output, corpus_version="policy-merge-v1")

            embedded = self._read_csv(output / "embedded-images.manifest.csv")
            self.assertEqual(
                [row["input_policy"] for row in embedded],
                [met_policy, met_policy, nga_policy, nga_policy],
            )
            images = self._read_csv(output / "images.manifest.csv")
            self.assertEqual(
                [row["image_input_policy"] for row in images],
                ["", "", nga_policy, nga_policy],
            )
            settings = json.loads(
                (output / "model-manifest.json").read_text(encoding="utf-8")
            )["model"]["settings"]
            self.assertEqual(settings["image_input_policy"], "per-record-mixed")
            self.assertEqual(
                settings["image_input_policies"], [nga_policy, met_policy]
            )

    def test_rejects_bundle_without_content_hash_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, nga = self._fixtures(root)

            corpus_path = nga / "corpus.csv"
            with corpus_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                corpus_rows = list(reader)
                corpus_fields = tuple(reader.fieldnames or ())
            for row in corpus_rows:
                row["image_sha256"] = ""
                row["visual_cluster_id"] = f"object:{row['artwork_id']}"
            with corpus_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=corpus_fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(corpus_rows)

            image_path = nga / "images.manifest.csv"
            with image_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                image_rows = list(reader)
                image_fields = tuple(reader.fieldnames or ())
            for row in image_rows:
                row["image_sha256"] = ""
            with image_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=image_fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(image_rows)

            self._refresh_corpus_artifacts(
                nga, ("corpus.csv", "images.manifest.csv")
            )
            with self.assertRaisesRegex(CorpusBuildError, "not reconciled"):
                merge_embedding_bundles(
                    (met, nga), root / "combined", corpus_version="invalid"
                )

    def test_rejects_contradictory_image_policy_and_declared_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, nga = self._fixtures(root)

            policy_bundle = root / "nga-policy-mismatch"
            shutil.copytree(nga, policy_bundle)
            image_path = policy_bundle / "images.manifest.csv"
            with image_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = (*tuple(reader.fieldnames or ()), "image_input_policy")
            for row in rows:
                row["image_input_policy"] = "contradictory-policy/v1"
            with image_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            self._refresh_corpus_artifacts(policy_bundle, ("images.manifest.csv",))
            with self.assertRaisesRegex(CorpusBuildError, "input policy differs"):
                merge_embedding_bundles(
                    (met, policy_bundle), root / "policy-output", corpus_version="invalid"
                )

            digest_bundle = root / "nga-digest-mismatch"
            shutil.copytree(nga, digest_bundle)
            embedded_path = digest_bundle / "embedded-images.manifest.csv"
            with embedded_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                embedded_rows = list(reader)
                embedded_fields = tuple(reader.fieldnames or ())
            embedded_rows[0]["declared_image_sha256"] = "0" * 64
            with embedded_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=embedded_fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(embedded_rows)
            self._refresh_model_artifact(
                digest_bundle, "embedded-images.manifest.csv"
            )
            with self.assertRaisesRegex(CorpusBuildError, "declared image digest"):
                merge_embedding_bundles(
                    (met, digest_bundle), root / "digest-output", corpus_version="invalid"
                )

    def test_allows_declared_original_digest_for_transformed_remote_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, _nga = self._fixtures(root)
            embedded_path = met / "embedded-images.manifest.csv"
            with embedded_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                embedded_rows = list(reader)
                embedded_fields = tuple(reader.fieldnames or ())
            embedded_rows[0].update(
                {
                    "image_url": "https://images.metmuseum.org/original/MET_1.jpg",
                    "input_kind": "remote-stream",
                    "input_source": "https://images.metmuseum.org/web-large/MET_1.jpg",
                    "declared_image_sha256": "0" * 64,
                }
            )
            with embedded_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=embedded_fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(embedded_rows)

            image_path = met / "images.manifest.csv"
            with image_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                image_rows = list(reader)
                image_fields = tuple(reader.fieldnames or ())
            image_rows[0]["image_url"] = embedded_rows[0]["image_url"]
            with image_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=image_fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(image_rows)

            corpus_path = met / "corpus.csv"
            with corpus_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                corpus_rows = list(reader)
                corpus_fields = tuple(reader.fieldnames or ())
            corpus_rows[0]["image_url"] = embedded_rows[0]["image_url"]
            with corpus_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=corpus_fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(corpus_rows)
            self._refresh_corpus_artifacts(
                met, ("corpus.csv", "images.manifest.csv")
            )
            self._refresh_model_artifact(met, "embedded-images.manifest.csv")

            loaded = _load_bundle(met)
            self.assertEqual(len(loaded.corpus_rows), 2)

    def test_load_bundle_accepts_a_relative_bundle_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, _nga = self._fixtures(root)
            relative = Path(os.path.relpath(met, Path.cwd()))

            loaded = _load_bundle(relative)

            self.assertEqual(loaded.root, met.resolve())

    def test_rejects_incompatible_date_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met = self._build_bundle(
                root,
                "met",
                "met",
                [{"artwork_id": "MET_1", "year": 1805}],
            )
            nga = self._build_bundle(
                root,
                "nga",
                "nga",
                [{"artwork_id": "NGA_1", "year": 1805}],
                date_config=DateConfig(bin_size=20),
            )
            output = root / "combined"
            with self.assertRaisesRegex(CorpusBuildError, "incompatible date rules"):
                merge_embedding_bundles((met, nga), output, corpus_version="invalid")
            self.assertFalse(output.exists())

    def test_mixed_physical_and_catalog_record_units_merge_as_catalog_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met = self._build_bundle(
                root,
                "met",
                "met",
                [{"artwork_id": "MET_1", "year": 1805}],
            )
            nga = self._build_bundle(
                root,
                "nga",
                "nga",
                [{"artwork_id": "NGA_1", "year": 1905}],
                counting_unit="catalog-record",
            )

            manifest = merge_embedding_bundles(
                (met, nga), root / "combined", corpus_version="mixed-units-v1"
            )

            self.assertEqual(manifest["corpus"]["countingUnit"], "catalog-record")
            self.assertEqual(
                manifest["merge"]["source_counting_units"],
                ["catalog-record", "physical-object"],
            )

    def test_records_an_optional_user_facing_label_in_both_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, nga = self._fixtures(root)
            output = root / "combined"

            manifest = merge_embedding_bundles(
                (met, nga),
                output,
                corpus_version="met-nga-v1",
                corpus_label="The Met + National Gallery of Art open-access image corpus",
            )

            build_manifest = json.loads(
                (output / "corpus-build-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["corpus"]["label"],
                "The Met + National Gallery of Art open-access image corpus",
            )
            self.assertEqual(build_manifest["corpus"], manifest["corpus"])

    def test_rejects_an_empty_explicit_corpus_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, nga = self._fixtures(root)

            with self.assertRaisesRegex(CorpusBuildError, "corpus_label"):
                merge_embedding_bundles(
                    (met, nga),
                    root / "combined",
                    corpus_version="met-nga-v1",
                    corpus_label="   ",
                )

    def test_merged_bundle_can_be_merged_again_with_same_model_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, nga = self._fixtures(root)
            combined = root / "combined"
            merge_embedding_bundles(
                (met, nga), combined, corpus_version="met-nga-v1"
            )
            aic = self._build_bundle(
                root,
                "aic",
                "aic",
                [{"artwork_id": "AIC_1", "year": 1955}],
            )

            manifest = merge_embedding_bundles(
                (combined, aic), root / "combined-again", corpus_version="three-v1"
            )

            self.assertEqual(manifest["corpus"]["count"], 5)
            self.assertEqual(
                [row["artwork_id"] for row in self._read_csv(root / "combined-again" / "corpus.csv")],
                ["MET_1", "MET_2", "NGA_1", "NGA_2", "AIC_1"],
            )

    def test_rejects_duplicate_artwork_ids_across_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met = self._build_bundle(
                root,
                "met",
                "met",
                [{"artwork_id": "SHARED_1", "year": 1805}],
            )
            nga = self._build_bundle(
                root,
                "nga",
                "nga",
                [{"artwork_id": "SHARED_1", "year": 1905}],
            )
            output = root / "combined"
            with self.assertRaisesRegex(CorpusBuildError, "duplicate artwork_id 'SHARED_1'"):
                merge_embedding_bundles((met, nga), output, corpus_version="invalid")
            self.assertFalse(output.exists())

    def test_rejects_tampered_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met, nga = self._fixtures(root)
            with (nga / "corpus.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            output = root / "combined"
            with self.assertRaisesRegex(CorpusBuildError, "integrity check failed"):
                merge_embedding_bundles((met, nga), output, corpus_version="invalid")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
