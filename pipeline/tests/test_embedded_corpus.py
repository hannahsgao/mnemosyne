from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.build import CANONICAL_FIELDS, CorpusBuildError, sha256_file
from pipeline.embedded_corpus import TRANSACTION_SCHEMA_VERSION, derive_embedded_corpus


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
        (bundle / "model-manifest.json").write_text(
            '{"schema_version":"fixture-model/v1"}\n', encoding="utf-8"
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
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return bundle, visual_manifest

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

            with self.assertRaisesRegex(CorpusBuildError, "64-character hex"):
                derive_embedded_corpus(bundle, visual_manifest, root / "clean.csv")

    def test_reconciles_a_previously_filtered_downstream_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, visual_manifest = self._write_bundle(root)
            for filename, fields in (
                ("corpus.csv", CANONICAL_FIELDS),
                ("embedded-images.manifest.csv", EMBEDDED_FIELDS),
            ):
                path = bundle / filename
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows[:1])

            manifest = derive_embedded_corpus(
                bundle, visual_manifest, root / "reconciled.csv"
            )

            self.assertEqual(manifest["counts"]["output_rows"], 1)
            self.assertEqual(manifest["counts"]["prior_downstream_excluded_rows"], 3)
            self.assertEqual(manifest["counts"]["total_excluded_from_upstream_request"], 4)


if __name__ == "__main__":
    unittest.main()
