from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scipy import sparse

from pipeline.build import CorpusBuildError, build_corpus


FIELDS = (
    "object_ID",
    "title",
    "object_name",
    "date_begin",
    "date_end",
    "date_begin_bce",
    "date_end_bce",
    "materials",
    "techniques",
    "culture",
    "artist_name",
    "image_url",
    "image_path",
    "public_domain",
)


def write_fixture(path: Path, rows: list[dict[str, object]], fields=FIELDS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class CorpusBuildTests(unittest.TestCase):
    def fixture_rows(self) -> list[dict[str, object]]:
        return [
            {
                "object_ID": "MET_2",
                "title": "Range",
                "object_name": "painting",
                "date_begin": "1880",
                "date_end": "1890",
                "date_begin_bce": "false",
                "date_end_bce": "false",
                "materials": '["canvas"]',
                "techniques": '["oil"]',
                "culture": "",
                "artist_name": "Artist B",
                "image_url": "https://example.test/2.jpg",
                "image_path": "images/2.jpg",
                "public_domain": "true",
            },
            {
                "object_ID": "AIC_1",
                "title": "Unknown",
                "object_name": "print",
                "date_begin": "",
                "date_end": "",
                "date_begin_bce": "false",
                "date_end_bce": "false",
                "materials": "",
                "techniques": "",
                "culture": "",
                "artist_name": "Artist A",
                "image_url": "https://example.test/1.jpg",
                "image_path": "images/1.jpg",
                "public_domain": "false",
            },
            {
                "object_ID": "RIJKS_3",
                "title": "Exact",
                "object_name": "drawing",
                "date_begin": "1885",
                "date_end": "1885",
                "date_begin_bce": "false",
                "date_end_bce": "false",
                "materials": '["paper"]',
                "techniques": "",
                "culture": "Dutch",
                "artist_name": "Artist C",
                "image_url": "https://example.test/3.jpg",
                "image_path": "images/3.jpg",
                "public_domain": "false",
            },
        ]

    def test_build_emits_sorted_corpus_sparse_weights_and_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ArtiFact_clean.csv"
            output = root / "build"
            write_fixture(source, self.fixture_rows())
            manifest = build_corpus(
                source,
                output,
                corpus_version="fixture-v1",
                source_revision="deadbeef",
                retrieved_at="2026-08-03T00:00:00Z",
            )

            self.assertEqual(manifest["counts"]["dated_rows"], 2)
            self.assertEqual(manifest["counts"]["unknown_date_rows"], 1)
            self.assertEqual(manifest["corpus"]["countingUnit"], "physical-object")
            self.assertEqual(manifest["files"]["dateWeights"], "date-weights.npz")
            self.assertTrue(all("key" in item for item in manifest["bins"]))
            self.assertTrue((output / "build-manifest.json").is_file())
            self.assertTrue((output / "source-payloads" / source.name).is_file())
            with (output / "corpus.csv").open(encoding="utf-8", newline="") as handle:
                corpus = list(csv.DictReader(handle))
            self.assertEqual([row["artwork_id"] for row in corpus], ["AIC_1", "MET_2", "RIJKS_3"])
            self.assertEqual([row["embedding_offset"] for row in corpus], ["0", "1", "2"])
            self.assertEqual(corpus[1]["medium"], "canvas; oil")
            self.assertEqual(corpus[0]["source_record_url"], "https://www.artic.edu/artworks/1")

            matrix = sparse.load_npz(output / "date-weights.npz")
            self.assertEqual(matrix.shape[0], 3)
            self.assertAlmostEqual(float(matrix[0].sum()), 0.0)
            self.assertAlmostEqual(float(matrix[1].sum()), 1.0)
            self.assertAlmostEqual(float(matrix[2].sum()), 1.0)

            with (output / "bin-denominators.csv").open(encoding="utf-8", newline="") as handle:
                denominators = list(csv.DictReader(handle))
            self.assertAlmostEqual(sum(float(row["eligible_weight"]) for row in denominators), 2.0)

            with (output / "images.manifest.csv").open(encoding="utf-8", newline="") as handle:
                images = {row["artwork_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(images["MET_2"]["permission_status"], "public-domain")
            self.assertEqual(images["AIC_1"]["permission_status"], "unreviewed")

    def test_manifest_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ArtiFact_clean.csv"
            write_fixture(source, self.fixture_rows())
            manifests = []
            for suffix in ("one", "two"):
                output = root / suffix
                build_corpus(
                    source,
                    output,
                    corpus_version="fixture-v1",
                    source_revision="deadbeef",
                    retrieved_at="2026-08-03T00:00:00Z",
                )
                manifests.append((output / "build-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifests[0], manifests[1])
            parsed = json.loads(manifests[0])
            self.assertEqual(parsed["source"]["input_filename"], "ArtiFact_clean.csv")

    def test_failed_build_leaves_no_partial_output_and_retry_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ArtiFact_clean.csv"
            write_fixture(source, self.fixture_rows())

            for initial_state in ("absent", "empty"):
                with self.subTest(initial_state=initial_state):
                    output = root / f"build-{initial_state}"
                    if initial_state == "empty":
                        output.mkdir()

                    with patch(
                        "pipeline.build._write_sparse_npz",
                        side_effect=CorpusBuildError("injected staging failure"),
                    ):
                        with self.assertRaisesRegex(
                            CorpusBuildError, "injected staging failure"
                        ):
                            build_corpus(
                                source,
                                output,
                                corpus_version="fixture-v1",
                                source_revision="deadbeef",
                                retrieved_at="2026-08-03T00:00:00Z",
                            )

                    if initial_state == "absent":
                        self.assertFalse(output.exists())
                    else:
                        self.assertEqual(list(output.iterdir()), [])
                    self.assertEqual(
                        list(root.glob(f".{output.name}.staging-*")), []
                    )

                    build_corpus(
                        source,
                        output,
                        corpus_version="fixture-v1",
                        source_revision="deadbeef",
                        retrieved_at="2026-08-03T00:00:00Z",
                    )
                    self.assertTrue((output / "build-manifest.json").is_file())
                    self.assertTrue((output / "corpus.csv").is_file())

    def test_image_input_policy_is_preserved_only_when_declared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "nga-visual.csv"
            output = root / "build"
            row = {
                **self.fixture_rows()[0],
                "object_ID": "NGA_1",
                "image_url": (
                    "https://api.nga.gov/iiif/example/full/!1024,1024/0/default.jpg"
                ),
                "image_path": "",
                "image_input_policy": "nga-iiif-fit-1024-short-side-256/v1",
            }
            write_fixture(source, [row], (*FIELDS, "image_input_policy"))

            build_corpus(
                source,
                output,
                corpus_version="nga-fixture-v1",
                source_revision="deadbeef",
                retrieved_at="2026-08-13T00:00:00Z",
            )

            with (output / "images.manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                images = list(reader)
                self.assertIn("image_input_policy", reader.fieldnames or ())
            self.assertEqual(
                images[0]["image_input_policy"],
                "nga-iiif-fit-1024-short-side-256/v1",
            )

    def test_records_an_explicit_catalog_record_counting_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            output = root / "build"
            write_fixture(source, self.fixture_rows(), FIELDS)

            manifest = build_corpus(
                source,
                output,
                corpus_version="catalog-record-fixture-v1",
                source_revision="deadbeef",
                retrieved_at="2026-08-14T00:00:00Z",
                counting_unit="catalog-record",
            )

            self.assertEqual(manifest["counting_unit"], "catalog-record")
            self.assertEqual(manifest["corpus"]["countingUnit"], "catalog-record")

    def test_rejects_unknown_counting_unit_before_expensive_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            write_fixture(source, self.fixture_rows(), FIELDS)

            with self.assertRaisesRegex(CorpusBuildError, "counting_unit must be one of"):
                build_corpus(
                    source,
                    root / "build",
                    corpus_version="bad-unit-v1",
                    source_revision="deadbeef",
                    counting_unit="physical-objects",
                )

    def test_dirty_filename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "ArtiFact_clean_dirty.csv"
            write_fixture(source, self.fixture_rows())
            with self.assertRaisesRegex(CorpusBuildError, "dirty"):
                build_corpus(
                    source,
                    Path(temporary) / "output",
                    corpus_version="fixture-v1",
                    source_revision="deadbeef",
                )

    def test_error_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "apparently-clean.csv"
            fields = (*FIELDS, "title_error")
            rows = [{**row, "title_error": "corrupted"} for row in self.fixture_rows()]
            write_fixture(source, rows, fields)
            with self.assertRaisesRegex(CorpusBuildError, "error columns"):
                build_corpus(
                    source,
                    Path(temporary) / "output",
                    corpus_version="fixture-v1",
                    source_revision="deadbeef",
                )


if __name__ == "__main__":
    unittest.main()
