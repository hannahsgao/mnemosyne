from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scipy import sparse

from pipeline.build import CorpusBuildError
from pipeline.met import MET_CC0_URI, build_met_corpus


FIELDS = (
    "Object ID",
    "Is Public Domain",
    "Department",
    "Object Name",
    "Title",
    "Culture",
    "Period",
    "Dynasty",
    "Artist Display Name",
    "Object Date",
    "Object Begin Date",
    "Object End Date",
    "Medium",
    "Credit Line",
    "City",
    "Country",
    "Classification",
    "Link Resource",
    "Object Wikidata URL",
    "Tags",
)


def write_met_csv(path: Path) -> None:
    rows = [
        {
            "Object ID": "10",
            "Is Public Domain": "True",
            "Department": "Paintings",
            "Object Name": "Painting",
            "Title": "Horse at the Shore",
            "Culture": "American",
            "Artist Display Name": "Ada Artist",
            "Object Date": "1880–1890",
            "Object Begin Date": "1880",
            "Object End Date": "1890",
            "Medium": "Oil on canvas",
            "Credit Line": "Gift",
            "Country": "United States",
            "Classification": "Paintings",
            "Link Resource": "https://www.metmuseum.org/art/collection/search/10",
            "Object Wikidata URL": "https://www.wikidata.org/wiki/Q10",
            "Tags": "Horses|Shorelines",
        },
        {
            "Object ID": "11",
            "Is Public Domain": "False",
            "Title": "Copyrighted",
            "Object Date": "1900",
            "Object Begin Date": "1900",
            "Object End Date": "1900",
        },
        {
            "Object ID": "12",
            "Is Public Domain": "True",
            "Title": "No image",
            "Object Date": "1910",
            "Object Begin Date": "1910",
            "Object End Date": "1910",
        },
        {
            "Object ID": "13",
            "Is Public Domain": "True",
            "Title": "Unknown date",
            "Object Date": "Unknown",
            "Object Begin Date": "0",
            "Object End Date": "0",
        },
        {
            "Object ID": "14",
            "Is Public Domain": "True",
            "Department": "Egyptian Art",
            "Object Name": "Relief",
            "Title": "Lion Relief",
            "Culture": "Egyptian",
            "Period": "New Kingdom",
            "Dynasty": "Dynasty 18",
            "Artist Display Name": "",
            "Object Date": "ca. 1400 BCE",
            "Object Begin Date": "-1405",
            "Object End Date": "-1395",
            "Medium": "Stone",
            "City": "Thebes",
            "Country": "Egypt",
            "Classification": "Stone Sculpture",
            "Tags": "Lions",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class MetCorpusBuildTests(unittest.TestCase):
    def test_filters_and_preserves_met_metadata_and_source_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MetObjects.csv"
            image_ids = root / "met-has-images.json"
            output = root / "build"
            write_met_csv(source)
            image_ids.write_text(json.dumps({"total": 3, "objectIDs": [10, 11, 13, 14]}))

            manifest = build_met_corpus(
                source,
                output,
                corpus_version="met-fixture-v1",
                source_revision="deadbeef",
                image_ids_path=image_ids,
                retrieved_at="2026-08-03T00:00:00Z",
            )

            self.assertEqual(manifest["source"]["kind"], "met-open-access-csv-with-api-image-snapshot")
            self.assertEqual(manifest["source"]["eligibility"], "public-domain AND has-image AND usable-date")
            self.assertEqual(manifest["counts"]["input_rows"], 5)
            self.assertEqual(manifest["counts"]["canonical_rows"], 2)
            self.assertEqual(manifest["counts"]["rejected_not_public_domain"], 1)
            self.assertEqual(manifest["counts"]["rejected_without_image"], 1)
            self.assertEqual(manifest["counts"]["rejected_without_usable_date"], 1)
            self.assertTrue((output / "source-payloads" / source.name).is_file())
            self.assertTrue((output / "source-payloads" / image_ids.name).is_file())

            with (output / "corpus.csv").open(encoding="utf-8", newline="") as handle:
                rows = {row["source_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(set(rows), {"10", "14"})
            self.assertEqual(rows["10"]["tags"], "Horses|Shorelines")
            self.assertEqual(rows["10"]["object_wikidata_url"], "https://www.wikidata.org/wiki/Q10")
            self.assertEqual(rows["14"]["geography"], "Thebes; Egypt")
            self.assertEqual(rows["14"]["date_start"], "-1405")
            self.assertEqual(rows["14"]["metadata_license"], MET_CC0_URI)
            self.assertEqual(rows["14"]["image_url"], "")

            weights = sparse.load_npz(output / "date-weights.npz")
            self.assertEqual(weights.shape[0], 2)
            self.assertAlmostEqual(float(weights.sum()), 2.0)

    def test_requires_the_official_met_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MetObjects.csv"
            source.write_text("Object ID,Title\n1,Incomplete\n", encoding="utf-8")
            image_ids = root / "image-ids.json"
            image_ids.write_text("[1]\n", encoding="utf-8")
            with self.assertRaisesRegex(CorpusBuildError, "missing required columns"):
                build_met_corpus(
                    source,
                    root / "build",
                    corpus_version="met-fixture-v1",
                    source_revision="deadbeef",
                    image_ids_path=image_ids,
                )


if __name__ == "__main__":
    unittest.main()
