from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scipy import sparse

from pipeline.build import CorpusBuildError
from pipeline.dates import DateConfig
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
        {
            "Object ID": "15",
            "Is Public Domain": "True",
            "Title": "Geological specimen",
            "Object Date": "400,000–300,000 BCE",
            "Object Begin Date": "-400000",
            "Object End Date": "-300000",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class MetCorpusBuildTests(unittest.TestCase):
    def test_indexes_all_dateable_rows_and_preserves_source_snapshots(self) -> None:
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
                date_config=DateConfig(min_year=-15_000, max_year=2029),
            )

            self.assertEqual(manifest["source"]["kind"], "met-open-access-csv-with-local-fts")
            self.assertEqual(manifest["source"]["eligibility"], "usable-date")
            self.assertEqual(manifest["counts"]["input_rows"], 6)
            self.assertEqual(manifest["counts"]["canonical_rows"], 4)
            self.assertEqual(manifest["counts"]["non_public_domain_rows"], 1)
            self.assertEqual(manifest["counts"]["rows_without_known_image"], 2)
            self.assertEqual(manifest["counts"]["rejected_without_usable_date"], 1)
            self.assertEqual(manifest["counts"]["rejected_outside_date_bounds"], 1)
            self.assertTrue((output / "source-payloads" / source.name).is_file())
            self.assertTrue((output / "source-payloads" / image_ids.name).is_file())

            with (output / "corpus.csv").open(encoding="utf-8", newline="") as handle:
                rows = {row["source_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(set(rows), {"10", "11", "12", "14"})
            self.assertEqual(rows["10"]["tags"], "Horses|Shorelines")
            self.assertEqual(rows["10"]["object_wikidata_url"], "https://www.wikidata.org/wiki/Q10")
            self.assertEqual(rows["14"]["geography"], "Thebes; Egypt")
            self.assertEqual(rows["14"]["date_start"], "-1405")
            self.assertEqual(rows["14"]["metadata_license"], MET_CC0_URI)
            self.assertEqual(rows["14"]["image_available"], "True")
            self.assertEqual(rows["14"]["image_url"], "")
            self.assertEqual(rows["11"]["public_domain"], "False")
            self.assertEqual(rows["12"]["image_available"], "False")

            with (output / "coverage.csv").open(encoding="utf-8", newline="") as handle:
                coverage = list(csv.DictReader(handle))
            met_coverage = next(
                row for row in coverage if row["dimension"] == "institution" and row["value"] == "met"
            )
            self.assertEqual(met_coverage["image_count"], "3")

            weights = sparse.load_npz(output / "date-weights.npz")
            self.assertEqual(weights.shape[0], 4)
            self.assertAlmostEqual(float(weights.sum()), 4.0)

            index_path = output / manifest["files"]["keywordIndex"]
            with sqlite3.connect(index_path) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM artworks").fetchone()[0], 4)
                matches = connection.execute(
                    """
                    SELECT artworks.source_id
                    FROM artwork_fts
                    JOIN artworks ON artworks.row_id = artwork_fts.rowid
                    WHERE artwork_fts MATCH '"horse"'
                    ORDER BY artworks.row_id
                    """
                ).fetchall()
            self.assertEqual(matches, [(10,)])

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
