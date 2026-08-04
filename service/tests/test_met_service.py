from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.met import build_met_corpus
from mnemosyne_search.met_artifacts import MetKeywordArtifacts
from mnemosyne_search.met_client import FixtureMetClient
from mnemosyne_search.met_service import METADATA_WARNING, MetKeywordConfig, MetKeywordSearchService
from mnemosyne_search.models import SearchRequest
from mnemosyne_search.parsing import parse_query


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


def build_fixture(root: Path) -> MetKeywordArtifacts:
    source = root / "MetObjects.csv"
    rows = [
        (10, "Horse One", "Ada", 1880, "Paintings", "Horses"),
        (11, "Horse Two", "Ben", 1881, "Paintings", "Horses"),
        (12, "Lion One", "Cy", 1900, "Sculpture", "Lions"),
        (13, "Ship One", "Dee", 1901, "Paintings", "Ships"),
        (14, "Private", "Eve", 1902, "Paintings", "Horses"),
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for object_id, title, artist, year, classification, tags in rows:
            writer.writerow(
                {
                    "Object ID": object_id,
                    "Is Public Domain": object_id != 14,
                    "Department": "Fixture Department",
                    "Object Name": "Artwork",
                    "Title": title,
                    "Artist Display Name": artist,
                    "Object Date": year,
                    "Object Begin Date": year,
                    "Object End Date": year,
                    "Medium": "Fixture medium",
                    "Classification": classification,
                    "Tags": tags,
                    "Link Resource": f"https://www.metmuseum.org/art/collection/search/{object_id}",
                }
            )
    image_ids = root / "met-has-images.json"
    image_ids.write_text(json.dumps({"objectIDs": [10, 11, 12, 13, 14]}), encoding="utf-8")
    output = root / "corpus"
    build_met_corpus(
        source,
        output,
        corpus_version="met-integration-v1",
        source_revision="pinned-fixture",
        image_ids_path=image_ids,
        retrieved_at="2026-08-03T00:00:00Z",
    )
    return MetKeywordArtifacts.load(output)


class MetKeywordSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = build_fixture(self.root)
        self.client = FixtureMetClient(
            {
                "horse": [10, 11, 14, 999999],
                "lion": [12],
                "none": [],
            },
            {
                10: {
                    "objectID": 10,
                    "isPublicDomain": True,
                    "title": "Horse One (live detail)",
                    "artistDisplayName": "Ada Artist",
                    "primaryImageSmall": "https://images.metmuseum.org/10-small.jpg",
                    "objectURL": "https://www.metmuseum.org/art/collection/search/10",
                },
                11: {
                    "objectID": 11,
                    "isPublicDomain": True,
                    "primaryImageSmall": "https://images.metmuseum.org/11-small.jpg",
                },
                12: {
                    "objectID": 12,
                    "isPublicDomain": True,
                    "primaryImageSmall": "https://images.metmuseum.org/12-small.jpg",
                },
            },
        )
        self.service = MetKeywordSearchService(
            self.artifacts,
            self.client,
            config=MetKeywordConfig(minimum_denominator=1, evidence_count=5),
            clock=lambda: datetime(2026, 8, 3, tzinfo=timezone.utc),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_multi_series_frequency_uses_local_eligible_denominator(self) -> None:
        response = self.service.search(SearchRequest(query="horse, lion"))

        self.assertEqual(response["metric"]["unit"], "frequency")
        self.assertEqual(response["metric"]["label"], "Met metadata frequency")
        self.assertEqual(response["corpus"]["count"], 4)
        self.assertEqual([series["k"] for series in response["series"]], [2, 1])
        self.assertEqual([series["totalMatches"] for series in response["series"]], [4, 1])
        self.assertEqual([item["denominator"] for item in response["bins"]], [2.0, 0.0, 2.0])
        horse_values = [point["value"] for point in response["series"][0]["points"]]
        lion_values = [point["value"] for point in response["series"][1]["points"]]
        self.assertEqual(horse_values, [1.0, 0.0, 0.0])
        self.assertEqual(lion_values, [0.0, 0.0, 0.5])
        self.assertEqual(response["warnings"][0], METADATA_WARNING)
        json.dumps(response, allow_nan=False)

    def test_cache_reuses_each_query_across_multi_series_requests(self) -> None:
        self.service.search(SearchRequest(query="horse, lion"))
        self.service.search(SearchRequest(query="lion, horse"))
        self.assertEqual(self.client.search_calls, [("horse", "broad"), ("lion", "broad")])

    def test_selected_evidence_fetches_only_matching_cards_for_the_bin(self) -> None:
        horse_id = parse_query("horse")[0].id
        response = self.service.search(
            SearchRequest(
                query="horse, lion",
                selected_query_id=horse_id,
                selected_bin_key="1880:1889",
            )
        )
        cards = response["selectedEvidence"]["slices"]["randomContributors"]
        self.assertEqual({card["artworkId"] for card in cards}, {"MET_10", "MET_11"})
        self.assertTrue(all(card["imageUrl"] for card in cards))
        self.assertTrue(all(card["rawScore"] is None for card in cards))
        self.assertEqual(set(self.client.object_calls), {10, 11})

    def test_filter_recomputes_the_denominator_and_cache_key(self) -> None:
        unfiltered = self.service.search(SearchRequest(query="lion"))
        filtered = self.service.search(
            SearchRequest(query="lion", filters={"classification": ("Sculpture",)})
        )
        self.assertNotEqual(unfiltered["series"][0]["cacheKey"], filtered["series"][0]["cacheKey"])
        self.assertEqual(filtered["corpus"]["count"], 1)
        lion_point = next(
            point for point in filtered["series"][0]["points"] if point["objectCount"] == 1
        )
        self.assertEqual(lion_point["value"], 1.0)

    def test_empty_local_match_is_a_zero_line_with_a_warning(self) -> None:
        response = self.service.search(SearchRequest(query="none"))
        self.assertTrue(all(point["value"] == 0 for point in response["series"][0]["points"]))
        self.assertTrue(any("No eligible corpus matches" in warning for warning in response["warnings"]))
        self.assertEqual(response["selectedEvidence"]["slices"]["randomContributors"], [])


if __name__ == "__main__":
    unittest.main()
