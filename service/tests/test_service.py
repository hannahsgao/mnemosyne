from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from mnemosyne_search.artifacts import ArtifactBundle
from mnemosyne_search.cache import InMemorySeriesCache
from mnemosyne_search.encoders import FixtureTextEncoder
from mnemosyne_search.models import SearchRequest
from mnemosyne_search.prompting import PromptEnsemble
from mnemosyne_search.service import (
    METRIC_ID,
    SearchConfig,
    SearchService,
    _display_image_url,
    _source_record_url,
)


FIXTURES = Path(__file__).parent / "fixtures"


class CountingFixtureEncoder(FixtureTextEncoder):
    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        super().__init__(
            payload["vectors"],
            model_id=payload["modelId"],
            model_version=payload["modelVersion"],
        )
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return super().encode(texts)


class SearchServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = ArtifactBundle.load(FIXTURES)
        self.encoder = CountingFixtureEncoder(FIXTURES / "query-embeddings.json")
        self.service = SearchService(
            self.artifacts,
            self.encoder,
            prompt_ensemble=PromptEnsemble(version="fixture-prompts-v1", templates=("{query}",)),
            config=SearchConfig(
                percentile=0.1,
                evidence_percentile=0.1,
                minimum_denominator=1,
                minimum_evidence_clusters=1,
                minimum_bin_evidence_clusters=1,
            ),
            prefer_faiss=False,
            clock=lambda: datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        )

    def test_multi_series_response_uses_independent_score_qualified_sets(self) -> None:
        response = self.service.search(SearchRequest(query="horse, ship, Horse"))
        self.assertEqual(response["schemaVersion"], "mnemosyne.search.v1")
        self.assertEqual([item["label"] for item in response["queries"]], ["horse", "ship"])
        self.assertEqual(response["metric"]["id"], METRIC_ID)
        self.assertEqual(response["metric"]["percentile"], 0.1)
        self.assertIn(
            "Zero means no score-qualified matches were found in this corpus",
            response["metric"]["description"],
        )
        self.assertEqual([item["denominator"] for item in response["bins"]], [4.0, 3.0, 3.0])
        self.assertEqual([item["k"] for item in response["series"]], [1, 1])

        horse, ship = response["series"]
        self.assertAlmostEqual(horse["points"][0]["hitMass"], 1.0)
        self.assertAlmostEqual(horse["points"][0]["share"], 0.25)
        self.assertAlmostEqual(horse["points"][0]["lift"], 2.5)
        self.assertAlmostEqual(ship["points"][0]["lift"], 2.5)
        self.assertEqual(response["generatedAt"], "2026-08-03T12:00:00Z")

    def test_met_evidence_uses_web_large_display_derivative(self) -> None:
        self.assertEqual(
            _display_image_url(
                "https://images.metmuseum.org/CRDImages/gr/original/My Image–One.jpg"
            ),
            "https://images.metmuseum.org/CRDImages/gr/web-large/My%20Image%E2%80%93One.jpg",
        )
        self.assertEqual(
            _source_record_url("http://www.metmuseum.org/art/collection/search/123"),
            "https://www.metmuseum.org/art/collection/search/123",
        )

    def test_reuses_each_cached_series_across_different_multi_query_requests(self) -> None:
        self.service.search(SearchRequest(query="horse, ship"))
        self.service.search(SearchRequest(query="horse, train"))
        self.assertEqual(self.encoder.calls, [["horse", "ship"], ["train"]])

    def test_multi_series_response_survives_cache_smaller_than_query_count(self) -> None:
        service = SearchService(
            self.artifacts,
            self.encoder,
            prompt_ensemble=PromptEnsemble(version="fixture-prompts-v1", templates=("{query}",)),
            prefer_faiss=False,
            cache=InMemorySeriesCache(max_entries=1),
        )

        response = service.search(SearchRequest(query="horse, ship"))

        self.assertEqual(
            [series["queryId"] for series in response["series"]],
            [response["queries"][0]["id"], response["queries"][1]["id"]],
        )

    def test_selects_requested_series_and_bin_for_evidence(self) -> None:
        parsed = self.service.search(SearchRequest(query="horse, train"))
        train_id = parsed["queries"][1]["id"]
        response = self.service.search(
            SearchRequest(
                query="horse, train",
                selected_query_id=train_id,
                selected_bin_key="1900-1949",
            )
        )
        evidence = response["selectedEvidence"]
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence["queryId"], train_id)
        self.assertEqual(evidence["binKey"], "1900-1949")
        self.assertEqual(evidence["slices"]["strongest"][0]["artworkId"], "fixture-004")
        self.assertTrue(evidence["slices"]["strongest"][0]["contributor"])
        self.assertTrue(evidence["slices"]["bestNonContributors"])
        self.assertTrue(evidence["slices"]["randomDenominator"])

    def test_evidence_uses_a_stricter_tail_than_the_timeline(self) -> None:
        service = SearchService(
            self.artifacts,
            self.encoder,
            prompt_ensemble=PromptEnsemble(
                version="fixture-prompts-v1", templates=("{query}",)
            ),
            config=SearchConfig(
                percentile=0.3,
                evidence_percentile=0.1,
                minimum_denominator=1,
                minimum_evidence_clusters=1,
                minimum_bin_evidence_clusters=1,
            ),
            prefer_faiss=False,
        )

        response = service.search(SearchRequest(query="horse"))
        evidence = response["selectedEvidence"]
        self.assertIsNotNone(evidence)
        assert evidence is not None

        self.assertEqual(response["series"][0]["k"], 1)
        self.assertEqual(response["series"][0]["candidateK"], 3)
        self.assertEqual(evidence["percentile"], 0.1)
        self.assertEqual(evidence["contributorCount"], 1)
        self.assertEqual(
            [card["artworkId"] for card in evidence["slices"]["strongest"]],
            ["fixture-000"],
        )
        self.assertEqual(len(response["series"][0]["points"]), 1)
        self.assertEqual(response["series"][0]["points"][0]["objectCount"], 1)
        self.assertEqual(evidence["threshold"], response["series"][0]["threshold"])
        self.assertGreater(
            evidence["threshold"], response["series"][0]["candidateThreshold"]
        )

    def test_strict_evidence_floor_can_abstain_instead_of_inventing_matches(self) -> None:
        service = SearchService(
            self.artifacts,
            self.encoder,
            prompt_ensemble=PromptEnsemble(
                version="fixture-prompts-v1", templates=("{query}",)
            ),
            config=SearchConfig(minimum_evidence_score=1.0),
            prefer_faiss=False,
        )

        response = service.search(SearchRequest(query="flat"))

        self.assertTrue(response["series"][0]["lowSignal"])
        self.assertIn(
            "visual-evidence score cutoff",
            response["series"][0]["diagnostics"]["reasons"][-1],
        )
        self.assertIsNone(response["selectedEvidence"])
        self.assertEqual(response["series"][0]["points"], [])

    def test_rejects_evidence_selection_for_an_unplotted_bin(self) -> None:
        response = self.service.search(
            SearchRequest(query="horse", selected_bin_key="1900-1949")
        )

        self.assertIsNone(response["selectedEvidence"])

    def test_score_qualified_rows_in_sparse_bins_do_not_create_invisible_points(self) -> None:
        service = SearchService(
            self.artifacts,
            self.encoder,
            prompt_ensemble=PromptEnsemble(
                version="fixture-prompts-v1", templates=("{query}",)
            ),
            config=SearchConfig(minimum_denominator=1000),
            prefer_faiss=False,
        )

        response = service.search(SearchRequest(query="horse"))

        self.assertEqual(response["series"][0]["points"], [])
        self.assertEqual(response["series"][0]["suppressedBinKeys"], [])
        self.assertTrue(response["series"][0]["lowSignal"])
        self.assertIn(
            "statistically unreliable periods",
            response["series"][0]["diagnostics"]["reasons"][-1],
        )
        self.assertIsNone(response["selectedEvidence"])

    def test_series_distinguishes_suppressed_positive_bins_from_honest_zeros(self) -> None:
        service = SearchService(
            self.artifacts,
            self.encoder,
            prompt_ensemble=PromptEnsemble(
                version="fixture-prompts-v1", templates=("{query}",)
            ),
            config=SearchConfig(
                minimum_denominator=1,
                minimum_evidence_clusters=2,
                minimum_bin_evidence_clusters=2,
            ),
            prefer_faiss=False,
        )

        response = service.search(SearchRequest(query="horse"))

        series = response["series"][0]
        self.assertEqual(series["points"], [])
        self.assertEqual(series["suppressedBinKeys"], ["1800-1849"])
        plotted = {point["binKey"] for point in series["points"]}
        suppressed = set(series["suppressedBinKeys"])
        reliable_absent = [
            bin_item["key"]
            for bin_item in response["bins"]
            if not bin_item["belowMinimumDenominator"]
            and bin_item["key"] not in plotted
            and bin_item["key"] not in suppressed
        ]
        self.assertEqual(reliable_absent, ["1850-1899", "1900-1949"])
        self.assertIsNone(response["selectedEvidence"])

    def test_filter_changes_eligible_corpus_denominators_and_cache_key(self) -> None:
        unfiltered = self.service.search(SearchRequest(query="horse"))
        filtered = self.service.search(
            SearchRequest(query="horse", filters={"institution": ("The Met",)})
        )
        self.assertEqual(filtered["corpus"]["count"], 4)
        self.assertEqual([item["denominator"] for item in filtered["bins"]], [3.0, 1.0, 0.0])
        self.assertNotEqual(unfiltered["series"][0]["cacheKey"], filtered["series"][0]["cacheKey"])

    def test_diagnostics_can_mark_and_explain_low_signal(self) -> None:
        strict_service = SearchService(
            self.artifacts,
            self.encoder,
            prompt_ensemble=PromptEnsemble(version="fixture-prompts-v1", templates=("{query}",)),
            config=SearchConfig(minimum_standardized_separation=1000),
            prefer_faiss=False,
        )
        response = strict_service.search(SearchRequest(query="horse"))
        self.assertTrue(response["series"][0]["lowSignal"])
        self.assertIn("deterministic controls", response["series"][0]["diagnostics"]["reasons"][0])
        self.assertTrue(response["warnings"])

    def test_response_is_strict_json_without_nan(self) -> None:
        response = self.service.search(SearchRequest(query='"still life, fruit", horse'))
        encoded = json.dumps(response, allow_nan=False)
        self.assertIn("still life, fruit", encoded)


if __name__ == "__main__":
    unittest.main()
