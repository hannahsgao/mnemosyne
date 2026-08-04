from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from mnemosyne_search.artifacts import ArtifactBundle
from mnemosyne_search.encoders import FixtureTextEncoder
from mnemosyne_search.models import SearchRequest
from mnemosyne_search.prompting import PromptEnsemble
from mnemosyne_search.service import METRIC_ID, SearchConfig, SearchService


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
            prefer_faiss=False,
            clock=lambda: datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        )

    def test_multi_series_response_uses_independent_one_percent_top_k(self) -> None:
        response = self.service.search(SearchRequest(query="horse, ship, Horse"))
        self.assertEqual(response["schemaVersion"], "mnemosyne.search.v1")
        self.assertEqual([item["label"] for item in response["queries"]], ["horse", "ship"])
        self.assertEqual(response["metric"]["id"], METRIC_ID)
        self.assertEqual(response["metric"]["percentile"], 0.01)
        self.assertEqual([item["denominator"] for item in response["bins"]], [4.0, 3.0, 3.0])
        self.assertEqual([item["k"] for item in response["series"]], [1, 1])

        horse, ship = response["series"]
        self.assertAlmostEqual(horse["points"][0]["hitMass"], 1.0)
        self.assertAlmostEqual(horse["points"][0]["share"], 0.25)
        self.assertAlmostEqual(horse["points"][0]["lift"], 25.0)
        self.assertAlmostEqual(ship["points"][0]["lift"], 25.0)
        self.assertEqual(response["generatedAt"], "2026-08-03T12:00:00Z")

    def test_reuses_each_cached_series_across_different_multi_query_requests(self) -> None:
        self.service.search(SearchRequest(query="horse, ship"))
        self.service.search(SearchRequest(query="horse, train"))
        self.assertEqual(self.encoder.calls, [["horse", "ship"], ["train"]])

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
        self.assertEqual(evidence["queryId"], train_id)
        self.assertEqual(evidence["binKey"], "1900-1949")
        self.assertEqual(evidence["slices"]["strongest"][0]["artworkId"], "fixture-004")
        self.assertTrue(evidence["slices"]["strongest"][0]["contributor"])
        self.assertTrue(evidence["slices"]["bestNonContributors"])
        self.assertTrue(evidence["slices"]["randomDenominator"])

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
