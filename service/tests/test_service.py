from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mnemosyne_search.artifacts import ArtifactBundle
from mnemosyne_search.cache import InMemorySeriesCache
from mnemosyne_search.encoders import FixtureTextEncoder
from mnemosyne_search.index import NumpyFlatIPIndex
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


class PromptAwareCountingEncoder(FixtureTextEncoder):
    """Map production prompt variants back to deterministic fixture vectors."""

    _PREFIXES = ("an artwork depicting ", "a work of art about ")

    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        super().__init__(
            payload["vectors"],
            model_id=payload["modelId"],
            model_version=payload["modelVersion"],
        )
        self.calls: list[list[str]] = []

    def encode(self, texts):
        prompts = list(texts)
        self.calls.append(prompts)
        canonical = []
        for prompt in prompts:
            canonical.append(
                next(
                    (
                        prompt[len(prefix) :]
                        for prefix in self._PREFIXES
                        if prompt.startswith(prefix)
                    ),
                    prompt,
                )
            )
        return super().encode(canonical)


class RecordingExactIndex:
    def __init__(self, embeddings: np.ndarray) -> None:
        self._delegate = NumpyFlatIPIndex(embeddings)
        self.backend = self._delegate.backend
        self.search_calls: list[dict[str, object]] = []

    def search(
        self,
        queries: np.ndarray,
        k: int,
        *,
        eligible_indices: np.ndarray | None = None,
    ):
        self.search_calls.append(
            {
                "queries": np.asarray(queries, dtype=np.float32).copy(),
                "k": k,
                "eligible_indices": (
                    None
                    if eligible_indices is None
                    else np.asarray(eligible_indices, dtype=np.int64).copy()
                ),
            }
        )
        return self._delegate.search(
            queries, k, eligible_indices=eligible_indices
        )

    def score(self, query: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return self._delegate.score(query, indices)


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

    def _prompt_service(
        self,
        *,
        cache: InMemorySeriesCache | None = None,
    ) -> tuple[SearchService, PromptAwareCountingEncoder, RecordingExactIndex]:
        encoder = PromptAwareCountingEncoder(FIXTURES / "query-embeddings.json")
        index = RecordingExactIndex(self.artifacts.embeddings)
        service = SearchService(
            self.artifacts,
            encoder,
            prompt_ensemble=PromptEnsemble(version="fixture-prompts-v3"),
            config=SearchConfig(
                percentile=0.1,
                evidence_percentile=0.1,
                minimum_denominator=1,
                minimum_evidence_clusters=1,
                minimum_bin_evidence_clusters=1,
            ),
            index=index,
            cache=cache,
            prefer_faiss=False,
            clock=lambda: datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        )
        return service, encoder, index

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

    def test_one_uncached_series_searches_combined_and_diagnostic_vectors_once(self) -> None:
        service, encoder, index = self._prompt_service()

        response = service.search(SearchRequest(query="horse"))

        self.assertEqual(len(index.search_calls), 1)
        search_call = index.search_calls[0]
        self.assertEqual(search_call["k"], 1)
        self.assertIsNone(search_call["eligible_indices"])
        np.testing.assert_allclose(
            search_call["queries"],
            np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (4, 1),
            ),
        )
        self.assertEqual(
            encoder.calls,
            [
                [
                    "horse",
                    "an artwork depicting horse",
                    "a work of art about horse",
                ]
            ],
        )
        self.assertEqual(response["series"][0]["candidateK"], 1)
        self.assertEqual(
            response["series"][0]["diagnostics"]["promptTopKJaccard"],
            1.0,
        )
        self.assertEqual(
            response["selectedEvidence"]["slices"]["strongest"][0]["artworkId"],
            "fixture-000",
        )

    def test_multi_series_one_scan_matches_independent_series_semantics(self) -> None:
        service, encoder, index = self._prompt_service()

        response = service.search(SearchRequest(query="horse, ship"))

        self.assertEqual(len(index.search_calls), 1)
        expected_vectors = np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            index.search_calls[0]["queries"], expected_vectors
        )
        self.assertEqual(
            encoder.calls,
            [
                [
                    "horse",
                    "an artwork depicting horse",
                    "a work of art about horse",
                    "ship",
                    "an artwork depicting ship",
                    "a work of art about ship",
                ]
            ],
        )

        horse_service, _horse_encoder, _horse_index = self._prompt_service()
        ship_service, _ship_encoder, _ship_index = self._prompt_service()
        horse = horse_service.search(SearchRequest(query="horse"))
        ship = ship_service.search(SearchRequest(query="ship"))

        self.assertEqual(response["series"], [horse["series"][0], ship["series"][0]])
        self.assertEqual(response["selectedEvidence"], horse["selectedEvidence"])
        self.assertEqual(
            [series["diagnostics"] for series in response["series"]],
            [horse["series"][0]["diagnostics"], ship["series"][0]["diagnostics"]],
        )

    def test_mixed_cached_and_uncached_series_only_scans_the_missing_vectors(self) -> None:
        service, encoder, index = self._prompt_service()
        horse = service.search(SearchRequest(query="horse"))
        index.search_calls.clear()

        mixed = service.search(SearchRequest(query="horse, ship"))

        self.assertEqual(len(index.search_calls), 1)
        np.testing.assert_allclose(
            index.search_calls[0]["queries"],
            np.tile(
                np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                (4, 1),
            ),
        )
        self.assertEqual(mixed["series"][0], horse["series"][0])
        self.assertEqual(
            encoder.calls,
            [
                [
                    "horse",
                    "an artwork depicting horse",
                    "a work of art about horse",
                ],
                [
                    "ship",
                    "an artwork depicting ship",
                    "a work of art about ship",
                ],
            ],
        )

        index.search_calls.clear()
        encoder_calls = list(encoder.calls)
        repeated = service.search(SearchRequest(query="horse, ship"))
        self.assertEqual(index.search_calls, [])
        self.assertEqual(encoder.calls, encoder_calls)
        self.assertEqual(repeated["series"], mixed["series"])

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

    def test_preserves_an_empty_injected_cache(self) -> None:
        cache: InMemorySeriesCache = InMemorySeriesCache(max_entries=2)
        service = SearchService(
            self.artifacts,
            self.encoder,
            prompt_ensemble=PromptEnsemble(
                version="fixture-prompts-v1", templates=("{query}",)
            ),
            prefer_faiss=False,
            cache=cache,
        )

        self.assertIs(service.cache, cache)

    def test_cache_capacity_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_entries must be positive"):
            InMemorySeriesCache(max_entries=0)

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

    def test_evidence_response_reuses_cached_series_and_omits_timeline(self) -> None:
        parsed = self.service.search(SearchRequest(query="horse, train"))
        train_id = parsed["queries"][1]["id"]
        encoder_calls = list(self.encoder.calls)

        response = self.service.evidence(
            SearchRequest(
                query="horse, train",
                selected_query_id=train_id,
                selected_bin_key="1900-1949",
            )
        )

        self.assertEqual(
            set(response), {"schemaVersion", "selectedEvidence", "generatedAt"}
        )
        self.assertEqual(response["schemaVersion"], "mnemosyne.evidence.v1")
        self.assertEqual(response["generatedAt"], "2026-08-03T12:00:00Z")
        self.assertNotIn("bins", response)
        self.assertNotIn("series", response)
        self.assertEqual(self.encoder.calls, encoder_calls)
        evidence = response["selectedEvidence"]
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence["queryId"], train_id)
        self.assertEqual(evidence["binKey"], "1900-1949")
        self.assertEqual(evidence["slices"]["strongest"][0]["artworkId"], "fixture-004")

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
            config=SearchConfig(
                percentile=1.0,
                evidence_percentile=0.1,
                minimum_evidence_score=1.0,
            ),
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
        nearest = response["series"][0]["nearestMatches"]
        self.assertEqual(
            [card["artworkId"] for card in nearest[:3]],
            ["fixture-006", "fixture-008", "fixture-009"],
        )
        self.assertLessEqual(len(nearest), 20)
        self.assertEqual(
            [card["rawScore"] for card in nearest],
            sorted(
                [card["rawScore"] for card in nearest],
                reverse=True,
            ),
        )
        self.assertTrue(all(not card["contributor"] for card in nearest))
        self.assertTrue(
            all(card["contributionWeight"] == 0.0 for card in nearest)
        )
        self.assertEqual(
            len({card["visualClusterId"] for card in nearest}), len(nearest)
        )

    def test_score_qualified_series_omits_exploratory_nearest_matches(self) -> None:
        response = self.service.search(SearchRequest(query="horse"))

        self.assertGreater(response["series"][0]["k"], 0)
        self.assertNotIn("nearestMatches", response["series"][0])

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
        self.assertNotIn("nearestMatches", response["series"][0])

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

    def test_filter_values_are_trimmed_before_matching(self) -> None:
        plain = self.service.search(
            SearchRequest(query="horse", filters={"institution": ("The Met",)})
        )
        padded = self.service.search(
            SearchRequest(query="horse", filters={"institution": ("  The Met  ",)})
        )

        self.assertEqual(plain["corpus"], padded["corpus"])
        self.assertEqual(plain["series"], padded["series"])

    def test_cache_key_versions_reliability_and_diagnostic_settings(self) -> None:
        baseline = self.service.search(SearchRequest(query="horse"))["series"][0][
            "cacheKey"
        ]
        changed = SearchService(
            self.artifacts,
            self.encoder,
            prompt_ensemble=PromptEnsemble(
                version="fixture-prompts-v1", templates=("{query}",)
            ),
            config=SearchConfig(
                percentile=0.1,
                evidence_percentile=0.1,
                minimum_denominator=2,
                minimum_evidence_clusters=1,
                minimum_bin_evidence_clusters=1,
                control_sample_size=2,
            ),
            prefer_faiss=False,
        ).search(SearchRequest(query="horse"))["series"][0]["cacheKey"]

        self.assertNotEqual(baseline, changed)

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
