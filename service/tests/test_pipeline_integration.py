from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.build import build_corpus
from pipeline.embeddings import DeterministicTestEncoder, build_embedding_index
from mnemosyne_search.artifacts import ArtifactBundle
from mnemosyne_search.encoders import FixtureTextEncoder
from mnemosyne_search.models import SearchRequest
from mnemosyne_search.prompting import PromptEnsemble
from mnemosyne_search.service import SearchConfig, SearchService


class PipelineToSearchIntegrationTests(unittest.TestCase):
    def test_real_pipeline_bundle_serves_independent_multi_series_lift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ArtiFact_clean.csv"
            fields = (
                "object_ID",
                "title",
                "date_begin",
                "date_end",
                "date_begin_bce",
                "date_end_bce",
                "image_url",
                "public_domain",
            )
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for index, year in enumerate((1800, 1810, 1820, 1830), start=1):
                    writer.writerow(
                        {
                            "object_ID": f"AIC_{index:03d}",
                            "title": f"Fixture {index}",
                            "date_begin": year,
                            "date_end": year,
                            "date_begin_bce": "false",
                            "date_end_bce": "false",
                            "image_url": f"https://example.test/{index}.jpg",
                            "public_domain": "true",
                        }
                    )

            corpus_dir = root / "corpus"
            build_corpus(
                source,
                corpus_dir,
                corpus_version="integration-v1",
                source_revision="pinned-fixture-revision",
                retrieved_at="2026-08-03T00:00:00Z",
            )
            bundle_dir = root / "bundle"
            encoder = DeterministicTestEncoder(16)
            build_embedding_index(corpus_dir, bundle_dir, encoder)

            artifacts = ArtifactBundle.load(bundle_dir)
            vectors = np.asarray(artifacts.embeddings)
            text_encoder = FixtureTextEncoder(
                {"horse": vectors[0].tolist(), "ship": vectors[1].tolist()},
                model_id=encoder.encoder_id,
                model_version=encoder.revision,
            )
            service = SearchService(
                artifacts,
                text_encoder,
                prompt_ensemble=PromptEnsemble(
                    version="integration-prompts-v1", templates=("{query}",)
                ),
                config=SearchConfig(percentile=0.25, minimum_denominator=1),
                prefer_faiss=False,
            )

            response = service.search(SearchRequest(query="horse, ship"))

            self.assertEqual(response["schemaVersion"], "mnemosyne.search.v1")
            self.assertEqual(response["corpus"]["countingUnit"], "physical-object")
            self.assertEqual([query["label"] for query in response["queries"]], ["horse", "ship"])
            self.assertEqual([series["k"] for series in response["series"]], [1, 1])
            self.assertEqual([item["denominator"] for item in response["bins"]], [1.0] * 4)
            self.assertEqual(response["series"][0]["points"][0]["lift"], 4.0)
            self.assertEqual(response["series"][1]["points"][1]["lift"], 4.0)


if __name__ == "__main__":
    unittest.main()
