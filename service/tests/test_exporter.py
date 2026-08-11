from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mnemosyne_search.artifacts import ArtifactBundle
from mnemosyne_search.encoders import FixtureTextEncoder
from mnemosyne_search.exporter import (
    Concept,
    ConceptSource,
    artifact_manifest_sha256,
    export_catalog,
)
from mnemosyne_search.models import SearchRequest
from mnemosyne_search.prompting import PromptEnsemble
from mnemosyne_search.service import SearchConfig, SearchService


FIXTURES = Path(__file__).parent / "fixtures"


class CountingEncoder(FixtureTextEncoder):
    def __init__(self) -> None:
        payload = json.loads(
            (FIXTURES / "query-embeddings.json").read_text(encoding="utf-8")
        )
        super().__init__(
            payload["vectors"],
            model_id=payload["modelId"],
            model_version=payload["modelVersion"],
        )
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return super().encode(texts)


class FailingEncoder(CountingEncoder):
    sensitive_error = "private failure at /Users/example/private-artifacts/embeddings.npy"

    def encode(self, texts):
        values = list(texts)
        if "ship" in values:
            self.calls.append(values)
            raise RuntimeError(self.sensitive_error)
        return super().encode(values)


class ConceptExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = ArtifactBundle.load(FIXTURES)
        self.source = ConceptSource(
            version="fixture-catalog-v1",
            concepts=(
                Concept("horse", "horse", ("stallion",), "animals"),
                Concept("ship", "ship", ("vessel",), "transport"),
            ),
        )
        self.prompt = PromptEnsemble(
            version="fixture-prompts-v1", templates=("{query}",)
        )
        self.config = SearchConfig(
            percentile=0.1,
            evidence_percentile=0.1,
            minimum_denominator=1,
            minimum_evidence_clusters=1,
            minimum_bin_evidence_clusters=1,
        )
        self.manifest_hash = artifact_manifest_sha256(FIXTURES)

    @staticmethod
    def _release(output: Path) -> Path:
        pointer = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        return output / pointer["release"]

    @staticmethod
    def _written_release(output: Path, manifest: dict) -> Path:
        return output / "releases" / manifest["releaseFingerprint"][:24]

    def _export(self, output: Path, encoder: CountingEncoder, *, resume=False):
        return export_catalog(
            self.artifacts,
            self.source,
            output,
            encoder,
            prompt_ensemble=self.prompt,
            config=self.config,
            artifact_manifest_hash=self.manifest_hash,
            batch_size=2,
            resume=resume,
        )

    def test_export_matches_live_series_and_visible_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data" / "v1"
            encoder = CountingEncoder()
            manifest, stats = self._export(output, encoder)
            release = self._release(output)

            self.assertTrue(manifest["complete"])
            self.assertEqual(stats.completed, 2)
            self.assertEqual(len(list((release / "series").glob("*.json"))), 2)
            self.assertEqual(len(list((release / "evidence").glob("*.json"))), 2)

            exported = json.loads(
                (release / "series" / "horse.json").read_text(encoding="utf-8")
            )
            live_service = SearchService(
                self.artifacts,
                CountingEncoder(),
                prompt_ensemble=self.prompt,
                config=self.config,
                prefer_faiss=False,
            )
            live = live_service.search(SearchRequest(query="horse"))
            live_series = live["series"][0]
            bin_index = {item["key"]: index for index, item in enumerate(live["bins"])}
            self.assertEqual(
                exported["pointIndices"],
                [bin_index[point["binKey"]] for point in live_series["points"]],
            )
            np.testing.assert_allclose(
                exported["values"],
                [point["value"] for point in live_series["points"]],
                rtol=0,
                atol=0,
            )
            self.assertEqual(
                exported["suppressedBinIndices"],
                [bin_index[key] for key in live_series.get("suppressedBinKeys", [])],
            )

            evidence = json.loads(
                (release / "evidence" / "horse.json").read_text(encoding="utf-8")
            )
            default_index = exported["defaultEvidenceBinIndex"]
            period = next(
                item for item in evidence["periods"] if item["binIndex"] == default_index
            )
            self.assertEqual(
                period["artworkIds"],
                [
                    card["artworkId"]
                    for card in live["selectedEvidence"]["slices"]["strongest"]
                ],
            )

    def test_output_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left"
            right = root / "right"
            self._export(left, CountingEncoder())
            self._export(right, CountingEncoder())

            left_files = sorted(path.relative_to(left) for path in left.rglob("*") if path.is_file())
            right_files = sorted(path.relative_to(right) for path in right.rglob("*") if path.is_file())
            self.assertEqual(left_files, right_files)
            for relative in left_files:
                self.assertEqual((left / relative).read_bytes(), (right / relative).read_bytes())

    def test_resume_skips_completed_concepts_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            encoder = CountingEncoder()
            self._export(output, encoder)
            calls_after_first_export = len(encoder.calls)

            _, stats = self._export(output, encoder, resume=True)

            self.assertEqual(len(encoder.calls), calls_after_first_export)
            self.assertEqual(stats.skipped, 2)
            self.assertEqual(stats.failures, ())

    def test_shared_bins_and_one_sparse_evidence_asset_per_concept(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            self._export(output, CountingEncoder())
            release = self._release(output)

            bins = json.loads((release / "bins.json").read_text(encoding="utf-8"))
            self.assertEqual(len(bins["keys"]), 3)
            self.assertFalse((release / "series" / "horse.json").read_text().find('"bins"') >= 0)
            self.assertEqual(
                sorted(path.name for path in (release / "evidence").glob("*.json")),
                ["horse.json", "ship.json"],
            )

    def test_subset_release_is_inspectable_without_replacing_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            self._export(output, CountingEncoder())
            pointer_before = (output / "manifest.json").read_bytes()

            manifest, stats = export_catalog(
                self.artifacts,
                self.source,
                output,
                CountingEncoder(),
                prompt_ensemble=self.prompt,
                config=self.config,
                artifact_manifest_hash=self.manifest_hash,
                selected_concept_ids=("horse",),
                batch_size=1,
            )

            release = self._written_release(output, manifest)
            self.assertEqual((output / "manifest.json").read_bytes(), pointer_before)
            self.assertTrue(manifest["complete"])
            self.assertFalse(manifest["fullCatalog"])
            self.assertEqual(manifest["selection"], "subset")
            self.assertEqual(stats.failures, ())
            self.assertTrue((release / "manifest.json").is_file())
            self.assertTrue((release / "series" / "horse.json").is_file())
            self.assertFalse((release / "series" / "ship.json").exists())

    def test_failed_release_preserves_pointer_and_sanitizes_public_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            self._export(output, CountingEncoder())
            pointer_before = (output / "manifest.json").read_bytes()
            progress: list[dict] = []

            manifest, stats = export_catalog(
                self.artifacts,
                self.source,
                output,
                FailingEncoder(),
                prompt_ensemble=self.prompt,
                config=self.config,
                artifact_manifest_hash=self.manifest_hash,
                batch_size=2,
                progress=progress.append,
            )

            release = self._written_release(output, manifest)
            public_manifest_text = (release / "manifest.json").read_text(
                encoding="utf-8"
            )
            public_manifest = json.loads(public_manifest_text)
            self.assertEqual((output / "manifest.json").read_bytes(), pointer_before)
            self.assertFalse(public_manifest["complete"])
            self.assertFalse(public_manifest["fullCatalog"])
            self.assertEqual(
                public_manifest["failures"],
                [{"conceptId": "ship", "code": "concept-export-failed"}],
            )
            self.assertNotIn(FailingEncoder.sensitive_error, public_manifest_text)
            self.assertEqual(
                stats.failures,
                ({"conceptId": "ship", "error": FailingEncoder.sensitive_error},),
            )
            self.assertTrue(
                any(
                    event.get("status") == "failed"
                    and event.get("error") == FailingEncoder.sensitive_error
                    for event in progress
                )
            )
            self.assertEqual(progress[-1]["status"], "release-written")
            self.assertFalse(progress[-1]["published"])


if __name__ == "__main__":
    unittest.main()
