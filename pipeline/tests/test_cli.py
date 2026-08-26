from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from pipeline.cli import _parser, main


class PipelineCliTests(unittest.TestCase):
    def test_build_accepts_repeatable_additional_source_payloads(self) -> None:
        args = _parser().parse_args(
            [
                "build",
                "--input",
                "source.csv",
                "--output",
                "artifacts",
                "--corpus-version",
                "v1",
                "--source-revision",
                "pinned",
                "--source-kind",
                "nga-open-data-local-csv",
                "--counting-unit",
                "catalog-record",
                "--source-payload",
                "rights.json",
                "--source-payload",
                "derivation.json",
            ]
        )
        self.assertEqual(
            args.source_payloads,
            [Path("rights.json"), Path("derivation.json")],
        )
        self.assertEqual(args.source_kind, "nga-open-data-local-csv")
        self.assertEqual(args.counting_unit, "catalog-record")

    def test_embed_can_omit_the_optional_faiss_copy(self) -> None:
        args = _parser().parse_args(
            [
                "embed",
                "--corpus-dir",
                "corpus",
                "--output",
                "index",
                "--no-build-faiss",
            ]
        )
        self.assertTrue(args.no_build_faiss)

    def test_prepare_nga_defaults_to_strict_dated_1024px_derivatives(self) -> None:
        args = _parser().parse_args(
            [
                "prepare-nga-visual",
                "--objects",
                "objects.csv",
                "--published-images",
                "published_images.csv",
                "--object-associations",
                "object_associations.csv",
                "--output-csv",
                "nga.csv",
                "--source-revision",
                "a" * 40,
            ]
        )
        self.assertEqual(args.max_dimension, 1024)
        self.assertEqual(args.min_short_side, 256)
        self.assertEqual(args.object_associations, Path("object_associations.csv"))
        self.assertFalse(args.include_undated)
        self.assertFalse(args.no_preflight)

    def test_prepare_nga_dispatches_required_association_source(self) -> None:
        with (
            patch(
                "pipeline.cli.prepare_nga_visual_subset",
                return_value={"schema_version": "fixture"},
            ) as prepare,
            patch("builtins.print"),
        ):
            result = main(
                [
                    "prepare-nga-visual",
                    "--objects",
                    "objects.csv",
                    "--published-images",
                    "published_images.csv",
                    "--object-associations",
                    "object_associations.csv",
                    "--output-csv",
                    "nga.csv",
                    "--source-revision",
                    "a" * 40,
                    "--no-preflight",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            prepare.call_args.kwargs["object_associations_csv"],
            Path("object_associations.csv"),
        )

    def test_merge_accepts_repeatable_source_bundles(self) -> None:
        args = _parser().parse_args(
            [
                "merge-embedded-bundles",
                "--bundle",
                "met",
                "--bundle",
                "nga",
                "--output",
                "combined",
                "--corpus-version",
                "met-nga-v1",
                "--corpus-label",
                "The Met + National Gallery of Art open-access image corpus",
            ]
        )
        self.assertEqual(args.bundles, [Path("met"), Path("nga")])
        self.assertEqual(
            args.corpus_label,
            "The Met + National Gallery of Art open-access image corpus",
        )


if __name__ == "__main__":
    unittest.main()
