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

    def test_embed_accepts_a_source_request_delay(self) -> None:
        args = _parser().parse_args(
            [
                "embed",
                "--corpus-dir",
                "corpus",
                "--output",
                "index",
                "--image-request-delay-seconds",
                "1",
            ]
        )
        self.assertEqual(args.image_request_delay_seconds, 1.0)

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

    def test_open_museum_visual_commands_preserve_source_specific_defaults(self) -> None:
        cma = _parser().parse_args(
            [
                "prepare-cma-visual",
                "--snapshot-json",
                "data.json",
                "--output-csv",
                "cma.csv",
                "--source-revision",
                "a" * 40,
            ]
        )
        self.assertEqual(cma.workers, 16)
        self.assertFalse(cma.include_undated)

        aic = _parser().parse_args(
            [
                "prepare-aic-visual",
                "--source-dump",
                "artic-api-data.tar.bz2",
                "--output-csv",
                "aic.csv",
                "--source-revision",
                "b" * 64,
            ]
        )
        self.assertEqual(aic.request_delay_seconds, 1.0)
        self.assertFalse(aic.no_preflight)

        smk = _parser().parse_args(
            [
                "prepare-smk-visual",
                "--snapshot",
                "smk_all_da.zip",
                "--output-csv",
                "smk.csv",
                "--source-revision",
                "c" * 64,
            ]
        )
        self.assertEqual(smk.workers, 16)
        self.assertFalse(smk.no_preflight)

    def test_open_museum_visual_commands_dispatch_the_pinned_snapshots(self) -> None:
        cases = (
            (
                "prepare-cma-visual",
                "pipeline.cli.prepare_cma_visual_subset",
                ["--snapshot-json", "data.json"],
                "snapshot_json",
                Path("data.json"),
                "a" * 40,
            ),
            (
                "prepare-aic-visual",
                "pipeline.cli.prepare_aic_visual_subset",
                ["--source-dump", "aic.tar.bz2"],
                "source_dump",
                Path("aic.tar.bz2"),
                "b" * 64,
            ),
            (
                "prepare-smk-visual",
                "pipeline.cli.prepare_smk_visual_subset",
                ["--snapshot", "smk.zip"],
                "snapshot",
                Path("smk.zip"),
                "c" * 64,
            ),
        )
        for command, target, source_args, _label, expected, revision in cases:
            with self.subTest(command=command), patch(
                target, return_value={"schema_version": "fixture"}
            ) as prepare, patch("builtins.print"):
                result = main(
                    [
                        command,
                        *source_args,
                        "--output-csv",
                        "output.csv",
                        "--source-revision",
                        revision,
                        "--no-preflight",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(prepare.call_args.args[0], expected)
            self.assertEqual(
                prepare.call_args.kwargs["source_revision"], revision
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
