from __future__ import annotations

from pathlib import Path
import unittest

from pipeline.cli import _parser


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


if __name__ == "__main__":
    unittest.main()
