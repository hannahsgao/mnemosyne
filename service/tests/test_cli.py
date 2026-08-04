from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from mnemosyne_search.cli import main


FIXTURES = Path(__file__).parent / "fixtures"


class CommandLineTests(unittest.TestCase):
    def test_keyboard_interrupt_stops_without_a_traceback(self) -> None:
        with patch("mnemosyne_search.cli.serve", side_effect=KeyboardInterrupt):
            status = main(
                [
                    "--artifacts",
                    str(FIXTURES),
                    "--fixture-vectors",
                    str(FIXTURES / "query-embeddings.json"),
                    "--prompt-template",
                    "{query}",
                    "--prompt-version",
                    "fixture-prompts-v1",
                    "--no-faiss",
                ]
            )

        self.assertEqual(status, 130)


if __name__ == "__main__":
    unittest.main()
