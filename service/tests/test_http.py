from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from mnemosyne_search.artifacts import ArtifactBundle
from mnemosyne_search.encoders import FixtureTextEncoder
from mnemosyne_search.http import handler_for
from mnemosyne_search.prompting import PromptEnsemble
from mnemosyne_search.service import SearchService


FIXTURES = Path(__file__).parent / "fixtures"


class HttpServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        artifacts = ArtifactBundle.load(FIXTURES)
        encoder = FixtureTextEncoder.from_json(FIXTURES / "query-embeddings.json")
        service = SearchService(
            artifacts,
            encoder,
            prompt_ensemble=PromptEnsemble(version="fixture-v1", templates=("{query}",)),
            prefer_faiss=False,
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_health_and_search_endpoints(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/healthz") as response:
            health = json.load(response)
        self.assertEqual(health["status"], "ok")
        request = urllib.request.Request(
            f"{self.base_url}/v1/search",
            data=json.dumps({"query": "horse, train", "selectedBinKey": "1900-1949"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        self.assertEqual(response.status, 200)
        self.assertEqual(len(payload["series"]), 2)

    def test_invalid_query_returns_400(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/v1/search",
            data=b'{"query":"horse,"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("empty series", caught.exception.read().decode())


if __name__ == "__main__":
    unittest.main()
