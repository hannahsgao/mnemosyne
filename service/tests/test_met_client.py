from __future__ import annotations

from io import BytesIO
import json
import unittest
from unittest.mock import patch

from mnemosyne_search.met_client import HttpMetClient


def response(payload: object) -> BytesIO:
    return BytesIO(json.dumps(payload).encode("utf-8"))


class HttpMetClientTests(unittest.TestCase):
    def test_search_uses_keyless_parameters_and_caches_ids(self) -> None:
        client = HttpMetClient("https://met.example/v1", attempts=1)
        with patch("mnemosyne_search.met_client.urlopen", return_value=response({"objectIDs": [3, 2, 3]})) as get:
            first = client.search("horse & rider", "tags")
            second = client.search("horse & rider", "tags")

        self.assertEqual(first, (3, 2))
        self.assertEqual(second, first)
        self.assertEqual(get.call_count, 1)
        request = get.call_args.args[0]
        self.assertIn("q=horse+%26+rider", request.full_url)
        self.assertIn("hasImages=true", request.full_url)
        self.assertIn("tags=true", request.full_url)
        self.assertNotIn("key=", request.full_url)

    def test_object_details_are_cached(self) -> None:
        client = HttpMetClient("https://met.example/v1", attempts=1)
        payload = {"objectID": 42, "primaryImageSmall": "https://images.example/42.jpg"}
        with patch("mnemosyne_search.met_client.urlopen", return_value=response(payload)) as get:
            self.assertEqual(client.object(42), payload)
            self.assertEqual(client.object(42), payload)
        self.assertEqual(get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
