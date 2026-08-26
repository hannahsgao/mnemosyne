from __future__ import annotations

import json
from http import HTTPStatus
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path

from mnemosyne_search.artifacts import ArtifactBundle
from mnemosyne_search.encoders import FixtureTextEncoder
from mnemosyne_search.http import HttpConfig, handler_for
from mnemosyne_search.models import SearchRequest
from mnemosyne_search.prompting import PromptEnsemble
from mnemosyne_search.service import SearchConfig, SearchService


FIXTURES = Path(__file__).parent / "fixtures"


def request_json(
    url: str,
    payload: dict[str, object],
    *,
    token: str | None = None,
) -> urllib.request.Request:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )


class BlockingService:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def health(self) -> dict[str, object]:
        return {"status": "ok", "mode": "fixture"}

    def search(self, request: SearchRequest) -> dict[str, object]:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("fixture wait timed out")
        return {"query": request.query}

    def evidence(self, request: SearchRequest) -> dict[str, object]:
        return self.search(request)


class FailingService:
    def health(self) -> dict[str, object]:
        return {"status": "ok"}

    def search(self, request: SearchRequest) -> dict[str, object]:
        raise Exception("private implementation detail")

    def evidence(self, request: SearchRequest) -> dict[str, object]:
        raise RuntimeError("private upstream detail")


class HttpServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        artifacts = ArtifactBundle.load(FIXTURES)
        self.temporary = tempfile.TemporaryDirectory()
        image_path = Path(self.temporary.name) / "fixture.jpg"
        image_path.write_bytes(b"fixture-image-bytes")
        artifacts = replace(artifacts, image_paths={"fixture-000": image_path})
        encoder = FixtureTextEncoder.from_json(FIXTURES / "query-embeddings.json")
        service = SearchService(
            artifacts,
            encoder,
            prompt_ensemble=PromptEnsemble(version="fixture-v1", templates=("{query}",)),
            config=SearchConfig(
                minimum_denominator=1,
                minimum_evidence_clusters=1,
                minimum_bin_evidence_clusters=1,
            ),
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
        self.temporary.cleanup()

    def test_health_and_search_endpoints(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/healthz") as response:
            health = json.load(response)
        self.assertEqual(health["status"], "ok")
        with urllib.request.urlopen(f"{self.base_url}/livez") as response:
            self.assertEqual(json.load(response), {"status": "ok"})
        with urllib.request.urlopen(f"{self.base_url}/readyz") as response:
            self.assertEqual(json.load(response), {"status": "ready"})
        request = request_json(
            f"{self.base_url}/v1/search",
            {"query": "horse, train", "selectedBinKey": "1900-1949"},
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

    def test_oversized_request_is_rejected_without_reading_it(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/v1/search",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(65 * 1024),
            },
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, HTTPStatus.BAD_REQUEST)
        self.assertIn("request body size is invalid", caught.exception.read().decode())

    def test_evidence_endpoint_returns_only_the_versioned_envelope(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/v1/evidence",
            data=json.dumps(
                {"query": "horse", "selectedBinKey": "1800-1849"}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request) as response:
            payload = json.load(response)

        self.assertEqual(response.status, 200)
        self.assertEqual(
            set(payload), {"schemaVersion", "selectedEvidence", "generatedAt"}
        )
        self.assertEqual(payload["schemaVersion"], "mnemosyne.evidence.v1")
        self.assertEqual(payload["selectedEvidence"]["binKey"], "1800-1849")
        self.assertNotIn("bins", payload)
        self.assertNotIn("series", payload)

    def test_serves_manifest_backed_evidence_images(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/v1/images/fixture-000") as response:
            self.assertEqual(response.read(), b"fixture-image-bytes")
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertIn("immutable", response.headers["Cache-Control"])


class ProductionHttpTests(unittest.TestCase):
    def start_server(self, service: object, config: HttpConfig) -> str:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), handler_for(service, config)  # type: ignore[arg-type]
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_required_bearer_auth_protects_search_but_not_probes(self) -> None:
        service = BlockingService()
        service.release.set()
        base_url = self.start_server(
            service,
            HttpConfig(auth_mode="required", bearer_token="service-secret"),
        )

        with urllib.request.urlopen(f"{base_url}/livez") as response:
            self.assertEqual(response.status, HTTPStatus.OK)
        with urllib.request.urlopen(f"{base_url}/readyz") as response:
            self.assertEqual(response.status, HTTPStatus.OK)

        with self.assertRaises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(request_json(f"{base_url}/v1/search", {"query": "horse"}))
        self.assertEqual(missing.exception.code, HTTPStatus.UNAUTHORIZED)
        self.assertEqual(missing.exception.headers["WWW-Authenticate"], "Bearer")

        with self.assertRaises(urllib.error.HTTPError) as wrong:
            urllib.request.urlopen(
                request_json(
                    f"{base_url}/v1/search", {"query": "horse"}, token="wrong"
                )
            )
        self.assertEqual(wrong.exception.code, HTTPStatus.UNAUTHORIZED)

        with urllib.request.urlopen(
            request_json(
                f"{base_url}/v1/search",
                {"query": "horse"},
                token="service-secret",
            )
        ) as response:
            self.assertEqual(json.load(response), {"query": "horse"})

    def test_required_bearer_auth_protects_images(self) -> None:
        artifacts = ArtifactBundle.load(FIXTURES)
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "fixture.jpg"
            image_path.write_bytes(b"private-image")
            artifacts = replace(artifacts, image_paths={"fixture-000": image_path})
            encoder = FixtureTextEncoder.from_json(FIXTURES / "query-embeddings.json")
            service = SearchService(
                artifacts,
                encoder,
                prompt_ensemble=PromptEnsemble(
                    version="fixture-v1", templates=("{query}",)
                ),
                config=SearchConfig(
                    minimum_denominator=1,
                    minimum_evidence_clusters=1,
                    minimum_bin_evidence_clusters=1,
                ),
                prefer_faiss=False,
            )
            base_url = self.start_server(
                service,
                HttpConfig(auth_mode="required", bearer_token="service-secret"),
            )
            request = urllib.request.Request(
                f"{base_url}/v1/images/fixture-000",
                headers={"Authorization": "Bearer service-secret"},
            )
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.read(), b"private-image")
                self.assertTrue(response.headers["Cache-Control"].startswith("private"))

    def test_busy_search_is_rejected_immediately_with_retry_after(self) -> None:
        service = BlockingService()
        base_url = self.start_server(
            service,
            HttpConfig(max_concurrent_searches=1, retry_after_seconds=3),
        )
        first_result: list[object] = []

        def first_request() -> None:
            try:
                with urllib.request.urlopen(
                    request_json(f"{base_url}/v1/search", {"query": "horse"})
                ) as response:
                    first_result.append(json.load(response))
            except Exception as error:  # pragma: no cover - failure surfaced below
                first_result.append(error)

        worker = threading.Thread(target=first_request)
        worker.start()
        self.assertTrue(service.started.wait(timeout=2))
        try:
            with self.assertRaises(urllib.error.HTTPError) as busy:
                urllib.request.urlopen(
                    request_json(f"{base_url}/v1/search", {"query": "ship"})
                )
            self.assertEqual(busy.exception.code, HTTPStatus.TOO_MANY_REQUESTS)
            self.assertEqual(busy.exception.headers["Retry-After"], "3")
        finally:
            service.release.set()
            worker.join(timeout=2)
        self.assertEqual(first_result, [{"query": "horse"}])

    def test_internal_errors_are_redacted(self) -> None:
        base_url = self.start_server(FailingService(), HttpConfig())
        with self.assertRaises(urllib.error.HTTPError) as failed:
            urllib.request.urlopen(
                request_json(f"{base_url}/v1/search", {"query": "horse"})
            )
        body = failed.exception.read().decode()
        self.assertEqual(failed.exception.code, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertNotIn("private implementation detail", body)
        self.assertEqual(json.loads(body), {"error": "internal server error"})

        with self.assertRaises(urllib.error.HTTPError) as upstream:
            urllib.request.urlopen(
                request_json(f"{base_url}/v1/evidence", {"query": "horse"})
            )
        upstream_body = upstream.exception.read().decode()
        self.assertEqual(upstream.exception.code, HTTPStatus.BAD_GATEWAY)
        self.assertNotIn("private upstream detail", upstream_body)
        self.assertEqual(json.loads(upstream_body), {"error": "search backend unavailable"})

    def test_draining_readiness_returns_503(self) -> None:
        draining = threading.Event()
        draining.set()
        service = BlockingService()
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), handler_for(service, HttpConfig(), draining=draining)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaises(urllib.error.HTTPError) as response:
                urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/readyz")
            self.assertEqual(response.exception.code, HTTPStatus.SERVICE_UNAVAILABLE)
            self.assertEqual(response.exception.headers["Retry-After"], "1")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
