"""Dependency-light persistent HTTP server for the search API."""

from __future__ import annotations

import json
import mimetypes
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .application import SearchApplication
from .models import SearchRequest
from .parsing import QuerySyntaxError


MAX_REQUEST_BYTES = 64 * 1024


def _request_from_json(payload: dict[str, Any]) -> SearchRequest:
    raw_filters = payload.get("filters", {})
    if not isinstance(raw_filters, dict):
        raise ValueError("filters must be an object")
    filters: dict[str, tuple[str, ...]] = {}
    for key, raw_value in raw_filters.items():
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        if not all(isinstance(value, str) for value in values):
            raise ValueError("filter values must be strings or lists of strings")
        filters[str(key)] = tuple(values)
    return SearchRequest(
        query=payload.get("query", ""),
        selected_query_id=payload.get("selectedQueryId"),
        selected_bin_key=payload.get("selectedBinKey"),
        corpus_view=payload.get("view", "all"),
        filters=filters,
    )


def handler_for(service: SearchApplication) -> type[BaseHTTPRequestHandler]:
    class SearchHandler(BaseHTTPRequestHandler):
        server_version = "MnemosyneSearch/1"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            if path == "/healthz":
                self._json(HTTPStatus.OK, service.health())
            elif path.startswith("/v1/images/"):
                artwork_id = unquote(path.removeprefix("/v1/images/"))
                artifacts = getattr(service, "artifacts", None)
                resolver = getattr(artifacts, "image_path_for", None)
                image_path = resolver(artwork_id) if callable(resolver) else None
                if image_path is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "image not found"})
                else:
                    self._file(image_path)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/v1/search":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("request body size is invalid")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                response = service.search(_request_from_json(payload))
            except (json.JSONDecodeError, QuerySyntaxError, TypeError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except RuntimeError as error:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
                return
            self._json(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: object) -> None:
            # Keep the library quiet. Deployments can wrap this handler with
            # structured request logging at the process boundary.
            return

        def _json(self, status: HTTPStatus, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)

    return SearchHandler


def serve(service: SearchApplication, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), handler_for(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()
