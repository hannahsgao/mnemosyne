"""Dependency-light persistent HTTP server for the search API."""

from __future__ import annotations

import hmac
import json
import logging
import mimetypes
import shutil
import signal
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import unquote, urlsplit

from .application import SearchApplication
from .models import SearchRequest
from .parsing import QuerySyntaxError


MAX_REQUEST_BYTES = 64 * 1024
MAX_REQUEST_TARGET_BYTES = 4 * 1024
AUTH_MODES = frozenset({"disabled", "if-configured", "required"})

_LOGGER = logging.getLogger("mnemosyne_search.http")


@dataclass(frozen=True)
class HttpConfig:
    """Production controls applied outside the scoring/search implementation."""

    auth_mode: Literal["disabled", "if-configured", "required"] = "disabled"
    bearer_token: str | None = None
    max_concurrent_searches: int = 2
    retry_after_seconds: int = 1
    request_io_timeout_seconds: float = 15.0
    max_request_bytes: int = MAX_REQUEST_BYTES
    max_request_target_bytes: int = MAX_REQUEST_TARGET_BYTES

    def __post_init__(self) -> None:
        if self.auth_mode not in AUTH_MODES:
            raise ValueError(f"auth_mode must be one of {sorted(AUTH_MODES)}")
        if self.bearer_token is not None:
            if not self.bearer_token or any(
                character in self.bearer_token for character in "\r\n"
            ):
                raise ValueError("bearer_token must be non-empty and cannot contain newlines")
        if self.auth_mode == "required" and self.bearer_token is None:
            raise ValueError("required bearer authentication needs a bearer token")
        if self.max_concurrent_searches < 1:
            raise ValueError("max_concurrent_searches must be positive")
        if self.retry_after_seconds < 1:
            raise ValueError("retry_after_seconds must be positive")
        if self.request_io_timeout_seconds <= 0:
            raise ValueError("request_io_timeout_seconds must be positive")
        if self.max_request_bytes < 1 or self.max_request_target_bytes < 1:
            raise ValueError("HTTP request limits must be positive")

    @property
    def authorization_required(self) -> bool:
        return self.auth_mode == "required" or (
            self.auth_mode == "if-configured" and self.bearer_token is not None
        )


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


def _log(event: str, *, level: int = logging.INFO, **fields: object) -> None:
    """Emit one compact JSON object without request bodies or credentials."""

    _LOGGER.log(
        level,
        json.dumps(
            {"event": event, **fields},
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ),
    )


class SearchHTTPServer(ThreadingHTTPServer):
    """Join active request threads when the listener shuts down."""

    allow_reuse_address = True
    block_on_close = True
    daemon_threads = False
    request_queue_size = 32


def handler_for(
    service: SearchApplication,
    config: HttpConfig | None = None,
    *,
    draining: threading.Event | None = None,
) -> type[BaseHTTPRequestHandler]:
    http_config = config or HttpConfig()
    search_slots = threading.BoundedSemaphore(http_config.max_concurrent_searches)

    class SearchHandler(BaseHTTPRequestHandler):
        server_version = "MnemosyneSearch/2"
        sys_version = ""

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(http_config.request_io_timeout_seconds)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._begin_request()
            if not self._target_is_bounded():
                return
            path = urlsplit(self.path).path
            self._request_path = path
            if path == "/livez":
                self._json(HTTPStatus.OK, {"status": "ok"})
                return
            if path == "/readyz":
                if draining is not None and draining.is_set():
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"status": "not-ready"},
                        headers={"Retry-After": str(http_config.retry_after_seconds)},
                    )
                    return
                try:
                    health = service.health()
                except Exception:
                    self._log_exception("readiness_check_failed")
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"status": "not-ready"},
                        headers={"Retry-After": str(http_config.retry_after_seconds)},
                    )
                    return
                status = (
                    HTTPStatus.OK
                    if health.get("status") == "ok"
                    else HTTPStatus.SERVICE_UNAVAILABLE
                )
                self._json(status, {"status": "ready" if status == 200 else "not-ready"})
                return
            if path == "/healthz":
                try:
                    self._json(HTTPStatus.OK, service.health())
                except Exception:
                    self._log_exception("health_check_failed")
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "service unavailable"},
                    )
                return
            if path.startswith("/v1/images/"):
                if not self._authorize():
                    return
                try:
                    artwork_id = unquote(path.removeprefix("/v1/images/"))
                    artifacts = getattr(service, "artifacts", None)
                    resolver = getattr(artifacts, "image_path_for", None)
                    image_path = resolver(artwork_id) if callable(resolver) else None
                    if image_path is None or not image_path.is_file():
                        self._json(HTTPStatus.NOT_FOUND, {"error": "image not found"})
                    else:
                        self._file(image_path)
                except Exception:
                    self._log_exception("image_response_failed")
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal server error"})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._begin_request()
            if not self._target_is_bounded():
                return
            path = urlsplit(self.path).path
            self._request_path = path
            if path not in {"/v1/search", "/v1/evidence"}:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self._authorize():
                return
            try:
                payload = self._read_json_body()
                request = _request_from_json(payload)
            except socket.timeout:
                self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "request body timed out"})
                return
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                QuerySyntaxError,
                TypeError,
                ValueError,
            ) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return

            if not search_slots.acquire(blocking=False):
                self._json(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    {"error": "search capacity is busy"},
                    headers={"Retry-After": str(http_config.retry_after_seconds)},
                )
                return
            try:
                response = (
                    service.evidence(request)
                    if path == "/v1/evidence"
                    else service.search(request)
                )
            except (QuerySyntaxError, TypeError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except RuntimeError:
                self._log_exception("search_backend_failed")
                self._json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": "search backend unavailable"},
                )
                return
            except Exception:
                self._log_exception("search_request_failed")
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal server error"},
                )
                return
            finally:
                search_slots.release()

            try:
                self._json(HTTPStatus.OK, response)
            except (TypeError, ValueError):
                self._log_exception("search_response_serialization_failed")
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal server error"},
                )

        def log_message(self, format: str, *args: object) -> None:
            return

        def log_error(self, format: str, *args: object) -> None:
            # BaseHTTPRequestHandler uses this for protocol/parser failures that
            # occur before a normal response can be constructed.
            _log(
                "http_protocol_error",
                level=logging.WARNING,
                requestId=getattr(self, "_request_id", "unassigned"),
                detail=format % args,
            )

        def version_string(self) -> str:
            return self.server_version

        def _begin_request(self) -> None:
            self._request_started = time.monotonic()
            self._request_id = uuid.uuid4().hex
            self._request_path = urlsplit(self.path).path
            self._response_logged = False

        def _target_is_bounded(self) -> bool:
            if len(self.path.encode("utf-8", errors="replace")) <= http_config.max_request_target_bytes:
                return True
            self._json(HTTPStatus.REQUEST_URI_TOO_LONG, {"error": "request target is too long"})
            return False

        def _authorize(self) -> bool:
            if not http_config.authorization_required:
                return True
            expected = http_config.bearer_token
            assert expected is not None
            values = self.headers.get_all("Authorization", failobj=[])
            authorized = False
            if len(values) == 1:
                scheme, separator, presented = values[0].partition(" ")
                authorized = bool(
                    separator
                    and scheme.casefold() == "bearer"
                    and presented
                    and hmac.compare_digest(
                        presented.encode("utf-8"), expected.encode("utf-8")
                    )
                )
            if authorized:
                return True
            self._json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "unauthorized"},
                headers={"WWW-Authenticate": "Bearer"},
            )
            return False

        def _read_json_body(self) -> dict[str, Any]:
            transfer_encoding = self.headers.get("Transfer-Encoding")
            if transfer_encoding and transfer_encoding.casefold() != "identity":
                raise ValueError("transfer-encoded request bodies are not supported")
            lengths = self.headers.get_all("Content-Length", failobj=[])
            if len(lengths) != 1:
                raise ValueError("exactly one Content-Length header is required")
            try:
                length = int(lengths[0], 10)
            except ValueError as error:
                raise ValueError("Content-Length must be an integer") from error
            if length <= 0 or length > http_config.max_request_bytes:
                raise ValueError("request body size is invalid")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("request body ended before Content-Length")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _json(
            self,
            status: HTTPStatus | int,
            payload: Any,
            *,
            headers: Mapping[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Request-ID", self._request_id)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass
            finally:
                self._log_response(int(status))

        def _file(self, path: Path) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "private, max-age=31536000, immutable")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Request-ID", self._request_id)
            self.end_headers()
            try:
                with path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass
            finally:
                self._log_response(HTTPStatus.OK)

        def _log_response(self, status: int | HTTPStatus) -> None:
            if self._response_logged:
                return
            self._response_logged = True
            _log(
                "http_request",
                requestId=self._request_id,
                method=self.command,
                path=self._request_path,
                status=int(status),
                durationMs=round((time.monotonic() - self._request_started) * 1000, 2),
            )

        def _log_exception(self, event: str) -> None:
            _LOGGER.exception(
                json.dumps(
                    {
                        "event": event,
                        "requestId": self._request_id,
                        "method": self.command,
                        "path": self._request_path,
                    },
                    separators=(",", ":"),
                )
            )

    return SearchHandler


def serve(
    service: SearchApplication,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    config: HttpConfig | None = None,
) -> None:
    """Serve until interrupted, draining active request threads on SIGTERM."""

    http_config = config or HttpConfig()
    draining = threading.Event()
    server = SearchHTTPServer(
        (host, port), handler_for(service, http_config, draining=draining)
    )
    server.timeout = 0.5
    previous_sigterm: signal.Handlers | None = None

    def request_shutdown(signum: int, _frame: object) -> None:
        _log("server_shutdown_requested", signal=signum)
        draining.set()

    if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGTERM"):
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, request_shutdown)

    _log(
        "server_started",
        host=host,
        port=server.server_port,
        authMode=http_config.auth_mode,
        authRequired=http_config.authorization_required,
        maxConcurrentSearches=http_config.max_concurrent_searches,
    )
    try:
        while not draining.is_set():
            server.handle_request()
    finally:
        draining.set()
        server.server_close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        _log("server_stopped")
