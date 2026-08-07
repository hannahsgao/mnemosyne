"""Small cached client for the keyless Met Collection API."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import ssl
from threading import RLock
import time
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import certifi


MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
SEARCH_MODES = frozenset({"broad", "title", "tags"})
MET_FTS_SCHEMA_VERSION = 1


class MetClient(Protocol):
    def search(self, query: str, mode: str = "broad") -> tuple[int, ...]: ...

    def object(self, object_id: int) -> Mapping[str, Any] | None: ...


class SqliteMetClient:
    """Read-only local Met full-text search and evidence metadata provider."""

    backend = "sqlite-fts5"

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database).resolve()
        if not self.database.is_file():
            raise ValueError(f"Met FTS database does not exist: {self.database}")
        encoded = quote(self.database.as_posix(), safe="/")
        self._database_connection = sqlite3.connect(
            f"file:{encoded}?mode=ro&immutable=1",
            uri=True,
            timeout=5,
            check_same_thread=False,
        )
        self._database_connection.row_factory = sqlite3.Row
        self._database_connection.execute("PRAGMA query_only=ON")
        self._database_lock = RLock()
        version = int(self._database_connection.execute("PRAGMA user_version").fetchone()[0])
        if version != MET_FTS_SCHEMA_VERSION:
            self._database_connection.close()
            raise ValueError(f"unsupported Met FTS schema version: {version}")

    def close(self) -> None:
        with self._database_lock:
            self._database_connection.close()

    @staticmethod
    def _expression(query: str, mode: str) -> str | None:
        if mode not in SEARCH_MODES:
            raise ValueError(f"unsupported Met search mode: {mode}")
        if not any(character.isalnum() for character in query):
            return None
        escaped = query.replace('"', '""')
        phrase = f'"{escaped}"'
        return phrase if mode == "broad" else f"{mode} : {phrase}"

    def search(self, query: str, mode: str = "broad") -> tuple[int, ...]:
        expression = self._expression(query, mode)
        if expression is None:
            return ()
        try:
            with self._database_lock:
                rows = self._database_connection.execute(
                    """
                    SELECT artworks.source_id
                    FROM artwork_fts
                    JOIN artworks ON artworks.row_id = artwork_fts.rowid
                    WHERE artwork_fts MATCH ?
                    ORDER BY artworks.row_id
                    """,
                    (expression,),
                )
                return tuple(int(row[0]) for row in rows)
        except sqlite3.Error as exc:
            raise RuntimeError(f"local Met FTS search failed: {exc}") from exc

    def object(self, object_id: int) -> Mapping[str, Any] | None:
        with self._database_lock:
            row = self._database_connection.execute(
                """
                SELECT source_id, title, artist, date_display, object_url,
                       image_url, credit_line, public_domain
                FROM artworks
                WHERE source_id = ?
                """,
                (object_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "objectID": int(row["source_id"]),
            "title": str(row["title"]),
            "artistDisplayName": str(row["artist"]),
            "objectDate": str(row["date_display"]),
            "objectURL": str(row["object_url"]),
            "primaryImageSmall": str(row["image_url"]),
            "creditLine": str(row["credit_line"]),
            "isPublicDomain": bool(row["public_domain"]),
        }


class HttpMetClient:
    def __init__(
        self,
        api_base: str = MET_API_BASE,
        *,
        timeout: float = 20.0,
        attempts: int = 3,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.attempts = attempts
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._search_cache: dict[tuple[str, str], tuple[int, ...]] = {}
        self._object_cache: dict[int, Mapping[str, Any] | None] = {}
        self._lock = RLock()

    def search(self, query: str, mode: str = "broad") -> tuple[int, ...]:
        if mode not in SEARCH_MODES:
            raise ValueError(f"unsupported Met search mode: {mode}")
        key = (query.casefold(), mode)
        with self._lock:
            cached = self._search_cache.get(key)
        if cached is not None:
            return cached
        parameters = {"q": query, "hasImages": "true"}
        if mode != "broad":
            parameters[mode] = "true"
        payload = self._get_json(f"/search?{urlencode(parameters)}")
        raw_ids = payload.get("objectIDs") or []
        if not isinstance(raw_ids, list):
            raise RuntimeError("Met search response did not contain an objectIDs list")
        try:
            result = tuple(dict.fromkeys(int(value) for value in raw_ids))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Met search returned an invalid object ID") from exc
        with self._lock:
            self._search_cache[key] = result
        return result

    def object(self, object_id: int) -> Mapping[str, Any] | None:
        with self._lock:
            if object_id in self._object_cache:
                return self._object_cache[object_id]
        try:
            payload = self._get_json(f"/objects/{object_id}")
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
            payload = None
        with self._lock:
            self._object_cache[object_id] = payload
        return payload

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(
            f"{self.api_base}{path}",
            headers={"Accept": "application/json", "User-Agent": "mnemosyne-search/1"},
        )
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                with urlopen(  # noqa: S310
                    request,
                    timeout=self.timeout,
                    context=self._ssl_context,
                ) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict):
                    raise RuntimeError("Met API returned a non-object JSON response")
                return payload
            except HTTPError as exc:
                last_error = RuntimeError(f"Met API HTTP {exc.code}")
                if exc.code < 500 and exc.code != 429:
                    break
            except (URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                last_error = exc
            if attempt + 1 < self.attempts:
                time.sleep(2**attempt)
        raise RuntimeError(f"Met API request failed: {last_error}")


class FixtureMetClient:
    """In-memory Met API substitute used by unit and integration tests."""

    def __init__(
        self,
        searches: Mapping[str, list[int] | tuple[int, ...]],
        objects: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> None:
        self.searches = {key.casefold(): tuple(value) for key, value in searches.items()}
        self.objects = dict(objects or {})
        self.search_calls: list[tuple[str, str]] = []
        self.object_calls: list[int] = []

    def search(self, query: str, mode: str = "broad") -> tuple[int, ...]:
        self.search_calls.append((query, mode))
        return self.searches.get(query.casefold(), ())

    def object(self, object_id: int) -> Mapping[str, Any] | None:
        self.object_calls.append(object_id)
        return self.objects.get(object_id)
