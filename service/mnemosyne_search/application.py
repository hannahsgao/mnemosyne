"""Shared interface for HTTP-search backends."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .models import SearchRequest, SearchResponse


class SearchApplication(Protocol):
    def search(self, request: SearchRequest) -> SearchResponse: ...

    def health(self) -> Mapping[str, Any]: ...
