"""Shared interface for HTTP-search backends."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .models import EvidenceResponse, SearchRequest, SearchResponse


class SearchApplication(Protocol):
    def search(self, request: SearchRequest) -> SearchResponse: ...

    def evidence(self, request: SearchRequest) -> EvidenceResponse: ...

    def health(self) -> Mapping[str, Any]: ...
