"""Google-Ngram-style comma parsing for one to five independent series."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from .models import QueryTerm


class QuerySyntaxError(ValueError):
    pass


_SPACE_RE = re.compile(r"\s+")
MAX_QUERY_LENGTH = 500


def normalize_term(value: str) -> str:
    """Normalize only for identity/cache use; preserve the user's display label."""

    # Match JavaScript's locale-insensitive lowercase behavior. Full Unicode
    # case-folding (for example Straße -> strasse) made frontend and backend
    # disagree about which terms were duplicates.
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).strip()).lower()


def parse_query(raw: str, *, max_series: int = 5) -> list[QueryTerm]:
    """Split commas outside quotes, supporting RFC-4180 doubled quote escapes."""

    if not isinstance(raw, str):
        raise QuerySyntaxError("query must be a string")
    if len(raw) > MAX_QUERY_LENGTH:
        raise QuerySyntaxError(f"query must be at most {MAX_QUERY_LENGTH} characters")

    fields: list[str] = []
    buffer: list[str] = []
    in_quotes = False
    i = 0
    while i < len(raw):
        char = raw[i]
        if char == '"':
            if in_quotes and i + 1 < len(raw) and raw[i + 1] == '"':
                buffer.append('"')
                i += 2
                continue
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            fields.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
        i += 1

    if in_quotes:
        raise QuerySyntaxError("query contains an unmatched double quote")
    fields.append("".join(buffer).strip())

    if any(not field for field in fields):
        raise QuerySyntaxError("query contains an empty series")

    terms: list[QueryTerm] = []
    seen: set[str] = set()
    for label in fields:
        normalized = normalize_term(label)
        if not normalized:
            raise QuerySyntaxError("query contains an empty series")
        if normalized in seen:
            continue
        seen.add(normalized)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        terms.append(QueryTerm(id=f"q-{digest}", label=label, normalized=normalized))

    if not terms:
        raise QuerySyntaxError("query must contain at least one series")
    if len(terms) > max_series:
        raise QuerySyntaxError(f"query supports at most {max_series} unique series")
    return terms
