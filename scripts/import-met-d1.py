#!/usr/bin/env python3
"""Upload a local Met SQLite corpus into the production D1 search service."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ARTWORK_COLUMNS = (
    "row_id",
    "source_id",
    "artwork_id",
    "title",
    "tags",
    "artist",
    "culture",
    "medium",
    "object_type",
    "classification",
    "period",
    "dynasty",
    "geography",
    "department",
    "date_display",
    "date_start",
    "date_end",
    "date_qualifier",
    "object_url",
    "credit_line",
    "public_domain",
)


def request_json(url: str, token: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = Request(
        url,
        data=body,
        method="GET" if payload is None else "POST",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise RuntimeError("import service returned a non-object response")
            return result
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
            if attempt < 5:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"import request failed: {last_error}")


def bin_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {
                "bin_index": int(row["bin_index"]),
                "bin_key": row["bin_key"],
                "bin_start": int(row["bin_start"]),
                "bin_end": int(row["bin_end"]),
                "bin_label": row["bin_label"],
                "denominator": float(row["eligible_weight"]),
                "object_count": int(row["physical_object_count"]),
                "cluster_count": int(row["visual_cluster_count"]),
            }
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--url", required=True, help="Base site URL or /_admin/met/import URL")
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()
    token = os.environ.get("MNEMOSYNE_IMPORT_TOKEN", "")
    if not token:
        raise SystemExit("MNEMOSYNE_IMPORT_TOKEN is required")
    if not 1 <= args.chunk_size <= 500:
        raise SystemExit("--chunk-size must be between 1 and 500")
    url = args.url.rstrip("/")
    if not url.endswith("/_admin/met/import"):
        url += "/_admin/met/import"

    database = args.artifacts / "met-search.sqlite3"
    denominators = args.artifacts / "bin-denominators.csv"
    if not database.is_file() or not denominators.is_file():
        raise SystemExit("artifact bundle is missing met-search.sqlite3 or bin-denominators.csv")

    status = request_json(url, token)
    max_row_id = int((status.get("artwork") or {}).get("max_row_id") or 0)
    existing_bins = int((status.get("bin") or {}).get("count") or 0)
    if existing_bins < 1:
        bins = bin_rows(denominators)
        for start in range(0, len(bins), 400):
            request_json(url, token, {"kind": "bins", "rows": bins[start : start + 400]})
        print(f"Imported {len(bins):,} timeline bins", flush=True)

    connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    total = int(connection.execute("SELECT COUNT(*) FROM artworks").fetchone()[0])
    placeholders = ", ".join(ARTWORK_COLUMNS)
    imported = max_row_id
    started = time.monotonic()
    while True:
        batch = connection.execute(
            f"SELECT {placeholders} FROM artworks WHERE row_id > ? ORDER BY row_id LIMIT ?",
            (imported, args.chunk_size),
        ).fetchall()
        if not batch:
            break
        payload = [{column: row[column] for column in ARTWORK_COLUMNS} for row in batch]
        request_json(url, token, {"kind": "artworks", "rows": payload})
        imported = int(batch[-1]["row_id"])
        if imported % 5000 < args.chunk_size or imported == total:
            elapsed = max(time.monotonic() - started, 0.001)
            rate = max(imported - max_row_id, 0) / elapsed
            print(f"Imported through row {imported:,}/{total:,} ({rate:,.0f} rows/s)", flush=True)
    connection.close()
    request_json(url, token, {"kind": "finalize"})
    print("Met D1 corpus is ready", flush=True)


if __name__ == "__main__":
    main()
