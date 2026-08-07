"""Build a dated Met corpus and local FTS index from the Open Access export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .build import CANONICAL_FIELDS, CorpusBuildError, build_corpus, sha256_file
from .dates import DateConfig, NormalizedDate, normalize_date, parse_bool


MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
MET_SOURCE_URL = "https://github.com/metmuseum/openaccess"
MET_CC0_URI = "https://creativecommons.org/publicdomain/zero/1.0/"
MET_SOURCE_KIND = "met-open-access-csv-with-local-fts"
MET_FTS_FILENAME = "met-search.sqlite3"
MET_FTS_SCHEMA_VERSION = 1
MET_FTS_FIELDS = (
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
)
MET_REQUIRED_FIELDS = frozenset(
    {
        "Object ID",
        "Is Public Domain",
        "Object Date",
        "Object Begin Date",
        "Object End Date",
    }
)


def _fetch_has_image_ids(api_base: str, *, attempts: int = 3, timeout: float = 120.0) -> dict[str, Any]:
    query = urlencode({"hasImages": "true", "q": "*"})
    request = Request(
        f"{api_base.rstrip('/')}/search?{query}",
        headers={"Accept": "application/json", "User-Agent": "mnemosyne-met-build/1"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed/user-selected API
                payload = json.load(response)
            if not isinstance(payload, dict) or not isinstance(payload.get("objectIDs"), list):
                raise CorpusBuildError("Met has-images response did not contain objectIDs")
            return payload
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise CorpusBuildError(f"could not fetch Met has-images object IDs: {last_error}")


def _read_image_ids(path: Path) -> tuple[set[int], int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusBuildError(f"could not read Met image-ID snapshot: {exc}") from exc
    raw_ids = payload.get("objectIDs") if isinstance(payload, dict) else payload
    if not isinstance(raw_ids, list):
        raise CorpusBuildError("Met image-ID snapshot must be a list or an object with objectIDs")
    try:
        image_ids = {int(value) for value in raw_ids}
    except (TypeError, ValueError) as exc:
        raise CorpusBuildError("Met image-ID snapshot contains a non-integer object ID") from exc
    return image_ids, len(raw_ids)


def _year(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        year = int(text)
    except ValueError:
        return text
    return "" if year == 0 else str(year)


def _join_geography(row: dict[str, str]) -> str:
    fields = ("City", "State", "Country", "Region", "Subregion", "Locale", "Locus")
    return "; ".join(value for field in fields if (value := row.get(field, "").strip()))


def _bounded_date(date: NormalizedDate, config: DateConfig) -> NormalizedDate | None:
    if not date.dated:
        return date
    assert date.start is not None and date.end is not None
    if config.min_year is not None and date.end < config.min_year:
        return None
    if config.max_year is not None and date.start > config.max_year:
        return None
    start = max(date.start, config.min_year) if config.min_year is not None else date.start
    end = min(date.end, config.max_year) if config.max_year is not None else date.end
    return NormalizedDate(
        display=date.display,
        start=start,
        end=end,
        qualifier=date.qualifier,
        parse_method=date.parse_method,
    )


def _canonical_met_row(
    row: dict[str, str], object_id: int, *, image_available: bool
) -> dict[str, object]:
    public_domain = parse_bool(row.get("Is Public Domain"))
    return {
        "artwork_id": f"MET_{object_id}",
        "physical_object_id": f"MET_{object_id}",
        "visual_cluster_id": f"object:MET_{object_id}",
        "institution": "met",
        "source_id": str(object_id),
        "source_record_url": row.get("Link Resource", "").strip()
        or f"https://www.metmuseum.org/art/collection/search/{object_id}",
        "title": row.get("Title", "").strip(),
        "artist": row.get("Artist Display Name", "").strip(),
        "object_type": row.get("Object Name", "").strip(),
        "medium": row.get("Medium", "").strip(),
        "culture": row.get("Culture", "").strip(),
        "department": row.get("Department", "").strip(),
        "classification": row.get("Classification", "").strip(),
        "period": row.get("Period", "").strip(),
        "dynasty": row.get("Dynasty", "").strip(),
        "geography": _join_geography(row),
        "tags": row.get("Tags", "").strip(),
        "object_wikidata_url": row.get("Object Wikidata URL", "").strip(),
        "date_display": row.get("Object Date", "").strip(),
        "date_start": _year(row.get("Object Begin Date")),
        "date_end": _year(row.get("Object End Date")),
        "metadata_license": MET_CC0_URI,
        "image_rights_uri": MET_CC0_URI if public_domain and image_available else "",
        "credit_line": row.get("Credit Line", "").strip(),
        "public_domain": public_domain,
        "image_available": image_available,
    }


def _optional_int(value: str) -> int | None:
    text = value.strip()
    return int(text) if text else None


def _build_fts_index(corpus_csv: Path, output: Path) -> int:
    """Build a self-contained, row-aligned Met metadata FTS5 database."""

    connection = sqlite3.connect(output)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA page_size=4096;
            CREATE TABLE artworks (
                row_id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL UNIQUE,
                artwork_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                tags TEXT NOT NULL,
                artist TEXT NOT NULL,
                culture TEXT NOT NULL,
                medium TEXT NOT NULL,
                object_type TEXT NOT NULL,
                classification TEXT NOT NULL,
                period TEXT NOT NULL,
                dynasty TEXT NOT NULL,
                geography TEXT NOT NULL,
                department TEXT NOT NULL,
                date_display TEXT NOT NULL,
                date_start INTEGER,
                date_end INTEGER,
                date_qualifier TEXT NOT NULL,
                object_url TEXT NOT NULL,
                metadata_license TEXT NOT NULL,
                image_rights_uri TEXT NOT NULL,
                credit_line TEXT NOT NULL,
                public_domain INTEGER NOT NULL,
                image_available INTEGER NOT NULL,
                image_url TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE artwork_fts USING fts5(
                title,
                tags,
                artist,
                culture,
                medium,
                object_type,
                classification,
                period,
                dynasty,
                geography,
                department,
                content='artworks',
                content_rowid='row_id',
                tokenize='porter unicode61 remove_diacritics 2'
            );
            """
        )
        count = 0
        batch: list[tuple[object, ...]] = []
        with corpus_csv.open(encoding="utf-8", newline="") as handle:
            for count, row in enumerate(csv.DictReader(handle), start=1):
                batch.append(
                    (
                        count,
                        int(row["source_id"]),
                        row["artwork_id"],
                        row["title"],
                        row["tags"],
                        row["artist"],
                        row["culture"],
                        row["medium"],
                        row["object_type"],
                        row["classification"],
                        row["period"],
                        row["dynasty"],
                        row["geography"],
                        row["department"],
                        row["date_display"],
                        _optional_int(row["date_start"]),
                        _optional_int(row["date_end"]),
                        row["date_qualifier"],
                        row["source_record_url"],
                        row["metadata_license"],
                        row["image_rights_uri"],
                        row["credit_line"],
                        int(row["public_domain"].casefold() == "true"),
                        int(row["image_available"].casefold() == "true"),
                        row["image_url"],
                    )
                )
                if len(batch) >= 10_000:
                    connection.executemany(
                        "INSERT INTO artworks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        batch,
                    )
                    batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO artworks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
        connection.execute("INSERT INTO artwork_fts(artwork_fts) VALUES('rebuild')")
        connection.execute("INSERT INTO artwork_fts(artwork_fts) VALUES('optimize')")
        connection.execute(f"PRAGMA user_version={MET_FTS_SCHEMA_VERSION}")
        connection.commit()
        connection.execute("ANALYZE")
        connection.commit()
        connection.execute("VACUUM")
    except sqlite3.Error as exc:
        raise CorpusBuildError(f"could not build Met FTS5 index: {exc}") from exc
    finally:
        connection.close()
    return count


def _attach_fts_manifest(root: Path, manifest: dict[str, object], row_count: int) -> None:
    index_path = root / MET_FTS_FILENAME
    files = manifest["files"]
    assert isinstance(files, dict)
    files["keywordIndex"] = MET_FTS_FILENAME
    manifest["search_index"] = {
        "backend": "sqlite-fts5",
        "schema_version": MET_FTS_SCHEMA_VERSION,
        "tokenizer": "porter unicode61 remove_diacritics 2",
        "fields": list(MET_FTS_FIELDS),
        "rows": row_count,
    }
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    artifacts.append(
        {
            "path": MET_FTS_FILENAME,
            "sha256": sha256_file(index_path),
            "bytes": index_path.stat().st_size,
            "rows": row_count,
        }
    )
    artifacts.sort(key=lambda entry: str(entry["path"]))
    (root / "build-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_met_corpus(
    input_csv: Path | str,
    output_dir: Path | str,
    *,
    corpus_version: str,
    source_revision: str,
    image_ids_path: Path | str | None = None,
    api_base: str = MET_API_BASE,
    fetch_image_ids: bool = False,
    retrieved_at: str | None = None,
    date_config: DateConfig | None = None,
    require_parquet: bool = False,
) -> dict[str, object]:
    """Create local FTS and aggregation artifacts without runtime Met API calls."""

    source = Path(input_csv).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise CorpusBuildError(f"Met Open Access CSV does not exist: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise CorpusBuildError(f"output directory must be absent or empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = date_config or DateConfig()
    config.validate()

    with tempfile.TemporaryDirectory(
        prefix=".mnemosyne-met-", dir=destination.parent
    ) as temporary:
        temporary_root = Path(temporary)
        image_snapshot: Path | None
        if image_ids_path is not None:
            image_snapshot = Path(image_ids_path).resolve()
        elif fetch_image_ids:
            image_snapshot = temporary_root / "met-has-images.json"
            image_snapshot.write_text(
                json.dumps(_fetch_has_image_ids(api_base), separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        else:
            image_snapshot = None
        if image_snapshot is None:
            image_ids: set[int] = set()
            image_id_rows = 0
        else:
            image_ids, image_id_rows = _read_image_ids(image_snapshot)

        normalized_csv = temporary_root / "met-normalized.csv"
        counts = {
            "rejected_missing_id": 0,
            "rejected_without_usable_date": 0,
            "rejected_outside_date_bounds": 0,
            "non_public_domain_rows": 0,
            "rows_without_known_image": 0,
        }
        input_rows = 0
        eligible_rows = 0
        with source.open("r", encoding="utf-8-sig", newline="") as source_handle, normalized_csv.open(
            "w", encoding="utf-8", newline=""
        ) as output_handle:
            reader = csv.DictReader(source_handle)
            if not reader.fieldnames or not MET_REQUIRED_FIELDS.issubset(reader.fieldnames):
                missing = sorted(MET_REQUIRED_FIELDS - set(reader.fieldnames or ()))
                raise CorpusBuildError(f"Met CSV is missing required columns: {', '.join(missing)}")
            writer = csv.DictWriter(output_handle, fieldnames=CANONICAL_FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                input_rows += 1
                try:
                    object_id = int(row.get("Object ID", ""))
                except ValueError:
                    counts["rejected_missing_id"] += 1
                    continue
                if not parse_bool(row.get("Is Public Domain")):
                    counts["non_public_domain_rows"] += 1
                image_available = object_id in image_ids
                if not image_available:
                    counts["rows_without_known_image"] += 1
                canonical = _canonical_met_row(
                    row, object_id, image_available=image_available
                )
                try:
                    date = normalize_date(canonical, config)
                except ValueError:
                    date = None
                if date is None or not date.dated:
                    counts["rejected_without_usable_date"] += 1
                    continue
                date = _bounded_date(date, config)
                if date is None:
                    counts["rejected_outside_date_bounds"] += 1
                    continue
                canonical["date_start"] = date.start
                canonical["date_end"] = date.end
                writer.writerow(canonical)
                eligible_rows += 1

        if eligible_rows == 0:
            raise CorpusBuildError("Met eligibility rules selected no artworks")
        built = temporary_root / "artifact"
        payloads = (source, image_snapshot) if image_snapshot is not None else (source,)
        payload_row_counts = {source.name: input_rows}
        if image_snapshot is not None:
            payload_row_counts[image_snapshot.name] = image_id_rows
        manifest = build_corpus(
            normalized_csv,
            built,
            corpus_version=corpus_version,
            source_revision=source_revision,
            source_url=MET_SOURCE_URL,
            retrieved_at=retrieved_at,
            metadata_license=MET_CC0_URI,
            date_config=config,
            require_parquet=require_parquet,
            source_kind=MET_SOURCE_KIND,
            source_payloads=payloads,
            source_payload_row_counts=payload_row_counts,
            input_row_count=input_rows,
            source_counts={
                **counts,
                "eligible_rows": eligible_rows,
                "has_image_id_count": len(image_ids),
            },
            source_metadata={
                "image_ids_filename": image_snapshot.name if image_snapshot else None,
                "eligibility": "usable-date",
                "metadata_license": MET_CC0_URI,
            },
        )
        fts_rows = _build_fts_index(built / "corpus.csv", built / MET_FTS_FILENAME)
        if fts_rows != eligible_rows:
            raise CorpusBuildError("Met FTS row count does not match the canonical corpus")
        _attach_fts_manifest(built, manifest, fts_rows)

        if destination.exists():
            destination.rmdir()
        built.replace(destination)
        return manifest
