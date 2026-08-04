"""Build a dated, public-domain Met corpus from the official Open Access export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .build import CANONICAL_FIELDS, CorpusBuildError, build_corpus
from .dates import DateConfig, normalize_date, parse_bool


MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
MET_SOURCE_URL = "https://github.com/metmuseum/openaccess"
MET_CC0_URI = "https://creativecommons.org/publicdomain/zero/1.0/"
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


def _canonical_met_row(row: dict[str, str], object_id: int) -> dict[str, object]:
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
        "image_rights_uri": MET_CC0_URI,
        "credit_line": row.get("Credit Line", "").strip(),
        "public_domain": "true",
    }


def build_met_corpus(
    input_csv: Path | str,
    output_dir: Path | str,
    *,
    corpus_version: str,
    source_revision: str,
    image_ids_path: Path | str | None = None,
    api_base: str = MET_API_BASE,
    retrieved_at: str | None = None,
    date_config: DateConfig | None = None,
    require_parquet: bool = False,
) -> dict[str, object]:
    """Create local aggregation artifacts without downloading the Met's images."""

    source = Path(input_csv).resolve()
    if not source.is_file():
        raise CorpusBuildError(f"Met Open Access CSV does not exist: {source}")
    config = date_config or DateConfig()
    config.validate()

    with tempfile.TemporaryDirectory(prefix="mnemosyne-met-") as temporary:
        temporary_root = Path(temporary)
        if image_ids_path is None:
            image_snapshot = temporary_root / "met-has-images.json"
            image_snapshot.write_text(
                json.dumps(_fetch_has_image_ids(api_base), separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        else:
            image_snapshot = Path(image_ids_path).resolve()
        image_ids, image_id_rows = _read_image_ids(image_snapshot)

        normalized_csv = temporary_root / "met-normalized.csv"
        counts = {
            "rejected_missing_id": 0,
            "rejected_not_public_domain": 0,
            "rejected_without_image": 0,
            "rejected_without_usable_date": 0,
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
                    counts["rejected_not_public_domain"] += 1
                    continue
                if object_id not in image_ids:
                    counts["rejected_without_image"] += 1
                    continue
                canonical = _canonical_met_row(row, object_id)
                try:
                    date = normalize_date(canonical, config)
                except ValueError:
                    date = None
                if date is None or not date.dated:
                    counts["rejected_without_usable_date"] += 1
                    continue
                writer.writerow(canonical)
                eligible_rows += 1

        if eligible_rows == 0:
            raise CorpusBuildError("Met eligibility rules selected no artworks")
        return build_corpus(
            normalized_csv,
            output_dir,
            corpus_version=corpus_version,
            source_revision=source_revision,
            source_url=MET_SOURCE_URL,
            retrieved_at=retrieved_at,
            metadata_license=MET_CC0_URI,
            date_config=config,
            require_parquet=require_parquet,
            source_kind="met-open-access-csv-with-api-image-snapshot",
            source_payloads=(source, image_snapshot),
            source_payload_row_counts={source.name: input_rows, image_snapshot.name: image_id_rows},
            input_row_count=input_rows,
            source_counts={**counts, "eligible_rows": eligible_rows, "has_image_id_count": len(image_ids)},
            source_metadata={
                "image_ids_filename": image_snapshot.name,
                "eligibility": "public-domain AND has-image AND usable-date",
                "metadata_license": MET_CC0_URI,
            },
        )
