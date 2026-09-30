"""Prepare a reproducible, rights-gated Cleveland Museum of Art image corpus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import ssl
import tempfile
import time
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from . import __version__
from .build import CANONICAL_FIELDS, CorpusBuildError, sha256_file


VISUAL_SUBSET_SCHEMA_VERSION = "mnemosyne-cma-visual-subset/v1"
CMA_CC0_URI = "https://creativecommons.org/publicdomain/zero/1.0/"
CMA_OPEN_DATA_URL = "https://github.com/ClevelandMuseumArt/openaccess"
CMA_API_URL = "https://openaccess-api.clevelandart.org/api/artworks/"
CMA_IMAGE_HOST = "openaccess-cdn.clevelandart.org"
CMA_IMAGE_INPUT_POLICY = "cma-open-access-web-jpeg/v1"
PHYSICAL_OBJECT_GROUPING_POLICY = "cma-one-catalog-record-per-object/v1"
CMA_TIMELINE_DATE_SANITY_POLICY = "cma-timeline-date-sanity/v1"
CMA_TIMELINE_MIN_YEAR = -15_000
CMA_TIMELINE_MAX_YEAR = 2_026
CMA_TIMELINE_MAX_SPAN_YEARS = 10_000

_PINNED_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNKNOWN_DATE = re.compile(
    r"^\s*(?:date\s+unknown|unknown|undated|n\s*\.\s*d\s*\.?)\s*$",
    flags=re.IGNORECASE,
)


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"null", "none", "nan"} else text


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).casefold() in {"1", "true", "t", "yes", "y"}


def _positive_integer(value: object) -> int | None:
    text = _clean(value)
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if (
        not number.is_finite()
        or number != number.to_integral_value()
        or number <= 0
    ):
        return None
    return int(number)


def _historical_year(value: object) -> str:
    text = _clean(value).replace(",", "")
    try:
        number = Decimal(text)
    except InvalidOperation:
        return ""
    if not number.is_finite() or number != number.to_integral_value() or number == 0:
        return ""
    return str(int(number))


def _rank(seed: str, artwork_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\x1f{artwork_id}".encode("utf-8")).digest()


def _text_list(value: object) -> str:
    if isinstance(value, str):
        return _clean(value)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ""
    values: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean(item)
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return "; ".join(values)


def _artist_text(value: object) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ""
    all_descriptions: list[str] = []
    caption_descriptions: list[str] = []
    for creator in value:
        if not isinstance(creator, Mapping):
            continue
        description = _clean(creator.get("description"))
        if not description:
            continue
        all_descriptions.append(description)
        if _is_true(creator.get("use_in_caption")):
            caption_descriptions.append(description)
    selected = caption_descriptions or all_descriptions
    return "; ".join(dict.fromkeys(selected))


def _wikidata_url(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    raw = value.get("wikidata")
    candidates = raw if isinstance(raw, list) else [raw]
    for candidate in candidates:
        url = _clean(candidate)
        if re.fullmatch(
            r"https://www\.wikidata\.org/wiki/Q[1-9][0-9]*", url, flags=re.IGNORECASE
        ):
            return url
    return ""


def _validate_cma_image_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("CMA image URL is invalid") from exc
    if parsed.scheme != "https":
        raise ValueError("CMA image URL must use https")
    if (parsed.hostname or "").casefold().rstrip(".") != CMA_IMAGE_HOST:
        raise ValueError(f"CMA image URL must use {CMA_IMAGE_HOST}")
    if parsed.username or parsed.password or port not in {None, 443}:
        raise ValueError("CMA image URL must not contain credentials or a custom port")
    if parsed.query or parsed.fragment:
        raise ValueError("CMA image URL must not contain a query or fragment")
    if not parsed.path.casefold().endswith((".jpg", ".jpeg")):
        raise ValueError("CMA web image URL must identify a JPEG")


def _validate_cma_record_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("CMA record URL is invalid") from exc
    if parsed.scheme != "https":
        raise ValueError("CMA record URL must use https")
    if (parsed.hostname or "").casefold().rstrip(".") not in {
        "clevelandart.org",
        "www.clevelandart.org",
    }:
        raise ValueError("CMA record URL must use clevelandart.org")
    if parsed.username or parsed.password or port not in {None, 443}:
        raise ValueError("CMA record URL must not contain credentials or a custom port")
    if parsed.query or parsed.fragment or not parsed.path.startswith("/art/"):
        raise ValueError("CMA record URL must identify an artwork page")


class _ValidatedCmaRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_cma_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@lru_cache(maxsize=1)
def _verified_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:  # pragma: no cover - normally supplied by the model stack
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


@lru_cache(maxsize=1)
def _verified_opener():
    return build_opener(
        HTTPSHandler(context=_verified_ssl_context()),
        _ValidatedCmaRedirectHandler(),
    )


def _remote_image_available(url: str, retries: int = 2) -> tuple[bool, str]:
    _validate_cma_image_url(url)
    request = Request(
        url,
        method="HEAD",
        headers={
            "Accept": "image/*",
            "User-Agent": "Mnemosyne CMA embedding preflight",
        },
    )
    for attempt in range(retries + 1):
        try:
            with _verified_opener().open(request, timeout=20) as response:
                _validate_cma_image_url(response.geturl())
                content_type = response.headers.get("Content-Type", "")
                if content_type and not content_type.casefold().startswith("image/"):
                    return False, f"unexpected content type: {content_type}"
                return True, ""
        except HTTPError as exc:
            retryable = exc.code in {408, 425, 429} or 500 <= exc.code < 600
            if not retryable or attempt >= retries:
                return False, f"HTTP {exc.code}"
        except (OSError, TimeoutError, ValueError) as exc:
            if attempt >= retries:
                return False, str(exc)
        time.sleep(0.25 * (2**attempt))
    return False, "unreachable retry state"


def _cacheable_availability(available: bool, reason: str) -> bool:
    if available or reason.startswith("unexpected content type:"):
        return True
    if reason.startswith("HTTP "):
        try:
            status = int(reason.removeprefix("HTTP "))
        except ValueError:
            return False
        return status not in {408, 425, 429} and not 500 <= status < 600
    return False


def _read_snapshot(
    path: Path,
) -> tuple[list[dict[str, object]], str, int | None]:
    if not path.is_file():
        raise CorpusBuildError(f"CMA JSON snapshot is missing: {path}")
    try:
        with path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise CorpusBuildError(f"CMA JSON snapshot is invalid: {path}") from exc

    reported_total: int | None = None
    if isinstance(payload, list):
        raw_rows = payload
        snapshot_format = "github-data-json"
    elif isinstance(payload, Mapping) and "data" in payload:
        data = payload["data"]
        if isinstance(data, list):
            raw_rows = data
            snapshot_format = "api-artworks-response"
        elif isinstance(data, Mapping):
            raw_rows = [data]
            snapshot_format = "api-artwork-response"
        else:
            raise CorpusBuildError("CMA API snapshot data must be an object or list")
        info = payload.get("info")
        if isinstance(info, Mapping):
            raw_total = info.get("total")
            if isinstance(raw_total, int) and not isinstance(raw_total, bool) and raw_total >= 0:
                reported_total = raw_total
    elif isinstance(payload, Mapping) and "id" in payload:
        raw_rows = [payload]
        snapshot_format = "direct-artwork-object"
    else:
        raise CorpusBuildError(
            "CMA JSON snapshot must be data.json, an API response, or an artwork object"
        )

    if not raw_rows:
        raise CorpusBuildError("CMA JSON snapshot contains no artwork records")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise CorpusBuildError(
                f"CMA JSON snapshot artwork at index {index} is not an object"
            )
        # json.load already created ordinary dictionaries. Retain those objects
        # instead of copying the very large official data.json payload.
        rows.append(row)
    return rows, snapshot_format, reported_total


def _timeline_date_sanity_reasons(
    display: str, start: int, end: int
) -> tuple[str, ...]:
    lowered = display.casefold()
    has_bce = bool(re.search(r"\b(?:bce|bc)\b", lowered))
    has_ce = bool(re.search(r"\b(?:ce|ad)\b", lowered))
    displayed_magnitudes = [
        int(value.replace(",", ""))
        for value in re.findall(r"\d[\d,]*", lowered)
    ]
    reasons: list[str] = []
    if start < CMA_TIMELINE_MIN_YEAR:
        reasons.append("before_supported_minimum")
    if end > CMA_TIMELINE_MAX_YEAR:
        reasons.append("after_supported_maximum")
    if end - start > CMA_TIMELINE_MAX_SPAN_YEARS:
        reasons.append("excessive_date_span")
    if (
        has_bce
        and displayed_magnitudes
        and max(displayed_magnitudes) > abs(CMA_TIMELINE_MIN_YEAR)
    ):
        reasons.append("bce_magnitude_outside_supported_range")
    if has_bce and not has_ce and end > 0:
        reasons.append("bce_display_positive_bound")
    if has_ce and not has_bce and start < 0:
        reasons.append("ce_display_negative_bound")
    if start < 0 < end and not (has_bce and has_ce):
        reasons.append("unexplained_cross_era_bounds")
    if lowered.strip() == "invalid":
        reasons.append("literal_invalid_display")
    if re.search(r"\bmodern\b", lowered):
        reasons.append("disjunctive_modern_date")
    if has_bce and has_ce and " or " in lowered:
        reasons.append("disjunctive_bce_ce_date")
    return tuple(reasons)


def _date_fields(
    source: Mapping[str, object],
) -> tuple[dict[str, str], str | None, tuple[str, ...]]:
    display = _clean(source.get("creation_date")) or _clean(source.get("date_text"))
    if not display:
        return (
            _unknown_date_fields("cma_missing_creation_date"),
            "without_creation_date",
            (),
        )
    if _UNKNOWN_DATE.fullmatch(display):
        return (
            _unknown_date_fields("cma_unknown_creation_date"),
            "unknown_creation_date",
            (),
        )
    start = _historical_year(source.get("creation_date_earliest"))
    end = _historical_year(source.get("creation_date_latest"))
    if not start or not end:
        return (
            _unknown_date_fields("cma_incomplete_numeric_date_bounds"),
            "without_numeric_date_bounds",
            (),
        )
    if int(start) > int(end):
        return (
            _unknown_date_fields("cma_invalid_numeric_date_bounds"),
            "invalid_numeric_date_bounds",
            (),
        )

    sanity_reasons = _timeline_date_sanity_reasons(display, int(start), int(end))
    if sanity_reasons:
        return (
            _unknown_date_fields("cma_untrusted_timeline_date"),
            "untrusted_timeline_date",
            sanity_reasons,
        )

    lowered = display.casefold()
    if re.match(r"^(?:c\.|ca\.?|circa|about|approximately)\s*", lowered):
        qualifier = "circa"
    elif start != end:
        qualifier = "range"
    else:
        qualifier = "exact"
    method = "cma_creation_date_bounds_range" if start != end else "cma_creation_date_bounds_exact"
    return {
        "date_display": display,
        "date_start": start,
        "date_end": end,
        "date_qualifier": qualifier,
        "date_parse_method": method,
    }, None, ()


def _unknown_date_fields(method: str) -> dict[str, str]:
    return {
        "date_display": "",
        "date_start": "",
        "date_end": "",
        "date_qualifier": "unknown",
        "date_parse_method": method,
    }


def _source_record_url(source: Mapping[str, object], source_id: str) -> str:
    supplied = _clean(source.get("url"))
    if supplied:
        try:
            _validate_cma_record_url(supplied)
        except ValueError:
            pass
        else:
            return supplied
    accession_number = _clean(source.get("accession_number"))
    identifier = accession_number or source_id
    return f"https://www.clevelandart.org/art/{quote(identifier, safe='')}"


def _canonical_row(
    source: Mapping[str, object],
    source_id: str,
    web_image: Mapping[str, object],
    date_fields: Mapping[str, str],
    source_revision: str,
) -> dict[str, str]:
    artwork_id = f"CMA_{source_id}"
    return {
        "artwork_id": artwork_id,
        "physical_object_id": artwork_id,
        "visual_cluster_id": "",
        "institution": "cma",
        "source_id": source_id,
        "source_record_url": _source_record_url(source, source_id),
        "source_dataset_version": source_revision,
        "title": _clean(source.get("title")),
        "artist": _artist_text(source.get("creators")),
        "object_type": _clean(source.get("type")),
        "medium": _clean(source.get("technique")),
        "culture": _text_list(source.get("culture")),
        "department": _clean(source.get("department")),
        "classification": _clean(source.get("collection")),
        "period": "",
        "dynasty": "",
        "geography": "",
        "tags": _text_list(source.get("artists_tags")),
        "object_wikidata_url": _wikidata_url(source.get("external_resources")),
        **date_fields,
        "metadata_license": CMA_CC0_URI,
        "image_rights_uri": CMA_CC0_URI,
        "credit_line": _clean(source.get("creditline")),
        "public_domain": "True",
        "image_available": "True",
        "image_url": _clean(web_image.get("url")),
        "image_sha256": "",
        "image_width": str(_positive_integer(web_image.get("width")) or ""),
        "image_height": str(_positive_integer(web_image.get("height")) or ""),
        "embedding_offset": "",
        "image_path": "",
        "image_use_permitted": "True",
        "image_input_policy": CMA_IMAGE_INPUT_POLICY,
    }


def _read_candidates(
    rows: Sequence[Mapping[str, object]],
    source_revision: str,
    *,
    include_undated: bool,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    stats = {
        "input_rows": 0,
        "rejected_invalid_id": 0,
        "rejected_non_object_record": 0,
        "rejected_not_cc0": 0,
        "rejected_nonblank_copyright": 0,
        "rejected_missing_web_image": 0,
        "rejected_invalid_web_image": 0,
        "objects_without_creation_date": 0,
        "objects_unknown_creation_date": 0,
        "objects_without_numeric_date_bounds": 0,
        "objects_invalid_numeric_date_bounds": 0,
        "objects_untrusted_timeline_date": 0,
        "rejected_without_creation_date": 0,
        "rejected_unknown_creation_date": 0,
        "rejected_without_numeric_date_bounds": 0,
        "rejected_invalid_numeric_date_bounds": 0,
        "rejected_untrusted_timeline_date": 0,
    }
    date_sanity_reason_counts: dict[str, int] = {}
    date_sanity_rows: list[dict[str, object]] = []
    for row_number, source in enumerate(rows, start=1):
        stats["input_rows"] += 1
        source_number = _positive_integer(source.get("id"))
        if source_number is None:
            stats["rejected_invalid_id"] += 1
            continue
        source_id = str(source_number)
        if source_id in seen_ids:
            raise CorpusBuildError(
                f"CMA JSON snapshot artwork {row_number}: duplicate id {source_id!r}"
            )
        seen_ids.add(source_id)

        record_type = _clean(source.get("record_type"))
        if record_type and record_type.casefold() != "object":
            stats["rejected_non_object_record"] += 1
            continue
        if _clean(source.get("share_license_status")) != "CC0":
            stats["rejected_not_cc0"] += 1
            continue
        if _clean(source.get("copyright")):
            stats["rejected_nonblank_copyright"] += 1
            continue

        images = source.get("images")
        web_image = images.get("web") if isinstance(images, Mapping) else None
        if not isinstance(web_image, Mapping) or not _clean(web_image.get("url")):
            stats["rejected_missing_web_image"] += 1
            continue
        try:
            _validate_cma_image_url(_clean(web_image.get("url")))
        except ValueError:
            stats["rejected_invalid_web_image"] += 1
            continue
        if (
            _positive_integer(web_image.get("width")) is None
            or _positive_integer(web_image.get("height")) is None
        ):
            stats["rejected_invalid_web_image"] += 1
            continue

        date_fields, date_issue, date_sanity_reasons = _date_fields(source)
        if date_sanity_reasons:
            for reason in date_sanity_reasons:
                date_sanity_reason_counts[reason] = (
                    date_sanity_reason_counts.get(reason, 0) + 1
                )
            date_sanity_rows.append(
                {
                    "artwork_id": f"CMA_{source_id}",
                    "date_display": _clean(source.get("creation_date"))
                    or _clean(source.get("date_text")),
                    "reasons": list(date_sanity_reasons),
                }
            )
        if date_issue is not None:
            stats[f"objects_{date_issue}"] += 1
            if not include_undated:
                stats[f"rejected_{date_issue}"] += 1
                continue
        candidates.append(
            _canonical_row(
                source,
                source_id,
                web_image,
                date_fields,
                source_revision,
            )
        )

    stats["rights_image_candidates"] = len(candidates) + sum(
        stats[key]
        for key in (
            "rejected_without_creation_date",
            "rejected_unknown_creation_date",
            "rejected_without_numeric_date_bounds",
            "rejected_invalid_numeric_date_bounds",
            "rejected_untrusted_timeline_date",
        )
    )
    stats["strict_timeline_candidates"] = len(candidates) - (
        sum(
            stats[key]
            for key in (
                "objects_without_creation_date",
                "objects_unknown_creation_date",
                "objects_without_numeric_date_bounds",
                "objects_invalid_numeric_date_bounds",
                "objects_untrusted_timeline_date",
            )
        )
        if include_undated
        else 0
    )
    return candidates, {
        **stats,
        "timeline_date_sanity": {
            "policy": CMA_TIMELINE_DATE_SANITY_POLICY,
            "supported_min_year": CMA_TIMELINE_MIN_YEAR,
            "supported_max_year": CMA_TIMELINE_MAX_YEAR,
            "maximum_span_years": CMA_TIMELINE_MAX_SPAN_YEARS,
            "quarantined_rows": stats["objects_untrusted_timeline_date"],
            "reason_counts": dict(sorted(date_sanity_reason_counts.items())),
            "sample_rows": sorted(
                date_sanity_rows, key=lambda row: str(row["artwork_id"])
            )[:50],
        },
    }


def _output_fields() -> tuple[str, ...]:
    fields = list(CANONICAL_FIELDS)
    for field in ("image_path", "image_use_permitted", "image_input_policy"):
        if field not in fields:
            fields.append(field)
    return tuple(fields)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.stem}-",
        suffix=".csv",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        writer = csv.DictWriter(
            temporary,
            fieldnames=_output_fields(),
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_cma_visual_subset(
    snapshot_json: Path | str,
    output_csv: Path | str,
    *,
    source_revision: str,
    sample_size: int = 0,
    seed: str = "cma-cc0-web-visual-v1",
    workers: int = 16,
    preflight: bool = True,
    include_undated: bool = False,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    """Prepare CMA CC0 web-image rows from a pinned local official JSON snapshot."""

    if sample_size < 0 or workers < 1:
        raise CorpusBuildError("sample size must be non-negative and workers positive")
    revision = _clean(source_revision).casefold()

    snapshot_path = Path(snapshot_json).resolve()
    output_path = Path(output_csv).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    incomplete_path = output_path.with_suffix(".incomplete.json")
    availability_path = output_path.with_suffix(".availability.csv")
    if snapshot_path in {
        output_path,
        manifest_path,
        incomplete_path,
        availability_path,
    }:
        raise CorpusBuildError("CMA adapter outputs must not overwrite the source snapshot")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if incomplete_path.is_file():
        try:
            incomplete = json.loads(incomplete_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusBuildError(
                f"invalid incomplete-build marker requires inspection: {incomplete_path}"
            ) from exc
        if (
            incomplete.get("schema_version") != VISUAL_SUBSET_SCHEMA_VERSION
            or incomplete.get("output") != output_path.name
        ):
            raise CorpusBuildError(
                f"incomplete-build marker does not own this output: {incomplete_path}"
            )
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    elif output_path.exists() or manifest_path.exists():
        raise CorpusBuildError(f"output CSV or manifest already exists: {output_path}")

    marker_temporary = incomplete_path.with_suffix(".tmp")
    marker_temporary.write_text(
        json.dumps(
            {"schema_version": VISUAL_SUBSET_SCHEMA_VERSION, "output": output_path.name},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    marker_temporary.replace(incomplete_path)

    rows, snapshot_format, reported_total = _read_snapshot(snapshot_path)
    snapshot_sha256 = sha256_file(snapshot_path)
    if snapshot_format == "github-data-json":
        if not _PINNED_GIT_REVISION.fullmatch(revision):
            raise CorpusBuildError(
                "CMA Git/LFS data.json source_revision must be a pinned "
                "40-character git SHA"
            )
        revision_kind = "git-commit"
    else:
        if not _SHA256.fullmatch(revision) or revision != snapshot_sha256:
            raise CorpusBuildError(
                "CMA API snapshot source_revision must equal the snapshot SHA-256"
            )
        revision_kind = "snapshot-sha256"
    snapshot_rows = len(rows)
    candidates, candidate_stats = _read_candidates(
        rows,
        revision,
        include_undated=include_undated,
    )
    # Full CMA records contain extensive bibliography and provenance sections
    # that are not carried into the compact visual corpus. Release them before
    # network preflight and embedding preparation.
    del rows
    candidates.sort(key=lambda row: (_rank(seed, row["artwork_id"]), row["artwork_id"]))
    if not candidates:
        raise CorpusBuildError("no eligible CMA visual candidates remain after selection")

    full_scan = sample_size == 0
    target_size = sample_size or len(candidates)
    if len(candidates) < target_size:
        raise CorpusBuildError(
            f"only {len(candidates)} eligible CMA candidates exist for sample size {target_size}"
        )

    selected: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    examined = 0
    if not preflight:
        selected.extend(candidates[:target_size])
        examined = target_size
        if progress:
            progress(examined, len(selected), len(candidates))
    else:
        cache_fields = ("artwork_id", "image_url", "available", "reason")
        candidate_by_id = {row["artwork_id"]: row for row in candidates}
        cached: dict[str, tuple[bool, str]] = {}
        if availability_path.is_file():
            with availability_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != cache_fields:
                    raise CorpusBuildError(
                        f"CMA availability cache has an incompatible schema: {availability_path}"
                    )
                for row in reader:
                    artwork_id = _clean(row.get("artwork_id"))
                    candidate = candidate_by_id.get(artwork_id)
                    if candidate is None or _clean(row.get("image_url")) != candidate["image_url"]:
                        continue
                    available = _is_true(row.get("available"))
                    reason = _clean(row.get("reason"))
                    if _cacheable_availability(available, reason):
                        cached[artwork_id] = available, reason

        cache_exists = availability_path.is_file() and availability_path.stat().st_size > 0
        with availability_path.open("a", encoding="utf-8", newline="") as cache_handle:
            cache_writer = csv.DictWriter(
                cache_handle, fieldnames=cache_fields, lineterminator="\n"
            )
            if not cache_exists:
                cache_writer.writeheader()
                cache_handle.flush()
            block_size = max(64, workers * 4)
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="cma-image-head"
            ) as executor:
                for offset in range(0, len(candidates), block_size):
                    block = candidates[offset : offset + block_size]
                    missing = [row for row in block if row["artwork_id"] not in cached]
                    checked = executor.map(
                        lambda row: _remote_image_available(row["image_url"]), missing
                    )
                    for candidate, (available, reason) in zip(missing, checked, strict=True):
                        cached[candidate["artwork_id"]] = available, reason
                        if _cacheable_availability(available, reason):
                            cache_writer.writerow(
                                {
                                    "artwork_id": candidate["artwork_id"],
                                    "image_url": candidate["image_url"],
                                    "available": available,
                                    "reason": reason,
                                }
                            )
                    cache_handle.flush()
                    for candidate in block:
                        examined += 1
                        available, reason = cached[candidate["artwork_id"]]
                        if available and len(selected) < target_size:
                            selected.append(candidate)
                        elif not available and len(failures) < 50:
                            failures.append(
                                {"artwork_id": candidate["artwork_id"], "reason": reason}
                            )
                    if progress:
                        progress(examined, len(selected), len(candidates))
                    if not full_scan and len(selected) >= target_size:
                        break

    if not full_scan and len(selected) < target_size:
        raise CorpusBuildError(
            f"prepared only {len(selected)} of {target_size} CMA images after {examined} candidates"
        )
    if not selected:
        raise CorpusBuildError("no reachable CMA images remain after preflight")

    selected.sort(key=lambda row: row["artwork_id"])
    _write_csv(output_path, selected)

    source_url = (
        CMA_OPEN_DATA_URL if snapshot_format == "github-data-json" else CMA_API_URL
    )
    source_kind = (
        "cma-open-access-data-json"
        if snapshot_format == "github-data-json"
        else "cma-open-access-api-snapshot"
    )
    snapshot_entry: dict[str, object] = {
        "filename": snapshot_path.name,
        "format": snapshot_format,
        "sha256": snapshot_sha256,
        "bytes": snapshot_path.stat().st_size,
        "rows": snapshot_rows,
    }
    if reported_total is not None:
        snapshot_entry["reported_total"] = reported_total
    source: dict[str, object] = {
        "kind": source_kind,
        "url": source_url,
        "revision": revision,
        "revision_kind": revision_kind,
        "metadata_license": CMA_CC0_URI,
        "snapshot": snapshot_entry,
    }

    manifest: dict[str, object] = {
        "schema_version": VISUAL_SUBSET_SCHEMA_VERSION,
        "builder_version": __version__,
        "source": source,
        "selection": {
            "algorithm": "rights-date-gated-sha256-seeded-sample-with-fallbacks",
            "seed": seed,
            "requested_rows": target_size,
            "prepared_rows": len(selected),
            "eligible_candidates": len(candidates),
            "examined_candidates": examined,
            "include_undated": include_undated,
            "included_undated_date_policy": "blank-canonical-date-zero-weight",
            "timeline_date_policy": (
                "require-known-creation-date-and-complete-cma-numeric-bounds/v1"
            ),
            **candidate_stats,
        },
        "rights_gate": {
            "institution": "cma",
            "requirement": (
                "share_license_status=CC0, copyright blank, and images.web on "
                f"https://{CMA_IMAGE_HOST}"
            ),
            "metadata_license": CMA_CC0_URI,
            "image_rights_uri": CMA_CC0_URI,
            "rejected_not_cc0": candidate_stats["rejected_not_cc0"],
            "rejected_nonblank_copyright": candidate_stats[
                "rejected_nonblank_copyright"
            ],
        },
        "physical_object_grouping": {
            "policy": PHYSICAL_OBJECT_GROUPING_POLICY,
            "rows_collapsed": 0,
            "selected_rows": len(selected),
            "selected_distinct_physical_object_ids": len(selected),
        },
        "placeholder_basenames": [],
        "images": {
            "storage": "stream-at-embed-time",
            "service": "Cleveland Museum of Art Open Access CDN",
            "variant": "images.web",
            "availability_preflight": preflight,
            "input_policy": CMA_IMAGE_INPUT_POLICY,
            "allowed_host": CMA_IMAGE_HOST,
            "stored_bytes": 0,
        },
        "output": {
            "csv": output_path.name,
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
        "sample_failures": failures,
    }
    manifest_temporary = manifest_path.with_suffix(".tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_temporary.replace(manifest_path)
    incomplete_path.unlink()
    return manifest


__all__ = [
    "CMA_API_URL",
    "CMA_CC0_URI",
    "CMA_IMAGE_HOST",
    "CMA_IMAGE_INPUT_POLICY",
    "CMA_OPEN_DATA_URL",
    "VISUAL_SUBSET_SCHEMA_VERSION",
    "prepare_cma_visual_subset",
]
