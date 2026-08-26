"""Prepare a reproducible, rights-gated NGA image subset for visual retrieval."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import ssl
import tempfile
import time
from typing import Callable, Mapping
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from . import __version__
from .build import CANONICAL_FIELDS, CorpusBuildError, sha256_file


VISUAL_SUBSET_SCHEMA_VERSION = "mnemosyne-nga-visual-subset/v1"
NGA_CC0_URI = "https://creativecommons.org/publicdomain/zero/1.0/"
NGA_OPEN_DATA_URL = "https://github.com/NationalGalleryOfArt/opendata"
NGA_IMAGE_INPUT_POLICY = "nga-iiif-fit-1024-short-side-256/v1"
DEFAULT_MAX_DIMENSION = 1024
DEFAULT_MIN_SHORT_SIDE = 256
PHYSICAL_OBJECT_GROUPING_POLICY = "nga-inseparable-association-root/v1"

_PINNED_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_OBJECT_FIELDS = {
    "objectid",
    "title",
    "displaydate",
    "beginyear",
    "endyear",
    "medium",
    "attribution",
    "creditline",
    "classification",
    "subclassification",
    "isvirtual",
    "departmentabbr",
    "wikidataid",
}
_UNKNOWN_DISPLAY_DATE = re.compile(
    r"(?:^\s*n\s*\.\s*d\s*\.?\s*$|"
    r"^\s*(?:date\s+unknown|unknown|undated)\s*$|"
    r"^\s*model\s+date\s+unknown\b)",
    flags=re.IGNORECASE,
)
_IMAGE_FIELDS = {
    "uuid",
    "iiifurl",
    "viewtype",
    "sequence",
    "width",
    "height",
    "openaccess",
    "depictstmsobjectid",
}
_OBJECT_ASSOCIATION_FIELDS = {
    "parentobjectid",
    "childobjectid",
    "relationship",
}


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"null", "none", "nan"} else text


def _is_true(value: object) -> bool:
    return _clean(value).casefold() in {"1", "true", "t", "yes", "y"}


def _object_id(value: object) -> str | None:
    text = _clean(value)
    if not text or not text.isascii() or not text.isdecimal():
        return None
    return str(int(text))


def _positive_integer(value: object) -> int | None:
    text = _clean(value)
    try:
        number = int(text)
    except ValueError:
        return None
    return number if number > 0 else None


def _historical_year(value: object) -> str:
    text = _clean(value).replace(",", "")
    try:
        number = Decimal(text)
    except InvalidOperation:
        return ""
    if not number.is_finite() or number != number.to_integral_value() or number == 0:
        return ""
    return str(int(number))


def _sequence_key(value: object) -> tuple[int, Decimal]:
    text = _clean(value)
    try:
        number = Decimal(text)
    except InvalidOperation:
        return (1, Decimal(0))
    if not number.is_finite():
        return (1, Decimal(0))
    return (0, number)


def _primary_key(row: Mapping[str, str]) -> tuple[int, Decimal, str]:
    rank, sequence = _sequence_key(row.get("sequence"))
    return rank, sequence, _clean(row.get("uuid")).casefold()


def _validate_nga_image_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError("NGA image URL must use https")
    if (parsed.hostname or "").casefold().rstrip(".") != "api.nga.gov":
        raise ValueError("NGA image URL must use api.nga.gov")
    if parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise ValueError("NGA image URL must not contain credentials or a custom port")
    if parsed.query or parsed.fragment:
        raise ValueError("NGA image URL must not contain a query or fragment")


def _iiif_derivative_url(
    iiif_url: str,
    image_uuid: str,
    width: int,
    height: int,
    *,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    min_short_side: int = DEFAULT_MIN_SHORT_SIDE,
) -> tuple[str, int, int]:
    """Return a bounded NGA IIIF derivative and its requested dimensions.

    A normal image fits within ``max_dimension``. Very wide or tall works use a
    larger long edge only when that is necessary to keep the short edge large
    enough for the 224 px image encoder. Images whose source short edge is too
    small are rejected rather than silently upscaled.
    """

    if max_dimension < min_short_side or min_short_side < 1:
        raise ValueError("IIIF derivative dimensions are invalid")
    if width < 1 or height < 1:
        raise ValueError("NGA source image dimensions must be positive")
    if min(width, height) < min_short_side:
        raise ValueError(
            f"NGA source image short side is below {min_short_side}px"
        )
    image_uuid = _clean(image_uuid).casefold()
    if not _UUID.fullmatch(image_uuid):
        raise ValueError("NGA image UUID is invalid")
    base = _clean(iiif_url).rstrip("/")
    _validate_nga_image_url(base)
    parsed = urlsplit(base)
    if parsed.path.casefold() != f"/iiif/{image_uuid}":
        raise ValueError("NGA IIIF URL does not match the published image UUID")

    longest = max(width, height)
    shortest = min(width, height)
    scale = min(1.0, max(max_dimension / longest, min_short_side / shortest))
    target_width = min(width, max(1, math.ceil(width * scale)))
    target_height = min(height, max(1, math.ceil(height * scale)))
    if min(target_width, target_height) < min_short_side:
        raise ValueError("NGA IIIF derivative cannot satisfy the short-side floor")
    derivative = (
        f"{base}/full/!{target_width},{target_height}/0/default.jpg"
    )
    _validate_nga_image_url(derivative)
    return derivative, target_width, target_height


class _ValidatedNgaRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_nga_image_url(newurl)
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
        _ValidatedNgaRedirectHandler(),
    )


def _remote_image_available(url: str, retries: int = 2) -> tuple[bool, str]:
    _validate_nga_image_url(url)
    request = Request(
        url,
        method="HEAD",
        headers={"Accept": "image/*", "User-Agent": "Mnemosyne NGA embedding preflight"},
    )
    for attempt in range(retries + 1):
        try:
            with _verified_opener().open(request, timeout=20) as response:
                _validate_nga_image_url(response.geturl())
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


def _rank(seed: str, artwork_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\x1f{artwork_id}".encode("utf-8")).digest()


def _image_input_policy(max_dimension: int, min_short_side: int) -> str:
    if (
        max_dimension == DEFAULT_MAX_DIMENSION
        and min_short_side == DEFAULT_MIN_SHORT_SIDE
    ):
        return NGA_IMAGE_INPUT_POLICY
    return f"nga-iiif-fit-{max_dimension}-short-side-{min_short_side}/v1"


def _read_primary_images(
    published_images_csv: Path,
    *,
    max_dimension: int,
    min_short_side: int,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    if not published_images_csv.is_file():
        raise CorpusBuildError(
            f"NGA published_images.csv is missing: {published_images_csv}"
        )
    selected: dict[str, dict[str, str]] = {}
    stats = {
        "input_image_rows": 0,
        "rejected_not_open_access": 0,
        "rejected_not_primary": 0,
        "rejected_missing_object_id": 0,
        "rejected_invalid_image_metadata": 0,
        "rejected_source_short_side": 0,
        "eligible_primary_image_rows": 0,
        "superseded_primary_image_rows": 0,
    }
    with published_images_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_IMAGE_FIELDS - fields)
        if missing:
            raise CorpusBuildError(
                "NGA published_images.csv is missing required fields: "
                + ", ".join(missing)
            )
        for source in reader:
            stats["input_image_rows"] += 1
            if not _is_true(source.get("openaccess")):
                stats["rejected_not_open_access"] += 1
                continue
            if _clean(source.get("viewtype")).casefold() != "primary":
                stats["rejected_not_primary"] += 1
                continue
            source_id = _object_id(source.get("depictstmsobjectid"))
            if source_id is None:
                stats["rejected_missing_object_id"] += 1
                continue
            image_uuid = _clean(source.get("uuid")).casefold()
            width = _positive_integer(source.get("width"))
            height = _positive_integer(source.get("height"))
            if not _UUID.fullmatch(image_uuid) or width is None or height is None:
                stats["rejected_invalid_image_metadata"] += 1
                continue
            try:
                image_url, target_width, target_height = _iiif_derivative_url(
                    _clean(source.get("iiifurl")),
                    image_uuid,
                    width,
                    height,
                    max_dimension=max_dimension,
                    min_short_side=min_short_side,
                )
            except ValueError as exc:
                if "short side" in str(exc):
                    stats["rejected_source_short_side"] += 1
                else:
                    stats["rejected_invalid_image_metadata"] += 1
                continue
            candidate = {
                "uuid": image_uuid,
                "sequence": _clean(source.get("sequence")),
                "image_url": image_url,
                "image_width": str(target_width),
                "image_height": str(target_height),
            }
            stats["eligible_primary_image_rows"] += 1
            current = selected.get(source_id)
            if current is None or _primary_key(candidate) < _primary_key(current):
                if current is not None:
                    stats["superseded_primary_image_rows"] += 1
                selected[source_id] = candidate
            else:
                stats["superseded_primary_image_rows"] += 1
    stats["selected_primary_image_objects"] = len(selected)
    return selected, stats


def _wikidata_url(value: object) -> str:
    identifier = _clean(value)
    if re.fullmatch(r"Q[1-9][0-9]*", identifier, flags=re.IGNORECASE):
        return f"https://www.wikidata.org/wiki/{identifier.upper()}"
    return ""


def _date_fields(source: Mapping[str, str]) -> dict[str, str]:
    display = _clean(source.get("displaydate"))
    if not display:
        return {
            "date_display": "",
            "date_start": "",
            "date_end": "",
            "date_qualifier": "unknown",
            "date_parse_method": "nga_missing_displaydate_numeric_ignored",
        }
    start = _historical_year(source.get("beginyear"))
    end = _historical_year(source.get("endyear"))
    lowered = display.casefold()
    if re.match(r"^(?:c\.|ca\.?|circa|about|approximately)\s*", lowered):
        qualifier = "circa"
    elif start and end and start != end:
        qualifier = "range"
    elif start or end:
        qualifier = "exact"
    else:
        qualifier = "unknown"
    if start and end:
        method = (
            "nga_displaydate_gated_source_range"
            if start != end
            else "nga_displaydate_gated_source_exact"
        )
    elif start or end:
        method = "nga_displaydate_gated_source_single_bound"
    else:
        method = "nga_displaydate_only"
    return {
        "date_display": display,
        "date_start": start,
        "date_end": end,
        "date_qualifier": qualifier,
        "date_parse_method": method,
    }


def _is_unknown_display_date(value: object) -> bool:
    """Recognize NGA display labels that explicitly disclaim a work date."""

    return bool(_UNKNOWN_DISPLAY_DATE.search(_clean(value)))


def _read_physical_object_groups(
    object_associations_csv: Path,
    object_ids: set[str],
) -> tuple[dict[str, str], dict[str, int]]:
    """Resolve inseparable association chains without removing catalog rows."""

    if not object_associations_csv.is_file():
        raise CorpusBuildError(
            "NGA object_associations.csv is missing: "
            f"{object_associations_csv}"
        )

    parent_by_child: dict[str, str] = {}
    seen_associations: set[tuple[str, str, str]] = set()
    stats = {
        "input_association_rows": 0,
        "inseparable_association_rows": 0,
        "separable_association_rows": 0,
        "other_relationship_rows": 0,
    }
    with object_associations_csv.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_OBJECT_ASSOCIATION_FIELDS - fields)
        if missing:
            raise CorpusBuildError(
                "NGA object_associations.csv is missing required fields: "
                + ", ".join(missing)
            )
        for row_number, source in enumerate(reader, start=2):
            stats["input_association_rows"] += 1
            parent_id = _object_id(source.get("parentobjectid"))
            child_id = _object_id(source.get("childobjectid"))
            if parent_id is None or child_id is None:
                raise CorpusBuildError(
                    "NGA object_associations.csv row "
                    f"{row_number}: invalid parentobjectid or childobjectid"
                )
            missing_ids = sorted({parent_id, child_id} - object_ids, key=int)
            if missing_ids:
                raise CorpusBuildError(
                    "NGA object_associations.csv row "
                    f"{row_number}: object ID(s) missing from objects.csv: "
                    + ", ".join(missing_ids)
                )

            relationship = _clean(source.get("relationship")).casefold()
            association = parent_id, child_id, relationship
            if association in seen_associations:
                raise CorpusBuildError(
                    "NGA object_associations.csv row "
                    f"{row_number}: duplicate association "
                    f"{parent_id}->{child_id} ({relationship or 'blank'})"
                )
            seen_associations.add(association)

            if relationship == "separable":
                stats["separable_association_rows"] += 1
                continue
            if relationship != "inseparable":
                stats["other_relationship_rows"] += 1
                continue
            stats["inseparable_association_rows"] += 1
            existing_parent = parent_by_child.get(child_id)
            if existing_parent is not None and existing_parent != parent_id:
                raise CorpusBuildError(
                    "NGA object_associations.csv row "
                    f"{row_number}: inseparable child {child_id} has multiple "
                    f"parents ({existing_parent}, {parent_id})"
                )
            parent_by_child[child_id] = parent_id

    root_by_member: dict[str, str] = {}
    depth_by_member: dict[str, int] = {}
    for start in sorted(parent_by_child, key=int):
        if start in root_by_member:
            continue
        path: list[str] = []
        positions: dict[str, int] = {}
        current = start
        while current in parent_by_child:
            if current in root_by_member:
                root = root_by_member[current]
                base_depth = depth_by_member[current]
                break
            if current in positions:
                cycle = path[positions[current] :] + [current]
                raise CorpusBuildError(
                    "NGA object_associations.csv contains an inseparable cycle: "
                    + " -> ".join(cycle)
                )
            positions[current] = len(path)
            path.append(current)
            current = parent_by_child[current]
        else:
            root = current
            base_depth = 0
        depth = base_depth
        for member in reversed(path):
            depth += 1
            root_by_member[member] = root
            depth_by_member[member] = depth

    stats["inseparable_child_objects"] = len(parent_by_child)
    stats["inseparable_physical_groups"] = len(set(root_by_member.values()))
    stats["maximum_inseparable_chain_depth"] = max(
        depth_by_member.values(), default=0
    )
    return root_by_member, stats


def _canonical_row(
    source: Mapping[str, str],
    source_id: str,
    image: Mapping[str, str],
    source_revision: str,
    *,
    timeline_eligible: bool,
    image_input_policy: str,
) -> dict[str, str]:
    artwork_id = f"NGA_{source_id}"
    classification = _clean(source.get("classification"))
    subclassification = _clean(source.get("subclassification"))
    return {
        "artwork_id": artwork_id,
        "physical_object_id": artwork_id,
        "visual_cluster_id": "",
        "institution": "nga",
        "source_id": source_id,
        "source_record_url": (
            f"https://www.nga.gov/collection/art-object-page.{source_id}.html"
        ),
        "source_dataset_version": source_revision,
        "title": _clean(source.get("title")),
        "artist": _clean(source.get("attribution")),
        "object_type": subclassification or classification,
        "medium": _clean(source.get("medium")),
        "culture": "",
        "department": _clean(source.get("departmentabbr")),
        "classification": classification,
        "period": "",
        "dynasty": "",
        "geography": "",
        "tags": "",
        "object_wikidata_url": _wikidata_url(source.get("wikidataid")),
        **(
            _date_fields(source)
            if timeline_eligible
            else {
                "date_display": "",
                "date_start": "",
                "date_end": "",
                "date_qualifier": "unknown",
                "date_parse_method": "nga_strict_timeline_gate_excluded",
            }
        ),
        "metadata_license": NGA_CC0_URI,
        "image_rights_uri": NGA_CC0_URI,
        "credit_line": _clean(source.get("creditline")),
        "public_domain": "True",
        "image_available": "True",
        "image_url": image["image_url"],
        "image_sha256": "",
        "image_width": image["image_width"],
        "image_height": image["image_height"],
        "embedding_offset": "",
        "image_path": "",
        "image_use_permitted": "True",
        "image_input_policy": image_input_policy,
    }


def _read_object_candidates(
    objects_csv: Path,
    primary_images: Mapping[str, dict[str, str]],
    source_revision: str,
    *,
    include_undated: bool,
    image_input_policy: str,
) -> tuple[list[dict[str, str]], dict[str, int], set[str]]:
    if not objects_csv.is_file():
        raise CorpusBuildError(f"NGA objects.csv is missing: {objects_csv}")
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    matched_image_ids: set[str] = set()
    stats = {
        "input_object_rows": 0,
        "rejected_invalid_object_id": 0,
        "rejected_virtual_objects": 0,
        "rejected_without_display_date": 0,
        "rejected_unknown_display_date": 0,
        "rejected_without_numeric_date_bounds": 0,
        "objects_without_display_date": 0,
        "objects_unknown_display_date": 0,
        "objects_without_numeric_date_bounds": 0,
    }
    with objects_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_OBJECT_FIELDS - fields)
        if missing:
            raise CorpusBuildError(
                "NGA objects.csv is missing required fields: " + ", ".join(missing)
            )
        for row_number, source in enumerate(reader, start=2):
            stats["input_object_rows"] += 1
            source_id = _object_id(source.get("objectid"))
            if source_id is None:
                stats["rejected_invalid_object_id"] += 1
                continue
            if source_id in seen:
                raise CorpusBuildError(
                    f"NGA objects.csv row {row_number}: duplicate objectid {source_id!r}"
                )
            seen.add(source_id)
            image = primary_images.get(source_id)
            if image is None:
                continue
            matched_image_ids.add(source_id)
            if _is_true(source.get("isvirtual")):
                stats["rejected_virtual_objects"] += 1
                continue
            has_display_date = bool(_clean(source.get("displaydate")))
            has_unknown_display_date = _is_unknown_display_date(
                source.get("displaydate")
            )
            has_numeric_bounds = bool(
                _historical_year(source.get("beginyear"))
                and _historical_year(source.get("endyear"))
            )
            timeline_eligible = (
                has_display_date
                and not has_unknown_display_date
                and has_numeric_bounds
            )
            if not has_display_date:
                stats["objects_without_display_date"] += 1
                if not include_undated:
                    stats["rejected_without_display_date"] += 1
                    continue
            elif has_unknown_display_date:
                stats["objects_unknown_display_date"] += 1
                if not include_undated:
                    stats["rejected_unknown_display_date"] += 1
                    continue
            elif not has_numeric_bounds:
                stats["objects_without_numeric_date_bounds"] += 1
                if not include_undated:
                    stats["rejected_without_numeric_date_bounds"] += 1
                    continue
            candidates.append(
                _canonical_row(
                    source,
                    source_id,
                    image,
                    source_revision,
                    timeline_eligible=timeline_eligible,
                    image_input_policy=image_input_policy,
                )
            )
    stats["primary_images_without_object"] = len(set(primary_images) - matched_image_ids)
    stats["rights_image_nonvirtual_candidates"] = (
        len(candidates)
        + stats["rejected_without_display_date"]
        + stats["rejected_unknown_display_date"]
        + stats["rejected_without_numeric_date_bounds"]
    )
    stats["strict_timeline_candidates"] = (
        len(candidates)
        - (
            stats["objects_without_display_date"]
            + stats["objects_unknown_display_date"]
            + stats["objects_without_numeric_date_bounds"]
            if include_undated
            else 0
        )
    )
    return candidates, stats, seen


def _output_fields() -> tuple[str, ...]:
    fields = list(CANONICAL_FIELDS)
    for field in ("image_path", "image_use_permitted", "image_input_policy"):
        if field not in fields:
            fields.append(field)
    return tuple(fields)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
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


def prepare_nga_visual_subset(
    objects_csv: Path | str,
    published_images_csv: Path | str,
    output_csv: Path | str,
    *,
    object_associations_csv: Path | str,
    source_revision: str,
    sample_size: int = 0,
    seed: str = "nga-open-access-primary-visual-v1",
    workers: int = 16,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    min_short_side: int = DEFAULT_MIN_SHORT_SIDE,
    preflight: bool = True,
    include_undated: bool = False,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    """Prepare NGA open-access primary-image rows for the canonical build.

    The input files must be local exports from one pinned revision of NGA's
    official Open Data repository. No museum pages or unreviewed images are
    crawled. By default, objects require a human ``displaydate`` plus both
    valid numeric bounds. NGA documents that numeric years without a display
    date commonly fall back to the creator's lifespan; requiring the complete
    trio also guarantees that every embedded row contributes timeline weight.
    """

    if sample_size < 0 or workers < 1:
        raise CorpusBuildError("sample size must be non-negative and workers positive")
    if max_dimension < min_short_side or min_short_side < 224:
        raise CorpusBuildError(
            "NGA IIIF max dimension must cover a short-side floor of at least 224"
        )
    image_input_policy = _image_input_policy(max_dimension, min_short_side)
    revision = _clean(source_revision)
    if not _PINNED_REVISION.fullmatch(revision):
        raise CorpusBuildError("NGA source_revision must be a pinned 40-character git SHA")

    objects_path = Path(objects_csv).resolve()
    images_path = Path(published_images_csv).resolve()
    associations_path = Path(object_associations_csv).resolve()
    output_path = Path(output_csv).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    incomplete_path = output_path.with_suffix(".incomplete.json")
    availability_path = output_path.with_suffix(".availability.csv")
    if {objects_path, images_path, associations_path} & {
        output_path,
        manifest_path,
        incomplete_path,
        availability_path,
    }:
        raise CorpusBuildError("NGA adapter outputs must not overwrite source CSV files")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if incomplete_path.is_file():
        try:
            incomplete = json.loads(incomplete_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusBuildError(
                f"invalid incomplete-build marker requires inspection: {incomplete_path}"
            ) from exc
        if incomplete.get("schema_version") != VISUAL_SUBSET_SCHEMA_VERSION or incomplete.get(
            "output"
        ) != output_path.name:
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

    primary_images, image_stats = _read_primary_images(
        images_path,
        max_dimension=max_dimension,
        min_short_side=min_short_side,
    )
    candidates, object_stats, object_ids = _read_object_candidates(
        objects_path,
        primary_images,
        revision,
        include_undated=include_undated,
        image_input_policy=image_input_policy,
    )
    root_by_member, grouping_stats = _read_physical_object_groups(
        associations_path,
        object_ids,
    )
    for candidate in candidates:
        root_id = root_by_member.get(candidate["source_id"], candidate["source_id"])
        candidate["physical_object_id"] = f"NGA_{root_id}"
    candidates.sort(key=lambda row: (_rank(seed, row["artwork_id"]), row["artwork_id"]))
    if not candidates:
        raise CorpusBuildError("no eligible NGA visual candidates remain after selection")

    full_scan = sample_size == 0
    target_size = sample_size or len(candidates)
    if len(candidates) < target_size:
        raise CorpusBuildError(
            f"only {len(candidates)} eligible NGA candidates exist for sample size {target_size}"
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
                        f"NGA availability cache has an incompatible schema: {availability_path}"
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
                max_workers=workers, thread_name_prefix="nga-image-head"
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
            f"prepared only {len(selected)} of {target_size} NGA images after "
            f"{examined} candidates"
        )
    if not selected:
        raise CorpusBuildError("no reachable NGA images remain after preflight")

    selected.sort(key=lambda row: row["artwork_id"])
    _write_csv(output_path, selected)

    source = {
        "kind": "nga-open-data-local-csv",
        "url": NGA_OPEN_DATA_URL,
        "revision": revision,
        "metadata_license": NGA_CC0_URI,
        "objects": {
            "filename": objects_path.name,
            "sha256": sha256_file(objects_path),
            "bytes": objects_path.stat().st_size,
            "rows": object_stats["input_object_rows"],
        },
        "published_images": {
            "filename": images_path.name,
            "sha256": sha256_file(images_path),
            "bytes": images_path.stat().st_size,
            "rows": image_stats["input_image_rows"],
        },
        "object_associations": {
            "filename": associations_path.name,
            "sha256": sha256_file(associations_path),
            "bytes": associations_path.stat().st_size,
            "rows": grouping_stats["input_association_rows"],
        },
    }
    selected_group_sizes: dict[str, int] = {}
    for row in selected:
        physical_object_id = row["physical_object_id"]
        selected_group_sizes[physical_object_id] = (
            selected_group_sizes.get(physical_object_id, 0) + 1
        )
    manifest: dict[str, object] = {
        "schema_version": VISUAL_SUBSET_SCHEMA_VERSION,
        "builder_version": __version__,
        "source": source,
        "selection": {
            "algorithm": "rights-date-gated-sha256-seeded-sample-with-fallbacks",
            "primary_image_order": "numeric-sequence-ascending-then-uuid-ascending",
            "seed": seed,
            "requested_rows": target_size,
            "prepared_rows": len(selected),
            "eligible_candidates": len(candidates),
            "examined_candidates": examined,
            "require_nonvirtual": True,
            "include_undated": include_undated,
            "included_undated_date_policy": "blank-canonical-date-zero-weight",
            "timeline_date_policy": (
                "require-known-human-displaydate-and-complete-nga-numeric-bounds/v2"
            ),
            **object_stats,
            **image_stats,
        },
        "rights_gate": {
            "institution": "nga",
            "requirement": "published_images.openaccess=1 and viewtype=primary",
            "image_rights_uri": NGA_CC0_URI,
            "rejected_not_open_access": image_stats["rejected_not_open_access"],
        },
        "physical_object_grouping": {
            "policy": PHYSICAL_OBJECT_GROUPING_POLICY,
            "relationship_match": "casefold-exact-inseparable",
            "rows_collapsed": 0,
            **grouping_stats,
            "eligible_rows": len(candidates),
            "eligible_grouped_rows": sum(
                row["physical_object_id"] != row["artwork_id"]
                for row in candidates
            ),
            "eligible_distinct_physical_object_ids": len(
                {row["physical_object_id"] for row in candidates}
            ),
            "selected_rows": len(selected),
            "selected_grouped_rows": sum(
                row["physical_object_id"] != row["artwork_id"]
                for row in selected
            ),
            "selected_distinct_physical_object_ids": len(selected_group_sizes),
            "selected_shared_physical_groups": sum(
                size > 1 for size in selected_group_sizes.values()
            ),
        },
        "placeholder_basenames": [],
        "images": {
            "storage": "stream-at-embed-time",
            "service": "NGA IIIF Image API",
            "availability_preflight": preflight,
            "input_policy": image_input_policy,
            "fit_max_dimension": max_dimension,
            "minimum_short_side": min_short_side,
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
