"""Prepare a reproducible, rights-gated SMK image corpus for embedding."""

from __future__ import annotations

import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
import io
import json
from pathlib import Path
import re
import ssl
import tempfile
import time
from typing import Callable, Iterator, Mapping, TextIO
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
import zipfile

from . import __version__
from .build import CANONICAL_FIELDS, CorpusBuildError, sha256_file


VISUAL_SUBSET_SCHEMA_VERSION = "mnemosyne-smk-visual-subset/v1"
SMK_NIGHTLY_SNAPSHOT_URL = "https://getallzip.open.smk.dk/smk_all_da.zip"
SMK_API_DOCUMENTATION_URL = "https://www.smk.dk/en/article/smk-api/"
SMK_PUBLIC_DOMAIN_MARK_URI = "https://creativecommons.org/publicdomain/mark/1.0/"
# The Public Domain Mark is a work/image status assertion, not a metadata
# license. Keep the canonical metadata-license field blank until SMK publishes
# an exact machine-readable dataset-license URI in the consumed snapshot.
SMK_METADATA_LICENSE_URI = ""
SMK_IMAGE_HOST = "iip-thumb.smk.dk"
SMK_IMAGE_INPUT_POLICY = "smk-declared-iiif-thumbnail/v1"

_PINNED_REVISION = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SOURCE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_YEAR_PREFIX = re.compile(r"^([+-]?\d{1,7})(?:-|$)")
_JSON_CHUNK_SIZE = 1024 * 1024
_REQUIRED_SNAPSHOT_FIELDS = frozenset(
    {
        "id",
        "production_date",
        "public_domain",
        "rights",
        "has_image",
        "image_thumbnail",
    }
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


def _list_values(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [] if value is None else [value]


def _mapping_values(value: object, field: str) -> list[str]:
    output: list[str] = []
    for item in _list_values(value):
        candidate = _clean(item.get(field)) if isinstance(item, Mapping) else ""
        if candidate and candidate not in output:
            output.append(candidate)
    return output


def _text_values(value: object) -> list[str]:
    output: list[str] = []
    for item in _list_values(value):
        candidate = _clean(item)
        if candidate and candidate not in output:
            output.append(candidate)
    return output


def _first_title(source: Mapping[str, object]) -> str:
    titles = _mapping_values(source.get("titles"), "title")
    return titles[0] if titles else ""


def _artist(source: Mapping[str, object]) -> str:
    artists = _text_values(source.get("artist"))
    if not artists:
        artists = _mapping_values(source.get("production"), "creator")
    return " | ".join(artists)


def _creator_nationalities(source: Mapping[str, object]) -> str:
    return "; ".join(
        _mapping_values(source.get("production"), "creator_nationality")
    )


def _fold_danish(value: str) -> str:
    return value.casefold().translate(str.maketrans({"æ": "ae", "ø": "o", "å": "a"}))


def _date_follows_creator_activity(value: object) -> bool:
    """Identify SMK's explicit creator-activity fallback disclosure."""

    for note in _text_values(value):
        folded = _fold_danish(note)
        follows = "folger" in folded or "follows" in folded
        creator = any(
            token in folded for token in ("kunstner", "artist", "creator", "monogram")
        )
        activity = any(
            token in folded
            for token in (
                "virkear",
                "active year",
                "activity",
                "working year",
                "lifespan",
            )
        )
        if follows and creator and activity:
            return True
    return False


def _historical_year(value: object) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value) if value != 0 else ""
    text = _clean(value)
    match = _YEAR_PREFIX.match(text)
    if match is None:
        return ""
    year = int(match.group(1))
    return str(year) if year != 0 else ""


def _date_fields(source: Mapping[str, object]) -> tuple[dict[str, str], str]:
    if _date_follows_creator_activity(source.get("production_dates_notes")):
        return (
            {
                "date_display": "",
                "date_start": "",
                "date_end": "",
                "date_qualifier": "unknown",
                "date_parse_method": "smk_creator_activity_date_excluded",
            },
            "creator-activity-excluded",
        )

    production_dates = _list_values(source.get("production_date"))
    first = production_dates[0] if production_dates else None
    if not isinstance(first, Mapping):
        return (
            {
                "date_display": "",
                "date_start": "",
                "date_end": "",
                "date_qualifier": "unknown",
                "date_parse_method": "smk_first_production_date_missing",
            },
            "missing-or-invalid",
        )
    start = _historical_year(first.get("start"))
    end = _historical_year(first.get("end"))
    if not start or not end:
        return (
            {
                "date_display": "",
                "date_start": "",
                "date_end": "",
                "date_qualifier": "unknown",
                "date_parse_method": "smk_first_production_date_invalid",
            },
            "missing-or-invalid",
        )
    if int(start) > int(end):
        start, end = end, start
    period = _clean(first.get("period"))
    folded_period = period.casefold()
    if any(token in folded_period for token in ("ca.", "circa", "approx")):
        qualifier = "circa"
    elif start != end:
        qualifier = "range"
    else:
        qualifier = "exact"
    return (
        {
            "date_display": period or (start if start == end else f"{start}-{end}"),
            "date_start": start,
            "date_end": end,
            "date_qualifier": qualifier,
            "date_parse_method": "smk_first_production_date",
        },
        "dated",
    )


def _validate_smk_image_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("SMK image URL is invalid") from exc
    if parsed.scheme != "https":
        raise ValueError("SMK image URL must use https")
    if (parsed.hostname or "").casefold().rstrip(".") != SMK_IMAGE_HOST:
        raise ValueError(f"SMK image URL must use {SMK_IMAGE_HOST}")
    if parsed.username or parsed.password or port not in {None, 443}:
        raise ValueError("SMK image URL must not contain credentials or a custom port")
    if not parsed.path or parsed.query or parsed.fragment:
        raise ValueError("SMK image URL must have a path and no query or fragment")


class _ValidatedSmkRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_smk_image_url(newurl)
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
        _ValidatedSmkRedirectHandler(),
    )


def _remote_image_available(url: str, retries: int = 2) -> tuple[bool, str]:
    _validate_smk_image_url(url)
    request = Request(
        url,
        method="HEAD",
        headers={"Accept": "image/*", "User-Agent": "Mnemosyne SMK embedding preflight"},
    )
    for attempt in range(retries + 1):
        try:
            with _verified_opener().open(request, timeout=20) as response:
                _validate_smk_image_url(response.geturl())
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


def _iter_json_array(handle: TextIO) -> Iterator[object]:
    """Incrementally decode a JSON array after its opening bracket was read."""

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    first = True

    def read_more(keep_from: int) -> tuple[str, int, bool]:
        remaining = buffer[keep_from:]
        chunk = handle.read(_JSON_CHUNK_SIZE)
        return remaining + chunk, 0, not chunk

    while True:
        while position >= len(buffer) and not eof:
            buffer, position, eof = read_more(position)
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) or eof:
                break
            buffer, position, eof = read_more(position)
        if position >= len(buffer):
            raise CorpusBuildError("SMK snapshot JSON array is not closed")

        if first:
            if buffer[position] == "]":
                position += 1
                break
        else:
            if buffer[position] == "]":
                position += 1
                break
            if buffer[position] != ",":
                raise CorpusBuildError("SMK snapshot JSON array requires comma delimiters")
            position += 1
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    break
                buffer, position, eof = read_more(position)
            if position >= len(buffer) or buffer[position] == "]":
                raise CorpusBuildError("SMK snapshot JSON array has a trailing comma")

        value_start = position
        while True:
            try:
                value, end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError as exc:
                if eof:
                    raise CorpusBuildError("SMK snapshot contains invalid JSON") from exc
                buffer, position, eof = read_more(value_start)
                value_start = 0
        yield value
        position = end
        first = False
        if position > _JSON_CHUNK_SIZE:
            buffer = buffer[position:]
            position = 0

    trailing = buffer[position:] + handle.read()
    if trailing.strip():
        raise CorpusBuildError("SMK snapshot has content after its JSON array")


def _iter_json_document(handle: TextIO) -> Iterator[object]:
    first = handle.read(1)
    while first and first.isspace():
        first = handle.read(1)
    if first == "[":
        yield from _iter_json_array(handle)
        return
    if first == "{":
        try:
            payload = json.loads(first + handle.read())
        except json.JSONDecodeError as exc:
            raise CorpusBuildError("SMK API snapshot contains invalid JSON") from exc
        items = payload.get("items") if isinstance(payload, Mapping) else None
        if not isinstance(items, list):
            raise CorpusBuildError("SMK API snapshot must contain an items array")
        yield from items
        return
    raise CorpusBuildError("SMK snapshot must be a JSON array or API items object")


def _zip_member(path: Path) -> zipfile.ZipInfo | None:
    if not zipfile.is_zipfile(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            members = [
                info
                for info in archive.infolist()
                if not info.is_dir() and Path(info.filename).suffix.casefold() == ".json"
            ]
    except zipfile.BadZipFile as exc:
        raise CorpusBuildError(f"SMK ZIP snapshot is invalid: {path}") from exc
    if len(members) != 1:
        raise CorpusBuildError("SMK ZIP snapshot must contain exactly one JSON member")
    member = members[0]
    if member.flag_bits & 0x1:
        raise CorpusBuildError("SMK ZIP snapshot JSON member must not be encrypted")
    return member


@contextmanager
def _snapshot_text(path: Path, member: zipfile.ZipInfo | None):
    if member is None:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield handle
        return
    try:
        with zipfile.ZipFile(path) as archive, archive.open(member, "r") as binary:
            with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as handle:
                yield handle
    except zipfile.BadZipFile as exc:
        raise CorpusBuildError(f"SMK ZIP snapshot is invalid: {path}") from exc


def _source_record_url(source: Mapping[str, object]) -> str:
    supplied = _clean(source.get("frontend_url"))
    if supplied:
        try:
            parsed = urlsplit(supplied)
            if (
                parsed.scheme == "https"
                and (parsed.hostname or "").casefold().rstrip(".") == "open.smk.dk"
                and not parsed.username
                and not parsed.password
                and parsed.port in {None, 443}
            ):
                return supplied
        except ValueError:
            pass
    object_number = _clean(source.get("object_number"))
    if object_number:
        return f"https://open.smk.dk/artwork/image/{quote(object_number, safe='')}"
    return ""


def _canonical_row(
    source: Mapping[str, object], source_id: str, source_revision: str
) -> tuple[dict[str, str], str, int]:
    artwork_id = f"SMK_{source_id}"
    object_names = _mapping_values(source.get("object_names"), "name")
    techniques = _text_values(source.get("techniques"))
    materials = _text_values(source.get("materials"))
    date_fields, date_status = _date_fields(source)
    production_dates = _list_values(source.get("production_date"))
    return (
        {
            "artwork_id": artwork_id,
            "physical_object_id": artwork_id,
            "visual_cluster_id": "",
            "institution": "smk",
            "source_id": source_id,
            "source_record_url": _source_record_url(source),
            "source_dataset_version": source_revision,
            "title": _first_title(source),
            "artist": _artist(source),
            "object_type": object_names[0] if object_names else "",
            "medium": "; ".join(techniques or materials),
            "culture": _creator_nationalities(source),
            "department": _clean(source.get("responsible_department")),
            "classification": "; ".join(object_names),
            "period": (
                _clean(production_dates[0].get("period"))
                if production_dates and isinstance(production_dates[0], Mapping)
                else ""
            ),
            "dynasty": "",
            "geography": "",
            "tags": "",
            "object_wikidata_url": "",
            **date_fields,
            "metadata_license": SMK_METADATA_LICENSE_URI,
            "image_rights_uri": SMK_PUBLIC_DOMAIN_MARK_URI,
            "credit_line": _clean(source.get("credit_line")),
            "public_domain": "True",
            "image_available": "True",
            "image_url": _clean(source.get("image_thumbnail")),
            "image_sha256": "",
            "image_width": "",
            "image_height": "",
            "embedding_offset": "",
            "image_path": "",
            "image_use_permitted": "True",
            "image_input_policy": SMK_IMAGE_INPUT_POLICY,
        },
        date_status,
        max(0, len(production_dates) - 1),
    )


def _read_candidates(
    snapshot: Path, member: zipfile.ZipInfo | None, source_revision: str
) -> tuple[list[dict[str, str]], dict[str, int]]:
    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_fields: set[str] = set()
    stats = {
        "input_rows": 0,
        "rejected_non_object_rows": 0,
        "rejected_invalid_id": 0,
        "rejected_not_public_domain": 0,
        "rejected_wrong_rights_uri": 0,
        "rejected_without_image": 0,
        "rejected_without_image_thumbnail": 0,
        "rejected_invalid_image_url": 0,
        "creator_activity_dates_excluded": 0,
        "missing_or_invalid_first_production_date": 0,
        "ignored_additional_production_dates": 0,
    }
    try:
        with _snapshot_text(snapshot, member) as handle:
            for row_number, raw in enumerate(_iter_json_document(handle), start=1):
                stats["input_rows"] += 1
                if not isinstance(raw, Mapping):
                    stats["rejected_non_object_rows"] += 1
                    continue
                source: Mapping[str, object] = raw
                seen_fields.update(str(field) for field in source)
                source_id = _clean(source.get("id"))
                if not source_id or not _SAFE_SOURCE_ID.fullmatch(source_id):
                    stats["rejected_invalid_id"] += 1
                    continue
                if source_id in seen_ids:
                    raise CorpusBuildError(
                        f"SMK snapshot row {row_number}: duplicate id {source_id!r}"
                    )
                seen_ids.add(source_id)
                if source.get("public_domain") is not True:
                    stats["rejected_not_public_domain"] += 1
                    continue
                if _clean(source.get("rights")) != SMK_PUBLIC_DOMAIN_MARK_URI:
                    stats["rejected_wrong_rights_uri"] += 1
                    continue
                if source.get("has_image") is not True:
                    stats["rejected_without_image"] += 1
                    continue
                image_url = _clean(source.get("image_thumbnail"))
                if not image_url:
                    stats["rejected_without_image_thumbnail"] += 1
                    continue
                try:
                    _validate_smk_image_url(image_url)
                except ValueError:
                    stats["rejected_invalid_image_url"] += 1
                    continue
                canonical, date_status, ignored_dates = _canonical_row(
                    source, source_id, source_revision
                )
                if date_status == "creator-activity-excluded":
                    stats["creator_activity_dates_excluded"] += 1
                elif date_status == "missing-or-invalid":
                    stats["missing_or_invalid_first_production_date"] += 1
                stats["ignored_additional_production_dates"] += ignored_dates
                candidates.append(canonical)
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise CorpusBuildError(f"could not read SMK snapshot: {snapshot}") from exc

    missing = sorted(_REQUIRED_SNAPSHOT_FIELDS - seen_fields)
    if missing:
        raise CorpusBuildError(
            "SMK snapshot is missing required fields: " + ", ".join(missing)
        )
    return candidates, stats


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


def prepare_smk_visual_subset(
    snapshot_path: Path | str,
    output_csv: Path | str,
    *,
    source_revision: str,
    sample_size: int = 0,
    seed: str = "smk-public-domain-thumbnail-visual-v1",
    workers: int = 16,
    preflight: bool = True,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    """Normalize a pinned local SMK JSON/ZIP snapshot into visual-corpus rows.

    The nightly feed itself is moving, so ``source_revision`` must be the exact
    SHA-256 of the local snapshot. Only records carrying all three independent
    visual gates are considered: ``public_domain=true``, the exact Public Domain
    Mark URI, and ``has_image=true``. Pixels are never retained by this adapter.
    """

    if sample_size < 0 or workers < 1:
        raise CorpusBuildError("sample size must be non-negative and workers positive")
    revision = _clean(source_revision)
    if not _PINNED_REVISION.fullmatch(revision):
        raise CorpusBuildError("SMK source_revision must be a 64-character lowercase SHA-256")

    snapshot = Path(snapshot_path).resolve()
    if not snapshot.is_file():
        raise CorpusBuildError(f"SMK snapshot does not exist: {snapshot}")
    snapshot_sha256 = sha256_file(snapshot)
    if revision != snapshot_sha256:
        raise CorpusBuildError(
            "SMK source_revision must match the SHA-256 of the local snapshot"
        )
    member = _zip_member(snapshot)

    output = Path(output_csv).resolve()
    manifest_path = output.with_suffix(".manifest.json")
    incomplete_path = output.with_suffix(".incomplete.json")
    availability_path = output.with_suffix(".availability.csv")
    if snapshot in {output, manifest_path, incomplete_path, availability_path}:
        raise CorpusBuildError("SMK adapter outputs must not overwrite the source snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)

    if incomplete_path.is_file():
        try:
            incomplete = json.loads(incomplete_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusBuildError(
                f"invalid incomplete-build marker requires inspection: {incomplete_path}"
            ) from exc
        if incomplete.get("schema_version") != VISUAL_SUBSET_SCHEMA_VERSION or incomplete.get(
            "output"
        ) != output.name:
            raise CorpusBuildError(
                f"incomplete-build marker does not own this output: {incomplete_path}"
            )
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    elif output.exists() or manifest_path.exists():
        raise CorpusBuildError(f"output CSV or manifest already exists: {output}")

    marker_temporary = incomplete_path.with_suffix(".tmp")
    marker_temporary.write_text(
        json.dumps(
            {"schema_version": VISUAL_SUBSET_SCHEMA_VERSION, "output": output.name},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    marker_temporary.replace(incomplete_path)

    candidates, stats = _read_candidates(snapshot, member, revision)
    candidates.sort(key=lambda row: (_rank(seed, row["artwork_id"]), row["artwork_id"]))
    if not candidates:
        raise CorpusBuildError("no eligible SMK visual candidates remain after rights gating")

    full_scan = sample_size == 0
    target_size = sample_size or len(candidates)
    if len(candidates) < target_size:
        raise CorpusBuildError(
            f"only {len(candidates)} eligible SMK candidates exist for sample size {target_size}"
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
                        f"SMK availability cache has an incompatible schema: {availability_path}"
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
                max_workers=workers, thread_name_prefix="smk-image-head"
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
            f"prepared only {len(selected)} of {target_size} SMK images after "
            f"{examined} candidates"
        )
    if not selected:
        raise CorpusBuildError("no reachable SMK images remain after preflight")

    selected.sort(key=lambda row: row["artwork_id"])
    _write_csv(output, selected)

    source_snapshot: dict[str, object] = {
        "filename": snapshot.name,
        "sha256": snapshot_sha256,
        "bytes": snapshot.stat().st_size,
        "rows": stats["input_rows"],
        "container": "zip" if member is not None else "json",
    }
    if member is not None:
        source_snapshot.update(
            {
                "json_member": member.filename,
                "json_member_bytes": member.file_size,
                "json_member_crc32": f"{member.CRC:08x}",
            }
        )
    manifest: dict[str, object] = {
        "schema_version": VISUAL_SUBSET_SCHEMA_VERSION,
        "builder_version": __version__,
        "source": {
            "kind": "smk-open-local-json-snapshot",
            "url": SMK_NIGHTLY_SNAPSHOT_URL,
            "documentation_url": SMK_API_DOCUMENTATION_URL,
            "revision": revision,
            "metadata_license": SMK_METADATA_LICENSE_URI,
            "snapshot": source_snapshot,
        },
        "selection": {
            "algorithm": "rights-gated-sha256-seeded-sample-with-fallbacks",
            "seed": seed,
            "requested_rows": target_size,
            "prepared_rows": len(selected),
            "eligible_candidates": len(candidates),
            "examined_candidates": examined,
            "date_policy": "first-production-date-only-no-creator-activity-fallback/v1",
            **stats,
        },
        "rights_gate": {
            "institution": "smk",
            "requirement": (
                "public_domain=true, exact Public Domain Mark 1.0 rights URI, "
                "has_image=true, and trusted image_thumbnail host"
            ),
            "image_rights_uri": SMK_PUBLIC_DOMAIN_MARK_URI,
            "image_host": SMK_IMAGE_HOST,
            "rejected_not_public_domain": stats["rejected_not_public_domain"],
            "rejected_wrong_rights_uri": stats["rejected_wrong_rights_uri"],
            "rejected_without_image": stats["rejected_without_image"],
            "rejected_without_image_thumbnail": stats[
                "rejected_without_image_thumbnail"
            ],
            "rejected_invalid_image_url": stats["rejected_invalid_image_url"],
        },
        "placeholder_basenames": [],
        "images": {
            "storage": "stream-at-embed-time",
            "service": "SMK declared IIIF thumbnail",
            "availability_preflight": preflight,
            "input_policy": SMK_IMAGE_INPUT_POLICY,
            "allowed_host": SMK_IMAGE_HOST,
            "stored_bytes": 0,
        },
        "output": {
            "csv": output.name,
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
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
    "SMK_IMAGE_HOST",
    "SMK_IMAGE_INPUT_POLICY",
    "SMK_API_DOCUMENTATION_URL",
    "SMK_METADATA_LICENSE_URI",
    "SMK_NIGHTLY_SNAPSHOT_URL",
    "SMK_PUBLIC_DOMAIN_MARK_URI",
    "VISUAL_SUBSET_SCHEMA_VERSION",
    "prepare_smk_visual_subset",
]
