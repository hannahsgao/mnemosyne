"""Prepare a reproducible, rights-gated AIC image subset for visual retrieval.

The adapter consumes an already-downloaded Art Institute of Chicago API data
dump.  It never discovers records from the live API: callers must provide
either the extracted dump directory or the original ``.tar.bz2`` archive and
pin it by checksum.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from functools import lru_cache
import math
from pathlib import Path, PurePosixPath
import re
import ssl
import tarfile
import tempfile
import time
from typing import Callable, Iterator, Mapping
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from . import __version__
from .build import CANONICAL_FIELDS, CorpusBuildError, sha256_file


VISUAL_SUBSET_SCHEMA_VERSION = "mnemosyne-aic-visual-subset/v1"
AIC_CC0_URI = "https://creativecommons.org/publicdomain/zero/1.0/"
AIC_DATA_DUMP_URL = (
    "https://artic-api-data.s3.amazonaws.com/artic-api-data.tar.bz2"
)
AIC_DATA_DOCUMENTATION_URL = (
    "https://github.com/art-institute-of-chicago/api-data"
)
AIC_IIIF_BASE = "https://www.artic.edu/iiif/2"
AIC_IMAGE_INPUT_POLICY = "aic-iiif-full-width-843/v1"
PHYSICAL_OBJECT_GROUPING_POLICY = "aic-catalog-record-is-physical-object/v1"
DEFAULT_REQUEST_DELAY_SECONDS = 1.0

_INVENTORY_ALGORITHM = "sha256-path-size-content-sha256/v1"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_UNKNOWN_DATES = frozenset(
    {"", "unknown", "undated", "date unknown", "n.d.", "n.d"}
)
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_ARTWORK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _InventoryEntry:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class _Snapshot:
    kind: str
    archive: dict[str, object] | None
    config: Mapping[str, object]
    config_entry: _InventoryEntry
    artwork_entries: tuple[_InventoryEntry, ...]
    candidates: tuple[dict[str, str], ...]
    stats: dict[str, int]
    content_sha256: str
    content_bytes: int


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _compact(value: object) -> str:
    return " ".join(_clean(value).split())


def _source_id(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return str(value)


def _filename_source_id(path: str) -> str:
    stem = PurePosixPath(path).stem
    if not stem.isascii() or not stem.isdecimal() or str(int(stem)) != stem:
        raise CorpusBuildError(
            f"AIC artwork filename must be a canonical positive integer: {path}"
        )
    if int(stem) < 1:
        raise CorpusBuildError(
            f"AIC artwork filename must be a canonical positive integer: {path}"
        )
    return stem


def _list_text(value: object) -> str:
    if not isinstance(value, list):
        return ""
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _compact(item)
        folded = text.casefold()
        if text and folded not in seen:
            seen.add(folded)
            output.append(text)
    return "; ".join(output)


def _historical_year(value: object) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value.strip()):
        number = int(value.strip())
    else:
        return ""
    return str(number) if number != 0 else ""


def _date_fields(source: Mapping[str, object]) -> dict[str, str]:
    display = _clean(source.get("date_display"))
    qualifier_title = _clean(source.get("date_qualifier_title"))
    lowered_display = display.casefold()
    lowered_qualifier = qualifier_title.casefold()

    # AIC sometimes places an artist's lifespan in date_start/date_end for an
    # undated work.  Preserve the human label but do not turn those bounds into
    # evidence about when the artwork was made.
    if (
        lowered_display in _UNKNOWN_DATES
        or lowered_qualifier in {"artist's working dates", "artists working dates"}
    ):
        return {
            "date_display": display,
            "date_start": "",
            "date_end": "",
            "date_qualifier": "unknown",
            "date_parse_method": "aic_non_work_date_bounds_excluded",
        }

    start = _historical_year(source.get("date_start"))
    end = _historical_year(source.get("date_end"))
    qualifier_aliases = {
        "about": "circa",
        "approximately": "circa",
        "c.": "circa",
        "ca": "circa",
        "ca.": "circa",
        "circa": "circa",
        "before": "before",
        "after": "after",
    }
    qualifier = qualifier_aliases.get(lowered_qualifier, "")
    if not qualifier:
        if start and end:
            qualifier = "exact" if start == end else "range"
        else:
            qualifier = "unknown"

    if start and end:
        method = "aic_source_exact" if start == end else "aic_source_range"
    elif start or end:
        method = "aic_source_single_bound"
    elif display:
        method = "aic_display_only"
    else:
        method = "aic_unknown"
    return {
        "date_display": display,
        "date_start": start,
        "date_end": end,
        "date_qualifier": qualifier,
        "date_parse_method": method,
    }


def _validate_iiif_base(value: object) -> str:
    base = _clean(value).rstrip("/")
    try:
        parsed = urlsplit(base)
        port = parsed.port
    except ValueError as exc:
        raise CorpusBuildError("AIC config.iiif_url is invalid") from exc
    if parsed.scheme != "https":
        raise CorpusBuildError("AIC config.iiif_url must use https")
    if (parsed.hostname or "").casefold().rstrip(".") != "www.artic.edu":
        raise CorpusBuildError(
            "AIC config.iiif_url must use the production www.artic.edu host"
        )
    if parsed.username or parsed.password or port not in {None, 443}:
        raise CorpusBuildError(
            "AIC config.iiif_url must not contain credentials or a custom port"
        )
    if parsed.path != "/iiif/2" or parsed.query or parsed.fragment:
        raise CorpusBuildError(
            "AIC config.iiif_url must be the production /iiif/2 base"
        )
    return AIC_IIIF_BASE


def _validate_aic_image_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("AIC image URL is invalid") from exc
    if parsed.scheme != "https":
        raise ValueError("AIC image URL must use https")
    if (parsed.hostname or "").casefold().rstrip(".") != "www.artic.edu":
        raise ValueError("AIC image URL must use www.artic.edu")
    if parsed.username or parsed.password or port not in {None, 443}:
        raise ValueError(
            "AIC image URL must not contain credentials or a custom port"
        )
    if parsed.query or parsed.fragment:
        raise ValueError("AIC image URL must not contain a query or fragment")
    match = re.fullmatch(
        r"/iiif/2/([0-9a-fA-F-]+)/full/843,/0/default\.jpg", parsed.path
    )
    if match is None or _UUID.fullmatch(match.group(1)) is None:
        raise ValueError(
            "AIC image URL must use the official /full/843,/0/default.jpg policy"
        )


def _iiif_image_url(iiif_base: str, image_id: str) -> str:
    normalized_id = _clean(image_id).casefold()
    if _UUID.fullmatch(normalized_id) is None:
        raise ValueError("AIC image_id must be a canonical UUID")
    if _validate_iiif_base(iiif_base) != AIC_IIIF_BASE:
        raise ValueError("AIC IIIF base is not trusted")  # pragma: no cover
    url = f"{AIC_IIIF_BASE}/{normalized_id}/full/843,/0/default.jpg"
    _validate_aic_image_url(url)
    return url


class _ValidatedAicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_aic_image_url(newurl)
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
        _ValidatedAicRedirectHandler(),
    )


def _remote_image_available(
    url: str,
    retries: int = 2,
    *,
    before_request: Callable[[], None] | None = None,
) -> tuple[bool, str]:
    _validate_aic_image_url(url)
    request = Request(
        url,
        method="HEAD",
        headers={
            "Accept": "image/*",
            "User-Agent": "Mnemosyne AIC embedding preflight",
            "AIC-User-Agent": "Mnemosyne AIC embedding preflight",
        },
    )
    for attempt in range(retries + 1):
        try:
            if before_request is not None:
                before_request()
            with _verified_opener().open(request, timeout=20) as response:
                _validate_aic_image_url(response.geturl())
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


def _locate_json_root(source: Path) -> Path:
    candidates: list[Path] = []
    if source.name == "json" and (source / "artworks").is_dir():
        candidates.append(source)
    if (source / "json" / "artworks").is_dir():
        candidates.append(source / "json")
    for child in source.iterdir():
        if child.is_dir() and (child / "json" / "artworks").is_dir():
            candidates.append(child / "json")
    unique = sorted({candidate.resolve() for candidate in candidates})
    if len(unique) != 1:
        raise CorpusBuildError(
            "AIC extracted dump must contain exactly one json/artworks directory"
        )
    json_root = unique[0]
    if json_root.is_symlink() or (json_root / "artworks").is_symlink():
        raise CorpusBuildError("AIC dump json directories must not be symlinks")
    return json_root


def _directory_payloads(source: Path) -> Iterator[tuple[str, bytes]]:
    json_root = _locate_json_root(source)
    config = json_root / "config.json"
    if not config.is_file() or config.is_symlink():
        raise CorpusBuildError(f"AIC config.json is missing or unsafe: {config}")
    if config.stat().st_size > _MAX_CONFIG_BYTES:
        raise CorpusBuildError("AIC config.json exceeds the safety size limit")
    yield "json/config.json", config.read_bytes()

    artwork_paths = sorted(
        (path for path in (json_root / "artworks").iterdir() if path.suffix == ".json"),
        key=lambda path: path.name,
    )
    if not artwork_paths:
        raise CorpusBuildError("AIC dump contains no json/artworks/*.json files")
    for path in artwork_paths:
        if not path.is_file() or path.is_symlink():
            raise CorpusBuildError(f"AIC artwork entry is not a regular file: {path}")
        if path.stat().st_size > _MAX_ARTWORK_BYTES:
            raise CorpusBuildError(f"AIC artwork JSON exceeds the safety size limit: {path}")
        yield f"json/artworks/{path.name}", path.read_bytes()


def _canonical_tar_member(name: str) -> str | None:
    parts = tuple(part for part in PurePosixPath(name).parts if part not in {"", "."})
    if ".." in parts:
        raise CorpusBuildError("AIC archive contains an unsafe parent path")
    if len(parts) >= 2 and parts[-2:] == ("json", "config.json"):
        return "json/config.json"
    if (
        len(parts) >= 3
        and parts[-3:-1] == ("json", "artworks")
        and parts[-1].endswith(".json")
    ):
        return f"json/artworks/{parts[-1]}"
    return None


def _archive_payloads(source: Path) -> Iterator[tuple[str, bytes]]:
    seen: set[str] = set()
    try:
        with tarfile.open(source, mode="r|bz2") as archive:
            for member in archive:
                canonical = _canonical_tar_member(member.name)
                if canonical is None:
                    continue
                if canonical in seen:
                    raise CorpusBuildError(
                        f"AIC archive contains duplicate member {canonical}"
                    )
                seen.add(canonical)
                if not member.isfile():
                    raise CorpusBuildError(
                        f"AIC archive member is not a regular file: {member.name}"
                    )
                limit = (
                    _MAX_CONFIG_BYTES
                    if canonical == "json/config.json"
                    else _MAX_ARTWORK_BYTES
                )
                if member.size > limit:
                    raise CorpusBuildError(
                        f"AIC archive member exceeds the safety size limit: {member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:  # pragma: no cover - guarded by member.isfile
                    raise CorpusBuildError(
                        f"AIC archive member cannot be read: {member.name}"
                    )
                payload = extracted.read(limit + 1)
                if len(payload) != member.size:
                    raise CorpusBuildError(
                        f"AIC archive member has an invalid size: {member.name}"
                    )
                yield canonical, payload
    except (tarfile.TarError, OSError, EOFError) as exc:
        if isinstance(exc, CorpusBuildError):  # pragma: no cover - defensive
            raise
        raise CorpusBuildError(f"invalid AIC tar.bz2 dump: {source}") from exc


def _snapshot_payloads(source: Path) -> Iterator[tuple[str, bytes]]:
    if source.is_dir():
        yield from _directory_payloads(source)
        return
    if source.is_file() and (
        source.name.endswith(".tar.bz2") or source.name.endswith(".tbz2")
    ):
        yield from _archive_payloads(source)
        return
    raise CorpusBuildError(
        "AIC source must be an extracted dump directory or a .tar.bz2 archive"
    )


def _json_object(payload: bytes, path: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusBuildError(f"AIC snapshot contains invalid JSON: {path}") from exc
    if not isinstance(decoded, dict):
        raise CorpusBuildError(f"AIC snapshot JSON must contain an object: {path}")
    return decoded


def _inventory_digest(entries: tuple[_InventoryEntry, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{_INVENTORY_ALGORITHM}\n".encode("ascii"))
    for entry in sorted(entries, key=lambda item: item.path):
        digest.update(
            json.dumps(
                [entry.path, entry.bytes, entry.sha256],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _blank_copyright(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _canonical_candidate(
    source: Mapping[str, object], source_id: str, image_id: str
) -> dict[str, str]:
    artwork_id = f"AIC_{source_id}"
    artist = (
        _list_text(source.get("artist_titles"))
        or _compact(source.get("artist_title"))
        or _compact(source.get("artist_display"))
    )
    return {
        "artwork_id": artwork_id,
        "physical_object_id": artwork_id,
        "visual_cluster_id": "",
        "institution": "aic",
        "source_id": source_id,
        "source_record_url": f"https://www.artic.edu/artworks/{source_id}",
        "source_dataset_version": "",
        "title": _clean(source.get("title")),
        "artist": artist,
        "object_type": _clean(source.get("artwork_type_title")),
        "medium": _clean(source.get("medium_display")),
        "culture": "",
        "department": _clean(source.get("department_title")),
        "classification": _clean(source.get("classification_title")),
        "period": "",
        "dynasty": "",
        "geography": _clean(source.get("place_of_origin")),
        "tags": _list_text(source.get("term_titles")),
        "object_wikidata_url": "",
        **_date_fields(source),
        "metadata_license": AIC_CC0_URI,
        "image_rights_uri": AIC_CC0_URI,
        "credit_line": _clean(source.get("credit_line")),
        "public_domain": "True",
        "image_available": "True",
        "image_url": "",
        "image_sha256": "",
        "image_width": "",
        "image_height": "",
        "embedding_offset": "",
        "image_path": "",
        "image_use_permitted": "True",
        "image_input_policy": AIC_IMAGE_INPUT_POLICY,
        "_image_id": image_id,
    }


def _read_snapshot(source: Path) -> _Snapshot:
    config: Mapping[str, object] | None = None
    config_entry: _InventoryEntry | None = None
    artwork_entries: list[_InventoryEntry] = []
    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    stats = {
        "input_artwork_files": 0,
        "rejected_not_public_domain": 0,
        "rejected_nonblank_copyright_notice": 0,
        "rejected_missing_image_id": 0,
        "rejected_invalid_image_id": 0,
        "eligible_rights_image_candidates": 0,
        "date_bounds_excluded_as_non_work_dates": 0,
    }

    for path, payload in _snapshot_payloads(source):
        entry = _InventoryEntry(
            path=path,
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        if path == "json/config.json":
            if config is not None:
                raise CorpusBuildError("AIC snapshot contains duplicate config.json")
            config = _json_object(payload, path)
            config_entry = entry
            continue

        stats["input_artwork_files"] += 1
        artwork_entries.append(entry)
        filename_id = _filename_source_id(path)
        source_record = _json_object(payload, path)
        source_id = _source_id(source_record.get("id"))
        if source_id is None:
            raise CorpusBuildError(f"AIC artwork has an invalid integer id: {path}")
        if source_id != filename_id:
            raise CorpusBuildError(
                f"AIC artwork id {source_id} does not match filename {filename_id}.json"
            )
        if source_id in seen_ids:
            raise CorpusBuildError(f"AIC snapshot contains duplicate artwork id {source_id}")
        seen_ids.add(source_id)
        if source_record.get("api_model") != "artworks":
            raise CorpusBuildError(
                f"AIC artwork {source_id} does not declare api_model=artworks"
            )

        if source_record.get("is_public_domain") is not True:
            stats["rejected_not_public_domain"] += 1
            continue
        if not _blank_copyright(source_record.get("copyright_notice")):
            stats["rejected_nonblank_copyright_notice"] += 1
            continue
        raw_image_id = source_record.get("image_id")
        if raw_image_id is None or (isinstance(raw_image_id, str) and not raw_image_id.strip()):
            stats["rejected_missing_image_id"] += 1
            continue
        image_id = _clean(raw_image_id).casefold()
        if not isinstance(raw_image_id, str) or _UUID.fullmatch(image_id) is None:
            stats["rejected_invalid_image_id"] += 1
            continue

        candidate = _canonical_candidate(source_record, source_id, image_id)
        if candidate["date_parse_method"] == "aic_non_work_date_bounds_excluded":
            stats["date_bounds_excluded_as_non_work_dates"] += 1
        candidates.append(candidate)
        stats["eligible_rights_image_candidates"] += 1

    if config is None or config_entry is None:
        raise CorpusBuildError("AIC snapshot is missing json/config.json")
    if not artwork_entries:
        raise CorpusBuildError("AIC snapshot contains no artwork JSON records")
    iiif_base = _validate_iiif_base(config.get("iiif_url"))
    for candidate in candidates:
        candidate["image_url"] = _iiif_image_url(iiif_base, candidate.pop("_image_id"))

    all_entries = (config_entry, *artwork_entries)
    archive: dict[str, object] | None = None
    if source.is_file():
        archive = {
            "filename": source.name,
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        }
    return _Snapshot(
        kind=(
            "aic-api-data-local-tar-bz2"
            if source.is_file()
            else "aic-api-data-local-directory"
        ),
        archive=archive,
        config=config,
        config_entry=config_entry,
        artwork_entries=tuple(artwork_entries),
        candidates=tuple(candidates),
        stats=stats,
        content_sha256=_inventory_digest(all_entries),
        content_bytes=sum(entry.bytes for entry in all_entries),
    )


def compute_aic_snapshot_revision(source_dump: Path | str) -> str:
    """Return the checksum callers must pin as ``source_revision``.

    Archives are pinned by their original bytes.  Extracted directories are
    pinned by a deterministic inventory of the consumed ``config.json`` and
    artwork JSON bytes.
    """

    source = Path(source_dump).resolve()
    if source.is_file() and (
        source.name.endswith(".tar.bz2") or source.name.endswith(".tbz2")
    ):
        return sha256_file(source)
    return _read_snapshot(source).content_sha256


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


def _read_availability_cache(
    availability_path: Path,
    candidates: list[dict[str, str]],
) -> dict[str, tuple[bool, str]]:
    cache_fields = ("artwork_id", "image_url", "available", "reason")
    candidate_by_id = {row["artwork_id"]: row for row in candidates}
    cached: dict[str, tuple[bool, str]] = {}
    if not availability_path.is_file():
        return cached
    with availability_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != cache_fields:
            raise CorpusBuildError(
                f"AIC availability cache has an incompatible schema: {availability_path}"
            )
        for row in reader:
            artwork_id = _clean(row.get("artwork_id"))
            candidate = candidate_by_id.get(artwork_id)
            if candidate is None or _clean(row.get("image_url")) != candidate["image_url"]:
                continue
            available = _clean(row.get("available")).casefold() == "true"
            reason = _clean(row.get("reason"))
            if _cacheable_availability(available, reason):
                cached[artwork_id] = available, reason
    return cached


def _availability_manifest(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def prepare_aic_visual_subset(
    source_dump: Path | str,
    output_csv: Path | str,
    *,
    source_revision: str,
    sample_size: int = 0,
    seed: str = "aic-public-domain-preferred-visual-v1",
    workers: int = 1,
    preflight: bool = True,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    """Prepare AIC public-domain preferred-image rows from a pinned dump.

    ``source_revision`` must be a SHA-256 checksum.  For an archive it is the
    archive checksum; for an extracted directory use
    :func:`compute_aic_snapshot_revision`.  Availability checks are deliberately
    sequential and rate-limited in accordance with AIC's image-access guidance.
    """

    if sample_size < 0:
        raise CorpusBuildError("AIC sample size must be non-negative")
    if workers != 1:
        raise CorpusBuildError("AIC image preflight must use exactly one worker")
    if (
        not math.isfinite(request_delay_seconds)
        or request_delay_seconds < DEFAULT_REQUEST_DELAY_SECONDS
    ):
        raise CorpusBuildError("AIC request delay must be finite and at least 1 second")
    if not seed:
        raise CorpusBuildError("AIC deterministic sample seed must not be empty")
    revision = _clean(source_revision).casefold()
    if _SHA256.fullmatch(revision) is None:
        raise CorpusBuildError("AIC source_revision must be a pinned SHA-256 checksum")

    source_path = Path(source_dump).resolve()
    output_path = Path(output_csv).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    incomplete_path = output_path.with_suffix(".incomplete.json")
    availability_path = output_path.with_suffix(".availability.csv")
    if source_path in {
        output_path,
        manifest_path,
        incomplete_path,
        availability_path,
    }:
        raise CorpusBuildError("AIC adapter outputs must not overwrite the source dump")
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

    snapshot = _read_snapshot(source_path)
    pinned_digest = (
        str(snapshot.archive["sha256"])
        if snapshot.archive is not None
        else snapshot.content_sha256
    )
    if revision != pinned_digest:
        raise CorpusBuildError(
            "AIC source_revision does not match the local source snapshot: "
            f"expected {revision}, computed {pinned_digest}"
        )

    candidates = [dict(candidate) for candidate in snapshot.candidates]
    for candidate in candidates:
        candidate["source_dataset_version"] = revision
    candidates.sort(key=lambda row: (_rank(seed, row["artwork_id"]), row["artwork_id"]))
    if not candidates:
        raise CorpusBuildError("no eligible AIC visual candidates remain after rights gating")

    full_scan = sample_size == 0
    target_size = sample_size or len(candidates)
    if len(candidates) < target_size:
        raise CorpusBuildError(
            f"only {len(candidates)} eligible AIC candidates exist for sample size {target_size}"
        )

    selected: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    examined = 0
    live_requests = 0

    def before_live_request() -> None:
        nonlocal live_requests
        if live_requests:
            time.sleep(request_delay_seconds)
        live_requests += 1

    if not preflight:
        selected.extend(candidates[:target_size])
        examined = target_size
        if progress:
            progress(examined, len(selected), len(candidates))
    else:
        cache_fields = ("artwork_id", "image_url", "available", "reason")
        cached = _read_availability_cache(availability_path, candidates)
        cache_exists = availability_path.is_file() and availability_path.stat().st_size > 0
        with availability_path.open("a", encoding="utf-8", newline="") as cache_handle:
            cache_writer = csv.DictWriter(
                cache_handle, fieldnames=cache_fields, lineterminator="\n"
            )
            if not cache_exists:
                cache_writer.writeheader()
                cache_handle.flush()
            for candidate in candidates:
                cached_result = cached.get(candidate["artwork_id"])
                if cached_result is None:
                    available, reason = _remote_image_available(
                        candidate["image_url"], before_request=before_live_request
                    )
                    cached_result = available, reason
                    cached[candidate["artwork_id"]] = cached_result
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
                available, reason = cached_result
                examined += 1
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
            f"prepared only {len(selected)} of {target_size} AIC images after "
            f"{examined} candidates"
        )
    if not selected:
        raise CorpusBuildError("no reachable AIC images remain after preflight")

    selected.sort(key=lambda row: row["artwork_id"])
    _write_csv(output_path, selected)

    artwork_inventory = tuple(snapshot.artwork_entries)
    source: dict[str, object] = {
        "kind": snapshot.kind,
        "url": AIC_DATA_DUMP_URL,
        "documentation_url": AIC_DATA_DOCUMENTATION_URL,
        "revision": revision,
        "revision_kind": (
            "archive-sha256"
            if snapshot.archive is not None
            else "content-inventory-sha256"
        ),
        "metadata_license": AIC_CC0_URI,
        "raw_snapshot": {
            "input": source_path.name,
            "sha256": snapshot.content_sha256,
            "bytes": snapshot.content_bytes,
            "files": len(artwork_inventory) + 1,
            "inventory_algorithm": _INVENTORY_ALGORITHM,
        },
        "config": {
            "filename": snapshot.config_entry.path,
            "sha256": snapshot.config_entry.sha256,
            "bytes": snapshot.config_entry.bytes,
            "iiif_url": AIC_IIIF_BASE,
        },
        "artworks": {
            "directory": "json/artworks",
            "sha256": _inventory_digest(artwork_inventory),
            "bytes": sum(entry.bytes for entry in artwork_inventory),
            "files": len(artwork_inventory),
        },
    }
    if snapshot.archive is not None:
        source["archive"] = snapshot.archive

    availability = _availability_manifest(availability_path) if preflight else None
    manifest: dict[str, object] = {
        "schema_version": VISUAL_SUBSET_SCHEMA_VERSION,
        "builder_version": __version__,
        "source": source,
        "selection": {
            "algorithm": "rights-gated-sha256-seeded-sample-with-fallbacks",
            "seed": seed,
            "requested_rows": target_size,
            "prepared_rows": len(selected),
            "eligible_candidates": len(candidates),
            "examined_candidates": examined,
            "preferred_image_policy": "artwork.image_id-only",
            **snapshot.stats,
        },
        "rights_gate": {
            "institution": "aic",
            "requirement": (
                "is_public_domain is JSON true; copyright_notice is blank; "
                "image_id is a canonical UUID; config.iiif_url is the trusted "
                "production AIC IIIF base"
            ),
            "image_rights_uri": AIC_CC0_URI,
            "rejected_not_public_domain": snapshot.stats[
                "rejected_not_public_domain"
            ],
            "rejected_nonblank_copyright_notice": snapshot.stats[
                "rejected_nonblank_copyright_notice"
            ],
            "rejected_missing_image_id": snapshot.stats[
                "rejected_missing_image_id"
            ],
            "rejected_invalid_image_id": snapshot.stats[
                "rejected_invalid_image_id"
            ],
        },
        "physical_object_grouping": {
            "policy": PHYSICAL_OBJECT_GROUPING_POLICY,
            "rows_collapsed": 0,
            "eligible_rows": len(candidates),
            "eligible_distinct_physical_object_ids": len(candidates),
            "selected_rows": len(selected),
            "selected_distinct_physical_object_ids": len(selected),
        },
        "placeholder_basenames": [],
        "images": {
            "storage": "stream-at-embed-time",
            "service": "Art Institute of Chicago IIIF Image API 2.0",
            "availability_preflight": preflight,
            "preflight_concurrency": 1,
            "preflight_live_requests": live_requests,
            "request_delay_seconds": request_delay_seconds,
            "input_policy": AIC_IMAGE_INPUT_POLICY,
            "iiif_base": AIC_IIIF_BASE,
            "stored_bytes": 0,
            "availability_cache": availability,
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
    "AIC_CC0_URI",
    "AIC_DATA_DOCUMENTATION_URL",
    "AIC_DATA_DUMP_URL",
    "AIC_IMAGE_INPUT_POLICY",
    "AIC_IIIF_BASE",
    "DEFAULT_REQUEST_DELAY_SECONDS",
    "PHYSICAL_OBJECT_GROUPING_POLICY",
    "VISUAL_SUBSET_SCHEMA_VERSION",
    "compute_aic_snapshot_revision",
    "prepare_aic_visual_subset",
]
