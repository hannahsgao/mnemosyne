"""Prepare a reproducible, rights-gated Met image subset for visual retrieval."""

from __future__ import annotations

import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
import json
import os
from pathlib import Path
import re
import ssl
import tempfile
import time
from typing import Callable, Mapping
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from . import __version__
from .build import CANONICAL_FIELDS, CorpusBuildError, sha256_file
from .met import MET_CC0_URI


VISUAL_SUBSET_SCHEMA_VERSION = "mnemosyne-met-visual-subset/v1"
DEFAULT_MAX_IMAGE_BYTES = 24 * 1024 * 1024
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _validate_met_image_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError("Met image URL must use https")
    if (parsed.hostname or "").lower().rstrip(".") != "images.metmuseum.org":
        raise ValueError("Met image URL must use images.metmuseum.org")
    if parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise ValueError("Met image URL must not contain credentials or a custom port")


class _ValidatedMetRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_met_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _optimized_image_url(url: str) -> str:
    _validate_met_image_url(url)
    parsed = urlsplit(url)
    if parsed.hostname == "images.metmuseum.org" and "/original/" in parsed.path:
        parsed = parsed._replace(path=parsed.path.replace("/original/", "/web-large/", 1))
    parsed = parsed._replace(path=quote(parsed.path, safe="/%:@"))
    return urlunsplit(parsed)


@lru_cache(maxsize=1)
def _verified_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:  # pragma: no cover - normally supplied by the HTTP/model stack
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


@lru_cache(maxsize=1)
def _verified_opener():
    return build_opener(
        HTTPSHandler(context=_verified_ssl_context()),
        _ValidatedMetRedirectHandler(),
    )


def _remote_image_available(url: str, retries: int = 2) -> tuple[bool, str]:
    optimized = _optimized_image_url(url)
    request = Request(
        optimized,
        method="HEAD",
        headers={"Accept": "image/*", "User-Agent": "Mnemosyne embedding preflight"},
    )
    for attempt in range(retries + 1):
        try:
            with _verified_opener().open(request, timeout=20) as response:
                _validate_met_image_url(response.geturl())
                content_type = response.headers.get("Content-Type", "")
                if content_type and not content_type.lower().startswith("image/"):
                    return False, f"unexpected content type: {content_type}"
                return True, ""
        except HTTPError as exc:
            retryable = exc.code in {408, 425, 429} or 500 <= exc.code < 600
            if not retryable or attempt >= retries:
                return False, f"HTTP {exc.code}"
        except (OSError, TimeoutError) as exc:
            if attempt >= retries:
                return False, str(exc)
        time.sleep(0.25 * (2**attempt))
    return False, "unreachable retry state"


def _cacheable_availability(available: bool, reason: str) -> bool:
    """Persist successes and permanent failures, never transient network state."""

    if available or reason.startswith("unexpected content type:"):
        return True
    if reason.startswith("HTTP "):
        try:
            status = int(reason.removeprefix("HTTP "))
        except ValueError:
            return False
        return status not in {408, 425, 429} and not 500 <= status < 600
    return False


def _is_true(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def _rank(seed: str, artwork_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\x1f{artwork_id}".encode("utf-8")).digest()


def _public_domain_met_rows(corpus_csv: Path) -> dict[str, dict[str, str]]:
    if not corpus_csv.is_file():
        raise CorpusBuildError(f"Met corpus metadata is missing: {corpus_csv}")
    rows: dict[str, dict[str, str]] = {}
    with corpus_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_id = str(row.get("source_id", "")).strip()
            if source_id and _is_true(row.get("public_domain")):
                rows[source_id] = dict(row)
    if not rows:
        raise CorpusBuildError("Met corpus contains no public-domain rows")
    return rows


def _candidate_rows(
    met_rows: Mapping[str, dict[str, str]],
    artifact_csv: Path,
    seed: str,
) -> list[dict[str, str]]:
    if not artifact_csv.is_file():
        raise CorpusBuildError(f"ArtiFact clean CSV is missing: {artifact_csv}")
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    with artifact_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"object_ID", "image_url"}
        if not required.issubset(reader.fieldnames or []):
            raise CorpusBuildError("ArtiFact clean CSV requires object_ID and image_url")
        for source in reader:
            object_id = str(source.get("object_ID", "")).strip()
            prefix, separator, source_id = object_id.partition("_")
            image_url = str(source.get("image_url", "")).strip()
            if prefix.upper() != "MET" or not separator or not image_url or object_id in seen:
                continue
            canonical = met_rows.get(source_id)
            if canonical is None:
                continue
            seen.add(object_id)
            candidates.append({**canonical, "artwork_id": object_id, "image_url": image_url})
    candidates.sort(key=lambda row: (_rank(seed, row["artwork_id"]), row["artwork_id"]))
    if not candidates:
        raise CorpusBuildError("no public-domain Met image candidates overlap the two sources")
    return candidates


def _read_remote_image(url: str, max_bytes: int) -> bytes:
    url = _optimized_image_url(url)
    request = Request(
        url,
        headers={
            "Accept": "image/avif,image/webp,image/jpeg,image/png,image/*;q=0.8",
            "User-Agent": "Mnemosyne visual corpus builder",
        },
    )
    with _verified_opener().open(request, timeout=30) as response:
        _validate_met_image_url(response.geturl())
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError("source image exceeds the byte limit")
        content_type = response.headers.get("Content-Type", "")
        if content_type and not content_type.lower().startswith("image/"):
            raise ValueError(f"unexpected image content type: {content_type}")
        payload = response.read(max_bytes + 1)
    if not payload or len(payload) > max_bytes:
        raise ValueError("source image is empty or exceeds the byte limit")
    return payload


def _normalized_jpeg(
    payload: bytes,
    destination: Path,
    max_dimension: int,
) -> tuple[str, int, int, int]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - optional production dependency
        raise CorpusBuildError("Pillow is required to prepare Met visual images") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    image = None
    try:
        with Image.open(BytesIO(payload)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("decoded image has invalid dimensions")
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}-",
            suffix=".jpg",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            image.save(
                temporary_path,
                format="JPEG",
                quality=85,
                optimize=True,
                progressive=True,
            )
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        if image is not None:
            image.close()
    return sha256_file(destination), width, height, destination.stat().st_size


def _existing_jpeg(path: Path) -> tuple[str, int, int, int] | None:
    if not path.is_file():
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return sha256_file(path), width, height, path.stat().st_size
    except Exception:
        path.unlink(missing_ok=True)
        return None


def _materialize_candidate(
    candidate: dict[str, str],
    image_dir: Path,
    output_parent: Path,
    max_dimension: int,
    max_bytes: int,
) -> tuple[dict[str, str] | None, int, str | None]:
    artwork_id = candidate["artwork_id"]
    if not _SAFE_ID.fullmatch(artwork_id):
        return None, 0, "artwork id is not filename-safe"
    destination = image_dir / f"{artwork_id}.jpg"
    try:
        prepared = _existing_jpeg(destination)
        if prepared is None:
            payload = _read_remote_image(candidate["image_url"], max_bytes)
            prepared = _normalized_jpeg(payload, destination, max_dimension)
        digest, width, height, stored_bytes = prepared
        image_path = Path(os.path.relpath(destination, output_parent)).as_posix()
        row = {
            **candidate,
            "public_domain": "True",
            "image_available": "True",
            "image_path": image_path,
            "image_sha256": digest,
            "image_width": str(width),
            "image_height": str(height),
            "image_rights_uri": MET_CC0_URI,
            "image_use_permitted": "True",
            "embedding_offset": "",
        }
        return row, stored_bytes, None
    except Exception as exc:
        return None, 0, str(exc)


def prepare_met_visual_subset(
    met_corpus_dir: Path | str,
    artifact_csv: Path | str,
    output_csv: Path | str,
    image_dir: Path | str | None = None,
    *,
    sample_size: int,
    source_revision: str,
    seed: str = "met-public-domain-visual-v1",
    workers: int = 16,
    max_dimension: int = 512,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    preflight: bool = True,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    """Download and normalize a deterministic public-domain Met image sample.

    Candidate image URLs come from the pinned ArtiFact clean table, while image
    permission and catalogue metadata come from the official Met Open Access
    artifact. Only their intersection is eligible.
    """

    if sample_size < 0 or workers < 1 or max_dimension < 64 or max_image_bytes < 1024:
        raise CorpusBuildError(
            "sample must be non-negative and worker, image-size, and byte-limit values positive"
        )
    if not source_revision.strip():
        raise CorpusBuildError("the ArtiFact source revision must be pinned")

    met_root = Path(met_corpus_dir).resolve()
    artifact_path = Path(artifact_csv).resolve()
    output_path = Path(output_csv).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    incomplete_path = output_path.with_suffix(".incomplete.json")
    images_root = Path(image_dir).resolve() if image_dir is not None else None
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
        # The marker proves these are pipeline-owned partial publications.
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    elif output_path.exists() or manifest_path.exists():
        raise CorpusBuildError(f"output CSV or manifest already exists: {output_path}")
    marker_temporary = incomplete_path.with_suffix(".tmp")
    marker_temporary.write_text(
        json.dumps(
            {
                "schema_version": VISUAL_SUBSET_SCHEMA_VERSION,
                "output": output_path.name,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    marker_temporary.replace(incomplete_path)
    if images_root is not None:
        images_root.mkdir(parents=True, exist_ok=True)

    met_rows = _public_domain_met_rows(met_root / "corpus.csv")
    candidates = _candidate_rows(met_rows, artifact_path, seed)
    full_scan = sample_size == 0
    target_size = sample_size or len(candidates)
    if len(candidates) < target_size:
        raise CorpusBuildError(
            f"only {len(candidates)} eligible image candidates exist for sample size {target_size}"
        )

    selected: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    image_bytes = 0
    examined = 0
    if images_root is None:
        # The production path deliberately keeps pixels ephemeral.  The image
        # encoder streams these URLs in bounded batches and the durable bundle
        # retains only vectors, Met IDs, and compact card metadata.
        def append_candidate(candidate: dict[str, str]) -> None:
            selected.append(
                {
                    **candidate,
                    "public_domain": "True",
                    "image_available": "True",
                    "image_path": "",
                    "image_rights_uri": MET_CC0_URI,
                    "image_use_permitted": "True",
                    "embedding_offset": "",
                }
            )

        if not preflight:
            for candidate in candidates[:target_size]:
                append_candidate(candidate)
            examined = target_size
            if progress:
                progress(examined, len(selected), len(candidates))
        else:
            availability_path = output_path.with_suffix(".availability.csv")
            cache_fields = ("artwork_id", "image_url", "available", "reason")
            cached: dict[str, tuple[bool, str]] = {}
            candidate_by_id = {row["artwork_id"]: row for row in candidates}
            if availability_path.is_file():
                with availability_path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if tuple(reader.fieldnames or ()) != cache_fields:
                        legacy_path = availability_path.with_name(
                            f"{availability_path.stem}.legacy.csv"
                        )
                        if legacy_path.exists():
                            raise CorpusBuildError(
                                "availability cache has a legacy schema and its backup already "
                                f"exists: {legacy_path}"
                            )
                        availability_path.replace(legacy_path)
                        reader = iter(())
                    for row in reader:
                        artwork_id = str(row.get("artwork_id") or "").strip()
                        candidate = candidate_by_id.get(artwork_id)
                        cached_url = str(row.get("image_url") or "").strip()
                        if candidate is None or not cached_url:
                            continue
                        try:
                            current_url = _optimized_image_url(candidate["image_url"])
                        except ValueError:
                            continue
                        if cached_url != current_url:
                            continue
                        available = _is_true(row.get("available"))
                        reason = str(row.get("reason") or "")
                        if _cacheable_availability(available, reason):
                            cached[artwork_id] = (available, reason)
            cache_exists = availability_path.is_file() and availability_path.stat().st_size > 0
            with availability_path.open("a", encoding="utf-8", newline="") as cache_handle:
                cache_writer = csv.DictWriter(
                    cache_handle,
                    fieldnames=cache_fields,
                    lineterminator="\n",
                )
                if not cache_exists:
                    cache_writer.writeheader()
                    cache_handle.flush()
                block_size = max(64, workers * 4)
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="met-image-head"
                ) as executor:
                    for offset in range(0, len(candidates), block_size):
                        block = candidates[offset : offset + block_size]
                        missing = [row for row in block if row["artwork_id"] not in cached]
                        checked = executor.map(
                            lambda row: _remote_image_available(row["image_url"]), missing
                        )
                        for candidate, (available, reason) in zip(missing, checked, strict=True):
                            cached[candidate["artwork_id"]] = (available, reason)
                            if _cacheable_availability(available, reason):
                                cache_writer.writerow(
                                    {
                                        "artwork_id": candidate["artwork_id"],
                                        "image_url": _optimized_image_url(candidate["image_url"]),
                                        "available": available,
                                        "reason": reason,
                                    }
                                )
                        cache_handle.flush()
                        for candidate in block:
                            examined += 1
                            available, reason = cached[candidate["artwork_id"]]
                            if available and len(selected) < target_size:
                                append_candidate(candidate)
                            elif not available and len(failures) < 50:
                                failures.append(
                                    {"artwork_id": candidate["artwork_id"], "reason": reason}
                                )
                        if progress:
                            progress(examined, len(selected), len(candidates))
                        if not full_scan and len(selected) >= target_size:
                            break
    else:
        batch_size = max(64, workers * 4)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="met-image") as executor:
            for offset in range(0, len(candidates), batch_size):
                block = candidates[offset : offset + batch_size]
                results = executor.map(
                    lambda candidate: _materialize_candidate(
                        candidate,
                        images_root,
                        output_path.parent,
                        max_dimension,
                        max_image_bytes,
                    ),
                    block,
                )
                for candidate, (row, stored_bytes, failure) in zip(block, results, strict=True):
                    examined += 1
                    if row is not None and len(selected) < target_size:
                        selected.append(row)
                        image_bytes += stored_bytes
                    elif failure and len(failures) < 50:
                        failures.append({"artwork_id": candidate["artwork_id"], "reason": failure})
                if progress:
                    progress(examined, len(selected), len(candidates))
                if len(selected) >= target_size:
                    break
                if examined >= batch_size * 3 and not selected:
                    reason = failures[0]["reason"] if failures else "unknown downloader failure"
                    raise CorpusBuildError(
                        f"no images were prepared after {examined} attempts; "
                        f"first failure: {reason}"
                    )

    if not full_scan and len(selected) < target_size:
        raise CorpusBuildError(
            f"prepared only {len(selected)} of {target_size} images after {examined} candidates"
        )

    selected.sort(key=lambda row: row["artwork_id"])
    fields = (*CANONICAL_FIELDS, "image_path", "image_use_permitted")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{output_path.stem}-",
        suffix=".csv",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        writer = csv.DictWriter(
            temporary,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(selected)
    try:
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest: dict[str, object] = {
        "schema_version": VISUAL_SUBSET_SCHEMA_VERSION,
        "builder_version": __version__,
        "selection": {
            "algorithm": "sha256-seeded-uniform-sample-with-fallbacks",
            "seed": seed,
            "requested_rows": target_size,
            "prepared_rows": len(selected),
            "eligible_candidates": len(candidates),
            "examined_candidates": examined,
        },
        "rights_gate": {
            "requirement": "official Met Open Access public_domain=true",
            "image_rights_uri": MET_CC0_URI,
        },
        "images": {
            "storage": (
                "normalized-local-cache"
                if images_root is not None
                else "stream-at-embed-time"
            ),
            "availability_preflight": bool(images_root is None and preflight),
            "max_dimension": max_dimension,
            "jpeg_quality": 85,
            "stored_bytes": image_bytes,
            "directory": (
                os.path.relpath(images_root, output_path.parent)
                if images_root is not None
                else None
            ),
        },
        "sources": {
            "met_corpus": str(met_root),
            "met_corpus_sha256": sha256_file(met_root / "corpus.csv"),
            "artifact_clean_csv": str(artifact_path),
            "artifact_revision": source_revision,
            "artifact_clean_sha256": sha256_file(artifact_path),
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
