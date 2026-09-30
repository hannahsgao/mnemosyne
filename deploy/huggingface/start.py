"""Hydrate the immutable artifact bundle, then replace this process with the API."""

from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Any, Mapping


RELEASE_NAME = "met-nga-cma-openaccess-240278-siglip2-v1"
ARTIFACT_SOURCE = Path("/artifacts/releases") / RELEASE_NAME
LOCAL_ARTIFACTS = Path("/tmp/mnemosyne-artifacts")
MANIFEST_NAME = "model-manifest.json"
HYDRATION_MARKER = ".mnemosyne-verified-hydration.json"
HYDRATION_SCHEMA = "mnemosyne.verified-hydration.v1"

EXPECTED_SCHEMA_VERSION = "mnemosyne-embedding-build/v1"
EXPECTED_CORPUS_ID = RELEASE_NAME
EXPECTED_CORPUS_LABEL = (
    "The Met, National Gallery of Art, and Cleveland Museum of Art "
    "open-access image catalog"
)
EXPECTED_CORPUS_COUNT = 240_278
EXPECTED_COUNTING_UNIT = "catalog-record"
EXPECTED_MODEL_ID = "google/siglip2-base-patch16-224"
EXPECTED_MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
EXPECTED_ROWS = 240_278
EXPECTED_DIMENSIONS = 768
EXPECTED_ARTIFACT_COUNT = 24
EXPECTED_MERGE_SOURCES = (
    (0, ("met", "nga"), 0, 199_474, 199_474),
    (1, ("cma",), 199_474, 240_278, 40_804),
)

PORT = 7860
HTTP_ADMISSION_LIMIT = 8

_LOGGER = logging.getLogger("mnemosyne_space_start")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"artifact manifest {field} must be an object")
    return value


def _artifact_paths(manifest: Mapping[str, Any]) -> tuple[PurePosixPath, ...]:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or len(entries) != EXPECTED_ARTIFACT_COUNT:
        raise ValueError(
            f"artifact manifest must declare exactly {EXPECTED_ARTIFACT_COUNT} payload files"
        )

    paths: list[PurePosixPath] = []
    for entry in entries:
        item = _mapping(entry, "artifacts[]")
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("artifact manifest paths must be non-empty strings")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
            raise ValueError("artifact manifest paths must stay within the bundle root")
        if not isinstance(item.get("bytes"), int) or int(item["bytes"]) < 0:
            raise ValueError(f"artifact manifest byte count is invalid: {raw_path}")
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise ValueError(f"artifact manifest checksum is invalid: {raw_path}")
        if path in paths:
            raise ValueError(f"artifact manifest path is duplicated: {raw_path}")
        paths.append(path)
    return tuple(paths)


def load_and_validate_manifest(root: Path) -> Mapping[str, Any]:
    """Read the pinned bundle manifest and reject a mismatched deployment."""

    manifest_path = root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"artifact mount is missing {MANIFEST_NAME}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact manifest could not be read") from error

    manifest = _mapping(manifest, "root")
    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("artifact manifest schema does not match this deployment")

    corpus = _mapping(manifest.get("corpus"), "corpus")
    if corpus.get("id") != EXPECTED_CORPUS_ID:
        raise ValueError("artifact corpus ID does not match this deployment")
    if corpus.get("version") != EXPECTED_CORPUS_ID:
        raise ValueError("artifact corpus version does not match this deployment")
    if corpus.get("count") != EXPECTED_CORPUS_COUNT:
        raise ValueError("artifact corpus count does not match this deployment")
    if corpus.get("label") != EXPECTED_CORPUS_LABEL:
        raise ValueError("artifact corpus label does not match this deployment")
    if corpus.get("countingUnit") != EXPECTED_COUNTING_UNIT:
        raise ValueError("artifact corpus counting unit does not match this deployment")

    merge = _mapping(manifest.get("merge"), "merge")
    raw_sources = merge.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("artifact merge sources do not match this deployment")
    sources: list[tuple[int, tuple[str, ...], int, int, int]] = []
    for raw_source in raw_sources:
        source = _mapping(raw_source, "merge.sources[]")
        institutions = source.get("institutions")
        if not isinstance(institutions, list) or any(
            not isinstance(institution, str) for institution in institutions
        ):
            raise ValueError("artifact merge sources do not match this deployment")
        sources.append(
            (
                source.get("bundle_index"),
                tuple(institutions),
                source.get("row_start"),
                source.get("row_end_exclusive"),
                source.get("row_count"),
            )
        )
    if tuple(sources) != EXPECTED_MERGE_SOURCES:
        raise ValueError("artifact merge sources do not match this deployment")

    model = _mapping(manifest.get("model"), "model")
    if model.get("id") != EXPECTED_MODEL_ID:
        raise ValueError("artifact model ID does not match this deployment")
    if model.get("revision") != EXPECTED_MODEL_REVISION:
        raise ValueError("artifact model revision does not match this deployment")

    matrix = _mapping(manifest.get("matrix"), "matrix")
    if (
        matrix.get("rows") != EXPECTED_ROWS
        or matrix.get("dimensions") != EXPECTED_DIMENSIONS
        or matrix.get("dtype") != "float32"
        or matrix.get("l2_normalized") is not True
    ):
        raise ValueError("artifact matrix contract does not match this deployment")

    bins = manifest.get("bins")
    if not isinstance(bins, list) or len(bins) != 1_703:
        raise ValueError("artifact timeline bins do not match this deployment")

    _artifact_paths(manifest)
    return manifest


def _validate_source_files(source: Path, manifest: Mapping[str, Any]) -> None:
    resolved_source = source.resolve()
    entries = {
        PurePosixPath(str(item["path"])): int(item["bytes"])
        for item in manifest["artifacts"]
    }
    for relative_path in _artifact_paths(manifest):
        path = source.joinpath(*relative_path.parts)
        try:
            path.resolve(strict=True).relative_to(resolved_source)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                f"artifact file resolves outside the bundle root: {relative_path}"
            ) from error
        if not path.is_file():
            raise ValueError(f"artifact mount is missing declared file: {relative_path}")
        if path.stat().st_size != entries[relative_path]:
            raise ValueError(f"artifact file has the wrong byte count: {relative_path}")


def _is_completed_hydration(destination: Path) -> bool:
    try:
        manifest = load_and_validate_manifest(destination)
        actual = json.loads(
            (destination / HYDRATION_MARKER).read_text(encoding="utf-8")
        )
        expected = _verification_payload(destination, manifest)
        return actual == expected
    except (OSError, ValueError):
        return False


def _verification_payload(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, object]:
    manifest_digest = hashlib.sha256((root / MANIFEST_NAME).read_bytes()).hexdigest()
    artifacts: list[dict[str, str | int]] = []
    resolved_root = root.resolve()
    for item in manifest["artifacts"]:
        relative_path = PurePosixPath(str(item["path"]))
        path = root.joinpath(*relative_path.parts).resolve(strict=True)
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("artifact manifest path escapes the bundle root") from error
        if not path.is_file():
            raise ValueError(f"manifest-declared artifact is not a file: {relative_path}")
        stat = path.stat()
        artifacts.append(
            {
                "path": str(relative_path),
                "bytes": stat.st_size,
                "sha256": str(item["sha256"]).lower(),
                "mtimeNs": stat.st_mtime_ns,
                "ctimeNs": stat.st_ctime_ns,
            }
        )
    return {
        "schemaVersion": HYDRATION_SCHEMA,
        "manifestSha256": manifest_digest,
        "artifacts": artifacts,
    }


def _write_verified_hydration(root: Path, manifest: Mapping[str, Any]) -> None:
    marker = root / HYDRATION_MARKER
    marker.write_text(
        json.dumps(
            _verification_payload(root, manifest),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_verified_artifact(
    source: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    digest = hashlib.sha256()
    copied_bytes = 0
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            destination_handle.write(chunk)
            digest.update(chunk)
            copied_bytes += len(chunk)
    if copied_bytes != expected_bytes:
        raise ValueError(f"artifact changed size while hydrating: {source.name}")
    if digest.hexdigest() != expected_sha256.lower():
        raise ValueError(f"artifact checksum does not match manifest: {source.name}")
    shutil.copystat(source, destination)


def hydrate_artifacts(source: Path, destination: Path) -> Path:
    """Copy the read-only mount into a local directory using one atomic rename."""

    if source.resolve() == destination.resolve():
        raise ValueError("artifact source and local destination must differ")
    if not source.is_dir():
        raise ValueError("artifact source mount is unavailable")

    started = time.perf_counter()
    manifest = load_and_validate_manifest(source)
    _validate_source_files(source, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if _is_completed_hydration(destination):
            _LOGGER.info("reusing completed local artifact hydration")
            return destination
        raise ValueError("local artifact destination already exists but is incomplete")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.hydrate-",
            dir=destination.parent,
        )
    )
    try:
        shutil.copy2(source / MANIFEST_NAME, temporary / MANIFEST_NAME)
        entries = {
            PurePosixPath(str(item["path"])): item
            for item in manifest["artifacts"]
        }
        for relative_path in _artifact_paths(manifest):
            source_path = source.joinpath(*relative_path.parts)
            destination_path = temporary.joinpath(*relative_path.parts)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            entry = entries[relative_path]
            _copy_verified_artifact(
                source_path,
                destination_path,
                expected_bytes=int(entry["bytes"]),
                expected_sha256=str(entry["sha256"]),
            )
        _write_verified_hydration(temporary, manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    _LOGGER.info(
        "artifact hydration copied and verified %d files in %.3fs",
        len(entries),
        time.perf_counter() - started,
    )
    return destination


def service_argv(artifacts: Path) -> list[str]:
    """Return the single-process private-Space runtime profile."""

    return [
        "mnemosyne-search",
        "--artifacts",
        str(artifacts),
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--http-auth-mode",
        "disabled",
        "--max-concurrent-searches",
        str(HTTP_ADMISSION_LIMIT),
        "--retry-after-seconds",
        "1",
        "--request-io-timeout-seconds",
        "15",
        "--siglip2",
        "--device",
        "cpu",
        "--no-faiss",
        "--trust-hydration-verification",
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    startup_started = time.perf_counter()
    local_artifacts = hydrate_artifacts(ARTIFACT_SOURCE, LOCAL_ARTIFACTS)
    argv = service_argv(local_artifacts)
    _LOGGER.info(
        "starting one offline search process on port %d with request admission limit %d",
        PORT,
        HTTP_ADMISSION_LIMIT,
    )
    _LOGGER.info("startup wrapper completed in %.3fs", time.perf_counter() - startup_started)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
