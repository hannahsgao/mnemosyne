"""Hydrate the immutable artifact bundle, then replace this process with the API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping


RELEASE_NAME = "met-nga-openaccess-199474-siglip2-v1"
ARTIFACT_SOURCE = Path("/artifacts/releases") / RELEASE_NAME
LOCAL_ARTIFACTS = Path("/tmp/mnemosyne-artifacts")
MANIFEST_NAME = "model-manifest.json"
HYDRATION_MARKER = ".mnemosyne-hydrated"

EXPECTED_SCHEMA_VERSION = "mnemosyne-embedding-build/v1"
EXPECTED_CORPUS_ID = RELEASE_NAME
EXPECTED_CORPUS_LABEL = "The Met and National Gallery of Art open-access image catalog"
EXPECTED_CORPUS_COUNT = 199_474
EXPECTED_COUNTING_UNIT = "catalog-record"
EXPECTED_MODEL_ID = "google/siglip2-base-patch16-224"
EXPECTED_MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
EXPECTED_ROWS = 199_474
EXPECTED_DIMENSIONS = 768
EXPECTED_ARTIFACT_COUNT = 17
EXPECTED_MERGE_SOURCES = (
    (0, ("met",), 0, 142_482, 142_482),
    (1, ("nga",), 142_482, 199_474, 56_992),
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


def _marker_payload() -> str:
    return f"{EXPECTED_CORPUS_ID}\n{EXPECTED_MODEL_ID}@{EXPECTED_MODEL_REVISION}\n"


def _is_completed_hydration(destination: Path) -> bool:
    marker = destination / HYDRATION_MARKER
    try:
        if marker.read_text(encoding="utf-8") != _marker_payload():
            return False
        load_and_validate_manifest(destination)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return True


def hydrate_artifacts(source: Path, destination: Path) -> Path:
    """Copy the read-only mount into a local directory using one atomic rename."""

    if source.resolve() == destination.resolve():
        raise ValueError("artifact source and local destination must differ")
    if not source.is_dir():
        raise ValueError("artifact source mount is unavailable")

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
        for relative_path in _artifact_paths(manifest):
            source_path = source.joinpath(*relative_path.parts)
            destination_path = temporary.joinpath(*relative_path.parts)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        (temporary / HYDRATION_MARKER).write_text(_marker_payload(), encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    _LOGGER.info("artifact hydration completed")
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
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    local_artifacts = hydrate_artifacts(ARTIFACT_SOURCE, LOCAL_ARTIFACTS)
    argv = service_argv(local_artifacts)
    _LOGGER.info(
        "starting one offline search process on port %d with request admission limit %d",
        PORT,
        HTTP_ADMISSION_LIMIT,
    )
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
