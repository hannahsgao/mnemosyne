"""Verification stamps for atomically hydrated artifact bundles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


VERIFIED_HYDRATION_MARKER = ".mnemosyne-verified-hydration.json"
VERIFIED_HYDRATION_SCHEMA = "mnemosyne.verified-hydration.v1"


def _manifest_digest(manifest_path: Path) -> str:
    digest = hashlib.sha256()
    with manifest_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_stats(
    root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, str | int]]:
    resolved_root = root.resolve()
    result: list[dict[str, str | int]] = []
    for raw_entry in manifest.get("artifacts", ()):
        entry = dict(raw_entry)
        relative_path = str(entry["path"])
        path = (root / relative_path).resolve(strict=True)
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("artifact manifest path escapes the bundle root") from error
        if not path.is_file():
            raise ValueError(f"manifest-declared artifact is not a file: {relative_path}")
        stat = path.stat()
        result.append(
            {
                "path": relative_path,
                "bytes": stat.st_size,
                "sha256": str(entry["sha256"]).lower(),
                "mtimeNs": stat.st_mtime_ns,
                "ctimeNs": stat.st_ctime_ns,
            }
        )
    return result


def verification_payload(
    root: Path, manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, object]:
    return {
        "schemaVersion": VERIFIED_HYDRATION_SCHEMA,
        "manifestSha256": _manifest_digest(manifest_path),
        "artifacts": _artifact_stats(root, manifest),
    }


def write_verified_hydration(
    root: Path, manifest_path: Path, manifest: Mapping[str, Any]
) -> None:
    marker = root / VERIFIED_HYDRATION_MARKER
    temporary = marker.with_suffix(f"{marker.suffix}.tmp")
    payload = verification_payload(root, manifest_path, manifest)
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)


def validate_verified_hydration(
    root: Path, manifest_path: Path, manifest: Mapping[str, Any]
) -> bool:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    marker = root / VERIFIED_HYDRATION_MARKER
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
        expected = verification_payload(root, manifest_path, manifest)
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    return actual == expected
