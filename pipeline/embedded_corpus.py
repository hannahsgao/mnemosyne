"""Derive a hash-identified canonical corpus from a completed embedding bundle."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping, Sequence
from urllib.parse import unquote, urlsplit

from . import __version__
from .build import BUILD_SCHEMA_VERSION, CANONICAL_FIELDS, CorpusBuildError, sha256_file
from .embeddings import EMBED_SCHEMA_VERSION
from .repack import (
    _declared_path,
    _manifest_count,
    _read_manifest,
    _require_declared,
    _verify_artifacts,
)


DERIVATION_SCHEMA_VERSION = "mnemosyne-embedded-corpus-derivation/v1"
TRANSACTION_SCHEMA_VERSION = "mnemosyne-embedded-corpus-publication/v1"
KNOWN_MET_PLACEHOLDERS = (
    "Images-Restricted.jpg",
    "image-number-only.jpg",
)
OUTPUT_FIELDS = (
    *CANONICAL_FIELDS,
    "image_path",
    "permission_status",
    "image_use_permitted",
    "input_kind",
    "input_source",
    "input_width",
    "input_height",
    "input_policy",
    "image_input_policy",
)
_REQUIRED_MODEL_FILES = (
    "metadata",
    "embeddings",
    "dateWeights",
    "binDenominators",
    "imageManifest",
    "embeddedImages",
)
_CORPUS_DEPLOYED_FILES = (
    "metadata",
    "dateWeights",
    "binDenominators",
    "imageManifest",
)


def _read_csv(path: Path, required_fields: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise CorpusBuildError(f"required artifact is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required_fields) - fields)
        if missing:
            raise CorpusBuildError(
                f"{path.name} is missing required fields: {', '.join(missing)}"
            )
        return list(reader)


def _rows_by_id(rows: Sequence[dict[str, str]], source_name: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        artwork_id = row.get("artwork_id", "").strip()
        if not artwork_id:
            raise CorpusBuildError(f"{source_name} row {row_number}: artwork_id is empty")
        if artwork_id in indexed:
            raise CorpusBuildError(
                f"{source_name} row {row_number}: duplicate artwork_id {artwork_id!r}"
            )
        indexed[artwork_id] = row
    return indexed


def _ordered_artwork_ids(
    rows: Sequence[Mapping[str, str]], source_name: str
) -> list[str]:
    artwork_ids: list[str] = []
    for index, row in enumerate(rows):
        try:
            offset = int(row.get("embedding_offset", ""))
        except (TypeError, ValueError) as exc:
            raise CorpusBuildError(
                f"{source_name} row {index + 2}: embedding_offset is not an integer"
            ) from exc
        if offset != index:
            raise CorpusBuildError(
                f"{source_name} embedding_offset values must be contiguous from zero"
            )
        artwork_ids.append(row.get("artwork_id", "").strip())
    return artwork_ids


def _basename(value: str) -> str:
    if not value.strip():
        return ""
    path = unquote(urlsplit(value.strip()).path).replace("\\", "/")
    return PurePosixPath(path).name


def _placeholder_name(
    corpus_row: Mapping[str, str],
    embedded_row: Mapping[str, str],
    placeholder_by_casefold: Mapping[str, str],
) -> str | None:
    for field, row in (
        ("input_source", embedded_row),
        ("image_url", embedded_row),
        ("image_url", corpus_row),
        ("image_path", embedded_row),
    ):
        basename = _basename(row.get(field, ""))
        known = placeholder_by_casefold.get(basename.casefold())
        if known:
            return known
    return None


def _placeholder_policy(
    visual_payload: Mapping[str, object],
) -> tuple[tuple[str, ...], str]:
    if "placeholder_basenames" not in visual_payload:
        # Backward compatibility for already-published Met preflight manifests.
        return KNOWN_MET_PLACEHOLDERS, "legacy-met-default"

    declared = visual_payload["placeholder_basenames"]
    if not isinstance(declared, list):
        raise CorpusBuildError("visual preflight placeholder_basenames must be a list")
    placeholders: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(declared):
        if not isinstance(value, str) or not value.strip():
            raise CorpusBuildError(
                f"visual preflight placeholder_basenames[{index}] must be a non-empty string"
            )
        name = value.strip()
        if _basename(name) != name:
            raise CorpusBuildError(
                f"visual preflight placeholder_basenames[{index}] must be a basename"
            )
        folded = name.casefold()
        if folded in seen:
            raise CorpusBuildError(
                "visual preflight placeholder_basenames contains a case-insensitive duplicate"
            )
        seen.add(folded)
        placeholders.append(name)
    return tuple(placeholders), "visual-manifest"


def _validated_sha256(value: str, artwork_id: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CorpusBuildError(
            f"{artwork_id}: embedded input_sha256 must be a 64-character hex digest"
        )
    return digest


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _source_entry(path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _sha256_digest(value: object, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CorpusBuildError(f"{label} must be a 64-character hex digest")
    return digest


def _validate_corpus_declarations(
    bundle: Path,
    corpus_manifest: Mapping[str, object],
    deployed: Mapping[str, Path],
) -> None:
    files = corpus_manifest.get("files")
    if not isinstance(files, dict):
        raise CorpusBuildError("corpus build manifest files must be an object")
    entries = corpus_manifest.get("artifacts")
    if not isinstance(entries, list):
        raise CorpusBuildError("corpus build manifest artifacts must be a list")
    by_path: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CorpusBuildError("corpus build artifact entries must be objects")
        raw_path = str(entry.get("path") or "")
        if not raw_path or raw_path in by_path:
            raise CorpusBuildError(
                f"corpus build artifact path is invalid or duplicated: {raw_path!r}"
            )
        _sha256_digest(entry.get("sha256"), f"corpus build artifact {raw_path} sha256")
        try:
            declared_bytes = int(entry["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusBuildError(
                f"corpus build artifact byte count is invalid: {raw_path}"
            ) from exc
        if declared_bytes < 0:
            raise CorpusBuildError(
                f"corpus build artifact byte count is invalid: {raw_path}"
            )
        by_path[raw_path] = entry

    for key, actual_path in deployed.items():
        relative = actual_path.relative_to(bundle).as_posix()
        if files.get(key) != relative:
            raise CorpusBuildError(
                f"corpus build manifest does not declare deployed {key}: {relative}"
            )
        entry = by_path.get(relative)
        if entry is None:
            raise CorpusBuildError(
                f"corpus build manifest does not checksum deployed {key}: {relative}"
            )
        if (
            int(entry["bytes"]) != actual_path.stat().st_size
            or _sha256_digest(
                entry.get("sha256"), f"corpus build artifact {relative} sha256"
            )
            != sha256_file(actual_path)
        ):
            raise CorpusBuildError(
                f"deployed artifact differs from corpus build manifest: {relative}"
            )


def _validate_matrix_declaration(
    path: Path, model_manifest: Mapping[str, object], expected_rows: int
) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - required by the pipeline runtime
        raise CorpusBuildError("embedding derivation requires NumPy") from exc
    try:
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CorpusBuildError(f"embedding matrix is not a valid NPY file: {path}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] < 1:
        raise CorpusBuildError(
            f"embedding matrix shape {matrix.shape} does not match {expected_rows} corpus rows"
        )
    declared = model_manifest.get("matrix")
    if not isinstance(declared, dict) or (
        declared.get("rows") != int(matrix.shape[0])
        or declared.get("dimensions") != int(matrix.shape[1])
        or declared.get("dtype") != str(matrix.dtype)
        or declared.get("l2_normalized") is not True
        or declared.get("row_order") != "corpus.csv embedding_offset"
    ):
        raise CorpusBuildError("model manifest does not match its embedding matrix")


def _validated_embedding_bundle(
    bundle: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, Path]]:
    if not bundle.is_dir():
        raise CorpusBuildError(f"embedding bundle is not a directory: {bundle}")
    model_manifest_path = bundle / "model-manifest.json"
    model_manifest = _read_manifest(model_manifest_path, EMBED_SCHEMA_VERSION)
    artifacts = _verify_artifacts(bundle, model_manifest)
    declared = {
        key: _require_declared(bundle, model_manifest, artifacts, key)
        for key in _REQUIRED_MODEL_FILES
    }

    files = model_manifest.get("files")
    if not isinstance(files, dict):
        raise CorpusBuildError("model manifest files must be an object")
    numpy_index = _declared_path(bundle, files.get("numpyIndex"), "numpyIndex")
    if numpy_index != declared["embeddings"]:
        raise CorpusBuildError("model manifest numpyIndex must match embeddings")
    faiss_index = files.get("faissIndex")
    if faiss_index:
        faiss_path = _declared_path(bundle, faiss_index, "faissIndex")
        if faiss_path.relative_to(bundle).as_posix() not in artifacts:
            raise CorpusBuildError("model faissIndex is not covered by artifact checksums")
    source_provenance = files.get("sourceProvenance", [])
    if not isinstance(source_provenance, list):
        raise CorpusBuildError("model sourceProvenance must be a list")
    for raw_path in source_provenance:
        path = _declared_path(bundle, raw_path, "a source provenance file")
        if path.relative_to(bundle).as_posix() not in artifacts:
            raise CorpusBuildError(
                "model source provenance is not covered by artifact checksums"
            )

    corpus_manifest_path = bundle / "corpus-build-manifest.json"
    corpus_manifest_relative = corpus_manifest_path.relative_to(bundle).as_posix()
    if corpus_manifest_relative not in artifacts:
        raise CorpusBuildError(
            "model manifest does not checksum corpus-build-manifest.json"
        )
    corpus_manifest = _read_manifest(corpus_manifest_path, BUILD_SCHEMA_VERSION)
    if _sha256_digest(
        model_manifest.get("corpus_manifest_sha256"),
        "model corpus_manifest_sha256",
    ) != sha256_file(corpus_manifest_path):
        raise CorpusBuildError("model manifest has the wrong corpus manifest checksum")
    if model_manifest.get("corpus") != corpus_manifest.get("corpus"):
        raise CorpusBuildError("model and corpus manifests have different corpus identities")
    expected_rows = _manifest_count(model_manifest, "model")
    if _manifest_count(corpus_manifest, "corpus build") != expected_rows:
        raise CorpusBuildError("model and corpus manifests have different row counts")

    model = model_manifest.get("model")
    model_id = str(model.get("id") or "").strip() if isinstance(model, dict) else ""
    model_revision = (
        str(model.get("revision") or "").strip() if isinstance(model, dict) else ""
    )
    if (
        not isinstance(model, dict)
        or not model_id
        or not model_revision
        or not isinstance(model.get("settings"), dict)
    ):
        raise CorpusBuildError("model manifest has an invalid model identity or settings")
    index = model_manifest.get("index")
    if not isinstance(index, dict) or (
        index.get("metric") != "inner-product-on-l2-normalized-vectors"
        or index.get("exact") is not True
    ):
        raise CorpusBuildError("embedding bundle does not declare the exact cosine contract")
    _validate_matrix_declaration(declared["embeddings"], model_manifest, expected_rows)
    _validate_corpus_declarations(
        bundle,
        corpus_manifest,
        {key: declared[key] for key in _CORPUS_DEPLOYED_FILES},
    )
    return model_manifest, corpus_manifest, declared


def _validate_visual_binding(
    visual_manifest_path: Path,
    visual_payload: Mapping[str, object],
    corpus_manifest: Mapping[str, object],
) -> None:
    output = visual_payload.get("output")
    if not isinstance(output, dict):
        raise CorpusBuildError("visual preflight output provenance must be an object")
    output_sha256 = _sha256_digest(
        output.get("sha256"), "visual preflight output.sha256"
    )
    source = corpus_manifest.get("source")
    if not isinstance(source, dict):
        raise CorpusBuildError("corpus build manifest is missing source provenance")
    input_sha256 = _sha256_digest(
        source.get("input_sha256"), "corpus build source.input_sha256"
    )
    payloads = source.get("payloads")
    if not isinstance(payloads, list):
        raise CorpusBuildError("corpus build source payloads must be a list")
    payload_sha256: set[str] = set()
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise CorpusBuildError("corpus build source payload entries must be objects")
        payload_sha256.add(
            _sha256_digest(
                payload.get("sha256"),
                f"corpus build source payloads[{index}].sha256",
            )
        )
    if (
        output_sha256 != input_sha256
        and sha256_file(visual_manifest_path) not in payload_sha256
    ):
        raise CorpusBuildError(
            "visual preflight manifest is not bound to this embedding bundle"
        )


def derive_embedded_corpus(
    bundle_dir: Path | str,
    visual_manifest: Path | str,
    output_csv: Path | str,
) -> dict[str, object]:
    """Join an embedding bundle and emit canonical, content-hash visual identities.

    Pixel files are deliberately not copied. The resulting CSV is intended as a
    new ``pipeline build`` input; its embedding offsets are blank because known
    placeholders have been removed and the old row offsets are no longer valid.
    """

    bundle = Path(bundle_dir).resolve()
    visual_manifest_path = Path(visual_manifest).resolve()
    destination = Path(output_csv).resolve()
    manifest_destination = destination.with_suffix(".manifest.json")
    transaction_path = destination.with_name(f".{destination.name}.derive-transaction.json")
    model_manifest_path = bundle / "model-manifest.json"

    source_paths = {
        model_manifest_path.resolve(),
        visual_manifest_path,
    }
    if (
        destination in source_paths
        or manifest_destination in source_paths
        or transaction_path in source_paths
    ):
        raise CorpusBuildError("outputs must not overwrite an embedding provenance source")
    recovery: dict[str, object] | None = None
    if transaction_path.exists():
        if not transaction_path.is_file():
            raise CorpusBuildError(f"publication transaction is not a file: {transaction_path}")
        try:
            recovery = json.loads(transaction_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusBuildError(
                f"publication transaction is invalid: {transaction_path}"
            ) from exc
        if (
            not isinstance(recovery, dict)
            or recovery.get("schema_version") != TRANSACTION_SCHEMA_VERSION
            or recovery.get("output_csv") != destination.name
            or recovery.get("output_manifest") != manifest_destination.name
        ):
            raise CorpusBuildError(
                f"publication transaction does not match this output: {transaction_path}"
            )
    elif destination.exists() or manifest_destination.exists():
        existing = destination if destination.exists() else manifest_destination
        raise CorpusBuildError(f"refusing to overwrite existing output: {existing}")
    if not visual_manifest_path.is_file():
        raise CorpusBuildError(
            f"required visual preflight manifest is missing: {visual_manifest_path}"
        )
    model_manifest, corpus_manifest, declared_paths = _validated_embedding_bundle(bundle)
    corpus_path = declared_paths["metadata"]
    embedded_path = declared_paths["embeddedImages"]
    declared_sources = {
        model_manifest_path.resolve(),
        corpus_path,
        embedded_path,
        visual_manifest_path,
    }
    if (
        destination in declared_sources
        or manifest_destination in declared_sources
        or transaction_path in declared_sources
    ):
        raise CorpusBuildError("outputs must not overwrite an embedding provenance source")
    try:
        visual_payload = json.loads(visual_manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CorpusBuildError(
            f"visual preflight manifest is not valid JSON: {visual_manifest_path}"
        ) from exc
    if not isinstance(visual_payload, dict):
        raise CorpusBuildError("visual preflight manifest must contain a JSON object")
    if not isinstance(visual_payload.get("rights_gate"), dict):
        raise CorpusBuildError("visual preflight manifest is missing rights_gate provenance")
    images = visual_payload.get("images")
    if not isinstance(images, dict) or images.get("availability_preflight") is not True:
        raise CorpusBuildError(
            "visual preflight manifest must declare "
            "images.availability_preflight=true"
        )
    _validate_visual_binding(visual_manifest_path, visual_payload, corpus_manifest)
    placeholder_basenames, placeholder_policy_source = _placeholder_policy(visual_payload)
    placeholder_by_casefold = {
        name.casefold(): name for name in placeholder_basenames
    }

    corpus_rows = _read_csv(corpus_path, ("artwork_id", "embedding_offset"))
    embedded_rows = _read_csv(
        embedded_path,
        ("artwork_id", "embedding_offset", "input_sha256", "image_url"),
    )
    corpus_artwork_ids = _ordered_artwork_ids(corpus_rows, corpus_path.name)
    embedded_artwork_ids = _ordered_artwork_ids(embedded_rows, embedded_path.name)
    if corpus_artwork_ids != embedded_artwork_ids:
        raise CorpusBuildError(
            "bundle artwork IDs do not match in embedding_offset order"
        )
    corpus_by_id = _rows_by_id(corpus_rows, corpus_path.name)
    embedded_by_id = _rows_by_id(embedded_rows, embedded_path.name)
    expected_rows = _manifest_count(model_manifest, "model")
    if len(corpus_rows) != expected_rows or len(embedded_rows) != expected_rows:
        raise CorpusBuildError(
            "embedding bundle row count does not match its completed model manifest"
        )

    selection = visual_payload.get("selection")
    if not isinstance(selection, dict):
        raise CorpusBuildError("visual preflight manifest is missing selection provenance")
    prepared_rows = selection.get("prepared_rows")
    requested_rows = selection.get("requested_rows")
    if (
        isinstance(prepared_rows, bool)
        or not isinstance(prepared_rows, int)
        or isinstance(requested_rows, bool)
        or not isinstance(requested_rows, int)
        or not 0 <= prepared_rows <= requested_rows
    ):
        raise CorpusBuildError(
            "visual preflight selection requires valid prepared_rows and requested_rows"
        )
    if prepared_rows < len(corpus_rows):
        raise CorpusBuildError(
            "embedded corpus cannot contain more rows than the visual preflight prepared: "
            f"{len(corpus_rows)} > {prepared_rows}"
        )

    output_rows: list[dict[str, object]] = []
    excluded = {name: 0 for name in placeholder_basenames}
    replaced_declared_hashes = 0
    input_hashes: set[str] = set()
    for corpus_row in corpus_rows:
        artwork_id = corpus_row["artwork_id"].strip()
        embedded_row = embedded_by_id[artwork_id]
        placeholder = _placeholder_name(
            corpus_row, embedded_row, placeholder_by_casefold
        )
        if placeholder:
            excluded[placeholder] += 1
            continue

        input_sha256 = _validated_sha256(
            embedded_row.get("input_sha256", ""), artwork_id
        )
        declared = corpus_row.get("image_sha256", "").strip().lower()
        if declared and declared != input_sha256:
            replaced_declared_hashes += 1
        input_hashes.add(input_sha256)
        permission_status = embedded_row.get("permission_status", "").strip()
        permitted = _truthy(corpus_row.get("public_domain", "")) or permission_status in {
            "public-domain",
            "explicitly-permitted",
        }
        output_rows.append(
            {
                **{field: corpus_row.get(field, "") for field in CANONICAL_FIELDS},
                "visual_cluster_id": f"sha256:{input_sha256}",
                "image_sha256": input_sha256,
                "image_width": embedded_row.get("input_width", "").strip()
                or corpus_row.get("image_width", ""),
                "image_height": embedded_row.get("input_height", "").strip()
                or corpus_row.get("image_height", ""),
                "embedding_offset": "",
                "image_path": "",
                "permission_status": permission_status,
                "image_use_permitted": "true" if permitted else "false",
                "input_kind": embedded_row.get("input_kind", ""),
                "input_source": embedded_row.get("input_source", ""),
                "input_width": embedded_row.get("input_width", ""),
                "input_height": embedded_row.get("input_height", ""),
                "input_policy": embedded_row.get("input_policy", ""),
                # Keep the adapter policy in the field consumed by the next
                # canonical build. This makes a derive -> build -> embed cycle
                # preserve the exact remote derivative contract.
                "image_input_policy": embedded_row.get("input_policy", ""),
            }
        )

    source_entries = {
        "corpus": _source_entry(corpus_path),
        "embedded_images": _source_entry(embedded_path),
        "visual_preflight_manifest": _source_entry(visual_manifest_path),
    }
    if model_manifest_path.is_file():
        source_entries["model_manifest"] = _source_entry(model_manifest_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.stem}.derive-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        staged_csv = staging / destination.name
        staged_manifest = staging / manifest_destination.name
        _write_csv(staged_csv, output_rows)
        total_excluded = sum(excluded.values())
        upstream_excluded = requested_rows - prepared_rows
        prior_downstream_excluded = prepared_rows - len(corpus_rows)
        operation: dict[str, object] = {
            "join_key": "artwork_id",
            "visual_identity": "sha256:<embedded input_sha256>",
            "known_placeholder_basenames": list(placeholder_basenames),
            "image_path_policy": "blank-no-durable-pixel-storage",
            "embedding_offset_policy": "blank-rebuild-required-after-filtering",
        }
        if placeholder_policy_source != "legacy-met-default":
            operation["placeholder_policy_source"] = placeholder_policy_source
        manifest: dict[str, object] = {
            "schema_version": DERIVATION_SCHEMA_VERSION,
            "builder_version": __version__,
            "operation": operation,
            "counts": {
                "corpus_rows": len(corpus_rows),
                "embedded_image_rows": len(embedded_rows),
                "joined_rows": len(corpus_rows),
                "output_rows": len(output_rows),
                "upstream_visual_requested_rows": requested_rows,
                "upstream_visual_prepared_rows": prepared_rows,
                "upstream_visual_excluded_rows": upstream_excluded,
                "prior_downstream_excluded_rows": prior_downstream_excluded,
                "excluded_placeholder_rows": total_excluded,
                "excluded_by_placeholder": excluded,
                "total_excluded_from_upstream_request": upstream_excluded
                + prior_downstream_excluded
                + total_excluded,
                "unique_input_sha256": len(input_hashes),
                "duplicate_visual_rows": len(output_rows) - len(input_hashes),
                "replaced_declared_image_hashes": replaced_declared_hashes,
            },
            "sources": source_entries,
            "visual_preflight": visual_payload,
            "output": {
                "filename": destination.name,
                "sha256": sha256_file(staged_csv),
                "bytes": staged_csv.stat().st_size,
                "rows": len(output_rows),
                "fields": list(OUTPUT_FIELDS),
            },
        }
        staged_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected = {
            "schema_version": TRANSACTION_SCHEMA_VERSION,
            "output_csv": destination.name,
            "output_csv_sha256": sha256_file(staged_csv),
            "output_manifest": manifest_destination.name,
            "output_manifest_sha256": sha256_file(staged_manifest),
        }
        if recovery is not None:
            if recovery != expected:
                raise CorpusBuildError(
                    "cannot recover publication because its inputs or outputs changed"
                )
        else:
            staged_transaction = staging / transaction_path.name
            staged_transaction.write_text(
                json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8"
            )
            try:
                os.link(staged_transaction, transaction_path)
            except FileExistsError as exc:
                raise CorpusBuildError(
                    f"publication transaction already exists: {transaction_path}"
                ) from exc

        # The CSV is the completion file: publish its manifest first, then the
        # CSV. A crash at either step leaves the transaction marker, allowing a
        # rerun to verify existing bytes and finish without overwriting them.
        for staged, published, expected_sha256 in (
            (
                staged_manifest,
                manifest_destination,
                str(expected["output_manifest_sha256"]),
            ),
            (staged_csv, destination, str(expected["output_csv_sha256"])),
        ):
            if published.exists():
                if not published.is_file() or sha256_file(published) != expected_sha256:
                    raise CorpusBuildError(
                        f"publication recovery found conflicting output: {published}"
                    )
            else:
                staged.replace(published)
        transaction_path.unlink()
    return manifest
