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
from .build import CANONICAL_FIELDS, CorpusBuildError, sha256_file


DERIVATION_SCHEMA_VERSION = "mnemosyne-embedded-corpus-derivation/v1"
TRANSACTION_SCHEMA_VERSION = "mnemosyne-embedded-corpus-publication/v1"
KNOWN_MET_PLACEHOLDERS = (
    "Images-Restricted.jpg",
    "image-number-only.jpg",
)
_PLACEHOLDER_BY_CASEFOLD = {name.casefold(): name for name in KNOWN_MET_PLACEHOLDERS}
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


def _basename(value: str) -> str:
    if not value.strip():
        return ""
    path = unquote(urlsplit(value.strip()).path).replace("\\", "/")
    return PurePosixPath(path).name


def _placeholder_name(
    corpus_row: Mapping[str, str], embedded_row: Mapping[str, str]
) -> str | None:
    for field, row in (
        ("input_source", embedded_row),
        ("image_url", embedded_row),
        ("image_url", corpus_row),
        ("image_path", embedded_row),
    ):
        basename = _basename(row.get(field, ""))
        known = _PLACEHOLDER_BY_CASEFOLD.get(basename.casefold())
        if known:
            return known
    return None


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
    corpus_path = bundle / "corpus.csv"
    embedded_path = bundle / "embedded-images.manifest.csv"
    model_manifest_path = bundle / "model-manifest.json"

    source_paths = {
        corpus_path.resolve(),
        embedded_path.resolve(),
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

    corpus_rows = _read_csv(corpus_path, ("artwork_id",))
    embedded_rows = _read_csv(
        embedded_path,
        ("artwork_id", "input_sha256", "image_url"),
    )
    corpus_by_id = _rows_by_id(corpus_rows, corpus_path.name)
    embedded_by_id = _rows_by_id(embedded_rows, embedded_path.name)
    missing_embedded = sorted(set(corpus_by_id) - set(embedded_by_id))
    extra_embedded = sorted(set(embedded_by_id) - set(corpus_by_id))
    if missing_embedded or extra_embedded:
        details = []
        if missing_embedded:
            details.append(
                f"{len(missing_embedded)} corpus rows lack embedded provenance "
                f"(first: {missing_embedded[0]})"
            )
        if extra_embedded:
            details.append(
                f"{len(extra_embedded)} embedded rows lack corpus metadata "
                f"(first: {extra_embedded[0]})"
            )
        raise CorpusBuildError("bundle artwork IDs do not match: " + "; ".join(details))

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
    excluded = {name: 0 for name in KNOWN_MET_PLACEHOLDERS}
    replaced_declared_hashes = 0
    input_hashes: set[str] = set()
    for corpus_row in corpus_rows:
        artwork_id = corpus_row["artwork_id"].strip()
        embedded_row = embedded_by_id[artwork_id]
        placeholder = _placeholder_name(corpus_row, embedded_row)
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
        manifest: dict[str, object] = {
            "schema_version": DERIVATION_SCHEMA_VERSION,
            "builder_version": __version__,
            "operation": {
                "join_key": "artwork_id",
                "visual_identity": "sha256:<embedded input_sha256>",
                "known_placeholder_basenames": list(KNOWN_MET_PLACEHOLDERS),
                "image_path_policy": "blank-no-durable-pixel-storage",
                "embedding_offset_policy": "blank-rebuild-required-after-filtering",
            },
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
