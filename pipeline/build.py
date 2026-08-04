"""Build canonical corpus, sparse date weights, denominators, and manifests."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

from . import __version__
from .dates import (
    DATE_RULES_VERSION,
    DateConfig,
    NormalizedDate,
    bin_label,
    make_bins,
    normalize_date,
    parse_bool,
    uniform_bin_weights,
)


BUILD_SCHEMA_VERSION = "mnemosyne-corpus-build/v1"

CANONICAL_FIELDS = (
    "artwork_id",
    "physical_object_id",
    "visual_cluster_id",
    "institution",
    "source_id",
    "source_record_url",
    "source_dataset_version",
    "title",
    "artist",
    "object_type",
    "medium",
    "culture",
    "department",
    "classification",
    "period",
    "dynasty",
    "geography",
    "tags",
    "object_wikidata_url",
    "date_display",
    "date_start",
    "date_end",
    "date_qualifier",
    "date_parse_method",
    "metadata_license",
    "image_rights_uri",
    "credit_line",
    "public_domain",
    "image_url",
    "image_sha256",
    "image_width",
    "image_height",
    "embedding_offset",
)

IMAGE_MANIFEST_FIELDS = (
    "artwork_id",
    "image_url",
    "image_path",
    "image_sha256",
    "image_rights_uri",
    "public_domain",
    "permission_status",
)


class CorpusBuildError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_date(source_date: str | None) -> str:
    if source_date:
        parsed = datetime.fromisoformat(source_date.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    source_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_epoch:
        return datetime.fromtimestamp(int(source_epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return "not-recorded"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise CorpusBuildError("input CSV has no header")
        fields = [field.strip() for field in reader.fieldnames]
        rows = [dict(row) for row in reader]
    return fields, rows


def reject_dirty_split(path: Path, fields: Sequence[str]) -> None:
    lowered_name = path.name.lower()
    dirty_fields = sorted(
        field for field in fields if field.lower() in {"error_type", "error_subtype"} or field.lower().endswith("_error")
    )
    if "dirty" in lowered_name or dirty_fields:
        details = f"; error columns: {', '.join(dirty_fields)}" if dirty_fields else ""
        raise CorpusBuildError(
            "refusing error-injected/dirty ArtiFact input; use the pinned ArtiFact_clean CSV"
            + details
        )


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"null", "none", "nan"} else text


def _first(row: Mapping[str, object], *names: str) -> str:
    for name in names:
        value = _clean(row.get(name))
        if value:
            return value
    return ""


def _list_text(value: object) -> str:
    text = _clean(value)
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, list):
        return "; ".join(str(item).strip() for item in parsed if str(item).strip())
    return text


def _medium(row: Mapping[str, object]) -> str:
    values = [_list_text(row.get("materials")), _list_text(row.get("techniques"))]
    return "; ".join(value for value in values if value)


def _institution_and_source(object_id: str, supplied: str) -> tuple[str, str]:
    prefix, separator, remainder = object_id.partition("_")
    institutions = {
        "MET": "met",
        "AIC": "aic",
        "RIJKS": "rijksmuseum",
    }
    institution = supplied or institutions.get(prefix.upper(), prefix.lower() if separator else "unknown")
    return institution, remainder if separator else object_id


def _record_url(institution: str, source_id: str) -> str:
    if institution == "met" and source_id:
        return f"https://www.metmuseum.org/art/collection/search/{source_id}"
    if institution == "aic" and source_id:
        return f"https://www.artic.edu/artworks/{source_id}"
    return ""


def _stable_cluster_id(row: Mapping[str, object], artwork_id: str) -> str:
    supplied = _first(row, "visual_cluster_id")
    if supplied:
        return supplied
    image_hash = _first(row, "image_sha256")
    if image_hash:
        return f"sha256:{image_hash.lower()}"
    return f"object:{artwork_id}"


def _canonical_record(
    row: Mapping[str, object],
    date: NormalizedDate,
    source_revision: str,
    default_metadata_license: str,
) -> dict[str, object]:
    object_id = _first(row, "artwork_id", "object_ID", "object_id", "id")
    if not object_id:
        raise CorpusBuildError("every input row must have object_ID, object_id, artwork_id, or id")
    institution, source_id = _institution_and_source(object_id, _first(row, "institution"))
    artwork_id = object_id
    public_domain_text = _first(row, "public_domain", "is_public_domain")
    public_domain = parse_bool(public_domain_text) if public_domain_text else None
    canonical = {
        "artwork_id": artwork_id,
        "physical_object_id": _first(row, "physical_object_id") or artwork_id,
        "visual_cluster_id": _stable_cluster_id(row, artwork_id),
        "institution": institution,
        "source_id": _first(row, "source_id") or source_id,
        "source_record_url": _first(row, "source_record_url", "object_url")
        or _record_url(institution, source_id),
        "source_dataset_version": _first(row, "source_dataset_version") or source_revision,
        "title": _first(row, "title"),
        "artist": _first(row, "artist", "artist_name"),
        "object_type": _first(row, "object_type", "object_name"),
        "medium": _first(row, "medium") or _medium(row),
        "culture": _first(row, "culture"),
        "department": _first(row, "department"),
        "classification": _first(row, "classification"),
        "period": _first(row, "period"),
        "dynasty": _first(row, "dynasty"),
        "geography": _first(row, "geography"),
        "tags": _first(row, "tags"),
        "object_wikidata_url": _first(row, "object_wikidata_url"),
        "date_display": date.display,
        "date_start": date.start if date.start is not None else "",
        "date_end": date.end if date.end is not None else "",
        "date_qualifier": date.qualifier,
        "date_parse_method": date.parse_method,
        "metadata_license": _first(row, "metadata_license") or default_metadata_license,
        "image_rights_uri": _first(row, "image_rights_uri", "rights_uri"),
        "credit_line": _first(row, "credit_line"),
        "public_domain": public_domain,
        "image_url": _first(row, "image_url"),
        "image_sha256": _first(row, "image_sha256").lower(),
        "image_width": _first(row, "image_width"),
        "image_height": _first(row, "image_height"),
        "embedding_offset": "",
    }
    return canonical


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet_if_available(path: Path, fields: Sequence[str], rows: list[Mapping[str, object]]) -> bool:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        return False
    table = pa.Table.from_pylist([{field: row.get(field) for field in fields} for row in rows])
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    return True


def _write_sparse_npz(
    path: Path,
    rows: list[dict[str, object]],
    dates: list[NormalizedDate],
    bins: list[tuple[int, int]],
    bin_size: int,
) -> list[dict[str, object]]:
    try:
        import numpy as np
        from scipy import sparse
    except ImportError as exc:
        raise CorpusBuildError(
            "NumPy and SciPy are required to write date-weights.npz; "
            "install pipeline/requirements.txt"
        ) from exc

    bin_index = {start: index for index, (start, _end) in enumerate(bins)}
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    denominator = [0.0] * len(bins)
    physical_ids: list[set[str]] = [set() for _ in bins]
    visual_ids: list[set[str]] = [set() for _ in bins]

    for row_index, (row, date) in enumerate(zip(rows, dates, strict=True)):
        for start, weight in uniform_bin_weights(date, bin_size).items():
            column = bin_index.get(start)
            if column is None:
                continue
            matrix_rows.append(row_index)
            matrix_columns.append(column)
            matrix_values.append(weight)
            denominator[column] += weight
            physical_ids[column].add(str(row["physical_object_id"]))
            visual_ids[column].add(str(row["visual_cluster_id"]))

    matrix = sparse.csr_matrix(
        (np.asarray(matrix_values, dtype=np.float64), (matrix_rows, matrix_columns)),
        shape=(len(rows), len(bins)),
    )
    sparse.save_npz(path, matrix, compressed=True)

    return [
        {
            "bin_index": index,
            "bin_key": f"{start}:{end}",
            "bin_start": start,
            "bin_end": end,
            "bin_label": bin_label(start, end),
            "eligible_weight": format(denominator[index], ".12g"),
            "physical_object_count": len(physical_ids[index]),
            "visual_cluster_count": len(visual_ids[index]),
        }
        for index, (start, end) in enumerate(bins)
    ]


def _coverage_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dimension in ("institution", "object_type", "medium", "public_domain", "date_qualifier"):
        values: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            value = str(row.get(dimension, "") or "unknown")
            values.setdefault(value, []).append(row)
        for value in sorted(values):
            group = values[value]
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "row_count": len(group),
                    "dated_count": sum(row["date_start"] != "" for row in group),
                    "public_domain_count": sum(row["public_domain"] is True for row in group),
                    "image_count": sum(bool(row["image_url"]) for row in group),
                }
            )
    return output


def _image_rows(raw_rows: list[dict[str, str]], canonical_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for raw, canonical in zip(raw_rows, canonical_rows, strict=True):
        rights_uri = str(canonical["image_rights_uri"])
        public_domain = canonical["public_domain"] is True
        explicit_permission = parse_bool(raw.get("image_use_permitted"))
        if public_domain:
            status = "public-domain"
        elif explicit_permission:
            status = "explicitly-permitted"
        else:
            status = "unreviewed"
        output.append(
            {
                "artwork_id": canonical["artwork_id"],
                "image_url": canonical["image_url"],
                "image_path": _first(raw, "image_path"),
                "image_sha256": canonical["image_sha256"],
                "image_rights_uri": rights_uri,
                "public_domain": public_domain,
                "permission_status": status,
            }
        )
    return output


def _artifact_entry(root: Path, path: Path, rows: int | None = None) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = rows
    return entry


def build_corpus(
    input_csv: Path | str,
    output_dir: Path | str,
    *,
    corpus_version: str,
    source_revision: str,
    source_url: str = "https://huggingface.co/datasets/deem-data/ArtiFact",
    retrieved_at: str | None = None,
    metadata_license: str = "",
    date_config: DateConfig | None = None,
    require_parquet: bool = False,
    source_kind: str = "artifact-clean-local-csv",
    source_payloads: Sequence[Path | str] | None = None,
    source_payload_row_counts: Mapping[str, int] | None = None,
    input_row_count: int | None = None,
    source_counts: Mapping[str, int] | None = None,
    source_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one immutable artifact directory from a local clean-split CSV."""

    source = Path(input_csv).resolve()
    destination = Path(output_dir).resolve()
    config = date_config or DateConfig()
    config.validate()
    if not corpus_version.strip():
        raise CorpusBuildError("corpus_version is required")
    if not source_revision.strip():
        raise CorpusBuildError("source_revision is required")
    if not source.is_file():
        raise CorpusBuildError(f"input CSV does not exist: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise CorpusBuildError(f"output directory must be absent or empty: {destination}")

    fields, raw_rows = _read_csv(source)
    reject_dirty_split(source, fields)
    if not raw_rows:
        raise CorpusBuildError("input CSV has no records")

    normalized: list[tuple[dict[str, str], NormalizedDate, dict[str, object]]] = []
    seen_ids: set[str] = set()
    for row_number, raw in enumerate(raw_rows, start=2):
        try:
            date = normalize_date(raw, config)
            canonical = _canonical_record(raw, date, source_revision, metadata_license)
        except (CorpusBuildError, ValueError) as exc:
            raise CorpusBuildError(f"row {row_number}: {exc}") from exc
        artwork_id = str(canonical["artwork_id"])
        if artwork_id in seen_ids:
            raise CorpusBuildError(f"row {row_number}: duplicate artwork_id {artwork_id!r}")
        seen_ids.add(artwork_id)
        normalized.append((raw, date, canonical))

    normalized.sort(key=lambda item: str(item[2]["artwork_id"]))
    sorted_raw = [item[0] for item in normalized]
    dates = [item[1] for item in normalized]
    canonical_rows = [item[2] for item in normalized]
    for index, row in enumerate(canonical_rows):
        row["embedding_offset"] = index

    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mnemosyne-build-", dir=destination))
    try:
        source_payload_dir = staging / "source-payloads"
        source_payload_dir.mkdir()
        payload_sources = [Path(path).resolve() for path in source_payloads] if source_payloads else [source]
        if not payload_sources:
            raise CorpusBuildError("at least one source payload is required")
        missing_payloads = [path for path in payload_sources if not path.is_file()]
        if missing_payloads:
            raise CorpusBuildError(f"source payload does not exist: {missing_payloads[0]}")
        payload_names = [path.name for path in payload_sources]
        if len(payload_names) != len(set(payload_names)):
            raise CorpusBuildError("source payload filenames must be unique")
        copied_payloads: list[tuple[Path, Path]] = []
        for payload in payload_sources:
            copied = source_payload_dir / payload.name
            shutil.copyfile(payload, copied)
            copied_payloads.append((payload, copied))

        corpus_csv = staging / "corpus.csv"
        _write_csv(corpus_csv, CANONICAL_FIELDS, canonical_rows)
        corpus_parquet = staging / "corpus.parquet"
        wrote_corpus_parquet = _write_parquet_if_available(corpus_parquet, CANONICAL_FIELDS, canonical_rows)
        if require_parquet and not wrote_corpus_parquet:
            raise CorpusBuildError("--require-parquet was set but PyArrow is not installed")

        image_rows = _image_rows(sorted_raw, canonical_rows)
        images_csv = staging / "images.manifest.csv"
        _write_csv(images_csv, IMAGE_MANIFEST_FIELDS, image_rows)
        images_parquet = staging / "images.manifest.parquet"
        wrote_images_parquet = _write_parquet_if_available(images_parquet, IMAGE_MANIFEST_FIELDS, image_rows)

        bins = make_bins(dates, config)
        weights_path = staging / "date-weights.npz"
        denominator_rows = _write_sparse_npz(
            weights_path, canonical_rows, dates, bins, config.bin_size
        )
        denominator_fields = (
            "bin_index",
            "bin_key",
            "bin_start",
            "bin_end",
            "bin_label",
            "eligible_weight",
            "physical_object_count",
            "visual_cluster_count",
        )
        denominators_csv = staging / "bin-denominators.csv"
        _write_csv(denominators_csv, denominator_fields, denominator_rows)
        denominators_parquet = staging / "bin-denominators.parquet"
        wrote_denominators_parquet = _write_parquet_if_available(
            denominators_parquet, denominator_fields, denominator_rows
        )

        coverage = _coverage_rows(canonical_rows)
        coverage_fields = (
            "dimension",
            "value",
            "row_count",
            "dated_count",
            "public_domain_count",
            "image_count",
        )
        coverage_csv = staging / "coverage.csv"
        _write_csv(coverage_csv, coverage_fields, coverage)
        coverage_parquet = staging / "coverage.parquet"
        wrote_coverage_parquet = _write_parquet_if_available(
            coverage_parquet, coverage_fields, coverage
        )

        artifacts = [
            _artifact_entry(staging, corpus_csv, len(canonical_rows)),
            _artifact_entry(staging, images_csv, len(image_rows)),
            _artifact_entry(staging, weights_path, len(canonical_rows)),
            _artifact_entry(staging, denominators_csv, len(denominator_rows)),
            _artifact_entry(staging, coverage_csv, len(coverage)),
        ]
        payload_counts = source_payload_row_counts or {}
        for original, copied in copied_payloads:
            artifacts.append(
                _artifact_entry(staging, copied, payload_counts.get(original.name))
            )
        for wrote, path, count in (
            (wrote_corpus_parquet, corpus_parquet, len(canonical_rows)),
            (wrote_images_parquet, images_parquet, len(image_rows)),
            (wrote_denominators_parquet, denominators_parquet, len(denominator_rows)),
            (wrote_coverage_parquet, coverage_parquet, len(coverage)),
        ):
            if wrote:
                artifacts.append(_artifact_entry(staging, path, count))
        artifacts.sort(key=lambda entry: str(entry["path"]))

        manifest: dict[str, object] = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "builder_version": __version__,
            "corpus_version": corpus_version,
            "corpus": {
                "id": corpus_version,
                "version": corpus_version,
                "count": len(canonical_rows),
                "countingUnit": "physical-object",
            },
            "source": {
                "kind": source_kind,
                "url": source_url,
                "revision": source_revision,
                "retrieved_at": _source_date(retrieved_at),
                "input_filename": payload_sources[0].name,
                "input_sha256": sha256_file(payload_sources[0]),
                "payloads": [
                    {
                        "filename": original.name,
                        "sha256": sha256_file(original),
                        "bytes": original.stat().st_size,
                    }
                    for original in payload_sources
                ],
                **dict(source_metadata or {}),
            },
            "date_rules": {"version": DATE_RULES_VERSION, **asdict(config)},
            "counting_unit": "physical-object",
            "counts": {
                "input_rows": input_row_count if input_row_count is not None else len(raw_rows),
                "canonical_rows": len(canonical_rows),
                "dated_rows": sum(date.dated for date in dates),
                "unknown_date_rows": sum(not date.dated for date in dates),
                "bins": len(bins),
                "unreviewed_images": sum(
                    row["permission_status"] == "unreviewed" for row in image_rows
                ),
                **dict(source_counts or {}),
            },
            "canonical_fields": list(CANONICAL_FIELDS),
            "files": {
                "metadata": "corpus.csv",
                "embeddings": None,
                "dateWeights": "date-weights.npz",
                "binDenominators": "bin-denominators.csv",
                "imageManifest": "images.manifest.csv",
            },
            "bins": [
                {
                    "index": int(row["bin_index"]),
                    "key": row["bin_key"],
                    "start": int(row["bin_start"]),
                    "end": int(row["bin_end"]),
                    "label": row["bin_label"],
                }
                for row in denominator_rows
            ],
            "artifacts": artifacts,
        }
        manifest_path = staging / "build-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for child in sorted(staging.iterdir(), key=lambda path: path.name):
            child.replace(destination / child.name)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return manifest
