"""Merge compatible completed embedding bundles without re-embedding images."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence
from urllib.parse import unquote, urlsplit

from . import __version__
from .build import (
    BUILD_SCHEMA_VERSION,
    CANONICAL_FIELDS,
    IMAGE_INPUT_POLICY_FIELD,
    IMAGE_MANIFEST_FIELDS,
    CorpusBuildError,
    _coverage_rows,
    _write_csv,
    _write_sparse_npz,
    sha256_file,
)
from .dates import (
    DATE_RULES_VERSION,
    DateConfig,
    NormalizedDate,
    bin_label,
    make_bins,
    parse_bool,
    parse_year,
    uniform_bin_weights,
)
from .embeddings import EMBED_SCHEMA_VERSION, SIGLIP_IMAGE_INPUT_POLICY
from .repack import (
    _declared_path,
    _ordered_rows,
    _read_manifest,
    _reject_overlapping_output,
    _require_declared,
    _validate_matrix,
    _verify_artifacts,
)


MERGE_SCHEMA_VERSION = "mnemosyne-embedding-merge/v1"

_EMBEDDED_REQUIRED_FIELDS = (
    "embedding_offset",
    "artwork_id",
    "image_path",
    "image_url",
    "declared_image_sha256",
    "input_sha256",
    "input_kind",
    "input_source",
    "input_width",
    "input_height",
    "input_policy",
    "permission_status",
)
_DENOMINATOR_FIELDS = (
    "bin_index",
    "bin_key",
    "bin_start",
    "bin_end",
    "bin_label",
    "eligible_weight",
    "physical_object_count",
    "visual_cluster_count",
)
_COVERAGE_FIELDS = (
    "dimension",
    "value",
    "row_count",
    "dated_count",
    "public_domain_count",
    "image_count",
)
_OPERATIONAL_MODEL_SETTINGS = frozenset(
    {
        "allowed_image_hosts",
        "checkpointed",
        "device",
        "download_workers",
        "fetch_retries",
        "image_input_policy",
        "image_input_policies",
        "merged_from_completed_bundles",
        "max_image_bytes",
        "max_image_pixels",
        "request_timeout_seconds",
    }
)


@dataclass(frozen=True)
class _SourceBundle:
    root: Path
    model_manifest_path: Path
    corpus_manifest_path: Path
    model_manifest: dict[str, object]
    corpus_manifest: dict[str, object]
    corpus_path: Path
    embeddings_path: Path
    date_weights_path: Path
    denominators_path: Path
    images_path: Path
    embedded_path: Path
    corpus_fields: tuple[str, ...]
    image_fields: tuple[str, ...]
    embedded_fields: tuple[str, ...]
    corpus_rows: tuple[dict[str, str], ...]
    image_rows: tuple[dict[str, str], ...]
    embedded_rows: tuple[dict[str, str], ...]
    dates: tuple[NormalizedDate, ...]
    date_config: DateConfig
    dimensions: int
    processor_contract: dict[str, object]
    provenance_paths: tuple[Path, ...]


def _csv_fields(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        fields = csv.DictReader(handle).fieldnames
    if not fields:
        raise CorpusBuildError(f"{path.name} has no CSV header")
    if len(fields) != len(set(fields)):
        raise CorpusBuildError(f"{path.name} contains duplicate CSV fields")
    return tuple(fields)


def _artifact(root: Path, path: Path, rows: int | None = None) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = rows
    return entry


def _manifest_count(manifest: Mapping[str, object], label: str) -> int:
    corpus = manifest.get("corpus")
    if not isinstance(corpus, dict):
        raise CorpusBuildError(f"{label} manifest is missing corpus identity")
    count = corpus.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CorpusBuildError(f"{label} manifest has an invalid corpus count")
    return count


def _date_config(manifest: Mapping[str, object]) -> tuple[DateConfig, dict[str, object]]:
    raw = manifest.get("date_rules")
    expected_fields = {"version", *asdict(DateConfig()).keys()}
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise CorpusBuildError("corpus manifest has an invalid date_rules contract")
    if raw.get("version") != DATE_RULES_VERSION:
        raise CorpusBuildError(
            f"corpus date rules must use {DATE_RULES_VERSION!r}"
        )
    integer_fields = ("bin_size", "circa_years", "open_range_years")
    optional_integer_fields = ("min_year", "max_year")
    if any(
        isinstance(raw.get(field), bool) or not isinstance(raw.get(field), int)
        for field in integer_fields
    ) or any(
        value is not None and (isinstance(value, bool) or not isinstance(value, int))
        for value in (raw.get(field) for field in optional_integer_fields)
    ):
        raise CorpusBuildError("corpus manifest date rule values must be integers")
    config = DateConfig(
        bin_size=int(raw["bin_size"]),
        circa_years=int(raw["circa_years"]),
        open_range_years=int(raw["open_range_years"]),
        min_year=raw["min_year"],  # type: ignore[arg-type]
        max_year=raw["max_year"],  # type: ignore[arg-type]
    )
    try:
        config.validate()
    except ValueError as exc:
        raise CorpusBuildError(f"corpus manifest date rules are invalid: {exc}") from exc
    return config, dict(raw)


def _normalized_dates(
    rows: Sequence[Mapping[str, str]], source_name: str
) -> tuple[NormalizedDate, ...]:
    output: list[NormalizedDate] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            start = parse_year(row.get("date_start"))
            end = parse_year(row.get("date_end"))
        except ValueError as exc:
            raise CorpusBuildError(
                f"{source_name} row {row_number}: invalid normalized date"
            ) from exc
        if (start is None) != (end is None):
            raise CorpusBuildError(
                f"{source_name} row {row_number}: date_start and date_end must both be set or empty"
            )
        if start is not None and end is not None and start > end:
            raise CorpusBuildError(
                f"{source_name} row {row_number}: date_start exceeds date_end"
            )
        output.append(
            NormalizedDate(
                display=str(row.get("date_display") or "Unknown"),
                start=start,
                end=end,
                qualifier=str(row.get("date_qualifier") or "unknown"),
                parse_method=str(row.get("date_parse_method") or "unknown"),
            )
        )
    return tuple(output)


def _expected_bins(
    dates: Sequence[NormalizedDate], config: DateConfig
) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "key": f"{start}:{end}",
            "start": start,
            "end": end,
            "label": bin_label(start, end),
        }
        for index, (start, end) in enumerate(make_bins(dates, config))
    ]


def _processor_contract(model: Mapping[str, object]) -> dict[str, object]:
    model_id = str(model.get("id") or "").strip()
    revision = str(model.get("revision") or "").strip()
    settings = model.get("settings")
    if not model_id or not revision or not isinstance(settings, dict):
        raise CorpusBuildError("model manifest has an invalid model identity or settings")
    semantic_settings = {
        key: value
        for key, value in sorted(settings.items())
        if key not in _OPERATIONAL_MODEL_SETTINGS
    }
    if not semantic_settings:
        raise CorpusBuildError("model settings do not declare a processor contract")
    return {
        "model_id": model_id,
        "model_revision": revision,
        "settings": semantic_settings,
    }


def _validate_corpus_artifacts(
    root: Path,
    manifest: Mapping[str, object],
    deployed: Mapping[str, Path],
) -> None:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise CorpusBuildError("corpus manifest artifacts must be a list")
    by_path: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CorpusBuildError("corpus manifest artifact entries must be objects")
        raw_path = str(entry.get("path") or "")
        if not raw_path or raw_path in by_path:
            raise CorpusBuildError(f"corpus manifest artifact path is invalid: {raw_path!r}")
        by_path[raw_path] = entry
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise CorpusBuildError("corpus manifest files must be an object")
    for key, actual_path in deployed.items():
        relative = str(files.get(key) or "")
        entry = by_path.get(relative)
        if entry is None:
            raise CorpusBuildError(
                f"corpus manifest does not checksum its deployed {key}: {relative!r}"
            )
        try:
            expected_size = int(entry["bytes"])
            expected_digest = str(entry["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusBuildError(
                f"corpus manifest artifact metadata is invalid: {relative}"
            ) from exc
        if (
            actual_path.stat().st_size != expected_size
            or sha256_file(actual_path) != expected_digest
        ):
            raise CorpusBuildError(
                f"deployed artifact differs from corpus manifest: {relative}"
            )


def _denominator_rows(
    matrix, rows: Sequence[Mapping[str, str]], bins: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    import numpy as np

    eligible = np.asarray(matrix.sum(axis=0), dtype=np.float64).reshape(-1)
    physical_ids: list[set[str]] = [set() for _ in bins]
    visual_ids: list[set[str]] = [set() for _ in bins]
    matrix = matrix.tocsr()
    for row_index, row in enumerate(rows):
        start, end = matrix.indptr[row_index : row_index + 2]
        for column in matrix.indices[start:end]:
            physical_ids[int(column)].add(str(row.get("physical_object_id") or ""))
            visual_ids[int(column)].add(str(row.get("visual_cluster_id") or ""))
    return [
        {
            "bin_index": index,
            "bin_key": item["key"],
            "bin_start": item["start"],
            "bin_end": item["end"],
            "bin_label": item["label"],
            "eligible_weight": format(float(eligible[index]), ".12g"),
            "physical_object_count": len(physical_ids[index]),
            "visual_cluster_count": len(visual_ids[index]),
        }
        for index, item in enumerate(bins)
    ]


def _validate_date_artifacts(
    path: Path,
    denominators_path: Path,
    rows: Sequence[Mapping[str, str]],
    dates: Sequence[NormalizedDate],
    config: DateConfig,
    bins: Sequence[Mapping[str, object]],
) -> None:
    try:
        import numpy as np
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - required by the pipeline runtime
        raise CorpusBuildError("embedding bundle merging requires NumPy and SciPy") from exc
    try:
        matrix = sparse.load_npz(path).tocsr()
    except (OSError, ValueError) as exc:
        raise CorpusBuildError(f"date weights are not a valid sparse NPZ: {path}") from exc
    if matrix.shape != (len(rows), len(bins)):
        raise CorpusBuildError("date weights do not align with corpus rows and bins")
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data <= 0):
        raise CorpusBuildError("date weights must contain only finite positive values")
    matrix.sort_indices()
    bin_index = {int(item["start"]): index for index, item in enumerate(bins)}
    for row_index, date in enumerate(dates):
        expected_by_start = uniform_bin_weights(date, config.bin_size)
        try:
            expected_columns = np.asarray(
                [bin_index[start] for start in expected_by_start], dtype=np.int64
            )
        except KeyError as exc:
            raise CorpusBuildError("date weights refer to a date outside declared bins") from exc
        expected_values = np.asarray(list(expected_by_start.values()), dtype=np.float64)
        start, end = matrix.indptr[row_index : row_index + 2]
        if not np.array_equal(matrix.indices[start:end], expected_columns) or not np.allclose(
            matrix.data[start:end], expected_values, rtol=0, atol=1e-12
        ):
            raise CorpusBuildError(
                f"date weights do not match canonical date at corpus row {row_index}"
            )

    fields = _csv_fields(denominators_path)
    missing = sorted(set(_DENOMINATOR_FIELDS) - set(fields))
    if missing:
        raise CorpusBuildError(
            f"{denominators_path.name} is missing required fields: {', '.join(missing)}"
        )
    with denominators_path.open(encoding="utf-8-sig", newline="") as handle:
        actual = list(csv.DictReader(handle))
    expected = _denominator_rows(matrix, rows, bins)
    if len(actual) != len(expected):
        raise CorpusBuildError("bin denominators do not align with declared bins")
    for index, (raw, calculated) in enumerate(zip(actual, expected, strict=True)):
        try:
            matches = (
                int(raw["bin_index"]) == index
                and raw["bin_key"] == calculated["bin_key"]
                and int(raw["bin_start"]) == calculated["bin_start"]
                and int(raw["bin_end"]) == calculated["bin_end"]
                and raw["bin_label"] == calculated["bin_label"]
                and np.isclose(
                    float(raw["eligible_weight"]),
                    float(calculated["eligible_weight"]),
                    rtol=0,
                    atol=1e-9,
                )
                and int(raw["physical_object_count"])
                == calculated["physical_object_count"]
                and int(raw["visual_cluster_count"])
                == calculated["visual_cluster_count"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusBuildError("bin denominator values are invalid") from exc
        if not matches:
            raise CorpusBuildError(
                f"bin denominators do not match date weights at bin {index}"
            )


def _validate_rights_and_order(
    corpus_rows: Sequence[Mapping[str, str]],
    image_rows: Sequence[Mapping[str, str]],
    embedded_rows: Sequence[Mapping[str, str]],
) -> None:
    corpus_ids = [row["artwork_id"].strip() for row in corpus_rows]
    if corpus_ids != [row["artwork_id"].strip() for row in image_rows]:
        raise CorpusBuildError("corpus and image manifest artwork order differ")
    if corpus_ids != [row["artwork_id"].strip() for row in embedded_rows]:
        raise CorpusBuildError("corpus and embedded-image provenance artwork order differ")
    for corpus, image, embedded in zip(
        corpus_rows, image_rows, embedded_rows, strict=True
    ):
        artwork_id = corpus["artwork_id"].strip()
        if parse_bool(corpus.get("public_domain")) != parse_bool(image.get("public_domain")):
            raise CorpusBuildError(f"{artwork_id}: public-domain rights metadata differ")
        for field in ("image_url", "image_sha256", "image_rights_uri"):
            if str(corpus.get(field) or "").strip() != str(image.get(field) or "").strip():
                raise CorpusBuildError(f"{artwork_id}: {field} differs in image manifest")
        if embedded.get("image_url", "").strip() != image.get("image_url", "").strip():
            raise CorpusBuildError(f"{artwork_id}: embedded image URL differs")
        if embedded.get("permission_status", "").strip() != image.get(
            "permission_status", ""
        ).strip():
            raise CorpusBuildError(f"{artwork_id}: embedded permission status differs")
        image_policy = image.get(IMAGE_INPUT_POLICY_FIELD, "").strip()
        if image_policy and embedded.get("input_policy", "").strip() != image_policy:
            raise CorpusBuildError(
                f"{artwork_id}: embedded input policy differs from image manifest"
            )
        digest = embedded.get("input_sha256", "").strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise CorpusBuildError(
                f"{artwork_id}: embedded input_sha256 is not a 64-character hex digest"
            )
        if corpus.get("image_sha256", "").strip().lower() != digest:
            raise CorpusBuildError(
                f"{artwork_id}: corpus image_sha256 is not reconciled to embedded bytes"
            )
        if corpus.get("visual_cluster_id", "").strip().lower() != f"sha256:{digest}":
            raise CorpusBuildError(
                f"{artwork_id}: visual_cluster_id is not reconciled to embedded bytes"
            )
        declared_digest = embedded.get("declared_image_sha256", "").strip().lower()
        if declared_digest and (
            len(declared_digest) != 64
            or any(character not in "0123456789abcdef" for character in declared_digest)
        ):
            raise CorpusBuildError(
                f"{artwork_id}: declared image digest is not a 64-character hex digest"
            )
        if (
            declared_digest
            and _uses_declared_image_resource(image, embedded)
            and declared_digest != digest
        ):
            raise CorpusBuildError(
                f"{artwork_id}: declared image digest differs from embedded bytes"
            )


def _remote_resource_identity(value: object) -> tuple[object, ...] | None:
    """Return a comparison key for URLs that identify the same fetched resource."""

    try:
        parsed = urlsplit(str(value or "").strip())
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if scheme not in {"http", "https"} or not hostname:
        return None
    effective_port = port if port is not None else (443 if scheme == "https" else 80)
    return (
        scheme,
        hostname,
        effective_port,
        parsed.username,
        parsed.password,
        unquote(parsed.path),
        parsed.query,
    )


def _uses_declared_image_resource(
    image: Mapping[str, str], embedded: Mapping[str, str]
) -> bool:
    """Whether the declared digest and embedded bytes describe one resource.

    Local-file inputs are checked against their declared file before embedding.
    Remote inputs are the same resource only when the actual fetched URL matches
    the manifest URL. This intentionally excludes Met's ``/original/`` to
    ``/web-large/`` optimization, whose two legitimate byte hashes differ.
    """

    input_kind = embedded.get("input_kind", "").strip().casefold()
    if input_kind == "local-file" and image.get("image_path", "").strip():
        return True
    # This policy may replace a declared Met ``/original/`` URL with
    # ``/web-large/`` and then fall back to the original URL when the derivative
    # is too small. The upstream digest is therefore informational for every
    # outcome of this adaptive policy, including a same-URL fallback.
    if embedded.get("input_policy", "").strip() == SIGLIP_IMAGE_INPUT_POLICY:
        return False
    declared = _remote_resource_identity(image.get("image_url", ""))
    fetched = _remote_resource_identity(embedded.get("input_source", ""))
    return declared is not None and declared == fetched


def _load_bundle(root: Path) -> _SourceBundle:
    root = root.resolve()
    model_manifest_path = root / "model-manifest.json"
    model_manifest = _read_manifest(model_manifest_path, EMBED_SCHEMA_VERSION)
    artifacts = _verify_artifacts(root, model_manifest)
    corpus_path = _require_declared(root, model_manifest, artifacts, "metadata")
    embeddings_path = _require_declared(root, model_manifest, artifacts, "embeddings")
    date_weights_path = _require_declared(root, model_manifest, artifacts, "dateWeights")
    denominators_path = _require_declared(
        root, model_manifest, artifacts, "binDenominators"
    )
    images_path = _require_declared(root, model_manifest, artifacts, "imageManifest")
    embedded_path = _require_declared(root, model_manifest, artifacts, "embeddedImages")

    corpus_manifest_path = root / "corpus-build-manifest.json"
    if corpus_manifest_path.relative_to(root).as_posix() not in artifacts:
        raise CorpusBuildError(
            "source model manifest does not checksum corpus-build-manifest.json"
        )
    corpus_manifest = _read_manifest(corpus_manifest_path, BUILD_SCHEMA_VERSION)
    if model_manifest.get("corpus_manifest_sha256") != sha256_file(corpus_manifest_path):
        raise CorpusBuildError("source model manifest has the wrong corpus manifest checksum")
    if model_manifest.get("corpus") != corpus_manifest.get("corpus"):
        raise CorpusBuildError("model and corpus manifests have different corpus identities")
    _validate_corpus_artifacts(
        root,
        corpus_manifest,
        {
            "metadata": corpus_path,
            "dateWeights": date_weights_path,
            "binDenominators": denominators_path,
            "imageManifest": images_path,
        },
    )

    corpus_fields = _csv_fields(corpus_path)
    missing_corpus = sorted(set(CANONICAL_FIELDS) - set(corpus_fields))
    if missing_corpus:
        raise CorpusBuildError(
            f"{corpus_path.name} is missing canonical fields: {', '.join(missing_corpus)}"
        )
    image_fields = _csv_fields(images_path)
    missing_images = sorted(set(IMAGE_MANIFEST_FIELDS) - set(image_fields))
    if missing_images:
        raise CorpusBuildError(
            f"{images_path.name} is missing required fields: {', '.join(missing_images)}"
        )
    allowed_image_schemas = {
        tuple(IMAGE_MANIFEST_FIELDS),
        (*IMAGE_MANIFEST_FIELDS, IMAGE_INPUT_POLICY_FIELD),
    }
    if image_fields not in allowed_image_schemas:
        raise CorpusBuildError(
            "image manifest has unsupported fields or field order; only the optional "
            f"{IMAGE_INPUT_POLICY_FIELD!r} extension is allowed"
        )
    embedded_fields = _csv_fields(embedded_path)
    missing_embedded = sorted(set(_EMBEDDED_REQUIRED_FIELDS) - set(embedded_fields))
    if missing_embedded:
        raise CorpusBuildError(
            f"{embedded_path.name} is missing required fields: {', '.join(missing_embedded)}"
        )

    corpus_rows = tuple(
        _ordered_rows(
            corpus_path,
            required_fields=CANONICAL_FIELDS,
            require_offsets=True,
        )
    )
    image_rows = tuple(
        _ordered_rows(
            images_path,
            required_fields=IMAGE_MANIFEST_FIELDS,
            require_offsets=False,
        )
    )
    embedded_rows = tuple(
        _ordered_rows(
            embedded_path,
            required_fields=_EMBEDDED_REQUIRED_FIELDS,
            require_offsets=True,
        )
    )
    _validate_rights_and_order(corpus_rows, image_rows, embedded_rows)
    count = len(corpus_rows)
    if (
        _manifest_count(model_manifest, "model") != count
        or _manifest_count(corpus_manifest, "corpus") != count
    ):
        raise CorpusBuildError("manifest corpus count does not match deployed rows")
    rows, dimensions = _validate_matrix(embeddings_path, model_manifest, count)
    if rows != count:
        raise AssertionError("validated embedding row count changed")
    matrix_contract = model_manifest.get("matrix")
    if not isinstance(matrix_contract, dict) or matrix_contract.get("row_order") != (
        "corpus.csv embedding_offset"
    ):
        raise CorpusBuildError("embedding matrix has an unsupported row-order contract")
    index = model_manifest.get("index")
    if not isinstance(index, dict) or (
        index.get("metric") != "inner-product-on-l2-normalized-vectors"
        or index.get("exact") is not True
    ):
        raise CorpusBuildError("embedding bundle does not declare the exact cosine contract")

    date_config, _date_rules = _date_config(corpus_manifest)
    dates = _normalized_dates(corpus_rows, corpus_path.name)
    bins = _expected_bins(dates, date_config)
    if corpus_manifest.get("bins") != bins or model_manifest.get("bins") != bins:
        raise CorpusBuildError("bundle bins do not match its canonical dates and date rules")
    _validate_date_artifacts(
        date_weights_path,
        denominators_path,
        corpus_rows,
        dates,
        date_config,
        bins,
    )

    files = model_manifest.get("files")
    if not isinstance(files, dict):
        raise CorpusBuildError("model manifest files must be an object")
    raw_provenance = files.get("sourceProvenance", [])
    if not isinstance(raw_provenance, list):
        raise CorpusBuildError("model sourceProvenance must be a list")
    provenance_paths: list[Path] = []
    for raw_path in raw_provenance:
        path = _declared_path(root, raw_path, "a source provenance file")
        relative = path.relative_to(root).as_posix()
        if relative not in artifacts:
            raise CorpusBuildError(
                f"source provenance is not covered by artifact checksums: {relative}"
            )
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusBuildError(f"source provenance is invalid JSON: {path}") from exc
        provenance_paths.append(path)

    model = model_manifest.get("model")
    if not isinstance(model, dict):
        raise CorpusBuildError("model manifest is missing model identity")
    return _SourceBundle(
        root=root,
        model_manifest_path=model_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        model_manifest=model_manifest,
        corpus_manifest=corpus_manifest,
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        date_weights_path=date_weights_path,
        denominators_path=denominators_path,
        images_path=images_path,
        embedded_path=embedded_path,
        corpus_fields=corpus_fields,
        image_fields=image_fields,
        embedded_fields=embedded_fields,
        corpus_rows=corpus_rows,
        image_rows=image_rows,
        embedded_rows=embedded_rows,
        dates=dates,
        date_config=date_config,
        dimensions=dimensions,
        processor_contract=_processor_contract(model),
        provenance_paths=tuple(provenance_paths),
    )


def _typed_coverage_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    typed: list[dict[str, object]] = []
    for row in rows:
        public_domain = row.get("public_domain", "").strip()
        typed.append(
            {
                **row,
                "public_domain": parse_bool(public_domain) if public_domain else None,
                "image_available": parse_bool(row.get("image_available")),
            }
        )
    return _coverage_rows(typed)


def _settings_digest(settings: object) -> str:
    return hashlib.sha256(
        json.dumps(
            settings, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _merged_model_contract(
    bundles: Sequence[_SourceBundle], first_model: Mapping[str, object]
) -> dict[str, object]:
    """Describe common semantics and the union of source-specific image inputs."""

    semantic_settings = dict(bundles[0].processor_contract["settings"])
    hosts: set[str] = set()
    policies: set[str] = set()
    checkpointed = True
    for bundle in bundles:
        model = bundle.model_manifest.get("model")
        settings = model.get("settings") if isinstance(model, dict) else None
        if not isinstance(settings, dict):
            raise CorpusBuildError("source model settings are invalid")
        raw_hosts = settings.get("allowed_image_hosts", [])
        if not isinstance(raw_hosts, list) or any(
            not isinstance(host, str) or not host.strip() for host in raw_hosts
        ):
            raise CorpusBuildError("source allowed_image_hosts must be a string list")
        hosts.update(host.strip().lower() for host in raw_hosts)
        checkpointed = checkpointed and settings.get("checkpointed") is True
        policies.update(
            row.get("input_policy", "").strip()
            for row in bundle.embedded_rows
            if row.get("input_policy", "").strip()
        )
    ordered_policies = sorted(policies)
    semantic_settings.update(
        {
            "allowed_image_hosts": sorted(hosts),
            "checkpointed": checkpointed,
            "device": "source-bundle-specific",
            "merged_from_completed_bundles": len(bundles),
        }
    )
    if ordered_policies:
        semantic_settings.update(
            {
                "image_input_policy": (
                    ordered_policies[0]
                    if len(ordered_policies) == 1
                    else "per-record-mixed"
                ),
                "image_input_policies": ordered_policies,
            }
        )
    return {
        "id": first_model["id"],
        "revision": first_model["revision"],
        "settings": semantic_settings,
    }


def merge_embedding_bundles(
    bundle_dirs: Sequence[Path | str],
    output_dir: Path | str,
    *,
    corpus_version: str,
    corpus_label: str | None = None,
) -> dict[str, object]:
    """Atomically concatenate compatible float32 embedding bundles.

    Source bundle order is preserved. Rows remain in each source bundle's
    ``embedding_offset`` order, while every combined offset and date-derived
    artifact is rebuilt for the new corpus identity.
    """

    if len(bundle_dirs) < 2:
        raise CorpusBuildError("at least two embedding bundles are required")
    if not corpus_version.strip():
        raise CorpusBuildError("corpus_version is required")
    normalized_label = corpus_label.strip() if corpus_label is not None else None
    if normalized_label == "":
        raise CorpusBuildError("corpus_label must be non-empty when provided")
    roots = tuple(Path(path).resolve() for path in bundle_dirs)
    if len(set(roots)) != len(roots):
        raise CorpusBuildError("source embedding bundle paths must be unique")
    missing = next((root for root in roots if not root.is_dir()), None)
    if missing is not None:
        raise CorpusBuildError(f"embedding bundle is not a directory: {missing}")
    destination = Path(output_dir).resolve()
    _reject_overlapping_output(destination, roots)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise CorpusBuildError(f"output directory must be absent or empty: {destination}")

    bundles = tuple(_load_bundle(root) for root in roots)
    first = bundles[0]
    first_model = first.model_manifest["model"]
    assert isinstance(first_model, dict)
    first_date_rules = first.corpus_manifest["date_rules"]
    source_counting_units = {
        str(bundle.corpus_manifest.get("counting_unit") or "physical-object")
        for bundle in bundles
    }
    supported_counting_units = {"physical-object", "catalog-record"}
    if not source_counting_units.issubset(supported_counting_units):
        raise CorpusBuildError("source bundles use incompatible counting units")
    output_counting_unit = (
        next(iter(source_counting_units))
        if len(source_counting_units) == 1
        else "catalog-record"
    )
    for bundle in bundles[1:]:
        model = bundle.model_manifest.get("model")
        if not isinstance(model, dict) or (
            model.get("id") != first_model.get("id")
            or model.get("revision") != first_model.get("revision")
        ):
            raise CorpusBuildError("source bundles use different model IDs or revisions")
        if bundle.dimensions != first.dimensions:
            raise CorpusBuildError("source bundles use different embedding dimensions")
        if bundle.processor_contract != first.processor_contract:
            raise CorpusBuildError("source bundles use different processor contracts")
        if bundle.corpus_manifest.get("date_rules") != first_date_rules:
            raise CorpusBuildError("source bundles use incompatible date rules")
        for label, fields, expected in (
            ("corpus", bundle.corpus_fields, first.corpus_fields),
            ("embedded-image provenance", bundle.embedded_fields, first.embedded_fields),
        ):
            if fields != expected:
                raise CorpusBuildError(f"source bundles have different {label} CSV schemas")

    output_image_fields = (
        (*IMAGE_MANIFEST_FIELDS, IMAGE_INPUT_POLICY_FIELD)
        if any(IMAGE_INPUT_POLICY_FIELD in bundle.image_fields for bundle in bundles)
        else tuple(IMAGE_MANIFEST_FIELDS)
    )
    merged_model = _merged_model_contract(bundles, first_model)

    all_corpus_rows: list[dict[str, str]] = []
    all_image_rows: list[dict[str, str]] = []
    all_embedded_rows: list[dict[str, str]] = []
    all_dates: list[NormalizedDate] = []
    seen_ids: set[str] = set()
    for bundle in bundles:
        for corpus, image, embedded, date in zip(
            bundle.corpus_rows,
            bundle.image_rows,
            bundle.embedded_rows,
            bundle.dates,
            strict=True,
        ):
            artwork_id = corpus["artwork_id"].strip()
            if artwork_id in seen_ids:
                raise CorpusBuildError(
                    f"source bundles contain duplicate artwork_id {artwork_id!r}"
                )
            seen_ids.add(artwork_id)
            offset = len(all_corpus_rows)
            all_corpus_rows.append({**corpus, "embedding_offset": str(offset)})
            all_image_rows.append(dict(image))
            all_embedded_rows.append({**embedded, "embedding_offset": str(offset)})
            all_dates.append(date)

    total_rows = len(all_corpus_rows)
    combined_bins = _expected_bins(all_dates, first.date_config)
    bin_pairs = [
        (int(item["start"]), int(item["end"])) for item in combined_bins
    ]
    coverage_rows = _typed_coverage_rows(all_corpus_rows)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    manifest: dict[str, object]
    try:
        corpus_path = staging / "corpus.csv"
        images_path = staging / "images.manifest.csv"
        embedded_path = staging / "embedded-images.manifest.csv"
        weights_path = staging / "date-weights.npz"
        denominators_path = staging / "bin-denominators.csv"
        coverage_path = staging / "coverage.csv"
        embeddings_path = staging / "embeddings.npy"
        _write_csv(corpus_path, first.corpus_fields, all_corpus_rows)
        _write_csv(images_path, output_image_fields, all_image_rows)
        _write_csv(embedded_path, first.embedded_fields, all_embedded_rows)
        denominator_rows = _write_sparse_npz(
            weights_path,
            all_corpus_rows,  # type: ignore[arg-type]
            all_dates,
            bin_pairs,
            first.date_config.bin_size,
        )
        _write_csv(denominators_path, _DENOMINATOR_FIELDS, denominator_rows)
        _write_csv(coverage_path, _COVERAGE_FIELDS, coverage_rows)

        import numpy as np

        output_matrix = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, first.dimensions),
        )
        next_offset = 0
        for bundle in bundles:
            source_matrix = np.load(
                bundle.embeddings_path, mmap_mode="r", allow_pickle=False
            )
            for source_offset in range(0, len(bundle.corpus_rows), 8192):
                block = source_matrix[source_offset : source_offset + 8192]
                output_matrix[
                    next_offset + source_offset : next_offset + source_offset + len(block)
                ] = block
            next_offset += len(bundle.corpus_rows)
            del source_matrix
        output_matrix.flush()
        del output_matrix

        provenance_paths: list[Path] = []
        merge_sources: list[dict[str, object]] = []
        row_start = 0
        provenance_root = staging / "source-provenance"
        provenance_root.mkdir()
        for index, bundle in enumerate(bundles):
            bundle_root = provenance_root / f"bundle-{index:03d}"
            bundle_root.mkdir()
            copied_model = bundle_root / "model-manifest.json"
            copied_corpus = bundle_root / "corpus-build-manifest.json"
            shutil.copyfile(bundle.model_manifest_path, copied_model)
            shutil.copyfile(bundle.corpus_manifest_path, copied_corpus)
            provenance_paths.extend((copied_model, copied_corpus))
            copied_upstream: list[str] = []
            if bundle.provenance_paths:
                upstream_root = bundle_root / "upstream"
                upstream_root.mkdir()
                for source_path in bundle.provenance_paths:
                    relative = source_path.relative_to(bundle.root)
                    target = upstream_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_path, target)
                    provenance_paths.append(target)
                    copied_upstream.append(target.relative_to(staging).as_posix())
            institutions = sorted(
                {
                    row.get("institution", "").strip() or "unknown"
                    for row in bundle.corpus_rows
                }
            )
            row_count = len(bundle.corpus_rows)
            source_model = bundle.model_manifest["model"]
            assert isinstance(source_model, dict)
            merge_sources.append(
                {
                    "bundle_index": index,
                    "corpus": bundle.model_manifest["corpus"],
                    "row_start": row_start,
                    "row_end_exclusive": row_start + row_count,
                    "row_count": row_count,
                    "institutions": institutions,
                    "model_manifest": copied_model.relative_to(staging).as_posix(),
                    "model_manifest_sha256": sha256_file(copied_model),
                    "corpus_build_manifest": copied_corpus.relative_to(staging).as_posix(),
                    "corpus_build_manifest_sha256": sha256_file(copied_corpus),
                    "model_settings_sha256": _settings_digest(
                        source_model.get("settings")
                    ),
                    "upstream_provenance": copied_upstream,
                }
            )
            row_start += row_count

        source_digest = _settings_digest(
            [
                {
                    "model_manifest_sha256": item["model_manifest_sha256"],
                    "corpus_build_manifest_sha256": item[
                        "corpus_build_manifest_sha256"
                    ],
                }
                for item in merge_sources
            ]
        )
        corpus_identity = {
            "id": corpus_version,
            "version": corpus_version,
            "count": total_rows,
            "countingUnit": output_counting_unit,
            **({"label": normalized_label} if normalized_label is not None else {}),
        }
        build_artifacts = [
            _artifact(staging, corpus_path, total_rows),
            _artifact(staging, images_path, total_rows),
            _artifact(staging, weights_path, total_rows),
            _artifact(staging, denominators_path, len(denominator_rows)),
            _artifact(staging, coverage_path, len(coverage_rows)),
            *(_artifact(staging, path) for path in provenance_paths),
        ]
        build_artifacts.sort(key=lambda entry: str(entry["path"]))
        build_manifest: dict[str, object] = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "builder_version": __version__,
            "corpus_version": corpus_version,
            "corpus": corpus_identity,
            "source": {
                "kind": "merged-completed-embedding-bundles",
                "revision": source_digest,
                "retrieved_at": "not-recorded",
                "payloads": [
                    {
                        "filename": item["model_manifest"],
                        "sha256": item["model_manifest_sha256"],
                    }
                    for item in merge_sources
                ],
            },
            "date_rules": first_date_rules,
            "counting_unit": output_counting_unit,
            "counts": {
                "input_rows": total_rows,
                "canonical_rows": total_rows,
                "dated_rows": sum(date.dated for date in all_dates),
                "unknown_date_rows": sum(not date.dated for date in all_dates),
                "bins": len(combined_bins),
                "unreviewed_images": sum(
                    row.get("permission_status") == "unreviewed"
                    for row in all_image_rows
                ),
                "source_bundles": len(bundles),
                "unique_visual_clusters": len(
                    {row.get("visual_cluster_id", "") for row in all_corpus_rows}
                ),
            },
            "canonical_fields": list(first.corpus_fields),
            "files": {
                "metadata": "corpus.csv",
                "embeddings": None,
                "dateWeights": "date-weights.npz",
                "binDenominators": "bin-denominators.csv",
                "imageManifest": "images.manifest.csv",
                "coverage": "coverage.csv",
            },
            "bins": combined_bins,
            "merge": {
                "schema_version": MERGE_SCHEMA_VERSION,
                "row_order": "source-bundle-argument-order-then-embedding-offset",
                "date_artifacts": "rebuilt-from-combined-canonical-dates",
                "sources": merge_sources,
                "source_counting_units": sorted(source_counting_units),
                "output_counting_unit": output_counting_unit,
            },
            "artifacts": build_artifacts,
        }
        corpus_manifest_path = staging / "corpus-build-manifest.json"
        corpus_manifest_path.write_text(
            json.dumps(build_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        model_artifact_paths = [
            corpus_path,
            images_path,
            embedded_path,
            weights_path,
            denominators_path,
            coverage_path,
            embeddings_path,
            corpus_manifest_path,
            *provenance_paths,
        ]
        model_artifacts = sorted(
            (_artifact(staging, path) for path in model_artifact_paths),
            key=lambda entry: str(entry["path"]),
        )
        manifest = {
            "schema_version": EMBED_SCHEMA_VERSION,
            "builder_version": __version__,
            "corpus": corpus_identity,
            "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
            "model": merged_model,
            "matrix": {
                "rows": total_rows,
                "dimensions": first.dimensions,
                "dtype": "float32",
                "l2_normalized": True,
                "row_order": "corpus.csv embedding_offset",
            },
            "index": {
                "metric": "inner-product-on-l2-normalized-vectors",
                "backend": "numpy-flat-ip",
                "exact": True,
                "numpy_fallback": "embeddings.npy",
                "faiss": None,
            },
            "files": {
                "metadata": "corpus.csv",
                "embeddings": "embeddings.npy",
                "dateWeights": "date-weights.npz",
                "binDenominators": "bin-denominators.csv",
                "imageManifest": "images.manifest.csv",
                "embeddedImages": "embedded-images.manifest.csv",
                "coverage": "coverage.csv",
                "numpyIndex": "embeddings.npy",
                "faissIndex": None,
                "sourceProvenance": [
                    path.relative_to(staging).as_posix() for path in provenance_paths
                ],
            },
            "bins": combined_bins,
            "merge": {
                "schema_version": MERGE_SCHEMA_VERSION,
                "operation": "concatenate-completed-same-model-bundles",
                "row_order": "source-bundle-argument-order-then-embedding-offset",
                "offsets": "rebuilt-contiguous-from-zero",
                "timeline_artifacts": "rebuilt-from-combined-canonical-dates",
                "faiss": "omitted",
                "processor_contract": first.processor_contract,
                "processor_contract_sha256": _settings_digest(
                    first.processor_contract
                ),
                "sources": merge_sources,
                "source_counting_units": sorted(source_counting_units),
                "output_counting_unit": output_counting_unit,
            },
            "artifacts": model_artifacts,
        }
        manifest_path = staging / "model-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            if any(destination.iterdir()):
                raise CorpusBuildError(
                    f"output directory changed during publication: {destination}"
                )
            destination.rmdir()
        staging.replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = ["MERGE_SCHEMA_VERSION", "merge_embedding_bundles"]
