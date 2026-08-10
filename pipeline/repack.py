"""Repack an embedding matrix against reconciled canonical corpus metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from . import __version__
from .build import BUILD_SCHEMA_VERSION, CorpusBuildError, sha256_file
from .embeddings import EMBED_SCHEMA_VERSION


REPACK_SCHEMA_VERSION = "mnemosyne-embedding-repack/v1"


def _read_manifest(path: Path, expected_schema: str) -> dict[str, object]:
    if not path.is_file():
        raise CorpusBuildError(f"required manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CorpusBuildError(f"manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != expected_schema:
        raise CorpusBuildError(
            f"{path.name} must use schema {expected_schema!r}"
        )
    return payload


def _declared_path(root: Path, raw_path: object, label: str) -> Path:
    value = str(raw_path or "").strip()
    if not value:
        raise CorpusBuildError(f"manifest does not declare {label}")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CorpusBuildError(f"manifest path escapes its bundle: {value}") from exc
    if not resolved.is_file():
        raise CorpusBuildError(f"manifest-declared file is missing: {value}")
    return resolved


def _verify_artifacts(
    root: Path, manifest: Mapping[str, object]
) -> dict[str, Mapping[str, object]]:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise CorpusBuildError("manifest artifacts must be a list")
    verified: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CorpusBuildError("manifest artifact entries must be objects")
        raw_path = str(entry.get("path") or "")
        path = _declared_path(root, raw_path, "an artifact path")
        relative = path.relative_to(root).as_posix()
        if relative != raw_path or relative in verified:
            raise CorpusBuildError(f"manifest artifact path is invalid or duplicated: {raw_path}")
        try:
            declared_bytes = int(entry["bytes"])
            declared_sha256 = str(entry["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusBuildError(f"manifest artifact metadata is invalid: {raw_path}") from exc
        if path.stat().st_size != declared_bytes or sha256_file(path) != declared_sha256:
            raise CorpusBuildError(f"manifest artifact integrity check failed: {raw_path}")
        verified[relative] = entry
    return verified


def _require_declared(
    root: Path,
    manifest: Mapping[str, object],
    artifacts: Mapping[str, Mapping[str, object]],
    file_key: str,
) -> Path:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise CorpusBuildError("manifest files must be an object")
    path = _declared_path(root, files.get(file_key), file_key)
    relative = path.relative_to(root).as_posix()
    if relative not in artifacts:
        raise CorpusBuildError(f"required file is not covered by artifact checksums: {relative}")
    return path


def _ordered_rows(
    path: Path,
    *,
    required_fields: Sequence[str],
    require_offsets: bool,
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required_fields) - fields)
        if missing:
            raise CorpusBuildError(
                f"{path.name} is missing required fields: {', '.join(missing)}"
            )
        rows = list(reader)
    seen: set[str] = set()
    for index, row in enumerate(rows):
        artwork_id = str(row.get("artwork_id") or "").strip()
        if not artwork_id:
            raise CorpusBuildError(f"{path.name} row {index + 2}: artwork_id is empty")
        if artwork_id in seen:
            raise CorpusBuildError(
                f"{path.name} row {index + 2}: duplicate artwork_id {artwork_id!r}"
            )
        seen.add(artwork_id)
        if require_offsets:
            try:
                offset = int(str(row.get("embedding_offset") or ""))
            except ValueError as exc:
                raise CorpusBuildError(
                    f"{path.name} row {index + 2}: embedding_offset is not an integer"
                ) from exc
            if offset != index:
                raise CorpusBuildError(
                    f"{path.name} embedding_offset values must be contiguous in row order"
                )
    return rows


def _manifest_count(manifest: Mapping[str, object], label: str) -> int:
    corpus = manifest.get("corpus")
    if not isinstance(corpus, dict):
        raise CorpusBuildError(f"{label} manifest is missing corpus identity")
    count = corpus.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CorpusBuildError(f"{label} manifest has an invalid corpus count")
    return count


def _validate_matrix(
    path: Path,
    source_manifest: Mapping[str, object],
    expected_rows: int,
) -> tuple[int, int]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - required by the pipeline runtime
        raise CorpusBuildError("embedding repacking requires NumPy") from exc

    if path.suffix != ".npy":
        raise CorpusBuildError("repacking requires a canonical embeddings.npy matrix")
    try:
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CorpusBuildError(f"embedding matrix is not a valid NPY file: {path}") from exc
    if matrix.ndim != 2 or matrix.dtype != np.dtype("float32"):
        raise CorpusBuildError("repacking requires a two-dimensional float32 embedding matrix")
    rows, dimensions = map(int, matrix.shape)
    if rows != expected_rows or dimensions < 1:
        raise CorpusBuildError(
            f"embedding matrix shape {matrix.shape} does not match {expected_rows} corpus rows"
        )
    declared = source_manifest.get("matrix")
    if not isinstance(declared, dict) or (
        declared.get("rows") != rows
        or declared.get("dimensions") != dimensions
        or declared.get("dtype") != "float32"
        or declared.get("l2_normalized") is not True
    ):
        raise CorpusBuildError("source model manifest does not match the float32 matrix")
    for offset in range(0, rows, 8192):
        block = np.asarray(matrix[offset : offset + 8192], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1)
        if np.any(~np.isfinite(block)) or np.any(~np.isfinite(norms)):
            raise CorpusBuildError("embedding matrix contains non-finite values")
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-5):
            raise CorpusBuildError("embedding matrix rows are not L2-normalized")
    return rows, dimensions


def _validate_reconciliation(
    corpus_rows: Sequence[Mapping[str, str]],
    embedded_rows: Sequence[Mapping[str, str]],
    image_rows: Sequence[Mapping[str, str]],
) -> None:
    if len(image_rows) != len(corpus_rows):
        raise CorpusBuildError("rebuilt image manifest row count does not match corpus rows")
    image_by_id = {str(row.get("artwork_id") or "").strip(): row for row in image_rows}
    if len(image_by_id) != len(image_rows):
        raise CorpusBuildError("rebuilt image manifest contains empty or duplicate artwork IDs")
    for corpus, embedded in zip(corpus_rows, embedded_rows, strict=True):
        artwork_id = str(corpus["artwork_id"]).strip()
        digest = str(embedded.get("input_sha256") or "").strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise CorpusBuildError(
                f"{artwork_id}: embedded input_sha256 is not a 64-character hex digest"
            )
        if str(corpus.get("image_sha256") or "").strip().lower() != digest:
            raise CorpusBuildError(
                f"{artwork_id}: rebuilt corpus image_sha256 does not match embedded input_sha256"
            )
        if str(corpus.get("visual_cluster_id") or "").strip() != f"sha256:{digest}":
            raise CorpusBuildError(
                f"{artwork_id}: rebuilt visual_cluster_id does not match embedded input_sha256"
            )
        image = image_by_id.get(artwork_id)
        if image is None or str(image.get("image_sha256") or "").strip().lower() != digest:
            raise CorpusBuildError(
                f"{artwork_id}: rebuilt image manifest hash does not match embedded input_sha256"
            )


def _artifact(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validated_faiss(path: Path, rows: int, dimensions: int) -> None:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CorpusBuildError("--copy-faiss requires faiss-cpu for validation") from exc
    try:
        index = faiss.read_index(str(path))
    except Exception as exc:  # pragma: no cover - depends on optional FAISS parser
        raise CorpusBuildError(f"could not read the source FAISS index: {path}") from exc
    if (
        index.d != dimensions
        or index.ntotal != rows
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
        or "IndexFlat" not in type(index).__name__
    ):
        raise CorpusBuildError("source FAISS index is not the aligned exact IndexFlatIP")


def _reject_overlapping_output(destination: Path, sources: Sequence[Path]) -> None:
    for source in sources:
        if destination == source or destination.is_relative_to(source):
            raise CorpusBuildError("output directory must not be inside an input artifact")


def repack_embedded_bundle(
    embedding_bundle: Path | str,
    corpus_dir: Path | str,
    output_dir: Path | str,
    *,
    copy_faiss: bool = False,
) -> dict[str, object]:
    """Publish existing vectors with corpus metadata reconciled to their input hashes."""

    source_root = Path(embedding_bundle).resolve()
    corpus_root = Path(corpus_dir).resolve()
    destination = Path(output_dir).resolve()
    if not source_root.is_dir() or not corpus_root.is_dir():
        raise CorpusBuildError("embedding bundle and rebuilt corpus must be directories")
    _reject_overlapping_output(destination, (source_root, corpus_root))
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise CorpusBuildError(f"output directory must be absent or empty: {destination}")

    source_manifest_path = source_root / "model-manifest.json"
    source_manifest = _read_manifest(source_manifest_path, EMBED_SCHEMA_VERSION)
    source_artifacts = _verify_artifacts(source_root, source_manifest)
    source_corpus_path = _require_declared(
        source_root, source_manifest, source_artifacts, "metadata"
    )
    embeddings_path = _require_declared(
        source_root, source_manifest, source_artifacts, "embeddings"
    )
    embedded_path = _require_declared(
        source_root, source_manifest, source_artifacts, "embeddedImages"
    )

    corpus_manifest_path = corpus_root / "build-manifest.json"
    corpus_manifest = _read_manifest(corpus_manifest_path, BUILD_SCHEMA_VERSION)
    corpus_artifacts = _verify_artifacts(corpus_root, corpus_manifest)
    rebuilt_corpus_path = _require_declared(
        corpus_root, corpus_manifest, corpus_artifacts, "metadata"
    )
    date_weights_path = _require_declared(
        corpus_root, corpus_manifest, corpus_artifacts, "dateWeights"
    )
    denominators_path = _require_declared(
        corpus_root, corpus_manifest, corpus_artifacts, "binDenominators"
    )
    images_path = _require_declared(
        corpus_root, corpus_manifest, corpus_artifacts, "imageManifest"
    )

    source_rows = _ordered_rows(
        source_corpus_path,
        required_fields=("artwork_id", "embedding_offset"),
        require_offsets=True,
    )
    embedded_rows = _ordered_rows(
        embedded_path,
        required_fields=("artwork_id", "embedding_offset", "input_sha256"),
        require_offsets=True,
    )
    rebuilt_rows = _ordered_rows(
        rebuilt_corpus_path,
        required_fields=(
            "artwork_id",
            "embedding_offset",
            "image_sha256",
            "visual_cluster_id",
        ),
        require_offsets=True,
    )
    image_rows = _ordered_rows(
        images_path,
        required_fields=("artwork_id", "image_sha256"),
        require_offsets=False,
    )
    ordered_id_sets = [
        [row["artwork_id"].strip() for row in rows]
        for rows in (source_rows, embedded_rows, rebuilt_rows)
    ]
    if ordered_id_sets[0] != ordered_id_sets[1] or ordered_id_sets[0] != ordered_id_sets[2]:
        raise CorpusBuildError(
            "source bundle, embedded provenance, and rebuilt corpus artwork order differ"
        )
    row_count = len(rebuilt_rows)
    if (
        row_count < 1
        or _manifest_count(source_manifest, "source model") != row_count
        or _manifest_count(corpus_manifest, "rebuilt corpus") != row_count
    ):
        raise CorpusBuildError("source and rebuilt corpus row counts do not match")
    matrix_rows, dimensions = _validate_matrix(
        embeddings_path, source_manifest, row_count
    )
    if matrix_rows != row_count:
        raise AssertionError("validated matrix row count changed")
    _validate_reconciliation(rebuilt_rows, embedded_rows, image_rows)

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - required by pipeline runtime
        raise CorpusBuildError("embedding repacking requires NumPy") from exc
    try:
        with np.load(date_weights_path, allow_pickle=False) as sparse:
            sparse_shape = tuple(map(int, sparse["shape"]))
    except (KeyError, OSError, ValueError) as exc:
        raise CorpusBuildError("rebuilt date weights are not a valid sparse NPZ") from exc
    if len(sparse_shape) != 2 or sparse_shape[0] != row_count:
        raise CorpusBuildError("rebuilt date-weight rows do not align with the corpus")

    source_faiss_path: Path | None = None
    if copy_faiss:
        source_faiss_path = _require_declared(
            source_root, source_manifest, source_artifacts, "faissIndex"
        )
        _validated_faiss(source_faiss_path, row_count, dimensions)

    model = source_manifest.get("model")
    corpus_identity = corpus_manifest.get("corpus")
    bins = corpus_manifest.get("bins")
    if not isinstance(model, dict) or not model.get("id") or not model.get("revision"):
        raise CorpusBuildError("source model manifest has an invalid model identity")
    if not isinstance(corpus_identity, dict) or not isinstance(bins, list):
        raise CorpusBuildError("rebuilt corpus manifest lacks corpus identity or bins")

    provenance_sources = sorted((corpus_root / "source-payloads").glob("*.json"))
    for path in provenance_sources:
        relative = path.relative_to(corpus_root).as_posix()
        if relative not in corpus_artifacts:
            raise CorpusBuildError(
                f"JSON source provenance is not covered by artifact checksums: {relative}"
            )
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusBuildError(f"JSON source provenance is invalid: {path}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    manifest: dict[str, object]
    try:
        copies = {
            "corpus.csv": rebuilt_corpus_path,
            "date-weights.npz": date_weights_path,
            "bin-denominators.csv": denominators_path,
            "images.manifest.csv": images_path,
            "corpus-build-manifest.json": corpus_manifest_path,
            "embeddings.npy": embeddings_path,
            "embedded-images.manifest.csv": embedded_path,
        }
        copied_paths: list[Path] = []
        for target_name, source_path in copies.items():
            target = staging / target_name
            shutil.copyfile(source_path, target)
            copied_paths.append(target)

        copied_provenance: list[Path] = []
        if provenance_sources:
            provenance_dir = staging / "source-provenance"
            provenance_dir.mkdir()
            for source_path in provenance_sources:
                target = provenance_dir / source_path.name
                shutil.copyfile(source_path, target)
                copied_provenance.append(target)
        copied_paths.extend(copied_provenance)

        copied_faiss: Path | None = None
        if source_faiss_path is not None:
            copied_faiss = staging / "index.faiss"
            shutil.copyfile(source_faiss_path, copied_faiss)
            copied_paths.append(copied_faiss)

        artifacts = sorted(
            (_artifact(staging, path) for path in copied_paths),
            key=lambda entry: str(entry["path"]),
        )
        manifest = {
            "schema_version": EMBED_SCHEMA_VERSION,
            "builder_version": __version__,
            "corpus": corpus_identity,
            "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
            "model": model,
            "matrix": {
                "rows": row_count,
                "dimensions": dimensions,
                "dtype": "float32",
                "l2_normalized": True,
                "row_order": "corpus.csv embedding_offset",
            },
            "index": {
                "metric": "inner-product-on-l2-normalized-vectors",
                "backend": (
                    "faiss-index-flat-ip" if copied_faiss is not None else "numpy-flat-ip"
                ),
                "exact": True,
                "numpy_fallback": "embeddings.npy",
                "faiss": "index.faiss" if copied_faiss is not None else None,
            },
            "files": {
                "metadata": "corpus.csv",
                "embeddings": "embeddings.npy",
                "dateWeights": "date-weights.npz",
                "binDenominators": "bin-denominators.csv",
                "imageManifest": "images.manifest.csv",
                "embeddedImages": "embedded-images.manifest.csv",
                "numpyIndex": "embeddings.npy",
                "faissIndex": "index.faiss" if copied_faiss is not None else None,
                "sourceProvenance": [
                    path.relative_to(staging).as_posix() for path in copied_provenance
                ],
            },
            "bins": bins,
            "repack": {
                "schema_version": REPACK_SCHEMA_VERSION,
                "source_model_manifest_sha256": sha256_file(source_manifest_path),
                "reconciled_fields": ["image_sha256", "visual_cluster_id"],
                "artwork_order_validation": "exact-id-and-offset-match",
            },
            "artifacts": artifacts,
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
