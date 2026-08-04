"""Offline, pluggable image embedding and exact-index construction."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Protocol, Sequence

from . import __version__
from .build import CorpusBuildError, sha256_file


EMBED_SCHEMA_VERSION = "mnemosyne-embedding-build/v1"


class ImageEncoder(Protocol):
    encoder_id: str
    revision: str
    dimension: int
    requires_local_images: bool

    def encode(
        self,
        records: Sequence[Mapping[str, str]],
        image_root: Path,
        batch_size: int,
    ): ...

    def settings(self) -> Mapping[str, object]: ...


class DeterministicTestEncoder:
    """Model-free encoder for fixtures only; never use it for product semantics."""

    encoder_id = "mnemosyne/deterministic-test-encoder"
    revision = "sha256-stream-v1"
    requires_local_images = False

    def __init__(self, dimension: int = 32) -> None:
        if dimension <= 0:
            raise ValueError("deterministic encoder dimension must be positive")
        self.dimension = dimension

    def encode(
        self,
        records: Sequence[Mapping[str, str]],
        image_root: Path,
        batch_size: int,
    ):
        del image_root, batch_size
        import numpy as np

        output = np.empty((len(records), self.dimension), dtype=np.float32)
        for row_index, record in enumerate(records):
            token = "\x1f".join(
                (
                    record.get("artwork_id", ""),
                    record.get("image_sha256", ""),
                    record.get("image_url", ""),
                    record.get("image_path", ""),
                )
            ).encode("utf-8")
            seed = hashlib.sha256(token).digest()
            values = bytearray()
            counter = 0
            while len(values) < self.dimension:
                values.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
                counter += 1
            output[row_index] = (
                np.frombuffer(bytes(values[: self.dimension]), dtype=np.uint8).astype(np.float32)
                - 127.5
            )
        return output

    def settings(self) -> Mapping[str, object]:
        return {"algorithm": self.revision, "fixture_only": True}


class Siglip2LocalEncoder:
    """Optional Transformers adapter for a pinned, locally cached SigLIP 2 model."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        allow_download: bool = False,
        device: str = "auto",
    ) -> None:
        if not revision or revision == "main":
            raise CorpusBuildError("SigLIP 2 builds require a pinned --model-revision, not 'main'")
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise CorpusBuildError(
                "SigLIP 2 encoding requires torch, transformers, and Pillow; "
                "install pipeline/requirements-embedding.txt"
            ) from exc
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=not allow_download,
        )
        self._model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=not allow_download,
        )
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device not in {"cpu", "cuda", "mps"}:
            raise CorpusBuildError("device must be auto, cpu, cuda, or mps")
        if device == "cuda" and not torch.cuda.is_available():
            raise CorpusBuildError("CUDA was requested but is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise CorpusBuildError("MPS was requested but is unavailable")
        self._device = device
        self._model.to(device)
        self._model.eval()
        self.encoder_id = model_id
        self.revision = revision
        projection_size = getattr(self._model.config, "projection_size", None)
        vision_config = getattr(self._model.config, "vision_config", None)
        self.dimension = int(
            projection_size
            or getattr(vision_config, "projection_size", 0)
            or getattr(vision_config, "hidden_size", 0)
        )
        if not self.dimension:
            raise CorpusBuildError("could not determine SigLIP 2 embedding dimension")
        self.requires_local_images = True

    def encode(
        self,
        records: Sequence[Mapping[str, str]],
        image_root: Path,
        batch_size: int,
    ):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise CorpusBuildError("SigLIP 2 encoding requires NumPy and Pillow") from exc

        batches = []
        for offset in range(0, len(records), batch_size):
            images = []
            for record in records[offset : offset + batch_size]:
                image_path = _resolve_image_path(record, image_root)
                _verify_image_hash(image_path, record.get("image_sha256", ""))
                with Image.open(image_path) as opened:
                    images.append(opened.convert("RGB"))
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
            with self._torch.inference_mode():
                features = self._model.get_image_features(**inputs)
            batches.append(features.detach().cpu().to(self._torch.float32).numpy())
        return np.concatenate(batches, axis=0) if batches else np.empty((0, self.dimension), dtype=np.float32)

    def settings(self) -> Mapping[str, object]:
        return {
            "processor_class": self._processor.__class__.__name__,
            "model_class": self._model.__class__.__name__,
            "local_weights": True,
            "device": self._device,
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise CorpusBuildError(f"required artifact is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_image_path(record: Mapping[str, str], image_root: Path) -> Path:
    raw_path = record.get("image_path", "").strip()
    if not raw_path:
        raise CorpusBuildError(f"{record.get('artwork_id', 'unknown')}: image_path is required")
    path = Path(raw_path)
    resolved = path if path.is_absolute() else image_root / path
    if not resolved.is_file():
        raise CorpusBuildError(
            f"{record.get('artwork_id', 'unknown')}: image does not exist: {resolved}"
        )
    return resolved


def _verify_image_hash(path: Path, expected: str) -> None:
    if expected and sha256_file(path) != expected.lower():
        raise CorpusBuildError(f"image checksum does not match manifest: {path}")


def _ordered_embedding_records(corpus_dir: Path, allow_unreviewed: bool) -> tuple[list[dict[str, str]], dict[str, object]]:
    corpus = _read_csv(corpus_dir / "corpus.csv")
    images = _read_csv(corpus_dir / "images.manifest.csv")
    image_by_id = {row["artwork_id"]: row for row in images}
    if len(image_by_id) != len(images):
        raise CorpusBuildError("images.manifest.csv contains duplicate artwork_id values")
    try:
        corpus.sort(key=lambda row: int(row["embedding_offset"]))
    except (KeyError, ValueError) as exc:
        raise CorpusBuildError("corpus.csv requires integer embedding_offset values") from exc
    if [int(row["embedding_offset"]) for row in corpus] != list(range(len(corpus))):
        raise CorpusBuildError("corpus embedding_offset values must be contiguous from zero")

    records: list[dict[str, str]] = []
    for canonical in corpus:
        artwork_id = canonical["artwork_id"]
        image = image_by_id.get(artwork_id)
        if image is None:
            raise CorpusBuildError(f"no image manifest row for {artwork_id}")
        if not image.get("image_url") and not image.get("image_path"):
            raise CorpusBuildError(f"{artwork_id}: no image URL or path")
        if image.get("permission_status") == "unreviewed" and not allow_unreviewed:
            raise CorpusBuildError(
                f"{artwork_id}: image rights are unreviewed; pass --allow-unreviewed-images "
                "only after an external rights review"
            )
        records.append({**canonical, **image})

    manifest_path = corpus_dir / "build-manifest.json"
    if not manifest_path.is_file():
        raise CorpusBuildError(f"required artifact is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = manifest.get("corpus", {}).get("count")
    if expected_count != len(records):
        raise CorpusBuildError(
            f"corpus manifest count {expected_count!r} does not match {len(records)} records"
        )
    return records, manifest


def _normalize(matrix):
    import numpy as np

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise CorpusBuildError(f"encoder returned shape {values.shape}; expected a matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(norms)):
        raise CorpusBuildError("encoder returned non-finite values")
    if np.any(norms == 0):
        raise CorpusBuildError("encoder returned a zero vector")
    return values / norms


def _artifact(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _embedded_image_rows(
    records: Sequence[Mapping[str, str]],
    image_root: Path,
    require_local_images: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, record in enumerate(records):
        raw_path = record.get("image_path", "").strip()
        actual_hash = ""
        resolved_path = ""
        if raw_path:
            candidate = Path(raw_path)
            resolved = candidate if candidate.is_absolute() else image_root / candidate
            if resolved.is_file():
                resolved_path = str(resolved)
                actual_hash = sha256_file(resolved)
                declared_hash = record.get("image_sha256", "").lower()
                if declared_hash and declared_hash != actual_hash:
                    raise CorpusBuildError(f"image checksum does not match manifest: {resolved}")
            elif require_local_images:
                raise CorpusBuildError(
                    f"{record.get('artwork_id', 'unknown')}: image does not exist: {resolved}"
                )
        elif require_local_images:
            raise CorpusBuildError(f"{record.get('artwork_id', 'unknown')}: image_path is required")
        rows.append(
            {
                "embedding_offset": offset,
                "artwork_id": record["artwork_id"],
                "image_path": resolved_path,
                "image_url": record.get("image_url", ""),
                "declared_image_sha256": record.get("image_sha256", "").lower(),
                "input_sha256": actual_hash,
                "permission_status": record.get("permission_status", ""),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "embedding_offset",
        "artwork_id",
        "image_path",
        "image_url",
        "declared_image_sha256",
        "input_sha256",
        "permission_status",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_embedding_index(
    corpus_dir: Path | str,
    output_dir: Path | str,
    encoder: ImageEncoder,
    *,
    image_root: Path | str | None = None,
    dtype: str = "float32",
    batch_size: int = 16,
    allow_unreviewed_images: bool = False,
) -> dict[str, object]:
    """Embed every canonical row in order and build an exact inner-product index."""

    import numpy as np

    corpus_root = Path(corpus_dir).resolve()
    destination = Path(output_dir).resolve()
    images_root = Path(image_root).resolve() if image_root else corpus_root
    if dtype not in {"float16", "float32"}:
        raise CorpusBuildError("dtype must be float16 or float32")
    if batch_size <= 0:
        raise CorpusBuildError("batch_size must be positive")
    if destination.exists() and any(destination.iterdir()):
        raise CorpusBuildError(f"output directory must be absent or empty: {destination}")

    records, corpus_manifest = _ordered_embedding_records(corpus_root, allow_unreviewed_images)
    embedded_image_rows = _embedded_image_rows(
        records, images_root, encoder.requires_local_images
    )
    matrix = _normalize(encoder.encode(records, images_root, batch_size))
    if matrix.shape != (len(records), encoder.dimension):
        raise CorpusBuildError(
            f"encoder returned shape {matrix.shape}; expected {(len(records), encoder.dimension)}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mnemosyne-embed-", dir=destination))
    try:
        deployment_copies = {
            "corpus.csv": corpus_root / "corpus.csv",
            "date-weights.npz": corpus_root / "date-weights.npz",
            "bin-denominators.csv": corpus_root / "bin-denominators.csv",
            "images.manifest.csv": corpus_root / "images.manifest.csv",
            "corpus-build-manifest.json": corpus_root / "build-manifest.json",
        }
        for target_name, source_path in deployment_copies.items():
            if not source_path.is_file():
                raise CorpusBuildError(f"required artifact is missing: {source_path}")
            shutil.copyfile(source_path, staging / target_name)

        embedded_images_path = staging / "embedded-images.manifest.csv"
        _write_csv(embedded_images_path, embedded_image_rows)

        storage_matrix = matrix.astype(np.float16 if dtype == "float16" else np.float32)
        raw_path = staging / f"embeddings.{dtype.replace('float', 'f')}"
        storage_matrix.tofile(raw_path)
        npy_path = staging / "embeddings.npy"
        np.save(npy_path, storage_matrix, allow_pickle=False)
        fallback_path = staging / "index-flat-ip.npz"
        np.savez_compressed(
            fallback_path,
            embeddings=matrix.astype(np.float32),
            artwork_ids=np.asarray([record["artwork_id"] for record in records]),
        )

        index_backend = "numpy-flat-ip"
        faiss_path = staging / "index.faiss"
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError:
            wrote_faiss = False
        else:
            index = faiss.IndexFlatIP(encoder.dimension)
            index.add(matrix.astype(np.float32))
            faiss.write_index(index, str(faiss_path))
            wrote_faiss = True
            index_backend = "faiss-index-flat-ip"

        artifact_paths = [
            raw_path,
            npy_path,
            fallback_path,
            embedded_images_path,
            *(staging / name for name in deployment_copies),
        ]
        if wrote_faiss:
            artifact_paths.append(faiss_path)
        artifact_entries = sorted(
            (_artifact(staging, path) for path in artifact_paths),
            key=lambda entry: str(entry["path"]),
        )
        corpus_identity = corpus_manifest["corpus"]
        manifest: dict[str, object] = {
            "schema_version": EMBED_SCHEMA_VERSION,
            "builder_version": __version__,
            "corpus": corpus_identity,
            "corpus_manifest_sha256": sha256_file(corpus_root / "build-manifest.json"),
            "model": {
                "id": encoder.encoder_id,
                "revision": encoder.revision,
                "settings": dict(encoder.settings()),
            },
            "matrix": {
                "rows": matrix.shape[0],
                "dimensions": matrix.shape[1],
                "dtype": dtype,
                "l2_normalized": True,
                "row_order": "corpus.csv embedding_offset",
            },
            "index": {
                "metric": "inner-product-on-l2-normalized-vectors",
                "backend": index_backend,
                "exact": True,
                "numpy_fallback": "index-flat-ip.npz",
                "faiss": "index.faiss" if wrote_faiss else None,
            },
            "files": {
                "metadata": "corpus.csv",
                "embeddings": "embeddings.npy",
                "dateWeights": "date-weights.npz",
                "binDenominators": "bin-denominators.csv",
                "imageManifest": "images.manifest.csv",
                "embeddedImages": "embedded-images.manifest.csv",
                "numpyIndex": "index-flat-ip.npz",
                "faissIndex": "index.faiss" if wrote_faiss else None,
            },
            "bins": corpus_manifest.get("bins", []),
            "artifacts": artifact_entries,
        }
        manifest_path = staging / "model-manifest.json"
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
