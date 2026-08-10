"""Offline, pluggable image embedding and exact-index construction."""

from __future__ import annotations

import csv
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
from io import BytesIO
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import shutil
import ssl
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from . import __version__
from .build import CorpusBuildError, sha256_file


EMBED_SCHEMA_VERSION = "mnemosyne-embedding-build/v1"
SIGLIP_IMAGE_INPUT_POLICY = "remote-original-met-web-large-min-side-224/v2"
DEFAULT_ALLOWED_IMAGE_HOSTS = ("images.metmuseum.org",)


class _ValidatedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, validate_url: Callable[[str], None]) -> None:
        super().__init__()
        self._validate_url = validate_url

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        self._validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _package_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:  # pragma: no cover - required in production installs
        return "unknown"


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
    """Pinned SigLIP 2 adapter with local or ephemeral streamed image inputs."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        allow_download: bool = False,
        device: str = "auto",
        download_workers: int = 8,
        request_timeout: float = 30,
        fetch_retries: int = 2,
        max_image_bytes: int = 64 * 1024 * 1024,
        max_image_pixels: int = 100_000_000,
        allowed_image_hosts: Sequence[str] = DEFAULT_ALLOWED_IMAGE_HOSTS,
        checkpoint_dir: Path | str | None = None,
        progress: Callable[[int, int], None] | None = None,
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
            use_fast=True,
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
        if download_workers < 1 or request_timeout <= 0 or fetch_retries < 0:
            raise CorpusBuildError(
                "stream workers/timeout must be positive and retries non-negative"
            )
        if max_image_bytes < 1024 or max_image_pixels < 1:
            raise CorpusBuildError("image byte and pixel limits must be positive")
        normalized_hosts = tuple(
            sorted({host.strip().lower().rstrip(".") for host in allowed_image_hosts if host.strip()})
        )
        if not normalized_hosts:
            raise CorpusBuildError("at least one allowed image host is required")
        self._device = device
        self._download_workers = download_workers
        self._request_timeout = request_timeout
        self._fetch_retries = fetch_retries
        self._max_image_bytes = max_image_bytes
        self._max_image_pixels = max_image_pixels
        self._allowed_image_hosts = normalized_hosts
        self._checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self._progress = progress
        self._input_provenance: tuple[dict[str, str], ...] = ()
        try:
            import certifi
        except ImportError:  # pragma: no cover - normally supplied by Transformers
            self._ssl_context = ssl.create_default_context()
        else:
            self._ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._url_opener = build_opener(
            HTTPSHandler(context=self._ssl_context),
            _ValidatedRedirectHandler(self._validate_remote_url),
        )
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
        self.requires_local_images = False

    def _validate_remote_url(self, url: str) -> None:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https":
            raise CorpusBuildError("remote image URL must use https")
        if parsed.username or parsed.password or parsed.port not in {None, 443}:
            raise CorpusBuildError("remote image URL must not contain credentials or a custom port")
        if hostname not in self._allowed_image_hosts:
            raise CorpusBuildError(f"remote image host is not allowed: {hostname or '<missing>'}")

    def _resolved_image_url(self, url: str, *, prefer_web_large: bool) -> str:
        self._validate_remote_url(url)
        parsed = urlsplit(url)
        if (
            prefer_web_large
            and parsed.hostname == "images.metmuseum.org"
            and "/original/" in parsed.path
        ):
            parsed = parsed._replace(path=parsed.path.replace("/original/", "/web-large/", 1))
        parsed = parsed._replace(path=quote(parsed.path, safe="/%:@"))
        return urlunsplit(parsed)

    def _download(
        self, raw_url: str, *, prefer_web_large: bool = True
    ) -> tuple[bytes, str]:
        url = self._resolved_image_url(raw_url, prefer_web_large=prefer_web_large)
        request = Request(
            url,
            headers={
                "Accept": "image/avif,image/webp,image/jpeg,image/png,image/*;q=0.8",
                "User-Agent": "Mnemosyne embedding corpus builder",
            },
        )
        for attempt in range(self._fetch_retries + 1):
            try:
                with self._url_opener.open(request, timeout=self._request_timeout) as response:
                    self._validate_remote_url(response.geturl())
                    resolved_url = response.geturl()
                    declared = response.headers.get("Content-Length")
                    if declared and int(declared) > self._max_image_bytes:
                        raise CorpusBuildError(f"remote image exceeds byte limit: {url}")
                    content_type = response.headers.get("Content-Type", "")
                    if content_type and not content_type.lower().startswith("image/"):
                        raise CorpusBuildError(
                            f"remote input is not an image ({content_type}): {url}"
                        )
                    payload = response.read(self._max_image_bytes + 1)
                if not payload or len(payload) > self._max_image_bytes:
                    raise CorpusBuildError(f"remote image is empty or exceeds byte limit: {url}")
                return payload, resolved_url
            except HTTPError as exc:
                retryable = exc.code in {408, 425, 429} or 500 <= exc.code < 600
                if not retryable or attempt >= self._fetch_retries:
                    raise CorpusBuildError(
                        f"image fetch failed with HTTP {exc.code}: {url}"
                    ) from exc
            except (OSError, TimeoutError) as exc:
                if attempt >= self._fetch_retries:
                    raise CorpusBuildError(f"image fetch failed: {url}: {exc}") from exc
            time.sleep(0.25 * (2**attempt))
        raise AssertionError("unreachable image retry state")

    def _decode_image(self, payload: bytes, input_source: str, artwork_id: str):
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise CorpusBuildError("SigLIP 2 encoding requires Pillow") from exc

        try:
            with Image.open(BytesIO(payload)) as opened:
                width, height = opened.size
                if width < 1 or height < 1 or width * height > self._max_image_pixels:
                    raise CorpusBuildError(
                        f"{artwork_id}: decoded image exceeds pixel limit: {input_source}"
                    )
                opened.load()
                return ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as exc:
            raise CorpusBuildError(
                f"{artwork_id}: image decode failed: {input_source}"
            ) from exc

    def _load_image(
        self,
        record: Mapping[str, str],
        image_root: Path,
    ):
        artwork_id = record.get("artwork_id", "unknown")

        raw_path = record.get("image_path", "").strip()
        if raw_path:
            candidate = Path(raw_path)
            source = candidate if candidate.is_absolute() else image_root / candidate
            if not source.is_file():
                raise CorpusBuildError(
                    f"{record.get('artwork_id', 'unknown')}: image does not exist: {source}"
                )
            if source.stat().st_size > self._max_image_bytes:
                raise CorpusBuildError(f"{artwork_id}: local image exceeds byte limit: {source}")
            with source.open("rb") as handle:
                payload = handle.read(self._max_image_bytes + 1)
            if not payload or len(payload) > self._max_image_bytes:
                raise CorpusBuildError(
                    f"{artwork_id}: local image is empty or exceeds byte limit: {source}"
                )
            input_kind = "local-file"
            input_source = str(source.resolve())
            input_policy = "declared-local-file"
            image = self._decode_image(payload, input_source, artwork_id)
        else:
            raw_url = record.get("image_url", "").strip()
            if not raw_url:
                raise CorpusBuildError(
                    f"{record.get('artwork_id', 'unknown')}: image URL or path is required"
                )
            payload, input_source = self._download(raw_url, prefer_web_large=True)
            input_kind = "remote-stream"
            input_policy = SIGLIP_IMAGE_INPUT_POLICY
            image = self._decode_image(payload, input_source, artwork_id)
            original_url = self._resolved_image_url(raw_url, prefer_web_large=False)
            if input_source != original_url and min(image.size) < 224:
                try:
                    original_payload, original_source = self._download(
                        raw_url, prefer_web_large=False
                    )
                    original_image = self._decode_image(
                        original_payload, original_source, artwork_id
                    )
                except Exception:
                    image.close()
                    raise
                else:
                    image.close()
                    image = original_image
                    payload = original_payload
                    input_source = original_source
                    input_kind = "remote-stream-original"
        width, height = image.size
        return image, {
            "artwork_id": record.get("artwork_id", ""),
            "input_kind": input_kind,
            "input_source": input_source,
            "input_sha256": hashlib.sha256(payload).hexdigest(),
            "input_width": str(width),
            "input_height": str(height),
            "input_policy": input_policy,
        }

    def encode(
        self,
        records: Sequence[Mapping[str, str]],
        image_root: Path,
        batch_size: int,
    ):
        try:
            import numpy as np
        except ImportError as exc:
            raise CorpusBuildError("SigLIP 2 encoding requires NumPy and Pillow") from exc

        fingerprint = hashlib.sha256()
        fingerprint.update(
            json.dumps(
                {
                    "encoder_id": self.encoder_id,
                    "revision": self.revision,
                    "dimension": self.dimension,
                    "device": str(self._device),
                    "image_input_policy": SIGLIP_IMAGE_INPUT_POLICY,
                    "processor_class": self._processor.__class__.__qualname__,
                    "processor_config": getattr(self._processor, "to_dict", lambda: {})(),
                    "image_root": str(image_root.resolve()),
                    "runtime_versions": self._runtime_versions(),
                },
                default=str,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for record in records:
            fingerprint.update(
                "\x1f".join(
                    (
                        record.get("artwork_id", ""),
                        record.get("image_url", ""),
                        record.get("image_path", ""),
                        record.get("image_sha256", ""),
                    )
                ).encode("utf-8")
            )
            fingerprint.update(b"\n")
        build_fingerprint = fingerprint.hexdigest()

        provenance_fields = (
            "artwork_id",
            "input_kind",
            "input_source",
            "input_sha256",
            "input_width",
            "input_height",
            "input_policy",
        )
        provenance: list[dict[str, str]] = []
        completed = 0
        checkpoint_state_path: Path | None = None
        checkpoint_provenance_path: Path | None = None
        if self._checkpoint_dir is None:
            output = np.empty((len(records), self.dimension), dtype=np.float32)
        else:
            checkpoint = self._checkpoint_dir
            checkpoint.mkdir(parents=True, exist_ok=True)
            checkpoint_state_path = checkpoint / "state.json"
            checkpoint_matrix_path = checkpoint / "embeddings.work.npy"
            checkpoint_provenance_path = checkpoint / "input-provenance.csv"
            if checkpoint_state_path.is_file():
                state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
                expected = {
                    "fingerprint": build_fingerprint,
                    "rows": len(records),
                    "dimensions": self.dimension,
                }
                if any(state.get(key) != value for key, value in expected.items()):
                    raise CorpusBuildError(
                        f"embedding checkpoint does not match this build: {checkpoint}"
                    )
                completed = int(state.get("completed", -1))
                if not 0 <= completed <= len(records):
                    raise CorpusBuildError("embedding checkpoint has an invalid completed offset")
                output = np.lib.format.open_memmap(checkpoint_matrix_path, mode="r+")
                if output.shape != (len(records), self.dimension) or output.dtype != np.float32:
                    raise CorpusBuildError("embedding checkpoint matrix shape or dtype is invalid")
                if checkpoint_provenance_path.is_file():
                    with checkpoint_provenance_path.open(
                        encoding="utf-8", newline=""
                    ) as handle:
                        provenance = list(csv.DictReader(handle))[:completed]
                if len(provenance) != completed:
                    raise CorpusBuildError("embedding checkpoint provenance is incomplete")
                # A crash can append provenance before advancing state. Rewrite
                # only committed rows so resuming cannot duplicate entries.
                with checkpoint_provenance_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=provenance_fields, lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(provenance)
            else:
                unexpected = list(checkpoint.iterdir())
                if unexpected:
                    raise CorpusBuildError(
                        f"embedding checkpoint is incomplete; move or remove it: {checkpoint}"
                    )
                output = np.lib.format.open_memmap(
                    checkpoint_matrix_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(records), self.dimension),
                )
                with checkpoint_provenance_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=provenance_fields, lineterminator="\n"
                    )
                    writer.writeheader()
                self._write_checkpoint_state(
                    checkpoint_state_path,
                    build_fingerprint,
                    len(records),
                    completed,
                )
            if completed and self._progress:
                self._progress(completed, len(records))

        def submit_batch(
            executor: ThreadPoolExecutor, offset: int
        ) -> list[Future[tuple[object, dict[str, str]]]]:
            return [
                executor.submit(self._load_image, record, image_root)
                for record in records[offset : offset + batch_size]
            ]

        def collect_batch(futures):
            loaded = []
            try:
                for future in futures:
                    loaded.append(future.result())
            except Exception:
                for image, _source in loaded:
                    image.close()
                for future in futures[len(loaded) :]:
                    try:
                        image, _source = future.result()
                    except Exception:
                        continue
                    image.close()
                raise
            return loaded

        def close_batch(futures) -> None:
            for future in futures:
                try:
                    image, _source = future.result()
                except Exception:
                    continue
                image.close()

        with ThreadPoolExecutor(
            max_workers=min(self._download_workers, batch_size),
            thread_name_prefix="siglip-image",
        ) as executor:
            pending_offset = completed
            pending = submit_batch(executor, completed) if completed < len(records) else []
            try:
                while pending:
                    offset = pending_offset
                    loaded = collect_batch(pending)
                    next_offset = offset + len(loaded)
                    next_pending = (
                        submit_batch(executor, next_offset) if next_offset < len(records) else []
                    )
                    pending = []
                    pending_offset = next_offset
                    images = [item[0] for item in loaded]
                    batch_provenance = [item[1] for item in loaded]
                    try:
                        inputs = self._processor(images=images, return_tensors="pt")
                        inputs = {
                            name: tensor.to(self._device) for name, tensor in inputs.items()
                        }
                        with self._torch.inference_mode():
                            features = self._model.get_image_features(**inputs)
                        feature_batch = features.detach().cpu().to(self._torch.float32).numpy()
                        expected_shape = (len(images), self.dimension)
                        if feature_batch.shape != expected_shape:
                            raise CorpusBuildError(
                                f"encoder returned batch shape {feature_batch.shape}; "
                                f"expected {expected_shape}"
                            )
                        output[offset : offset + len(feature_batch)] = feature_batch
                        provenance.extend(batch_provenance)
                        completed = offset + len(feature_batch)
                        if checkpoint_state_path is not None:
                            output.flush()
                            assert checkpoint_provenance_path is not None
                            with checkpoint_provenance_path.open(
                                "a", encoding="utf-8", newline=""
                            ) as handle:
                                writer = csv.DictWriter(
                                    handle, fieldnames=provenance_fields, lineterminator="\n"
                                )
                                writer.writerows(batch_provenance)
                                handle.flush()
                            self._write_checkpoint_state(
                                checkpoint_state_path,
                                build_fingerprint,
                                len(records),
                                completed,
                            )
                        if self._progress:
                            self._progress(completed, len(records))
                    except Exception:
                        # The next batch may already have decoded while MPS was
                        # running. Drain it so no PIL objects survive an abort.
                        close_batch(next_pending)
                        raise
                    finally:
                        for image in images:
                            image.close()
                    pending = next_pending
            finally:
                if pending:
                    close_batch(pending)
        self._input_provenance = tuple(provenance)
        return output

    @property
    def input_provenance(self) -> tuple[dict[str, str], ...]:
        return self._input_provenance

    def _write_checkpoint_state(
        self,
        path: Path,
        fingerprint: str,
        rows: int,
        completed: int,
    ) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": "mnemosyne-embedding-checkpoint/v1",
                    "fingerprint": fingerprint,
                    "rows": rows,
                    "dimensions": self.dimension,
                    "completed": completed,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def cleanup_checkpoint(self) -> None:
        if self._checkpoint_dir is not None and self._checkpoint_dir.exists():
            for name in (
                "state.json",
                "state.tmp",
                "embeddings.work.npy",
                "input-provenance.csv",
            ):
                (self._checkpoint_dir / name).unlink(missing_ok=True)
            try:
                self._checkpoint_dir.rmdir()
            except OSError:
                # Never recursively delete a user-selected directory. Unknown
                # entries remain untouched for explicit inspection.
                return

    def settings(self) -> Mapping[str, object]:
        return {
            "processor_class": self._processor.__class__.__name__,
            "model_class": self._model.__class__.__name__,
            "local_weights": True,
            "device": self._device,
            "image_input": "local-file-or-ephemeral-remote-stream",
            "image_input_policy": SIGLIP_IMAGE_INPUT_POLICY,
            "download_workers": self._download_workers,
            "request_timeout_seconds": self._request_timeout,
            "fetch_retries": self._fetch_retries,
            "max_image_bytes": self._max_image_bytes,
            "max_image_pixels": self._max_image_pixels,
            "allowed_image_hosts": list(self._allowed_image_hosts),
            "runtime_versions": self._runtime_versions(),
            "checkpointed": self._checkpoint_dir is not None,
        }

    @staticmethod
    def _runtime_versions() -> Mapping[str, str]:
        return {
            "numpy": _package_version("numpy"),
            "pillow": _package_version("Pillow"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise CorpusBuildError(f"required artifact is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ordered_embedding_records(
    corpus_dir: Path, allow_unreviewed: bool
) -> tuple[list[dict[str, str]], dict[str, object]]:
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
    np.divide(values, norms, out=values)
    return values


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
                "input_kind": "local-file" if resolved_path else "",
                "input_source": resolved_path,
                "input_width": record.get("image_width", ""),
                "input_height": record.get("image_height", ""),
                "input_policy": "declared-local-file" if resolved_path else "",
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
        "input_kind",
        "input_source",
        "input_width",
        "input_height",
        "input_policy",
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
    write_faiss: bool = True,
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
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise CorpusBuildError(f"output directory must be absent or empty: {destination}")

    records, corpus_manifest = _ordered_embedding_records(corpus_root, allow_unreviewed_images)
    embedded_image_rows = _embedded_image_rows(records, images_root, encoder.requires_local_images)
    matrix = _normalize(encoder.encode(records, images_root, batch_size))
    provenance = getattr(encoder, "input_provenance", ())
    if provenance:
        if len(provenance) != len(records):
            raise CorpusBuildError("encoder input provenance does not align with corpus rows")
        for record, embedded, source in zip(
            records, embedded_image_rows, provenance, strict=True
        ):
            if source.get("artwork_id") != record["artwork_id"]:
                raise CorpusBuildError("encoder input provenance artwork order is inconsistent")
            embedded.update(
                {
                    "input_sha256": source.get("input_sha256", ""),
                    "input_kind": source.get("input_kind", ""),
                    "input_source": source.get("input_source", ""),
                    "input_width": source.get("input_width", ""),
                    "input_height": source.get("input_height", ""),
                    "input_policy": source.get("input_policy", ""),
                }
            )
    if matrix.shape != (len(records), encoder.dimension):
        raise CorpusBuildError(
            f"encoder returned shape {matrix.shape}; expected {(len(records), encoder.dimension)}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
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

        source_provenance_paths: list[Path] = []
        source_payload_dir = corpus_root / "source-payloads"
        if source_payload_dir.is_dir():
            provenance_sources = sorted(source_payload_dir.glob("*.json"))
            if provenance_sources:
                provenance_dir = staging / "source-provenance"
                provenance_dir.mkdir()
                for source_path in provenance_sources:
                    target_path = provenance_dir / source_path.name
                    shutil.copyfile(source_path, target_path)
                    source_provenance_paths.append(target_path)

        embedded_images_path = staging / "embedded-images.manifest.csv"
        _write_csv(embedded_images_path, embedded_image_rows)

        storage_matrix = matrix.astype(
            np.float16 if dtype == "float16" else np.float32, copy=False
        )
        # All retrieval backends must index the exact values the service will
        # load from embeddings.npy. Building FAISS from the pre-quantized
        # matrix makes nominally exact backends disagree for close neighbors.
        deployed_matrix = np.asarray(storage_matrix, dtype=np.float32)
        npy_path = staging / "embeddings.npy"
        np.save(npy_path, storage_matrix, allow_pickle=False)

        index_backend = "numpy-flat-ip"
        faiss_path = staging / "index.faiss"
        if not write_faiss:
            wrote_faiss = False
        else:
            try:
                import faiss  # type: ignore[import-not-found]
            except ImportError:
                wrote_faiss = False
            else:
                index = faiss.IndexFlatIP(encoder.dimension)
                index.add(np.ascontiguousarray(deployed_matrix, dtype=np.float32))
                faiss.write_index(index, str(faiss_path))
                wrote_faiss = True
                index_backend = "faiss-index-flat-ip"

        artifact_paths = [
            npy_path,
            embedded_images_path,
            *(staging / name for name in deployment_copies),
            *source_provenance_paths,
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
                "numpy_fallback": "embeddings.npy",
                "faiss": "index.faiss" if wrote_faiss else None,
            },
            "files": {
                "metadata": "corpus.csv",
                "embeddings": "embeddings.npy",
                "dateWeights": "date-weights.npz",
                "binDenominators": "bin-denominators.csv",
                "imageManifest": "images.manifest.csv",
                "embeddedImages": "embedded-images.manifest.csv",
                "numpyIndex": "embeddings.npy",
                "faissIndex": "index.faiss" if wrote_faiss else None,
                "sourceProvenance": [
                    path.relative_to(staging).as_posix() for path in source_provenance_paths
                ],
            },
            "bins": corpus_manifest.get("bins", []),
            "artifacts": artifact_entries,
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
    cleanup_checkpoint = getattr(encoder, "cleanup_checkpoint", None)
    if callable(cleanup_checkpoint):
        cleanup_checkpoint()
    return manifest
