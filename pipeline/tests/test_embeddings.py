from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.build import CorpusBuildError, build_corpus
from pipeline.embeddings import (
    AIC_IMAGE_INPUT_POLICY,
    DECLARED_REMOTE_IMAGE_INPUT_POLICY,
    DeterministicTestEncoder,
    SIGLIP_IMAGE_INPUT_POLICY,
    Siglip2LocalEncoder,
    build_embedding_index,
)


class EmbeddingBuildTests(unittest.TestCase):
    def test_stream_encoder_rejects_nonfinite_request_delays_before_model_load(self) -> None:
        for delay in (float("nan"), float("inf"), -1.0):
            with self.subTest(delay=delay), self.assertRaisesRegex(
                CorpusBuildError, "delay/retries non-negative"
            ):
                Siglip2LocalEncoder(
                    "model-that-must-not-load",
                    "pinned-revision",
                    request_delay_seconds=delay,
                )

    def _build_corpus(
        self, root: Path, extra_source_payloads: tuple[Path, ...] = ()
    ) -> Path:
        source = root / "ArtiFact_clean.csv"
        fields = (
            "object_ID",
            "title",
            "date_begin",
            "date_end",
            "date_begin_bce",
            "date_end_bce",
            "image_url",
            "public_domain",
            "image_rights_uri",
        )
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "object_ID": "MET_20",
                        "title": "Twenty",
                        "date_begin": "1900",
                        "date_end": "1900",
                        "date_begin_bce": "false",
                        "date_end_bce": "false",
                        "image_url": "https://example.test/20.jpg",
                        "public_domain": "true",
                        "image_rights_uri": "https://creativecommons.org/publicdomain/zero/1.0/",
                    },
                    {
                        "object_ID": "AIC_10",
                        "title": "Ten",
                        "date_begin": "1800",
                        "date_end": "1800",
                        "date_begin_bce": "false",
                        "date_end_bce": "false",
                        "image_url": "https://example.test/10.jpg",
                        "public_domain": "true",
                        "image_rights_uri": "https://creativecommons.org/publicdomain/zero/1.0/",
                    },
                ]
            )
        corpus_dir = root / "corpus"
        build_corpus(
            source,
            corpus_dir,
            corpus_version="fixture-v1",
            source_revision="deadbeef",
            retrieved_at="2026-08-03T00:00:00Z",
            source_payloads=(source, *extra_source_payloads),
        )
        return corpus_dir

    def test_embedding_carries_json_and_availability_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "source.manifest.json"
            audit.write_text('{"rights_gate":"fixture"}\n', encoding="utf-8")
            availability = root / "source.availability.csv"
            availability.write_text(
                "artwork_id,image_url,available,reason\n"
                "AIC_10,https://example.test/10.jpg,true,\n",
                encoding="utf-8",
            )
            corpus_dir = self._build_corpus(root, (audit, availability))
            output = root / "embeddings"

            manifest = build_embedding_index(
                corpus_dir,
                output,
                DeterministicTestEncoder(8),
                write_faiss=False,
            )

            self.assertEqual(
                manifest["files"]["sourceProvenance"],
                [
                    "source-provenance/source.availability.csv",
                    "source-provenance/source.manifest.json",
                ],
            )
            for relative in manifest["files"]["sourceProvenance"]:
                self.assertTrue((output / relative).is_file())
            artifact_paths = {entry["path"] for entry in manifest["artifacts"]}
            self.assertIn(
                "source-provenance/source.availability.csv", artifact_paths
            )

    def test_embedding_rejects_permitted_status_without_image_rights_uri(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._build_corpus(root)
            image_manifest = corpus_dir / "images.manifest.csv"
            with image_manifest.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = tuple(reader.fieldnames or ())
            rows[0]["image_rights_uri"] = ""
            with image_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(
                CorpusBuildError,
                "image rights differ between corpus and image manifest",
            ):
                build_embedding_index(
                    corpus_dir,
                    root / "embeddings",
                    DeterministicTestEncoder(8),
                )

    def test_embedding_fails_closed_when_all_rights_fields_are_blank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._build_corpus(root)
            for name in ("corpus.csv", "images.manifest.csv"):
                path = corpus_dir / name
                with path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                    fields = tuple(reader.fieldnames or ())
                rows[0]["image_rights_uri"] = ""
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=fields, lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(rows)

            with self.assertRaisesRegex(
                CorpusBuildError,
                "image rights are unreviewed or missing",
            ):
                build_embedding_index(
                    corpus_dir,
                    root / "embeddings",
                    DeterministicTestEncoder(8),
                )

    def test_deterministic_encoder_is_normalized_and_in_corpus_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._build_corpus(root)
            output = root / "embeddings"
            manifest = build_embedding_index(
                corpus_dir,
                output,
                DeterministicTestEncoder(16),
            )
            matrix = np.load(output / "embeddings.npy", allow_pickle=False)
            self.assertEqual(matrix.shape, (2, 16))
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(2), rtol=1e-6)
            self.assertEqual(matrix.dtype, np.float32)
            with (output / "corpus.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(
                    [row["artwork_id"] for row in csv.DictReader(handle)],
                    ["AIC_10", "MET_20"],
                )
            self.assertEqual(manifest["corpus"]["id"], "fixture-v1")
            self.assertEqual(manifest["index"]["metric"], "inner-product-on-l2-normalized-vectors")
            self.assertEqual(manifest["files"]["metadata"], "corpus.csv")
            self.assertEqual(manifest["files"]["embeddings"], "embeddings.npy")
            self.assertTrue((output / "embedded-images.manifest.csv").is_file())

    def test_embedding_manifest_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._build_corpus(root)
            manifests = []
            for suffix in ("one", "two"):
                output = root / suffix
                build_embedding_index(corpus_dir, output, DeterministicTestEncoder(8))
                manifests.append((output / "model-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifests[0], manifests[1])
            parsed = json.loads(manifests[0])
            self.assertTrue(parsed["model"]["settings"]["fixture_only"])

    def test_failed_publication_does_not_normalize_resumable_checkpoint(self) -> None:
        class RawCheckpointEncoder:
            encoder_id = "fixture/raw-checkpoint-encoder"
            revision = "raw-v1"
            dimension = 2
            requires_local_images = False

            def __init__(self, checkpoint: Path) -> None:
                self.checkpoint = checkpoint

            def encode(self, records, image_root, batch_size):
                del image_root, batch_size
                self.checkpoint.mkdir(parents=True, exist_ok=True)
                matrix_path = self.checkpoint / "embeddings.work.npy"
                if matrix_path.is_file():
                    return np.lib.format.open_memmap(matrix_path, mode="r+")
                matrix = np.lib.format.open_memmap(
                    matrix_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(records), self.dimension),
                )
                matrix[:] = np.asarray([[3, 4], [5, 12]], dtype=np.float32)
                matrix.flush()
                return matrix

            def cleanup_checkpoint(self) -> None:
                (self.checkpoint / "embeddings.work.npy").unlink(missing_ok=True)
                self.checkpoint.rmdir()

            def settings(self):
                return {"fixture_only": True, "checkpointed": True}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._build_corpus(root)
            checkpoint = root / "retry-checkpoint"
            retry_encoder = RawCheckpointEncoder(checkpoint)

            with patch(
                "pipeline.embeddings._write_csv",
                side_effect=RuntimeError("failure after normalization"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "failure after normalization"
                ):
                    build_embedding_index(
                        corpus_dir,
                        root / "failed-output",
                        retry_encoder,
                        write_faiss=False,
                    )

            raw_checkpoint = np.load(
                checkpoint / "embeddings.work.npy",
                allow_pickle=False,
            )
            np.testing.assert_array_equal(
                raw_checkpoint,
                np.asarray([[3, 4], [5, 12]], dtype=np.float32),
            )

            retry_output = root / "retry-output"
            build_embedding_index(
                corpus_dir,
                retry_output,
                retry_encoder,
                write_faiss=False,
            )
            self.assertFalse(checkpoint.exists())

            clean_output = root / "clean-output"
            build_embedding_index(
                corpus_dir,
                clean_output,
                RawCheckpointEncoder(root / "clean-checkpoint"),
                write_faiss=False,
            )
            self.assertEqual(
                (retry_output / "embeddings.npy").read_bytes(),
                (clean_output / "embeddings.npy").read_bytes(),
            )

    def test_nga_derivative_is_not_rewritten_and_requires_explicit_host(self) -> None:
        url = "https://api.nga.gov/iiif/example/full/!1024,1024/0/default.jpg"
        encoder = Siglip2LocalEncoder.__new__(Siglip2LocalEncoder)
        encoder._allowed_image_hosts = ("api.nga.gov",)

        self.assertEqual(
            encoder._resolved_image_url(url, prefer_web_large=True),
            url,
        )

        encoder._allowed_image_hosts = ("images.metmuseum.org",)
        with self.assertRaisesRegex(CorpusBuildError, "host is not allowed"):
            encoder._resolved_image_url(url, prefer_web_large=True)

    def test_stream_provenance_uses_declared_nga_input_policy(self) -> None:
        from PIL import Image

        url = "https://api.nga.gov/iiif/example/full/!1024,1024/0/default.jpg"
        policy = "nga-iiif-fit-1024-short-side-256/v1"
        payload_buffer = BytesIO()
        Image.new("RGB", (512, 384), (12, 34, 56)).save(payload_buffer, format="PNG")
        payload = payload_buffer.getvalue()

        encoder = Siglip2LocalEncoder.__new__(Siglip2LocalEncoder)
        encoder._allowed_image_hosts = ("api.nga.gov",)
        encoder._max_image_pixels = 1_000_000
        encoder._download = lambda raw_url, *, prefer_web_large: (
            payload,
            encoder._resolved_image_url(raw_url, prefer_web_large=prefer_web_large),
        )

        image, provenance = encoder._load_image(
            {
                "artwork_id": "NGA_1",
                "image_url": url,
                "image_input_policy": policy,
            },
            Path.cwd(),
        )
        try:
            self.assertEqual(image.size, (512, 384))
            self.assertEqual(provenance["input_source"], url)
            self.assertEqual(provenance["input_kind"], "remote-stream")
            self.assertEqual(provenance["input_policy"], policy)
            self.assertEqual(provenance["input_sha256"], hashlib.sha256(payload).hexdigest())
        finally:
            image.close()

    def test_embedding_manifest_preserves_per_record_input_policy(self) -> None:
        class ProvenanceEncoder(DeterministicTestEncoder):
            def encode(self, records, image_root, batch_size):
                self.seen_policies = [
                    record.get("image_input_policy", "") for record in records
                ]
                self._input_provenance = tuple(
                    {
                        "artwork_id": record["artwork_id"],
                        "input_kind": "remote-stream",
                        "input_source": record["image_url"],
                        "input_sha256": hashlib.sha256(
                            record["artwork_id"].encode("utf-8")
                        ).hexdigest(),
                        "input_width": "512",
                        "input_height": "512",
                        "input_policy": record.get("image_input_policy", ""),
                    }
                    for record in records
                )
                return super().encode(records, image_root, batch_size)

            @property
            def input_provenance(self):
                return self._input_provenance

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._build_corpus(root)
            image_manifest = corpus_dir / "images.manifest.csv"
            with image_manifest.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = (*tuple(reader.fieldnames or ()), "image_input_policy")
            for row in rows:
                row["image_input_policy"] = "adapter-declared-derivative/v1"
            with image_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

            encoder = ProvenanceEncoder(8)
            output = root / "embeddings"
            build_embedding_index(corpus_dir, output, encoder, write_faiss=False)

            self.assertEqual(
                encoder.seen_policies,
                ["adapter-declared-derivative/v1", "adapter-declared-derivative/v1"],
            )
            with (output / "embedded-images.manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                embedded_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["input_policy"] for row in embedded_rows],
                ["adapter-declared-derivative/v1", "adapter-declared-derivative/v1"],
            )

    def test_non_met_remote_policy_is_source_neutral_when_not_declared(self) -> None:
        policy = Siglip2LocalEncoder._remote_input_policy(
            {}, "https://api.nga.gov/iiif/example/full/!512,512/0/default.jpg"
        )
        self.assertEqual(policy, DECLARED_REMOTE_IMAGE_INPUT_POLICY)

    def test_remote_request_delay_paces_request_starts(self) -> None:
        class FakeOpener:
            def open(self, request, *, timeout):
                del request, timeout
                return object()

        encoder = Siglip2LocalEncoder.__new__(Siglip2LocalEncoder)
        encoder._request_delay_seconds = 1.0
        encoder._request_rate_lock = threading.Lock()
        encoder._next_request_at = 0.0
        encoder._request_timeout = 30
        encoder._url_opener = FakeOpener()
        request = object()

        with (
            patch("pipeline.embeddings.time.monotonic", side_effect=(100.0, 100.2)),
            patch("pipeline.embeddings.time.sleep") as sleep,
        ):
            encoder._open_remote_request(request)
            encoder._open_remote_request(request)

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.8)

    def test_remote_request_delay_is_strict_across_concurrent_workers(self) -> None:
        starts: list[float] = []

        class FakeOpener:
            def open(self, request, *, timeout):
                del request, timeout
                starts.append(time.monotonic())
                return object()

        encoder = Siglip2LocalEncoder.__new__(Siglip2LocalEncoder)
        encoder._request_delay_seconds = 0.02
        encoder._request_rate_lock = threading.Lock()
        encoder._next_request_at = 0.0
        encoder._request_timeout = 30
        encoder._url_opener = FakeOpener()

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(encoder._open_remote_request, (object(), object())))

        self.assertEqual(len(starts), 2)
        self.assertGreaterEqual(starts[1] - starts[0], 0.015)

    def test_aic_embedding_enforces_its_download_policy(self) -> None:
        encoder = Siglip2LocalEncoder.__new__(Siglip2LocalEncoder)
        records = ({"image_input_policy": AIC_IMAGE_INPUT_POLICY},)

        encoder._download_workers = 2
        encoder._request_delay_seconds = 1.0
        with self.assertRaisesRegex(CorpusBuildError, "download-workers 1"):
            encoder._validate_source_request_policy(records)

        encoder._download_workers = 1
        encoder._request_delay_seconds = 0.5
        with self.assertRaisesRegex(CorpusBuildError, "delay-seconds 1"):
            encoder._validate_source_request_policy(records)

        encoder._request_delay_seconds = 1.0
        encoder._validate_source_request_policy(records)

    def test_aic_image_requests_include_the_museum_project_header(self) -> None:
        aic_headers = Siglip2LocalEncoder._remote_request_headers(
            "https://www.artic.edu/iiif/2/example/full/843,/0/default.jpg"
        )
        other_headers = Siglip2LocalEncoder._remote_request_headers(
            "https://images.metmuseum.org/example.jpg"
        )

        self.assertEqual(
            aic_headers["AIC-User-Agent"], "Mnemosyne embedding corpus builder"
        )
        self.assertNotIn("AIC-User-Agent", other_headers)

    def test_met_web_large_falls_back_to_original_below_model_floor(self) -> None:
        from PIL import Image

        raw_url = "https://images.metmuseum.org/CRDImages/ep/original/example.jpg"

        def image_bytes(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
            buffer = BytesIO()
            Image.new("RGB", size, color).save(buffer, format="PNG")
            return buffer.getvalue()

        web_large = image_bytes((200, 300), (1, 2, 3))
        original = image_bytes((800, 1200), (4, 5, 6))
        calls: list[bool] = []

        encoder = Siglip2LocalEncoder.__new__(Siglip2LocalEncoder)
        encoder._allowed_image_hosts = ("images.metmuseum.org",)
        encoder._max_image_pixels = 2_000_000

        def download(_raw_url: str, *, prefer_web_large: bool = True):
            calls.append(prefer_web_large)
            if prefer_web_large:
                return web_large, raw_url.replace("/original/", "/web-large/")
            return original, raw_url

        encoder._download = download
        image, provenance = encoder._load_image(
            {"artwork_id": "MET_1", "image_url": raw_url}, Path.cwd()
        )
        try:
            self.assertEqual(calls, [True, False])
            self.assertEqual(image.size, (800, 1200))
            self.assertEqual(provenance["input_source"], raw_url)
            self.assertEqual(provenance["input_kind"], "remote-stream-original")
            self.assertEqual(provenance["input_policy"], SIGLIP_IMAGE_INPUT_POLICY)
            self.assertEqual(
                provenance["input_sha256"], hashlib.sha256(original).hexdigest()
            )
        finally:
            image.close()

    def test_stream_encoder_resumes_from_committed_batch(self) -> None:
        from PIL import Image

        class FakeTensor:
            def __init__(self, values: object) -> None:
                self.values = np.asarray(values, dtype=np.float32)

            def to(self, *_args: object, **_kwargs: object) -> "FakeTensor":
                return self

            def detach(self) -> "FakeTensor":
                return self

            def cpu(self) -> "FakeTensor":
                return self

            def numpy(self) -> np.ndarray:
                return self.values

        class FakeProcessor:
            def __call__(
                self, *, images: list[Image.Image], return_tensors: str
            ) -> dict[str, FakeTensor]:
                self.return_tensors = return_tensors
                return {
                    "pixel_values": FakeTensor(
                        [image.getpixel((0, 0))[0] for image in images]
                    )
                }

        class FakeModel:
            def get_image_features(self, *, pixel_values: FakeTensor) -> FakeTensor:
                values = pixel_values.values.reshape(-1, 1)
                return FakeTensor(np.concatenate((values, np.ones_like(values)), axis=1))

        class FakeTorch:
            float32 = np.float32

            @staticmethod
            def inference_mode():
                return nullcontext()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            records = [
                {"artwork_id": str(value), "image_url": f"https://example.test/{value}.jpg"}
                for value in (1, 2, 3)
            ]

            def make_encoder(
                progress,
                calls: list[str],
                *,
                device: str = "cpu",
                allowed_hosts: tuple[str, ...] = ("example.test",),
                max_image_bytes: int = 1024 * 1024,
                max_image_pixels: int = 10_000,
                download_workers: int = 2,
                request_delay_seconds: float = 0,
            ) -> Siglip2LocalEncoder:
                encoder = Siglip2LocalEncoder.__new__(Siglip2LocalEncoder)
                encoder.encoder_id = "fake/siglip"
                encoder.revision = "pinned"
                encoder.dimension = 2
                encoder._torch = FakeTorch()
                encoder._processor = FakeProcessor()
                encoder._model = FakeModel()
                encoder._device = device
                encoder._download_workers = download_workers
                encoder._request_timeout = 30
                encoder._request_delay_seconds = request_delay_seconds
                encoder._fetch_retries = 2
                encoder._max_image_bytes = max_image_bytes
                encoder._max_image_pixels = max_image_pixels
                encoder._allowed_image_hosts = allowed_hosts
                encoder._checkpoint_dir = checkpoint
                encoder._progress = progress
                encoder._input_provenance = ()

                def load_image(record, _image_root):
                    calls.append(record["artwork_id"])
                    value = int(record["artwork_id"])
                    return Image.new("RGB", (1, 1), (value, 0, 0)), {
                        "artwork_id": record["artwork_id"],
                        "input_kind": "remote-stream",
                        "input_source": record["image_url"],
                        "input_sha256": str(value) * 64,
                    }

                encoder._load_image = load_image
                return encoder

            first_calls: list[str] = []

            def interrupt(completed: int, _total: int) -> None:
                if completed == 2:
                    raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                make_encoder(interrupt, first_calls).encode(records, root, batch_size=2)
            state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["completed"], 2)
            self.assertEqual(state["batch_size"], 2)

            changed_batch_calls: list[str] = []
            with self.assertRaisesRegex(
                CorpusBuildError, "checkpoint does not match this build"
            ):
                make_encoder(None, changed_batch_calls).encode(
                    records, root, batch_size=1
                )
            self.assertEqual(changed_batch_calls, [])

            for changed_setting in (
                {"allowed_hosts": ("cdn.example.test",)},
                {"max_image_bytes": 2 * 1024 * 1024},
                {"max_image_pixels": 20_000},
                {"download_workers": 1},
                {"request_delay_seconds": 0.5},
            ):
                with self.subTest(changed_setting=changed_setting):
                    with self.assertRaisesRegex(
                        CorpusBuildError, "checkpoint does not match this build"
                    ):
                        make_encoder(None, [], **changed_setting).encode(
                            records, root, batch_size=2
                        )

            with self.assertRaisesRegex(
                CorpusBuildError, "checkpoint does not match this build"
            ):
                make_encoder(None, [], device="mps").encode(records, root, batch_size=2)

            resumed_calls: list[str] = []
            encoder = make_encoder(None, resumed_calls)
            matrix = encoder.encode(records, root, batch_size=2)
            self.assertEqual(resumed_calls, ["3"])
            np.testing.assert_array_equal(matrix, np.asarray([[1, 1], [2, 1], [3, 1]]))
            self.assertEqual(
                [row["artwork_id"] for row in encoder.input_provenance], ["1", "2", "3"]
            )
            encoder.cleanup_checkpoint()
            self.assertFalse(checkpoint.exists())

    def test_stream_encoder_rejects_short_model_batch(self) -> None:
        from PIL import Image

        class FakeTensor:
            def __init__(self, values: object) -> None:
                self.values = np.asarray(values, dtype=np.float32)

            def to(self, *_args: object, **_kwargs: object) -> "FakeTensor":
                return self

            def detach(self) -> "FakeTensor":
                return self

            def cpu(self) -> "FakeTensor":
                return self

            def numpy(self) -> np.ndarray:
                return self.values

        class FakeProcessor:
            def __call__(self, *, images, return_tensors):
                del images, return_tensors
                return {"pixel_values": FakeTensor([[1], [2]])}

        class FakeModel:
            def get_image_features(self, **_inputs):
                return FakeTensor([[1, 1]])

        class FakeTorch:
            float32 = np.float32

            @staticmethod
            def inference_mode():
                return nullcontext()

        encoder = Siglip2LocalEncoder.__new__(Siglip2LocalEncoder)
        encoder.encoder_id = "fake/siglip"
        encoder.revision = "pinned"
        encoder.dimension = 2
        encoder._torch = FakeTorch()
        encoder._processor = FakeProcessor()
        encoder._model = FakeModel()
        encoder._device = "cpu"
        encoder._download_workers = 2
        encoder._request_timeout = 30
        encoder._request_delay_seconds = 0
        encoder._fetch_retries = 2
        encoder._allowed_image_hosts = ("example.test",)
        encoder._max_image_bytes = 1024 * 1024
        encoder._max_image_pixels = 10_000
        encoder._checkpoint_dir = None
        encoder._progress = None
        encoder._input_provenance = ()
        encoder._load_image = lambda record, _root: (
            Image.new("RGB", (1, 1)),
            {
                "artwork_id": record["artwork_id"],
                "input_kind": "fixture",
                "input_source": "fixture",
                "input_sha256": "0" * 64,
                "input_width": "1",
                "input_height": "1",
                "input_policy": "fixture",
            },
        )
        records = [{"artwork_id": "1"}, {"artwork_id": "2"}]
        with self.assertRaisesRegex(Exception, "returned batch shape"):
            encoder.encode(records, Path.cwd(), batch_size=2)


if __name__ == "__main__":
    unittest.main()
