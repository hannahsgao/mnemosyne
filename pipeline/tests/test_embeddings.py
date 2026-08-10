from __future__ import annotations

import csv
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.build import CorpusBuildError, build_corpus
from pipeline.embeddings import (
    DeterministicTestEncoder,
    Siglip2LocalEncoder,
    build_embedding_index,
)


class EmbeddingBuildTests(unittest.TestCase):
    def _build_corpus(self, root: Path) -> Path:
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
        )
        return corpus_dir

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
                progress, calls: list[str], *, device: str = "cpu"
            ) -> Siglip2LocalEncoder:
                encoder = Siglip2LocalEncoder.__new__(Siglip2LocalEncoder)
                encoder.encoder_id = "fake/siglip"
                encoder.revision = "pinned"
                encoder.dimension = 2
                encoder._torch = FakeTorch()
                encoder._processor = FakeProcessor()
                encoder._model = FakeModel()
                encoder._device = device
                encoder._download_workers = 2
                encoder._max_image_pixels = 10_000
                encoder._allowed_image_hosts = ("example.test",)
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
