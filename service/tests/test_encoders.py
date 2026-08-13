from __future__ import annotations

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mnemosyne_search.encoders import Siglip2TextEncoder


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.devices: list[str] = []

    def to(self, device: str) -> "_FakeTensor":
        self.devices.append(device)
        return self

    def detach(self) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


def _fake_siglip_dependencies(
    *,
    max_position_embeddings: object = 64,
    cuda_available: bool = False,
    mps_available: bool = False,
) -> tuple[dict[str, ModuleType], dict[str, object]]:
    calls: dict[str, object] = {
        "tokenizer_loads": [],
        "model_loads": [],
        "tokenizer_calls": [],
    }

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            calls["tokenizer_calls"].append({"texts": list(texts), **kwargs})
            tensor = _FakeTensor([[1], [2]])
            calls["input_tensor"] = tensor
            return {"input_ids": tensor}

    tokenizer = Tokenizer()

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["tokenizer_loads"].append((args, kwargs))
            return tokenizer

    class LoadedTextModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                max_position_embeddings=max_position_embeddings
            )
            self.devices: list[str] = []
            self.eval_calls = 0
            self.calls: list[dict[str, object]] = []

        def to(self, device: str) -> "LoadedTextModel":
            self.devices.append(device)
            return self

        def eval(self) -> None:
            self.eval_calls += 1

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(pooler_output=_FakeTensor([[3, 4], [0, 2]]))

    model = LoadedTextModel()

    class SiglipTextModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["model_loads"].append((args, kwargs))
            return model

    class AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - guard rail
            raise AssertionError("the combined image-and-text model must not be loaded")

    torch = ModuleType("torch")
    torch.__version__ = "test-torch"
    torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: mps_available)
    )
    torch.inference_mode = lambda: nullcontext()

    transformers = ModuleType("transformers")
    transformers.__version__ = "test-transformers"
    transformers.AutoModel = AutoModel
    transformers.AutoTokenizer = AutoTokenizer
    transformers.SiglipTextModel = SiglipTextModel

    calls["model"] = model
    return {"torch": torch, "transformers": transformers}, calls


class Siglip2TextEncoderTests(unittest.TestCase):
    def test_loads_only_the_text_tower_at_the_exact_revision_and_device(self) -> None:
        modules, calls = _fake_siglip_dependencies(mps_available=True)

        with patch.dict(sys.modules, modules):
            encoder = Siglip2TextEncoder(
                "example/siglip-checkpoint",
                revision="pinned-commit-sha",
                device="mps",
                local_files_only=True,
            )
            encoded = encoder.encode(["cat", "ship"])

        self.assertEqual(
            calls["tokenizer_loads"],
            [
                (
                    ("example/siglip-checkpoint",),
                    {
                        "revision": "pinned-commit-sha",
                        "local_files_only": True,
                        "use_fast": True,
                    },
                )
            ],
        )
        self.assertEqual(
            calls["model_loads"],
            [
                (
                    ("example/siglip-checkpoint",),
                    {
                        "revision": "pinned-commit-sha",
                        "local_files_only": True,
                    },
                )
            ],
        )
        model = calls["model"]
        self.assertEqual(model.devices, ["mps"])
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(encoder.device, "mps")
        self.assertEqual(encoder.model_id, "example/siglip-checkpoint")
        self.assertEqual(encoder.model_version, "pinned-commit-sha")
        self.assertEqual(encoder._text_max_length, 64)
        self.assertEqual(calls["input_tensor"].devices, ["mps"])
        self.assertEqual(list(model.calls[0]), ["input_ids"])
        self.assertEqual(
            calls["tokenizer_calls"],
            [
                {
                    "texts": ["cat", "ship"],
                    "padding": "max_length",
                    "max_length": 64,
                    "truncation": True,
                    "return_tensors": "pt",
                }
            ],
        )
        np.testing.assert_allclose(
            encoded,
            np.asarray([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(encoded.dtype, np.float32)

    def test_uses_the_text_model_config_for_the_fixed_tokenizer_length(self) -> None:
        modules, calls = _fake_siglip_dependencies(max_position_embeddings=37)

        with patch.dict(sys.modules, modules):
            encoder = Siglip2TextEncoder(device="cpu")
            encoder.encode(["cat", "ship"])

        self.assertEqual(encoder._text_max_length, 37)
        self.assertEqual(calls["tokenizer_calls"][0]["max_length"], 37)

    def test_auto_device_selection_prefers_cuda_then_mps_then_cpu(self) -> None:
        cases = (
            (True, True, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        )
        for cuda_available, mps_available, expected in cases:
            with self.subTest(expected=expected):
                modules, calls = _fake_siglip_dependencies(
                    cuda_available=cuda_available,
                    mps_available=mps_available,
                )
                with patch.dict(sys.modules, modules):
                    encoder = Siglip2TextEncoder(device="auto")

                self.assertEqual(encoder.device, expected)
                self.assertEqual(calls["model"].devices, [expected])

    def test_rejects_invalid_or_unavailable_requested_devices(self) -> None:
        cases = (
            ("tpu", "device must be auto, cpu, cuda, or mps"),
            ("cuda", "CUDA was requested but is unavailable"),
            ("mps", "MPS was requested but is unavailable"),
        )
        for device, message in cases:
            with self.subTest(device=device):
                modules, _calls = _fake_siglip_dependencies()
                with patch.dict(sys.modules, modules):
                    with self.assertRaisesRegex(ValueError, message):
                        Siglip2TextEncoder(device=device)

    def test_rejects_invalid_text_max_position_configuration(self) -> None:
        for invalid_value in (None, True, 0, -1, 64.0):
            with self.subTest(max_position_embeddings=invalid_value):
                modules, _calls = _fake_siglip_dependencies(
                    max_position_embeddings=invalid_value
                )
                with patch.dict(sys.modules, modules):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "positive text max_position_embeddings",
                    ):
                        Siglip2TextEncoder(device="cpu")

    def test_missing_optional_dependencies_fail_with_a_clear_error(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(
                RuntimeError,
                "requires the 'siglip2' optional dependencies",
            ) as raised:
                Siglip2TextEncoder()

        self.assertIsInstance(raised.exception.__cause__, ImportError)

    def test_encode_uses_the_required_fixed_length_text_contract(self) -> None:
        calls: list[dict[str, object]] = []

        def tokenizer(texts, **kwargs):
            calls.append({"texts": texts, **kwargs})
            return {"input_ids": _FakeTensor([[1], [2]])}

        class Model:
            def __call__(self, *, input_ids: _FakeTensor):
                class Output:
                    pooler_output = _FakeTensor([[3, 4], [0, 2]])

                return Output()

        class Torch:
            @staticmethod
            def inference_mode():
                return nullcontext()

        encoder = Siglip2TextEncoder.__new__(Siglip2TextEncoder)
        encoder._device = "cpu"
        encoder._text_max_length = 64
        encoder._tokenizer = tokenizer
        encoder._model = Model()
        encoder._torch = Torch()

        encoded = encoder.encode(["cat", "ship"])

        self.assertEqual(
            calls,
            [
                {
                    "texts": ["cat", "ship"],
                    "padding": "max_length",
                    "max_length": 64,
                    "truncation": True,
                    "return_tensors": "pt",
                }
            ],
        )
        np.testing.assert_allclose(
            encoded,
            np.asarray([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(encoded.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
