from __future__ import annotations

from contextlib import nullcontext
import unittest

import numpy as np

from mnemosyne_search.encoders import Siglip2TextEncoder


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def to(self, _device: str) -> "_FakeTensor":
        return self

    def detach(self) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class Siglip2TextEncoderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
