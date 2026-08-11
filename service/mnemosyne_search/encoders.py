"""Local text-embedding interfaces and adapters.

The image tower is deliberately absent. Production artifact builds run it offline;
this request path owns only a matching local text tower.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


def l2_normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embeddings must be non-zero")
    return values / norms


class TextEncoder(Protocol):
    model_id: str
    model_version: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalized float32 row per input string."""


class FixtureTextEncoder:
    """Exact, deterministic vectors for checked-in fixtures and contract tests."""

    def __init__(
        self,
        vectors: dict[str, Sequence[float]],
        *,
        model_id: str = "fixture-static",
        model_version: str = "v1",
    ) -> None:
        if not vectors:
            raise ValueError("fixture vector map cannot be empty")
        self._vectors = {
            key.casefold(): l2_normalize(np.asarray([value], dtype=np.float32))[0]
            for key, value in vectors.items()
        }
        dimensions = {value.shape[0] for value in self._vectors.values()}
        if len(dimensions) != 1:
            raise ValueError("all fixture vectors must have the same dimension")
        self.model_id = model_id
        self.model_version = model_version

    @classmethod
    def from_json(cls, path: str | Path) -> "FixtureTextEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            payload["vectors"],
            model_id=payload["modelId"],
            model_version=payload["modelVersion"],
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = [text for text in texts if text.casefold() not in self._vectors]
        if missing:
            raise ValueError(f"fixture encoder has no vectors for: {missing!r}")
        return np.stack([self._vectors[text.casefold()] for text in texts]).astype(np.float32)


class DeterministicHashEncoder:
    """Small local smoke-test encoder; deterministic, but not semantically trained."""

    model_id = "deterministic-hash"
    model_version = "sha256-token-v1"

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE) or [text.casefold()]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                for offset in range(0, 16, 2):
                    index = int.from_bytes(digest[offset : offset + 2], "big") % self.dimension
                    sign = 1.0 if digest[16 + offset // 2] & 1 else -1.0
                    rows[row_index, index] += sign
        return l2_normalize(rows)


class Siglip2TextEncoder:
    """Lazy local Hugging Face SigLIP 2 text-tower adapter.

    This adapter downloads model files only when Hugging Face is not already
    cached and ``local_files_only`` is false. It never calls a hosted inference
    API and never requires an end-user API key.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        *,
        revision: str = "main",
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoTokenizer, SiglipTextModel
        except ImportError as error:  # pragma: no cover - optional production dependency
            raise RuntimeError(
                "Siglip2TextEncoder requires the 'siglip2' optional dependencies"
            ) from error

        self.model_id = model_id
        self.model_version = revision
        self._torch = torch
        self.runtime_versions = {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "numpy": str(np.__version__),
        }
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            use_fast=True,
        )
        # This checkpoint advertises ``model_type: siglip`` and its standalone
        # request-time tower is ``SiglipTextModel``.  Loading ``AutoModel`` here
        # materializes the unused vision tower as well as the text tower.
        self._model = SiglipTextModel.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        ).to(device)
        self._model.eval()
        self._text_max_length = getattr(
            self._model.config, "max_position_embeddings", None
        )
        if (
            isinstance(self._text_max_length, bool)
            or not isinstance(self._text_max_length, int)
            or self._text_max_length < 1
        ):
            raise RuntimeError(
                "SigLIP 2 model config must declare a positive text max_position_embeddings"
            )

    @property
    def device(self) -> str:
        return self._device

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # SigLIP 2's text tower was trained with a fixed 64-token input. Dynamic
        # batch padding changes the pooling position and can collapse otherwise
        # unrelated query vectors toward one another, destroying retrieval
        # quality even though the model and image embeddings are correct.
        inputs = self._tokenizer(
            list(texts),
            padding="max_length",
            max_length=self._text_max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model(**inputs)
            features = output.pooler_output
        return l2_normalize(features.detach().float().cpu().numpy())
