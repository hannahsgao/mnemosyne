"""Versioned prompt ensembles, encoded in one local batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .encoders import TextEncoder, l2_normalize
from .models import QueryTerm


@dataclass(frozen=True)
class EncodedSeries:
    combined: np.ndarray
    prompt_vectors: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PromptEnsemble:
    version: str = "art-concept-v1"
    templates: tuple[str, ...] = (
        "{query}",
        "an artwork depicting {query}",
        "a work of art about {query}",
    )

    def __post_init__(self) -> None:
        if not self.templates or any("{query}" not in template for template in self.templates):
            raise ValueError("prompt templates must be non-empty and contain {query}")

    def encode(self, terms: Sequence[QueryTerm], encoder: TextEncoder) -> list[EncodedSeries]:
        prompts = [template.format(query=term.normalized) for term in terms for template in self.templates]
        encoded = encoder.encode(prompts)
        if encoded.shape[0] != len(prompts):
            raise ValueError("encoder returned the wrong number of vectors")

        results: list[EncodedSeries] = []
        prompt_count = len(self.templates)
        for index in range(len(terms)):
            block = l2_normalize(encoded[index * prompt_count : (index + 1) * prompt_count])
            combined = l2_normalize(block.mean(axis=0, keepdims=True))[0]
            results.append(EncodedSeries(combined=combined, prompt_vectors=tuple(block)))
        return results
