"""Offline corpus, date-artifact, and embedding builder for Mnemosyne."""

__version__ = "0.1.0"

from .build import BUILD_SCHEMA_VERSION, build_corpus

__all__ = ["BUILD_SCHEMA_VERSION", "build_corpus"]
