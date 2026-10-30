"""Deterministic fake embedder for tests: hashed bag-of-words, L2-normalised.

No model, no network. Texts that share words get a higher cosine score, which
is enough to test retrieval, filtering and thresholds.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache


@lru_cache(maxsize=None)
def _hashed_vector(text: str, dim: int) -> tuple[float, ...]:
    """Vector for one text, computed once per process and shared by every FakeEmbedder (the result is a pure
    function of its arguments). Example: the RAG-eval tests embed the same 60 articles in every test."""
    vec = [0.0] * dim
    for token in re.findall(r"[a-z]+", text.lower()):
        vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return tuple(x / norm for x in vec) if norm else tuple(vec)


class FakeEmbedder:
    """Stands in for SentenceTransformerEmbedder."""

    model_name = "fake-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        return list(_hashed_vector(text, self.dim))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
