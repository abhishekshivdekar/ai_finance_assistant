"""Deterministic fake embedder for tests: hashed bag-of-words, L2-normalised.

No model, no network. Texts that share words get a higher cosine score, which
is enough to test retrieval, filtering and thresholds.
"""
from __future__ import annotations

import hashlib
import math
import re


class FakeEmbedder:
    """Stands in for SentenceTransformerEmbedder."""

    model_name = "fake-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z]+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
