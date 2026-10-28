"""Local text embeddings.

Design pattern: SINGLETON (cached getter). get_embedder() builds one embedder
per process, and the model inside is loaded lazily on first use. Example: the
retriever and the ingest script both call get_embedder(), so the
all-MiniLM-L6-v2 weights are loaded into memory once, not twice.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.utils.config import ConfigError, get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class SentenceTransformerEmbedder:
    """Wraps a local sentence-transformers model; vectors are L2-normalised."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        """Load the model on first use (slow, so never at import time)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts into normalised vectors."""
        vectors = self._load().encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query into a normalised vector."""
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformerEmbedder:
    """Return the shared embedder chosen by config (Singleton)."""
    cfg = get_config()["embeddings"]
    provider = str(cfg.get("provider", "")).lower()
    if provider != "huggingface":
        raise ConfigError(f"Unsupported embeddings.provider: {provider!r}")
    return SentenceTransformerEmbedder(cfg["model"])
