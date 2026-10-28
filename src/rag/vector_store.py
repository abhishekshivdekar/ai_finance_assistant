"""Vector store interface and FAISS implementation.

Design patterns:
- STRATEGY: VectorStore is the interface and FaissStore one implementation.
  Example: setting `rag.vector_store: chroma` later would pick a ChromaStore
  without the retriever changing.
- SINGLETON (cached getter): get_vector_store() loads the index once per
  process, like get_config() reads the YAML once.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.rag.manifest import resolve_dir, warn_if_stale
from src.utils.config import ConfigError, get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


class IndexNotFoundError(Exception):
    """The saved index files are missing (knowledge base unavailable)."""


class VectorStore(ABC):
    """Interface: add vectors with metadata, search, save and load."""

    @abstractmethod
    def add(self, vectors: list[list[float]], records: list[dict[str, Any]]) -> None:
        """Add normalised vectors with one metadata record each."""

    @abstractmethod
    def search(self, query_vector: list[float], k: int) -> list[tuple[dict[str, Any], float]]:
        """Return up to k (record, score) pairs, best first."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persist the store into a directory."""

    @abstractmethod
    def load(self, directory: Path) -> None:
        """Load a previously saved store from a directory."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored vectors."""


class FaissStore(VectorStore):
    """FAISS inner-product index; with normalised vectors the score is cosine."""

    def __init__(self) -> None:
        self._index: Any = None
        self._records: list[dict[str, Any]] = []

    def add(self, vectors: list[list[float]], records: list[dict[str, Any]]) -> None:
        if len(vectors) != len(records):
            raise ValueError("vectors and records must have the same length")
        if not vectors:
            return
        import faiss

        matrix = np.asarray(vectors, dtype="float32")
        if self._index is None:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
        elif matrix.shape[1] != self._index.d:
            raise ValueError(
                f"Vector dimension {matrix.shape[1]} does not match index dimension {self._index.d}"
            )
        self._index.add(matrix)
        self._records.extend(records)

    def search(self, query_vector: list[float], k: int) -> list[tuple[dict[str, Any], float]]:
        if self._index is None or k <= 0:
            return []
        k = min(k, len(self._records))
        query = np.asarray([query_vector], dtype="float32")
        scores, ids = self._index.search(query, k)
        return [
            (self._records[i], float(score))
            for score, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(directory / INDEX_FILE))
        (directory / CHUNKS_FILE).write_text(json.dumps(self._records), encoding="utf-8")

    def load(self, directory: Path) -> None:
        import faiss

        index_path, chunks_path = directory / INDEX_FILE, directory / CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            raise IndexNotFoundError(f"No saved index in {directory}")
        self._index = faiss.read_index(str(index_path))
        self._records = json.loads(chunks_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self._records)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Load the shared vector store chosen by config (Singleton) and check its manifest."""
    cfg = get_config()["rag"]
    name = str(cfg.get("vector_store", "")).lower()
    if name != "faiss":
        raise ConfigError(f"Unsupported rag.vector_store: {name!r}")
    directory = resolve_dir(cfg["vector_store_dir"])
    store = FaissStore()
    store.load(directory)
    warn_if_stale(directory)
    log.info("Loaded %d chunks from %s", len(store), directory)
    return store
