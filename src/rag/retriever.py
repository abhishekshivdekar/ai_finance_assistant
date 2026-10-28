"""Query-time retrieval with optional topic filter and score threshold.

Design pattern: STRATEGY consumer. retrieve() depends only on the embedder and
VectorStore interfaces, so tests inject fakes. Example: retrieve("What is a
bond?", topic="bonds") over-fetches top_k x candidate_multiplier chunks (FAISS
has no metadata filter), keeps the bonds ones, drops scores under
rag.score_threshold and returns at most top_k; an off-topic question returns [].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.models import Source
from src.rag.embeddings import get_embedder
from src.rag.vector_store import get_vector_store
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with the metadata needed for citations."""

    text: str
    title: str
    topic: str
    references: list[dict[str, str]]
    chunk_id: str
    score: float


def retrieve(
    query: str,
    topic: str | None = None,
    top_k: int | None = None,
    *,
    embedder: Any = None,
    store: Any = None,
) -> list[RetrievedChunk]:
    """Return the best chunks for `query`, or [] when nothing passes the threshold."""
    rag = get_config()["rag"]
    if not query or not query.strip():
        return []
    if topic is not None and topic not in rag["topics"]:
        log.warning("Unknown topic filter %r; returning no results", topic)
        return []

    k = rag["top_k"] if top_k is None else top_k
    candidates = k if topic is None else k * rag["candidate_multiplier"]
    embedder = embedder or get_embedder()
    store = store or get_vector_store()

    hits = store.search(embedder.embed_query(query), candidates)
    if topic is not None:
        hits = [(rec, score) for rec, score in hits if rec["topic"] == topic]
    hits = [(rec, score) for rec, score in hits if score >= rag["score_threshold"]]
    hits.sort(key=lambda pair: pair[1], reverse=True)

    return [
        RetrievedChunk(
            text=rec["text"],
            title=rec["title"],
            topic=rec["topic"],
            references=rec["references"],
            chunk_id=rec["chunk_id"],
            score=score,
        )
        for rec, score in hits[:k]
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered [1]..[n] blocks for the prompt, so the answer can cite them."""
    return "\n\n".join(f"[{i}] {c.title}\n{c.text}" for i, c in enumerate(chunks, start=1))


def to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    """Convert chunks to Source entries (url = the first reference's root site)."""
    return [
        Source(
            title=c.title,
            url=c.references[0]["url"] if c.references else None,
            chunk_id=c.chunk_id,
            score=c.score,
        )
        for c in chunks
    ]
