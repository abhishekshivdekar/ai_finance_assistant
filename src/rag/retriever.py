"""Query-time retrieval with optional topic filter and score threshold.

Design pattern: STRATEGY consumer. retrieve() depends only on the embedder and
VectorStore interfaces, so tests inject fakes. Example: retrieve("What is a
bond?", topic="bonds") searches every chunk (FAISS has
no metadata filter), keeps the bonds ones, drops scores under
rag.score_threshold, applies the diversity caps (select_chunks) and returns at
most top_k; an off-topic question returns [].
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


def diversity_caps(rag: dict[str, Any]) -> tuple[int | None, dict[str, int]]:
    """Read (max_per_article, max_per_topic) from the rag config section."""
    return rag.get("max_chunks_per_article"), dict(rag.get("max_chunks_per_topic") or {})


def select_chunks(
    hits: list[tuple[dict[str, Any], float]],
    top_k: int,
    threshold: float,
    max_per_article: int | None = None,
    max_per_topic: dict[str, int] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """Threshold, sort best-first, cap per article and per topic, keep top_k.

    Design pattern: SINGLE SOURCE OF TRUTH (one pure function shared by two
    callers). retrieve() uses it for the app and rank_from_hits() in the eval
    harness uses it for the sweep, so the eval measures exactly what the app
    does. Example: with max_per_article=2 and max_per_topic={"glossary": 1}, hits
    scoring 0.9 (stocks/a#0), 0.8 (stocks/a#1), 0.7 (stocks/a#2), 0.6
    (glossary/g#0), 0.5 (glossary/g#1) give stocks/a#0, stocks/a#1, glossary/g#0:
    a#2 is the third chunk of one article and g#1 is the second glossary chunk.

    A cap of None/0, or an empty max_per_topic, turns that rule off. The topic
    and article come from the record's chunk_id ("topic/slug#n") so records
    need no extra fields.
    """
    caps = max_per_topic or {}
    per_article: dict[str, int] = {}
    per_topic: dict[str, int] = {}
    kept: list[tuple[dict[str, Any], float]] = []
    ranked = sorted(
        (pair for pair in hits if pair[1] >= threshold), key=lambda pair: pair[1], reverse=True
    )
    for rec, score in ranked:
        if len(kept) >= top_k:
            break
        article = rec["chunk_id"].split("#")[0]
        topic = article.split("/")[0]
        if max_per_article and per_article.get(article, 0) >= max_per_article:
            continue
        if topic in caps and per_topic.get(topic, 0) >= caps[topic]:
            continue
        per_article[article] = per_article.get(article, 0) + 1
        per_topic[topic] = per_topic.get(topic, 0) + 1
        kept.append((rec, score))
    return kept


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
    if embedder is None:
        embedder = get_embedder()
    if store is None:
        store = get_vector_store()
    # Search every chunk: FAISS has no metadata filter, and the diversity caps
    # may drop top hits, so the pool must be larger than k. (The eval harness
    # searches every chunk too, so both see the same candidates.)
    hits = store.search(embedder.embed_query(query), len(store))
    if topic is not None:
        hits = [(rec, score) for rec, score in hits if rec["topic"] == topic]
    max_per_article, max_per_topic = diversity_caps(rag)
    hits = select_chunks(hits, k, rag["score_threshold"], max_per_article, max_per_topic)

    return [
        RetrievedChunk(
            text=rec["text"],
            title=rec["title"],
            topic=rec["topic"],
            references=rec["references"],
            chunk_id=rec["chunk_id"],
            score=score,
        )
        for rec, score in hits
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
