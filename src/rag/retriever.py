"""Query-time retrieval with optional topic filter and score threshold.

Design pattern: STRATEGY consumer. retrieve() depends only on the embedder and
VectorStore interfaces, so tests inject fakes. Example: retrieve("What is a
bond?", topic="bonds") searches every chunk (FAISS has
no metadata filter), keeps the bonds ones, drops scores under
rag.score_threshold, applies the diversity caps (select_chunks) and returns at
most top_k; an off-topic question returns [].
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from src.core.models import Source
from src.rag.embeddings import get_embedder
from src.rag.vector_store import get_vector_store
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


WRITTEN_FIRST = 1.0   # sort bonus: written-query chunks rank above expansion-only ones


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


def expand_query(query: str, expansions: dict[str, str]) -> str | None:
    """The query with each acronym replaced by its plain words, or None when no acronym is found.

    Whole words only, any case: "What is ltcg?" -> "What is long-term capital gains tax?", while
    "PPFX" and "stcgs" are left alone. No LLM call: the table is rag.query_expansions in config.
    """
    expanded = query
    for acronym, words in expansions.items():
        expanded = re.sub(rf"(?<!\w){re.escape(acronym)}(?!\w)", lambda _m, w=words: w, expanded, flags=re.IGNORECASE)
    return None if expanded == query else expanded


def merge_retrieval(
    retrieve_fn: Callable[..., list[RetrievedChunk]], queries: list[str], topic: str | None
) -> list[RetrievedChunk]:
    """Retrieve for each query as written first, then fill the free slots from the acronym-expanded forms.

    A query that holds an acronym such as LTCG is also searched expanded (see expand_query), because
    the articles use the long form. The expansion only FILLS: chunks found by a query as written come
    first with their own scores, and a chunk the expansion also finds keeps the written query's score.
    Example: "How does PPF work?" finds the deposit and interest chunks as written; its expansion
    ("public provident fund") would rank the article's heading chunk first and push them out of the
    two-per-article cap, so it may only add what is left. "What is LTCG?" finds nothing as written,
    so its expansion fills every slot.

    retrieve() already applied the threshold and caps per query; select_chunks runs again on the
    merged list so the per-article and glossary caps and top_k still hold (no second implementation).
    Its sort key is the score plus WRITTEN_FIRST for a written-query chunk (a cosine score is at most 1),
    which keeps the written results ahead without a second sorting rule. Used by the Finance Q&A agent
    (original + rewritten query) and the Tax agent (one query, locked topic).
    """
    rag = get_config()["rag"]
    expansions = rag.get("query_expansions") or {}
    searches: dict[str, bool] = {}          # query text -> searched as written? (insertion order = search order)
    for query in queries:
        for text, is_written in ((query, True), (expand_query(query, expansions), False)):
            if text:
                searches[text] = searches.get(text, False) or is_written
    found: dict[bool, dict[str, RetrievedChunk]] = {True: {}, False: {}}
    for text, is_written in searches.items():
        for chunk in retrieve_fn(text, topic=topic):
            known = found[is_written].get(chunk.chunk_id)
            if known is None or chunk.score > known.score:
                found[is_written][chunk.chunk_id] = chunk
    first = set(found[True])
    best = {**found[False], **found[True]}   # a written-query chunk keeps its own score even if the expansion scored higher
    hits = [
        ({"chunk_id": cid}, chunk.score + (WRITTEN_FIRST if cid in first else 0.0))
        for cid, chunk in best.items() if chunk.score >= rag["score_threshold"]
    ]
    max_per_article, max_per_topic = diversity_caps(rag)
    kept = select_chunks(hits, rag["top_k"], 0.0, max_per_article, max_per_topic)
    return [best[rec["chunk_id"]] for rec, _ in kept]


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
