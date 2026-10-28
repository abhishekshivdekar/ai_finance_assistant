"""RAG evaluation harness: hit-rate@k, MRR and a parameter sweep.

Run manually with the real embedder (no API tokens): run_rag_eval.py in the
project root. The unit tests call the same functions with a fake embedder.

Reading the table: hit_rate is the share of questions whose labelled article
appears in the top_k results; mrr is the mean of 1/rank of the first correct
article; off_topic_false is how many off-topic questions still returned
something above the threshold (should be 0). Pick the threshold between the
answerable and off-topic score distributions (architecture section 5.3).
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import yaml

from src.rag.embeddings import get_embedder
from src.rag.ingest import Article, build_store, load_articles
from src.utils.logger import get_logger

log = get_logger(__name__)

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")


def load_questions(path: Path | None = None) -> dict[str, Any]:
    """Load questions, off-topic questions and the sweep grid."""
    with (path or QUESTIONS_PATH).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def article_id(chunk_id: str) -> str:
    """'topic/slug#3' -> 'topic/slug'."""
    return chunk_id.split("#")[0]


def rank_from_hits(
    hits: list[tuple[dict[str, Any], float]], top_k: int, threshold: float
) -> list[str]:
    """Article ids in rank order after applying the threshold and top_k to chunks."""
    kept = [rec for rec, score in hits if score >= threshold][:top_k]
    ranked: list[str] = []
    for rec in kept:
        art = article_id(rec["chunk_id"])
        if art not in ranked:
            ranked.append(art)
    return ranked


def rank_of_first_hit(ranked: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first expected article, or None if absent."""
    for position, art in enumerate(ranked, start=1):
        if art in expected:
            return position
    return None


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions whose first correct article is within the top k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank (a miss counts as 0)."""
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None) / len(ranks)


def sweep(
    grid: dict[str, list[Any]],
    articles: list[Article],
    embedder: Any,
    questions: list[dict[str, Any]],
    off_topic: list[str],
) -> list[dict[str, Any]]:
    """Evaluate every grid combination. One store per (chunk_size, overlap)."""
    max_k = max(grid["top_k"])
    q_vecs = [embedder.embed_query(q["question"]) for q in questions]
    off_vecs = [embedder.embed_query(q) for q in off_topic]
    rows: list[dict[str, Any]] = []

    for size, overlap in product(grid["chunk_size"], grid["chunk_overlap"]):
        if overlap >= size:
            log.warning("Skipping overlap %d >= chunk_size %d", overlap, size)
            continue
        store = build_store(articles, embedder, size, overlap)
        q_hits = [store.search(v, max_k) for v in q_vecs]
        off_hits = [store.search(v, max_k) for v in off_vecs]
        for top_k, threshold in product(grid["top_k"], grid["threshold"]):
            ranks = [
                rank_of_first_hit(rank_from_hits(h, top_k, threshold), q["expected"])
                for h, q in zip(q_hits, questions)
            ]
            false_matches = sum(1 for h in off_hits if rank_from_hits(h, top_k, threshold))
            rows.append(
                {
                    "chunk_size": size, "chunk_overlap": overlap, "top_k": top_k,
                    "threshold": threshold, "hit_rate": hit_rate_at_k(ranks, top_k),
                    "mrr": mrr(ranks), "off_topic_false": false_matches,
                }
            )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    """Render the sweep rows as a fixed-width text table."""
    columns = ["chunk_size", "chunk_overlap", "top_k", "threshold",
               "hit_rate", "mrr", "off_topic_false"]
    lines = ["  ".join(f"{c:>15}" for c in columns)]
    for row in rows:
        cells = []
        for c in columns:
            value = row[c]
            cells.append(f"{value:>15.3f}" if isinstance(value, float) else f"{value:>15}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main() -> None:
    """Run the sweep with the real embedder and print the table."""
    data = load_questions()
    articles = load_articles()
    rows = sweep(data["grid"], articles, get_embedder(), data["questions"], data["off_topic"])
    print(format_table(rows))


if __name__ == "__main__":
    main()
