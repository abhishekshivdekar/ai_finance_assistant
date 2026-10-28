"""RAG evaluation harness: hit-rate@k, MRR and a parameter sweep.

Run manually with the real embedder (no API tokens): run_rag_eval.py in the
project root. The unit tests call the same functions with a fake embedder.

Reading the table: hit_rate is the share of questions whose labelled article
appears in the top_k results; mrr is the mean of 1/rank of the first correct
article; off_topic_false is how many off-topic questions still returned
something above the threshold (should be 0). Pick the threshold between the
answerable and off-topic score distributions (architecture section 5.3).

Three sets are scored separately: "development" (questions), "held-out"
(independent questions; skipped until labelled) and the off-topic set made of
off_topic plus near_miss. The extra columns held_out_hit_rate, held_out_mrr and
near_miss_false appear only once at least one held-out item is labelled.
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


def split_held_out(
    held_out: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return (labelled held-out items, number skipped for an empty `expected`)."""
    labelled = [q for q in held_out if q.get("expected")]
    return labelled, len(held_out) - len(labelled)


def summarize_sets(data: dict[str, Any]) -> list[str]:
    """One line per evaluation set, including how many held-out items are labelled."""
    held_out = data.get("held_out") or []
    labelled, _ = split_held_out(held_out)
    return [
        f"development: {len(data['questions'])} questions",
        f"held-out: {len(labelled)} of {len(held_out)} labelled yet",
        f"off-topic: {len(data['off_topic'])} off_topic + "
        f"{len(data.get('near_miss') or [])} near_miss (reported separately)",
    ]


def sweep(
    grid: dict[str, list[Any]],
    articles: list[Article],
    embedder: Any,
    questions: list[dict[str, Any]],
    off_topic: list[str],
    held_out: list[dict[str, Any]] | None = None,
    near_miss: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every grid combination. One store per (chunk_size, overlap).

    The development, held-out and off-topic sets are scored independently.
    Unlabelled held-out items are skipped; near_miss false matches are counted
    apart from off_topic ones.
    """
    max_k = max(grid["top_k"])
    held_out, _ = split_held_out(held_out or [])
    q_vecs = [embedder.embed_query(q["question"]) for q in questions]
    ho_vecs = [embedder.embed_query(q["question"]) for q in held_out]
    off_vecs = [embedder.embed_query(q) for q in off_topic]
    near_vecs = [embedder.embed_query(q) for q in near_miss or []]
    rows: list[dict[str, Any]] = []

    for size, overlap in product(grid["chunk_size"], grid["chunk_overlap"]):
        if overlap >= size:
            log.warning("Skipping overlap %d >= chunk_size %d", overlap, size)
            continue
        store = build_store(articles, embedder, size, overlap)
        q_hits = [store.search(v, max_k) for v in q_vecs]
        ho_hits = [store.search(v, max_k) for v in ho_vecs]
        off_hits = [store.search(v, max_k) for v in off_vecs]
        near_hits = [store.search(v, max_k) for v in near_vecs]
        for top_k, threshold in product(grid["top_k"], grid["threshold"]):
            ranks = [
                rank_of_first_hit(rank_from_hits(h, top_k, threshold), q["expected"])
                for h, q in zip(q_hits, questions)
            ]
            ho_ranks = [
                rank_of_first_hit(rank_from_hits(h, top_k, threshold), q["expected"])
                for h, q in zip(ho_hits, held_out)
            ]
            rows.append(
                {
                    "chunk_size": size, "chunk_overlap": overlap, "top_k": top_k,
                    "threshold": threshold, "hit_rate": hit_rate_at_k(ranks, top_k),
                    "mrr": mrr(ranks),
                    "off_topic_false": sum(
                        1 for h in off_hits if rank_from_hits(h, top_k, threshold)
                    ),
                    "held_out_hit_rate": hit_rate_at_k(ho_ranks, top_k),
                    "held_out_mrr": mrr(ho_ranks),
                    "near_miss_false": sum(
                        1 for h in near_hits if rank_from_hits(h, top_k, threshold)
                    ),
                }
            )
    return rows


BASE_COLUMNS = ["chunk_size", "chunk_overlap", "top_k", "threshold",
                "hit_rate", "mrr", "off_topic_false"]
HELD_OUT_COLUMNS = ["held_out_hit_rate", "held_out_mrr", "near_miss_false"]


def format_table(rows: list[dict[str, Any]], show_held_out: bool = False) -> str:
    """Render the sweep rows as a fixed-width text table.

    The held-out and near_miss columns are appended only when show_held_out is
    True (that is, when at least one held-out item is labelled).
    """
    columns = BASE_COLUMNS + (HELD_OUT_COLUMNS if show_held_out else [])
    width = max(len(c) for c in columns)
    lines = ["  ".join(f"{c:>{width}}" for c in columns)]
    for row in rows:
        cells = []
        for c in columns:
            value = row[c]
            cells.append(f"{value:>{width}.3f}" if isinstance(value, float) else f"{value:>{width}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main() -> None:
    """Run the sweep with the real embedder and print the set summary and table."""
    data = load_questions()
    articles = load_articles()
    held_out = data.get("held_out") or []
    rows = sweep(
        data["grid"], articles, get_embedder(), data["questions"], data["off_topic"],
        held_out, data.get("near_miss") or [],
    )
    print("\n".join(summarize_sets(data)))
    print()
    print(format_table(rows, show_held_out=bool(split_held_out(held_out)[0])))


if __name__ == "__main__":
    main()
