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
off_topic plus near_miss. near_miss_false is always shown; held_out_hit_rate
and held_out_mrr appear only once at least one held-out item is labelled.

Every question goes through the SAME retrieval path as the app: `merge_retrieval`
over `retrieve` (acronym expansion from rag.query_expansions, written query first,
threshold, caps, top_k). Only the store, the embedder and the rag settings are the
eval's own (`production_chunks`), so the numbers cannot drift from production.

The "diversity" column says whether the result-diversity caps of the app
(rag.max_chunks_per_article / rag.max_chunks_per_topic, applied by the shared
`select_chunks`) were on or off, so the sweep shows what the caps are worth.

After the table, a diagnostics report lists the development misses and the
score gap between answerable and off-topic/near-miss questions (see
`diagnostics`), for the configuration currently in config.yaml and for the
best row of the sweep.
"""
from __future__ import annotations

from functools import partial
from itertools import product
from statistics import median
from pathlib import Path
from typing import Any

import yaml

from src.rag.embeddings import SentenceTransformerEmbedder, get_embedder
from src.rag.ingest import Article, build_store, load_articles
from src.rag.retriever import RetrievedChunk, expand_query, merge_retrieval, retrieve
from src.utils.config import get_config
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


def make_embedder(model_name: str | None = None) -> Any:
    """Embedder for the eval: its own instance, with the progress bar switched off.

    The model is `model_name`, or embeddings.model from config.yaml when None.
    The eval gets its OWN SentenceTransformerEmbedder (show_progress_bar=False,
    so the output is not flooded), never the cached get_embedder() singleton the
    app uses. The eval only builds in-memory stores (build_store) and never
    saves, so the app's index and manifest are not affected either.
    """
    return SentenceTransformerEmbedder(
        model_name or get_config()["embeddings"]["model"], show_progress_bar=False
    )


class CachedQueryEmbedder:
    """Wraps an embedder and remembers query vectors (the sweep embeds the same question for every row)."""

    def __init__(self, embedder: Any) -> None:
        self.embedder, self.cache = embedder, {}

    def embed_query(self, text: str) -> Any:
        """The query vector, computed once per text."""
        if text not in self.cache:
            self.cache[text] = self.embedder.embed_query(text)
        return self.cache[text]

    def __getattr__(self, name: str) -> Any:
        """Everything else (embed_documents ...) goes to the wrapped embedder."""
        return getattr(self.embedder, name)


def rag_settings(top_k: int, threshold: float, diversity: bool = True) -> dict[str, Any]:
    """The rag section of config.yaml with this row's top_k and threshold; diversity=False turns the caps off."""
    rag = {**get_config()["rag"], "top_k": top_k, "score_threshold": threshold}
    if not diversity:
        rag.update(max_chunks_per_article=None, max_chunks_per_topic={})
    return rag


def production_chunks(
    question: str, embedder: Any, store: Any, rag: dict[str, Any], topic: str | None = None
) -> list[RetrievedChunk]:
    """The chunks the app would return for `question`: merge_retrieval over retrieve, on the eval's store.

    Design pattern: SINGLE SOURCE OF TRUTH. This calls the function the agents call; only the store,
    embedder and rag settings differ. Example: "What is LTCG?" is also searched as
    "What is long-term capital gains tax?", exactly as in the app.
    """
    search = partial(retrieve, embedder=embedder, store=store, rag=rag)
    return merge_retrieval(search, [question], topic, rag=rag)


def rank_articles(chunks: list[RetrievedChunk]) -> list[str]:
    """Article ids in rank order of the chunks (an article counts once, at its first chunk)."""
    ranked: list[str] = []
    for chunk in chunks:
        art = article_id(chunk.chunk_id)
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
    diversity_modes: tuple[bool, ...] = (True, False),
    configs: list[tuple[int, int, int, float]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every grid combination. One store per (chunk_size, overlap).

    `configs`, a list of (chunk_size, chunk_overlap, top_k, threshold), replaces
    the grid product so that only those configurations run.

    Every combination is scored once per entry of `diversity_modes` (caps on,
    caps off), shown in the "diversity" column. Each question goes through
    production_chunks (the app's merge_retrieval over retrieve).

    The development, held-out and off-topic sets are scored independently.
    Unlabelled held-out items are skipped; near_miss false matches are counted
    apart from off_topic ones.
    """
    held_out, _ = split_held_out(held_out or [])
    embedder = CachedQueryEmbedder(embedder)
    rows: list[dict[str, Any]] = []

    if configs is None:
        sizes = list(product(grid["chunk_size"], grid["chunk_overlap"]))
        settings = {pair: list(product(grid["top_k"], grid["threshold"])) for pair in sizes}
    else:
        sizes = list(dict.fromkeys((c[0], c[1]) for c in configs))
        settings = {pair: [(c[2], c[3]) for c in configs if (c[0], c[1]) == pair] for pair in sizes}

    for size, overlap in sizes:
        if overlap >= size:
            log.warning("Skipping overlap %d >= chunk_size %d", overlap, size)
            continue
        store = build_store(articles, embedder, size, overlap)
        for (top_k, threshold), diversity in product(settings[(size, overlap)], diversity_modes):
            rag = rag_settings(top_k, threshold, diversity)

            def rank(question: str) -> list[str]:
                return rank_articles(production_chunks(question, embedder, store, rag))

            ranks = [rank_of_first_hit(rank(q["question"]), q["expected"]) for q in questions]
            ho_ranks = [rank_of_first_hit(rank(q["question"]), q["expected"]) for q in held_out]
            rows.append(
                {
                    "chunk_size": size, "chunk_overlap": overlap, "top_k": top_k,
                    "threshold": threshold, "diversity": "on" if diversity else "off",
                    "hit_rate": hit_rate_at_k(ranks, top_k), "mrr": mrr(ranks),
                    "off_topic_false": sum(1 for q in off_topic if rank(q)),
                    "held_out_hit_rate": hit_rate_at_k(ho_ranks, top_k),
                    "held_out_mrr": mrr(ho_ranks),
                    "near_miss_false": sum(1 for q in near_miss or [] if rank(q)),
                }
            )
    return rows


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The best sweep row: no false matches first, then hit_rate, then mrr.

    Rows that let an off-topic or near-miss question through lose to rows that
    do not, because an answer built on an unrelated chunk is worse than none.
    Ties keep the earlier row.
    """
    def key(row: dict[str, Any]) -> tuple[bool, float, float]:
        clean = row["off_topic_false"] + row["near_miss_false"] == 0
        return clean, row["hit_rate"], row["mrr"]

    best: dict[str, Any] | None = None
    for row in rows:
        if best is None or key(row) > key(best):
            best = row
    return best


CONFIG_COLUMNS = ["chunk_size", "chunk_overlap", "top_k", "threshold", "diversity"]
DEV_COLUMNS = ["hit_rate", "mrr", "off_topic_false", "near_miss_false"]
HELD_OUT_COLUMNS = ["held_out_hit_rate", "held_out_mrr"]


def format_table(rows: list[dict[str, Any]], show_held_out: bool = False) -> str:
    """Render the sweep rows as a fixed-width text table.

    The held-out columns are the primary ones: they come right after the
    configuration, and appear only when show_held_out is True (that is, when at
    least one held-out item is labelled). near_miss_false is always shown.
    """
    columns = CONFIG_COLUMNS + (HELD_OUT_COLUMNS if show_held_out else []) + DEV_COLUMNS
    width = max(len(c) for c in columns)
    lines = ["  ".join(f"{c:>{width}}" for c in columns)]
    for row in rows:
        cells = []
        for c in columns:
            value = row[c]
            cells.append(f"{value:>{width}.3f}" if isinstance(value, float) else f"{value:>{width}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def best_by_article(hits: list[tuple[dict[str, Any], float]]) -> dict[str, float]:
    """Best chunk score per article id, in order of first appearance."""
    best: dict[str, float] = {}
    for rec, score in hits:
        art = article_id(rec["chunk_id"])
        best[art] = max(score, best.get(art, score))
    return best


def _fmt_scores(scores: list[tuple[str, float]]) -> str:
    return ", ".join(f"{art} ({score:.3f})" for art, score in scores)


def diagnostics(
    data: dict[str, Any],
    articles: list[Article],
    embedder: Any,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    threshold: float,
    diversity: bool = True,
    label: str = "config.yaml",
) -> str:
    """Report for ONE configuration: development misses and the score gap.

    `label` names the configuration in the heading ("config.yaml" or "best
    sweep row"); `diversity` says whether the result caps apply.

    (a) Each development question whose expected article is not returned, with
    the top 3 articles actually returned (best chunk score each). (b) The lowest
    and median best-chunk score of the expected article over answerable
    questions, against the highest best-chunk score of any off_topic or
    near_miss question, and whether the highest of those is below the lowest
    answerable score (a threshold between them separates the two sets).
    """
    embedder = CachedQueryEmbedder(embedder)
    store = build_store(articles, embedder, chunk_size, chunk_overlap)
    every_chunk = len(store)
    rag = rag_settings(top_k, threshold, diversity)
    expansions = rag.get("query_expansions") or {}

    def search(question: str) -> list[tuple[dict[str, Any], float]]:
        """Raw scores of every chunk for the question as written and, like the app, its acronym-expanded form."""
        texts = [question, expand_query(question, expansions)]
        return [hit for text in texts if text for hit in store.search(embedder.embed_query(text), every_chunk)]

    lines = [
        f"Diagnostics for {label}: chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}, top_k={top_k}, score_threshold={threshold}, "
        f"diversity={'on' if diversity else 'off'}",
        "",
    ]
    questions = data["questions"]
    answerable: list[float] = []
    misses: list[str] = []
    for q in questions:
        hits = search(q["question"])
        chunks = production_chunks(q["question"], embedder, store, rag)
        returned = rank_articles(chunks)
        best = best_by_article(hits)
        answerable.append(max((best.get(art, 0.0) for art in q["expected"]), default=0.0))
        if rank_of_first_hit(returned, q["expected"]) is None:
            kept = best_by_article([({"chunk_id": c.chunk_id}, c.score) for c in chunks])
            shown = _fmt_scores(list(kept.items())[:3]) or "nothing above threshold"
            misses.append(
                f"  Q: {q['question']}\n"
                f"     expected: {', '.join(q['expected'])}\n"
                f"     returned: {shown}"
            )
    lines.append(f"a) Development misses: {len(misses)} of {len(questions)}")
    lines.extend(misses or ["  none"])

    lines += ["", "b) Score gap (best chunk score per question)"]
    if answerable:
        lowest = min(answerable)
        lines.append(
            f"  answerable (expected article): lowest {lowest:.3f}, "
            f"median {median(answerable):.3f}"
        )
    unanswerable: list[float] = []
    for label, items in (("off_topic", data["off_topic"]), ("near_miss", data.get("near_miss") or [])):
        scores = [max((s for _, s in search(item)), default=0.0) for item in items]
        unanswerable += scores
        if scores:
            lines.append(f"  {label}: highest {max(scores):.3f}")
    if answerable and unanswerable:
        top_off = max(unanswerable)
        verdict = "BELOW" if top_off < lowest else "NOT below"
        lines.append(
            f"  The highest off-topic/near-miss score ({top_off:.3f}) is {verdict} "
            f"the lowest answerable score ({lowest:.3f})."
        )
    return "\n".join(lines)


def question_report(
    title: str,
    items: list[dict[str, Any]],
    articles: list[Article],
    embedder: Any,
    chunk_size: int,
    chunk_overlap: int,
    rag: dict[str, Any],
) -> str:
    """One block per question for ONE configuration: expected article (if any), its rank, and every returned chunk.

    Used for the held-out and near-miss sets, so a reader sees what the app returns, not only the totals.
    A near-miss item has no `expected`; any chunk listed for it is a false match.
    """
    embedder = CachedQueryEmbedder(embedder)
    store = build_store(articles, embedder, chunk_size, chunk_overlap)
    lines = [f"{title} ({len(items)} questions)"]
    for item in items:
        chunks = production_chunks(item["question"], embedder, store, rag)
        expected = item.get("expected")
        rank = rank_of_first_hit(rank_articles(chunks), expected) if expected else None
        verdict = "" if not expected else f"  expected: {', '.join(expected)}  rank: {rank if rank else 'MISS'}"
        lines.append(f"  Q: {item['question']}{verdict}")
        lines.extend(f"     {c.score:.3f}  {c.chunk_id}" for c in chunks)
        if not chunks:
            lines.append("     nothing above threshold")
    return "\n".join(lines)


def main(
    embedding_model: str | None = None,
    configs: list[tuple[int, int, int, float]] | None = None,
) -> None:
    """Run the sweep; print the summary, table and diagnostics (config row, best row).

    embedding_model None uses config.yaml; a name builds the eval stores with
    that model instead (see make_embedder). The app's saved index is untouched.
    `configs` (chunk_size, chunk_overlap, top_k, threshold) limits the sweep to
    those configurations instead of the questions.yaml grid.
    """
    data = load_questions()
    articles = load_articles()
    embedder = make_embedder(embedding_model)
    held_out = data.get("held_out") or []
    rows = sweep(
        data["grid"], articles, embedder, data["questions"], data["off_topic"],
        held_out, data.get("near_miss") or [], configs=configs,
    )
    print(f"embedding model: {embedding_model or get_config()['embeddings']['model']}")
    print("\n".join(summarize_sets(data)))
    print()
    print(format_table(rows, show_held_out=bool(split_held_out(held_out)[0])))
    cfg = get_config()["rag"]
    print()
    print(diagnostics(
        data, articles, embedder, cfg["chunk_size"], cfg["chunk_overlap"],
        cfg["top_k"], cfg["score_threshold"],
    ))
    rag = rag_settings(cfg["top_k"], cfg["score_threshold"])
    near = [{"question": q} if isinstance(q, str) else q for q in data.get("near_miss") or []]
    labelled_held_out = split_held_out(held_out)[0]
    for title, items in (("Held-out, config.yaml row", labelled_held_out), ("Near-miss, config.yaml row", near)):
        print()
        print(question_report(title, items, articles, embedder, cfg["chunk_size"], cfg["chunk_overlap"], rag))
    best = best_row(rows)
    if best:
        print()
        print(diagnostics(
            data, articles, embedder, best["chunk_size"], best["chunk_overlap"],
            best["top_k"], best["threshold"], best["diversity"] == "on", "best sweep row",
        ))


if __name__ == "__main__":
    main()
