import importlib.util
from pathlib import Path

import pytest

from fake_embedder import FakeEmbedder
from src.rag.ingest import Article, load_articles

# Load the harness by file path: "tests" is not an importable package here.
_spec = importlib.util.spec_from_file_location(
    "rag_eval_harness", Path(__file__).parent / "rag_eval" / "rag_eval.py"
)
rag_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag_eval)

article_id = rag_eval.article_id
format_table = rag_eval.format_table
hit_rate_at_k = rag_eval.hit_rate_at_k
load_questions = rag_eval.load_questions
mrr = rag_eval.mrr
rank_from_hits = rag_eval.rank_from_hits
rank_of_first_hit = rag_eval.rank_of_first_hit
sweep = rag_eval.sweep

REFS = [{"name": "SEBI", "url": "https://www.sebi.gov.in"}]


def test_article_id():
    assert article_id("bonds/how-bonds-work#3") == "bonds/how-bonds-work"


def test_rank_of_first_hit():
    assert rank_of_first_hit(["a", "b", "c"], ["c"]) == 3
    assert rank_of_first_hit(["a", "b", "c"], ["c", "b"]) == 2
    assert rank_of_first_hit(["a"], ["z"]) is None
    assert rank_of_first_hit([], ["z"]) is None


def test_hit_rate_at_k():
    ranks = [1, None, 3, 2]
    assert hit_rate_at_k(ranks, 1) == 0.25
    assert hit_rate_at_k(ranks, 2) == 0.5
    assert hit_rate_at_k(ranks, 3) == 0.75
    assert hit_rate_at_k([], 3) == 0.0


def test_mrr():
    assert mrr([1, None, 2]) == pytest.approx((1 + 0 + 0.5) / 3)
    assert mrr([1, 1]) == 1.0
    assert mrr([]) == 0.0


def test_rank_from_hits_applies_threshold_topk_and_dedupes():
    hits = [
        ({"chunk_id": "t/a#0"}, 0.9),
        ({"chunk_id": "t/a#1"}, 0.8),   # same article, counted once
        ({"chunk_id": "t/b#0"}, 0.5),
        ({"chunk_id": "t/c#0"}, 0.2),   # under threshold
    ]
    assert rank_from_hits(hits, top_k=10, threshold=0.3) == ["t/a", "t/b"]
    assert rank_from_hits(hits, top_k=2, threshold=0.3) == ["t/a"]   # top_k counts chunks
    assert rank_from_hits(hits, top_k=10, threshold=0.95) == []


def _toy_articles():
    return [
        Article(None, "shares", "Shares", "stocks", REFS, "shares company ownership dividend"),
        Article(None, "bonds", "Bonds", "bonds", REFS, "bond issuer interest coupon maturity"),
    ]


QUESTIONS = [
    {"question": "shares company ownership", "expected": ["stocks/shares"]},
    {"question": "bond coupon maturity", "expected": ["bonds/bonds"]},
    {"question": "bond coupon maturity", "expected": ["stocks/shares"]},   # deliberately wrong label
]
OFF_TOPIC = ["chocolate cake recipe"]
GRID = {"chunk_size": [200, 50], "chunk_overlap": [20, 60], "top_k": [1, 2], "threshold": [0.3, 0.99]}


def test_sweep_metrics_on_toy_fixture():
    rows = sweep(GRID, _toy_articles(), FakeEmbedder(), QUESTIONS, OFF_TOPIC)
    # overlap 60 >= chunk_size 50 is skipped: 3 valid size/overlap pairs x 2 x 2
    assert len(rows) == 3 * 2 * 2
    easy = next(r for r in rows if r["chunk_size"] == 200 and r["top_k"] == 1 and r["threshold"] == 0.3)
    assert easy["hit_rate"] == pytest.approx(2 / 3)
    assert easy["mrr"] == pytest.approx(2 / 3)
    assert easy["off_topic_false"] == 0
    strict = next(r for r in rows if r["threshold"] == 0.99 and r["chunk_size"] == 200)
    assert strict["hit_rate"] == 0.0 and strict["mrr"] == 0.0


def test_format_table():
    rows = sweep(GRID, _toy_articles(), FakeEmbedder(), QUESTIONS, OFF_TOPIC)
    lines = format_table(rows).splitlines()
    assert len(lines) == len(rows) + 1
    assert "hit_rate" in lines[0] and "mrr" in lines[0] and "off_topic_false" in lines[0]


def test_real_questions_file_is_well_formed():
    data = load_questions()
    assert len(data["questions"]) == 12 and len(data["off_topic"]) == 3
    known = {f"{a.topic}/{a.slug}" for a in load_articles()}
    for q in data["questions"]:
        assert q["question"] and set(q["expected"]) <= known
    assert set(data["grid"]) == {"chunk_size", "chunk_overlap", "top_k", "threshold"}


def test_main_prints_table_with_fake_embedder(monkeypatch, capsys):
    monkeypatch.setattr(rag_eval, "get_embedder", lambda: FakeEmbedder())
    rag_eval.main()
    out = capsys.readouterr().out
    assert "hit_rate" in out and len(out.splitlines()) > 5
