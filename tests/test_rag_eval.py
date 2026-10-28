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
split_held_out = rag_eval.split_held_out
summarize_sets = rag_eval.summarize_sets
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
    assert len(data["questions"]) == 38 and len(data["off_topic"]) == 3
    assert len(data["held_out"]) == 10 and len(data["near_miss"]) == 5
    assert all(q["question"] and q["expected"] == [] for q in data["held_out"])
    known = {f"{a.topic}/{a.slug}" for a in load_articles()}
    for q in data["questions"]:
        assert q["question"] and set(q["expected"]) <= known
    assert set(data["grid"]) == {"chunk_size", "chunk_overlap", "top_k", "threshold"}


def test_main_prints_table_with_fake_embedder(monkeypatch, capsys):
    monkeypatch.setattr(rag_eval, "get_embedder", lambda: FakeEmbedder())
    rag_eval.main()
    out = capsys.readouterr().out
    assert "hit_rate" in out and len(out.splitlines()) > 5
    assert "held-out: 0 of 10 labelled yet" in out
    assert "held_out_hit_rate" not in out and "near_miss_false" not in out


HELD_OUT = [
    {"question": "shares company ownership", "expected": ["stocks/shares"]},
    {"question": "bond coupon maturity", "expected": []},   # unlabelled: skipped
]
NEAR_MISS = ["bond coupon maturity", "chocolate cake recipe"]   # first one false-matches
EASY = {"chunk_size": 200, "chunk_overlap": 20, "top_k": 1, "threshold": 0.3}


def _easy_row(**kwargs):
    rows = sweep(GRID, _toy_articles(), FakeEmbedder(), QUESTIONS, OFF_TOPIC, **kwargs)
    return next(r for r in rows if all(r[k] == v for k, v in EASY.items()))


def test_split_held_out_skips_and_counts_unlabelled():
    labelled, skipped = split_held_out(HELD_OUT)
    assert labelled == [HELD_OUT[0]] and skipped == 1
    assert split_held_out([{"question": "q", "expected": []}] * 10) == ([], 10)


def test_summarize_sets_reports_labelled_count():
    data = {"questions": QUESTIONS, "off_topic": OFF_TOPIC, "near_miss": NEAR_MISS, "held_out": HELD_OUT}
    lines = summarize_sets(data)
    assert lines[0] == "development: 3 questions"
    assert lines[1] == "held-out: 1 of 2 labelled yet"
    assert "1 off_topic + 2 near_miss" in lines[2]
    assert summarize_sets({"questions": [], "off_topic": []})[1] == "held-out: 0 of 0 labelled yet"


def test_unlabelled_held_out_scores_zero_and_leaves_other_sets_alone():
    row = _easy_row(held_out=[{"question": "q", "expected": []}] * 10)
    assert row["held_out_hit_rate"] == 0.0 and row["held_out_mrr"] == 0.0
    assert row["hit_rate"] == pytest.approx(2 / 3) and row["off_topic_false"] == 0


def test_labelled_held_out_item_is_scored_correctly():
    row = _easy_row(held_out=HELD_OUT)   # only the labelled item counts, found at rank 1
    assert row["held_out_hit_rate"] == 1.0 and row["held_out_mrr"] == 1.0
    wrong = _easy_row(held_out=[{"question": "bond coupon maturity", "expected": ["stocks/shares"]}])
    assert wrong["held_out_hit_rate"] == 0.0 and wrong["held_out_mrr"] == 0.0


def test_three_sets_are_scored_independently():
    row = _easy_row(held_out=HELD_OUT, near_miss=NEAR_MISS)
    assert row["hit_rate"] == pytest.approx(2 / 3)   # development only
    assert row["held_out_hit_rate"] == 1.0           # held-out only
    assert row["off_topic_false"] == 0 and row["near_miss_false"] == 1


def test_near_miss_false_is_counted_apart_from_off_topic():
    rows = sweep(GRID, _toy_articles(), FakeEmbedder(), QUESTIONS, ["bond coupon maturity"],
                 near_miss=["chocolate cake recipe"])
    easy = next(r for r in rows if all(r[k] == v for k, v in EASY.items()))
    assert easy["off_topic_false"] == 1 and easy["near_miss_false"] == 0


def test_format_table_shows_held_out_columns_only_when_asked():
    rows = sweep(GRID, _toy_articles(), FakeEmbedder(), QUESTIONS, OFF_TOPIC, HELD_OUT, NEAR_MISS)
    plain = format_table(rows).splitlines()
    full = format_table(rows, show_held_out=True).splitlines()
    assert "near_miss_false" not in plain[0] and "held_out_mrr" not in plain[0]
    for name in ("held_out_hit_rate", "held_out_mrr", "near_miss_false"):
        assert name in full[0]
    assert len({len(line) for line in full}) == 1   # fixed width


def test_main_shows_extra_columns_when_a_held_out_item_is_labelled(monkeypatch, capsys):
    data = load_questions()
    data["held_out"][0]["expected"] = ["stocks/what-is-a-stock"]
    monkeypatch.setattr(rag_eval, "load_questions", lambda: data)
    monkeypatch.setattr(rag_eval, "get_embedder", lambda: FakeEmbedder())
    rag_eval.main()
    out = capsys.readouterr().out
    assert "held-out: 1 of 10 labelled yet" in out
    assert "held_out_hit_rate" in out and "near_miss_false" in out
