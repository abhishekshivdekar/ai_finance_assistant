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
    # overlap 60 >= chunk_size 50 is skipped: 3 valid size/overlap pairs x 2 x 2 x (diversity on/off)
    assert len(rows) == 3 * 2 * 2 * 2
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
    assert len(data["questions"]) == 121 and len(data["off_topic"]) == 3
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
    assert "held_out_hit_rate" not in out and "near_miss_false" in out
    assert "Diagnostics for config.yaml: chunk_size=800, chunk_overlap=120, top_k=4" in out


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
    assert "held_out_hit_rate" not in plain[0] and "held_out_mrr" not in plain[0]
    assert "near_miss_false" in plain[0] and "near_miss_false" in full[0]   # always shown
    for name in ("held_out_hit_rate", "held_out_mrr"):
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


DIAG_CFG = dict(chunk_size=200, chunk_overlap=20, top_k=1, threshold=0.3)


def _diag(questions, off_topic=OFF_TOPIC, near_miss=("chocolate cake recipe",), **cfg):
    data = {"questions": questions, "off_topic": off_topic, "near_miss": list(near_miss)}
    return rag_eval.diagnostics(data, _toy_articles(), FakeEmbedder(), **{**DIAG_CFG, **cfg})


def test_best_by_article_keeps_highest_chunk_score():
    hits = [({"chunk_id": "t/a#0"}, 0.4), ({"chunk_id": "t/a#1"}, 0.7), ({"chunk_id": "t/b#0"}, 0.5)]
    assert rag_eval.best_by_article(hits) == {"t/a": 0.7, "t/b": 0.5}


def test_diagnostics_lists_each_miss_with_expected_and_returned():
    report = _diag(QUESTIONS)   # the third question carries a deliberately wrong label
    assert "chunk_size=200, chunk_overlap=20, top_k=1, score_threshold=0.3" in report
    assert "a) Development misses: 1 of 3" in report
    assert report.count("  Q: ") == 1
    assert "expected: stocks/shares" in report
    assert "returned: bonds/bonds (" in report   # what was actually returned, with a score


def test_diagnostics_miss_with_nothing_above_threshold():
    report = _diag(QUESTIONS[:1], threshold=0.99)
    assert "a) Development misses: 1 of 1" in report
    assert "returned: nothing above threshold" in report


def test_diagnostics_reports_score_gap_and_says_below():
    report = _diag(QUESTIONS[:2])   # both correctly labelled
    assert "a) Development misses: 0 of 2\n  none" in report
    assert "answerable (expected article): lowest 0.775, median 0.820" in report
    assert "off_topic: highest" in report and "near_miss: highest" in report
    assert "is BELOW the lowest answerable score" in report


def test_diagnostics_says_not_below_when_off_topic_scores_as_high():
    report = _diag(QUESTIONS[:1], off_topic=["shares company ownership"])   # off-topic = a real match
    assert "is NOT below the lowest answerable score" in report


def test_diagnostics_handles_empty_sets():
    report = _diag([], off_topic=[], near_miss=())
    assert "Development misses: 0 of 0" in report
    assert "lowest" not in report and "highest" not in report


# --- diversity switch, shared rule, best row, embedding model ---
from src.rag import retriever as retriever_module
from src.rag.retriever import select_chunks


def _hit(chunk_id, score):
    return ({"chunk_id": chunk_id, "topic": chunk_id.split("/")[0]}, score)


CROWDED = [
    _hit("glossary/g#0", 0.9), _hit("glossary/h#0", 0.8), _hit("stocks/a#0", 0.7),
    _hit("stocks/a#1", 0.6), _hit("stocks/a#2", 0.5), _hit("bonds/b#0", 0.4),
]


def test_rank_from_hits_diversity_switch():
    # caps on (config: 2 per article, 1 glossary): h#0 and a#2 are dropped
    assert rank_from_hits(CROWDED, 4, 0.3, diversity=True) == ["glossary/g", "stocks/a", "bonds/b"]
    # caps off: top 4 chunks are g, h, a, a
    assert rank_from_hits(CROWDED, 4, 0.3, diversity=False) == ["glossary/g", "glossary/h", "stocks/a"]


def test_harness_and_retrieve_agree_on_the_same_hits(monkeypatch):
    records = [({**rec, "text": "t", "title": "T", "references": REFS}, score) for rec, score in CROWDED]

    class Store:
        def __len__(self):
            return len(records)

        def search(self, vector, k):
            return records[:k]

    for top_k in (2, 3, 4, 6):
        chunks = retriever_module.retrieve("anything", top_k=top_k, embedder=FakeEmbedder(), store=Store())
        app_ids = [c.chunk_id for c in chunks]
        rag = rag_eval.get_config()["rag"]
        shared = select_chunks(
            records, top_k, rag["score_threshold"],
            rag["max_chunks_per_article"], rag["max_chunks_per_topic"],
        )
        assert app_ids == [rec["chunk_id"] for rec, _ in shared]
        harness = rag_eval.rank_from_hits(records, top_k, rag["score_threshold"], diversity=True)
        assert harness == list(dict.fromkeys(i.split("#")[0] for i in app_ids))


def test_sweep_reports_diversity_on_and_off():
    rows = sweep(GRID, _toy_articles(), FakeEmbedder(), QUESTIONS, OFF_TOPIC)
    assert {r["diversity"] for r in rows} == {"on", "off"}
    assert "diversity" in format_table(rows).splitlines()[0]
    only_on = sweep(GRID, _toy_articles(), FakeEmbedder(), QUESTIONS, OFF_TOPIC, diversity_modes=(True,))
    assert {r["diversity"] for r in only_on} == {"on"} and len(only_on) == len(rows) // 2


def test_best_row_prefers_clean_rows_then_hit_rate_then_mrr():
    base = {"off_topic_false": 0, "near_miss_false": 0, "hit_rate": 0.5, "mrr": 0.4}
    leaky = {**base, "hit_rate": 0.9, "off_topic_false": 1}
    clean_low = {**base, "hit_rate": 0.6}
    clean_high = {**base, "hit_rate": 0.7, "mrr": 0.3}
    clean_high_mrr = {**base, "hit_rate": 0.7, "mrr": 0.5}
    assert rag_eval.best_row([leaky, clean_low]) is clean_low
    assert rag_eval.best_row([clean_low, clean_high, clean_high_mrr]) is clean_high_mrr
    assert rag_eval.best_row([leaky]) is leaky
    assert rag_eval.best_row([]) is None


def test_main_prints_diagnostics_for_config_and_best_row(monkeypatch, capsys):
    monkeypatch.setattr(rag_eval, "get_embedder", lambda: FakeEmbedder())
    rag_eval.main()
    out = capsys.readouterr().out
    assert "Diagnostics for config.yaml:" in out and "Diagnostics for best sweep row:" in out
    assert out.count("diversity=") == 2   # one heading per diagnostics block
    assert " on " in out and " off " in out   # table rows for both modes


class FakeSTEmbedder(FakeEmbedder):
    """Stands in for SentenceTransformerEmbedder; records the model name."""

    created: list = []

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        FakeSTEmbedder.created.append(model_name)


def test_make_embedder_default_uses_the_shared_embedder(monkeypatch):
    shared = FakeEmbedder()
    monkeypatch.setattr(rag_eval, "get_embedder", lambda: shared)
    assert rag_eval.make_embedder(None) is shared


def test_embedding_model_override_builds_its_own_embedder_and_leaves_the_app_alone(monkeypatch, capsys):
    FakeSTEmbedder.created = []
    monkeypatch.setattr(rag_eval, "SentenceTransformerEmbedder", FakeSTEmbedder)

    def boom(*args, **kwargs):
        raise AssertionError("the app's embedder / index must not be touched")

    monkeypatch.setattr(rag_eval, "get_embedder", boom)
    import src.rag.ingest as ingest
    monkeypatch.setattr(ingest, "build_index", boom)
    monkeypatch.setattr(ingest, "write_manifest", boom)

    rag_eval.main("some/other-model")
    out = capsys.readouterr().out
    assert FakeSTEmbedder.created == ["some/other-model"]
    assert "embedding model: some/other-model" in out


def test_run_rag_eval_passes_the_constant_to_main(monkeypatch):
    import run_rag_eval
    assert run_rag_eval.EMBEDDING_MODEL is None
    seen = []
    monkeypatch.setattr(run_rag_eval.runpy, "run_path", lambda path, run_name: {"main": seen.append})
    run_rag_eval.run()
    monkeypatch.setattr(run_rag_eval, "EMBEDDING_MODEL", "some/other-model")
    run_rag_eval.run()
    assert seen == [None, "some/other-model"]
