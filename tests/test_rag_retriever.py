import copy
import logging

import pytest

from src.core.models import Source
from src.rag import retriever
from src.rag.ingest import Article, build_store
from src.rag.retriever import RetrievedChunk, format_context, retrieve, to_sources
from src.utils.config import get_config
from fake_embedder import FakeEmbedder

REFS = [{"name": "SEBI", "url": "https://www.sebi.gov.in"}]


def _article(slug, topic, body):
    return Article(path=None, slug=slug, title=slug.title(), topic=topic, references=REFS, body=body)


@pytest.fixture
def store():
    arts = [
        _article("shares", "stocks", "shares company ownership dividend investors market price"),
        _article("bonds", "bonds", "bond issuer interest investors market price coupon"),
    ]
    return build_store(arts, FakeEmbedder(), 800, 120)


class SpyStore:
    def __init__(self, inner):
        self.inner, self.requested = inner, []

    def __len__(self):
        return len(self.inner)

    def search(self, vector, k):
        self.requested.append(k)
        return self.inner.search(vector, k)


def _retrieve(query, store, **kwargs):
    return retrieve(query, embedder=FakeEmbedder(), store=store, **kwargs)


def _patch_rag(monkeypatch, **overrides):
    cfg = copy.deepcopy(get_config())
    cfg["rag"].update(overrides)
    monkeypatch.setattr(retriever, "get_config", lambda: cfg)


def test_results_carry_metadata(store):
    chunks = _retrieve("bond coupon maturity interest issuer", store)
    top = chunks[0]
    assert isinstance(top, RetrievedChunk)
    assert (top.title, top.topic, top.chunk_id) == ("Bonds", "bonds", "bonds/bonds#0")
    assert top.references == REFS and "coupon" in top.text and top.score > 0.35


def test_scores_are_sorted_best_first(store):
    chunks = _retrieve("investors market price shares company ownership", store, top_k=2)
    assert [c.chunk_id for c in chunks] == ["stocks/shares#0", "bonds/bonds#0"]
    assert chunks[0].score >= chunks[1].score


QUERY = "investors market price shares company ownership"  # closest to stocks, still above 0.35 for bonds


def test_topic_filter_keeps_only_that_topic(store):
    chunks = _retrieve(QUERY, store, topic="bonds")
    assert chunks and {c.topic for c in chunks} == {"bonds"}


def test_topic_filter_searches_all_chunks(store):
    spy = SpyStore(store)
    _retrieve(QUERY, spy, topic="bonds", top_k=1)
    _retrieve(QUERY, spy, top_k=1)
    assert spy.requested == [len(store), len(store)]   # no-topic search also scans all (caps need spare hits)


class EmptyStore:
    def __init__(self):
        self.searched = False

    def __len__(self):
        return 0

    def search(self, vector, k):
        self.searched = True
        return []


def test_explicit_empty_store_is_used_not_replaced(store, monkeypatch):
    monkeypatch.setattr(retriever, "get_vector_store", lambda: store)
    empty = EmptyStore()
    assert _retrieve("bond coupon maturity interest issuer", empty) == []
    assert empty.searched


def test_topic_filter_finds_a_chunk_that_plain_top_k_would_miss(store):
    assert _retrieve(QUERY, store, top_k=1)[0].topic == "stocks"
    assert [c.topic for c in _retrieve(QUERY, store, topic="bonds", top_k=1)] == ["bonds"]


def test_threshold_gives_empty_list(store):
    assert _retrieve("chocolate cake recipe", store) == []


def test_threshold_is_read_from_config(store, monkeypatch):
    _patch_rag(monkeypatch, score_threshold=0.99)
    assert _retrieve("bond coupon maturity", store) == []


def test_top_k_limits_results(store, monkeypatch):
    _patch_rag(monkeypatch, score_threshold=0.0)
    assert len(_retrieve(QUERY, store, top_k=1)) == 1
    assert len(_retrieve(QUERY, store, top_k=5)) == 2


def test_empty_query_and_unknown_topic(store, caplog):
    assert _retrieve("   ", store) == []
    assert _retrieve("", store) == []
    with caplog.at_level(logging.WARNING):
        assert _retrieve(QUERY, store, topic="astrology") == []
    assert "Unknown topic" in caplog.text


def test_defaults_use_shared_embedder_and_store(store, monkeypatch):
    monkeypatch.setattr(retriever, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(retriever, "get_vector_store", lambda: store)
    assert retrieve("bond coupon maturity interest issuer")[0].topic == "bonds"


def _chunk(chunk_id="t/a#0", refs=REFS):
    return RetrievedChunk("body text", "Title", "stocks", refs, chunk_id, 0.5)


def test_format_context_numbers_blocks():
    text = format_context([_chunk("t/a#0"), _chunk("t/b#0")])
    assert text == "[1] Title\nbody text\n\n[2] Title\nbody text"
    assert format_context([]) == ""


def test_to_sources_reuses_source_model():
    sources = to_sources([_chunk(), _chunk("t/b#1", refs=[])])
    assert sources[0] == Source(
        title="Title", url="https://www.sebi.gov.in", chunk_id="t/a#0", score=0.5
    )
    assert sources[1]["url"] is None and sources[1]["chunk_id"] == "t/b#1"


# --- select_chunks: the shared diversity rule (fake data, no embedder) ---
from src.rag.retriever import diversity_caps, select_chunks


def _hit(chunk_id, score):
    return ({"chunk_id": chunk_id, "topic": chunk_id.split("/")[0]}, score)


HITS = [
    _hit("stocks/a#0", 0.9), _hit("stocks/a#1", 0.8), _hit("stocks/a#2", 0.7),
    _hit("glossary/g#0", 0.6), _hit("glossary/h#0", 0.5), _hit("bonds/b#0", 0.4),
    _hit("bonds/c#0", 0.2),
]


def _ids(hits):
    return [rec["chunk_id"] for rec, _ in hits]


def test_select_chunks_caps_per_article_and_per_topic():
    kept = select_chunks(HITS, 10, 0.3, max_per_article=2, max_per_topic={"glossary": 1})
    # a#2 is a third chunk of one article; glossary/h#0 is a second glossary chunk
    assert _ids(kept) == ["stocks/a#0", "stocks/a#1", "glossary/g#0", "bonds/b#0"]


def test_select_chunks_cap_off_when_config_is_empty():
    for article_cap, topic_cap in ((None, {}), (0, None), (None, None)):
        kept = select_chunks(HITS, 10, 0.3, article_cap, topic_cap)
        assert _ids(kept) == [r["chunk_id"] for r, s in HITS if s >= 0.3]


def test_select_chunks_keeps_order_threshold_and_top_k():
    shuffled = [HITS[3], HITS[0], HITS[6], HITS[1]]
    kept = select_chunks(shuffled, 10, 0.3)
    assert [s for _, s in kept] == [0.9, 0.8, 0.6]            # sorted best first, 0.2 dropped
    assert _ids(select_chunks(HITS, 2, 0.3, 2, {})) == ["stocks/a#0", "stocks/a#1"]
    assert select_chunks(HITS, 10, 0.95, 2, {}) == []


def test_capped_hits_make_room_for_lower_ranked_articles():
    # without the cap top_k=3 is all stocks/a; with it the third slot goes to glossary
    assert _ids(select_chunks(HITS, 3, 0.3)) == ["stocks/a#0", "stocks/a#1", "stocks/a#2"]
    assert _ids(select_chunks(HITS, 3, 0.3, 2, {"glossary": 1})) == [
        "stocks/a#0", "stocks/a#1", "glossary/g#0"]


def test_diversity_caps_reads_config():
    assert diversity_caps({"max_chunks_per_article": 2, "max_chunks_per_topic": {"glossary": 1}}) == (2, {"glossary": 1})
    assert diversity_caps({}) == (None, {})
    assert get_config()["rag"]["max_chunks_per_article"] == 2


class ScriptedStore:
    """Returns fixed hits so retrieve() can be checked against select_chunks."""

    def __init__(self, hits):
        self.hits = hits

    def __len__(self):
        return len(self.hits)

    def search(self, vector, k):
        return self.hits[:k]


def _rec(chunk_id, score):
    topic = chunk_id.split("/")[0]
    rec = {"text": "t", "title": "T", "topic": topic, "references": REFS, "chunk_id": chunk_id}
    return rec, score


def test_retrieve_applies_the_caps_from_config(monkeypatch):
    hits = [_rec(r["chunk_id"], s) for r, s in HITS]
    store = ScriptedStore(hits)
    _patch_rag(monkeypatch, top_k=4, score_threshold=0.3)
    got = [c.chunk_id for c in _retrieve("anything", store)]
    assert got == _ids(select_chunks(hits, 4, 0.3, 2, {"glossary": 1}))
    assert got == ["stocks/a#0", "stocks/a#1", "glossary/g#0", "bonds/b#0"]


def test_retrieve_caps_off_when_config_is_empty(monkeypatch):
    hits = [_rec(r["chunk_id"], s) for r, s in HITS]
    _patch_rag(monkeypatch, top_k=3, score_threshold=0.3,
               max_chunks_per_article=None, max_chunks_per_topic={})
    got = [c.chunk_id for c in _retrieve("anything", ScriptedStore(hits))]
    assert got == ["stocks/a#0", "stocks/a#1", "stocks/a#2"]
