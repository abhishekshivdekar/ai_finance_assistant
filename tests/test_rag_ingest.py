import json
import logging

import pytest

from src.rag import ingest
from src.rag.ingest import (
    Article,
    ArticleError,
    build_index,
    build_store,
    chunk_article,
    load_articles,
    parse_article,
)
from src.rag.manifest import compute_article_hash
from src.utils.config import get_config
from fake_embedder import FakeEmbedder

GOOD_REFS = "references:\n  - name: SEBI\n    url: https://www.sebi.gov.in\n"


def _write(path, front, body="Shares are units of ownership in a company.\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{front}---\n{body}", encoding="utf-8")
    return path


def _good_front(topic="stocks", title="A Title"):
    return f"title: {title}\ntopic: {topic}\n{GOOD_REFS}"


def test_parse_valid_article(tmp_path):
    path = _write(tmp_path / "stocks" / "what-is-a-stock.md", _good_front())
    art = parse_article(path)
    assert art.slug == "what-is-a-stock" and art.title == "A Title" and art.topic == "stocks"
    assert art.references == [{"name": "SEBI", "url": "https://www.sebi.gov.in"}]
    assert art.verify_count == 0


def test_verify_markers_are_counted(tmp_path):
    path = _write(tmp_path / "a.md", _good_front(), "Text <!-- VERIFY --> more <!--VERIFY-->\n")
    assert parse_article(path).verify_count == 2


@pytest.mark.parametrize(
    "text",
    [
        "no front matter at all",
        "---\ntitle: [unclosed\n---\nbody",
        "---\n- just\n- a list\n---\nbody",
        f"---\ntopic: stocks\n{GOOD_REFS}---\nbody",                       # missing title
        f"---\ntitle: T\ntopic: nonsense\n{GOOD_REFS}---\nbody",           # unknown topic
        "---\ntitle: T\ntopic: stocks\n---\nbody",                         # no references
        "---\ntitle: T\ntopic: stocks\nreferences: []\n---\nbody",         # empty references
        "---\ntitle: T\ntopic: stocks\nreferences: [a, b]\n---\nbody",     # wrong shape
        "---\ntitle: T\ntopic: stocks\nreferences:\n  - name: X\n    url: ftp://x\n---\nbody",
        f"---\ntitle: T\ntopic: stocks\n{GOOD_REFS}---\n   \n",             # empty body
        f"---\ntitle: T\ntopic: stocks\n{GOOD_REFS}---\n<!-- only a comment -->\n",
    ],
)
def test_invalid_articles_raise(tmp_path, text):
    path = tmp_path / "bad.md"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ArticleError):
        parse_article(path)


def test_load_articles_skips_bad_ones_and_logs(tmp_path, caplog):
    _write(tmp_path / "stocks" / "good.md", _good_front(), "Body <!-- VERIFY --> text\n")
    (tmp_path / "stocks" / "bad.md").write_text("no front matter", encoding="utf-8")
    (tmp_path / "stocks" / "binary.md").write_bytes(b"\xff\xfe\x00bad")
    with caplog.at_level(logging.WARNING):
        articles = load_articles(tmp_path)
    assert [a.slug for a in articles] == ["good"]
    assert "Skipping article" in caplog.text and "bad.md" in caplog.text
    assert "VERIFY" in caplog.text


def test_load_articles_defaults_to_configured_knowledge_base():
    assert len(load_articles()) >= 8


def _article(body, slug="demo", topic="stocks"):
    return Article(
        path=None, slug=slug, title="Demo", topic=topic,
        references=[{"name": "SEBI", "url": "https://www.sebi.gov.in"}], body=body,
    )


def _shared_prefix_len(previous, following):
    return max(k for k in range(len(following) + 1) if previous.endswith(following[:k]))


def test_chunk_size_and_overlap_are_respected():
    body = " ".join(f"word{i:04d}" for i in range(200))
    chunks = [c["text"] for c in chunk_article(_article(body), 100, 30)]
    assert len(chunks) > 3
    assert all(len(c) <= 100 for c in chunks)
    for previous, following in zip(chunks, chunks[1:]):
        shared = _shared_prefix_len(previous, following)
        assert 0 < shared <= 30


def test_zero_overlap_means_no_shared_text():
    body = " ".join(f"word{i:04d}" for i in range(100))
    chunks = [c["text"] for c in chunk_article(_article(body), 100, 0)]
    for previous, following in zip(chunks, chunks[1:]):
        assert _shared_prefix_len(previous, following) == 0


def test_chunk_records_carry_metadata_and_strip_comments():
    records = chunk_article(_article("Hello <!-- VERIFY --> world"), 800, 120)
    assert len(records) == 1
    rec = records[0]
    assert "VERIFY" not in rec["text"] and "<!--" not in rec["text"]
    assert rec["chunk_id"] == "stocks/demo#0"
    assert (rec["title"], rec["topic"]) == ("Demo", "stocks")
    assert rec["references"][0]["url"] == "https://www.sebi.gov.in"


def test_build_store_counts_chunks_and_handles_empty():
    arts = [_article("alpha beta gamma", "one"), _article("delta epsilon", "two", "bonds")]
    assert len(build_store(arts, FakeEmbedder(), 800, 120)) == 2
    assert len(build_store([], FakeEmbedder(), 800, 120)) == 0


def test_build_index_end_to_end(tmp_path, monkeypatch):
    kb, out = tmp_path / "kb", tmp_path / "idx"
    _write(kb / "stocks" / "a.md", _good_front("stocks", "A"), "shares company ownership\n")
    _write(kb / "bonds" / "b.md", _good_front("bonds", "B"), "bond coupon maturity\n")
    summary = build_index(FakeEmbedder(), kb, out, chunk_size=300, chunk_overlap=30)
    assert summary == {"articles": 2, "chunks": 2}
    assert (out / "index.faiss").exists() and (out / "chunks.json").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "embedding_model": get_config()["embeddings"]["model"],
        "chunk_size": 300, "chunk_overlap": 30,
        "article_hash": compute_article_hash(kb),
        "topic_chunking": get_config()["rag"]["topic_chunking"],
    }


def test_build_index_uses_config_defaults_and_default_embedder(tmp_path, monkeypatch):
    kb, out = tmp_path / "kb", tmp_path / "idx"
    _write(kb / "stocks" / "a.md", _good_front())
    monkeypatch.setattr(ingest, "get_embedder", lambda: FakeEmbedder())
    build_index(kb_dir=kb, out_dir=out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_size"] == get_config()["rag"]["chunk_size"]
    assert manifest["chunk_overlap"] == get_config()["rag"]["chunk_overlap"]


def test_build_index_with_no_valid_articles_raises(tmp_path):
    (tmp_path / "kb").mkdir()
    with pytest.raises(ValueError):
        build_index(FakeEmbedder(), tmp_path / "kb", tmp_path / "idx")
    assert not (tmp_path / "idx").exists()


GLOSSARY_BODY = "\n\n".join(f"**Term{i}** - definition number {i} has some words." for i in range(12))


def test_topic_override_applies_only_to_its_topic():
    overrides = {"glossary": {"chunk_size": 100, "chunk_overlap": 0}}
    glossary = chunk_article(_article(GLOSSARY_BODY, topic="glossary"), 800, 120, overrides)
    stocks = chunk_article(_article(GLOSSARY_BODY, topic="stocks"), 800, 120, overrides)
    assert len(glossary) > 3 and all(len(c["text"]) <= 100 for c in glossary)
    assert len(stocks) == 1


def test_topic_override_defaults_to_config():
    cfg = get_config()["rag"]["topic_chunking"]["glossary"]
    chunks = chunk_article(_article(GLOSSARY_BODY * 3, topic="glossary"), 800, 120)
    assert all(len(c["text"]) <= cfg["chunk_size"] for c in chunks)
    assert len(chunk_article(_article(GLOSSARY_BODY, topic="stocks"), 800, 120)) == 1


def test_build_store_honours_override():
    glossary = _article(GLOSSARY_BODY, topic="glossary")
    plain = build_store([glossary], FakeEmbedder(), 800, 0, {})
    split = build_store([glossary], FakeEmbedder(), 800, 0, {"glossary": {"chunk_size": 100, "chunk_overlap": 0}})
    assert len(plain) == 1 and len(split) > 3
