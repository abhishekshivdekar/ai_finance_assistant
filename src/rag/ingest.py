"""Offline ingestion: Markdown articles -> chunks -> embeddings -> saved index.

Design pattern: FACTORY-style builder around the STRATEGY interfaces. build_store()
takes any embedder and returns a VectorStore, so tests pass a fake embedder and
the eval harness builds throw-away in-memory stores. Example: build_index()
reads src/data/knowledge_base/stocks/what-is-a-stock.md, splits it into chunks
with ids like "stocks/what-is-a-stock#0", and saves the index plus manifest.json.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.embeddings import get_embedder
from src.rag.manifest import (
    build_manifest,
    compute_article_hash,
    resolve_dir,
    write_manifest,
)
from src.rag.vector_store import FaissStore, VectorStore
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_VERIFY = re.compile(r"<!--\s*VERIFY\s*(?::[^>]*?)?\s*-->")


class ArticleError(ValueError):
    """An article has missing or invalid front matter or body."""


@dataclass(frozen=True)
class Article:
    """A validated knowledge base article."""

    path: Path
    slug: str
    title: str
    topic: str
    references: list[dict[str, str]]
    body: str
    verify_count: int = 0


def parse_article(path: Path) -> Article:
    """Read and validate one article; raise ArticleError if it is unusable."""
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(text)
    if not match:
        raise ArticleError("missing front matter")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ArticleError(f"front matter is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise ArticleError("front matter must be a mapping")

    title = meta.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ArticleError("missing title")
    topic = meta.get("topic")
    if topic not in get_config()["rag"]["topics"]:
        raise ArticleError(f"unknown topic {topic!r}")
    refs = meta.get("references")
    if not isinstance(refs, list) or not refs:
        raise ArticleError("references must be a non-empty list")
    for ref in refs:
        if not (
            isinstance(ref, dict)
            and isinstance(ref.get("name"), str)
            and isinstance(ref.get("url"), str)
            and ref["url"].startswith(("http://", "https://"))
        ):
            raise ArticleError("each reference needs a name and an http(s) url")
    body = match.group(2)
    if not _COMMENT.sub("", body).strip():
        raise ArticleError("empty body")

    return Article(
        path=path,
        slug=path.stem,
        title=title.strip(),
        topic=topic,
        references=[{"name": r["name"], "url": r["url"]} for r in refs],
        body=body,
        verify_count=len(_VERIFY.findall(body)),
    )


def load_articles(kb_dir: Path | None = None) -> list[Article]:
    """Load every valid article; a bad one is logged and skipped, never fatal."""
    directory = kb_dir or resolve_dir(get_config()["rag"]["knowledge_base_dir"])
    articles: list[Article] = []
    for file in sorted(directory.rglob("*.md")):
        try:
            article = parse_article(file)
        except (ArticleError, OSError, UnicodeDecodeError) as exc:
            log.warning("Skipping article %s: %s", file, exc)
            continue
        if article.verify_count:
            log.warning("%s has %d <!-- VERIFY --> marker(s) to check", file.name, article.verify_count)
        articles.append(article)
    return articles


def chunk_article(
    article: Article,
    chunk_size: int,
    chunk_overlap: int,
    topic_chunking: dict[str, dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Split an article into chunk records that carry its metadata.

    Design pattern: STRATEGY chosen per topic. The topic's override in
    rag.topic_chunking (None means read it from config) replaces the default
    size and overlap. Example: glossary articles use 250/0, so each tiny
    "**NAV** - ..." definition becomes its own chunk; stocks articles keep 800/120.
    """
    if topic_chunking is None:
        topic_chunking = get_config()["rag"].get("topic_chunking") or {}
    override = topic_chunking.get(article.topic) or {}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=override.get("chunk_size", chunk_size),
        chunk_overlap=override.get("chunk_overlap", chunk_overlap),
    )
    pieces = splitter.split_text(_COMMENT.sub("", article.body).strip())
    return [
        {
            "text": piece,
            "title": article.title,
            "topic": article.topic,
            "references": article.references,
            "chunk_id": f"{article.topic}/{article.slug}#{i}",
        }
        for i, piece in enumerate(pieces)
    ]


def build_store(
    articles: list[Article],
    embedder: Any,
    chunk_size: int,
    chunk_overlap: int,
    topic_chunking: dict[str, dict[str, int]] | None = None,
) -> VectorStore:
    """Chunk and embed articles into an in-memory store (not saved)."""
    records = [
        record
        for article in articles
        for record in chunk_article(article, chunk_size, chunk_overlap, topic_chunking)
    ]
    store = FaissStore()
    if records:
        store.add(embedder.embed_documents([r["text"] for r in records]), records)
    return store


def build_index(
    embedder: Any = None,
    kb_dir: Path | None = None,
    out_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, int]:
    """Build and save the index plus manifest.json; returns {articles, chunks}."""
    cfg = get_config()
    rag = cfg["rag"]
    kb_path = kb_dir or resolve_dir(rag["knowledge_base_dir"])
    out_path = out_dir or resolve_dir(rag["vector_store_dir"])
    size = rag["chunk_size"] if chunk_size is None else chunk_size
    overlap = rag["chunk_overlap"] if chunk_overlap is None else chunk_overlap

    articles = load_articles(kb_path)
    store = build_store(
        articles, embedder or get_embedder(), size, overlap, rag.get("topic_chunking") or {}
    )
    if len(store) == 0:
        raise ValueError(f"No valid articles found in {kb_path}")
    store.save(out_path)
    write_manifest(
        out_path,
        build_manifest(
            cfg["embeddings"]["model"],
            size,
            overlap,
            compute_article_hash(kb_path),
            rag.get("topic_chunking") or {},
        ),
    )
    log.info("Indexed %d articles into %d chunks at %s", len(articles), len(store), out_path)
    return {"articles": len(articles), "chunks": len(store)}
