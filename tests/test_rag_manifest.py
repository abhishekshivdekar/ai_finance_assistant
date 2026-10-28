import copy
import logging
from pathlib import Path

import pytest

from src.rag import manifest
from src.rag.manifest import (
    build_manifest,
    compute_article_hash,
    manifest_mismatches,
    read_manifest,
    resolve_dir,
    warn_if_stale,
    write_manifest,
)
from src.utils.config import PROJECT_ROOT, get_config


def _kb(tmp_path):
    kb = tmp_path / "kb"
    (kb / "stocks").mkdir(parents=True)
    (kb / "stocks" / "a.md").write_text("alpha", encoding="utf-8")
    return kb


def _patch_cfg(monkeypatch, kb):
    cfg = copy.deepcopy(get_config())
    cfg["rag"]["knowledge_base_dir"] = str(kb)
    monkeypatch.setattr(manifest, "get_config", lambda: cfg)
    return cfg


def _current(cfg, kb):
    return build_manifest(
        cfg["embeddings"]["model"], cfg["rag"]["chunk_size"],
        cfg["rag"]["chunk_overlap"], compute_article_hash(kb),
    )


def test_resolve_dir():
    assert resolve_dir("src/rag") == PROJECT_ROOT / "src/rag"
    absolute = Path(PROJECT_ROOT).resolve()
    assert resolve_dir(absolute) == absolute


def test_hash_is_stable_and_changes_with_content_or_new_files(tmp_path):
    kb = _kb(tmp_path)
    first = compute_article_hash(kb)
    assert first == compute_article_hash(kb)
    (kb / "stocks" / "a.md").write_text("alpha!", encoding="utf-8")
    changed = compute_article_hash(kb)
    assert changed != first
    (kb / "stocks" / "b.md").write_text("beta", encoding="utf-8")
    assert compute_article_hash(kb) != changed


def test_write_read_roundtrip_and_missing(tmp_path):
    data = build_manifest("m", 800, 120, "h")
    write_manifest(tmp_path / "new", data)
    assert read_manifest(tmp_path / "new") == data
    assert read_manifest(tmp_path / "absent") is None
    (tmp_path / "manifest.json").write_text("{broken", encoding="utf-8")
    assert read_manifest(tmp_path) is None


def test_manifest_mismatches():
    current = build_manifest("m", 800, 120, "h")
    assert manifest_mismatches(dict(current), current) == []
    stale = {**current, "chunk_size": 600, "article_hash": "x"}
    assert manifest_mismatches(stale, current) == ["chunk_size", "article_hash"]


def test_warn_if_stale_clean_manifest_is_silent(tmp_path, monkeypatch, caplog):
    kb = _kb(tmp_path)
    cfg = _patch_cfg(monkeypatch, kb)
    write_manifest(tmp_path / "idx", _current(cfg, kb))
    with caplog.at_level(logging.WARNING):
        assert warn_if_stale(tmp_path / "idx") == []
    assert caplog.text == ""


def test_warn_if_stale_reports_mismatch(tmp_path, monkeypatch, caplog):
    kb = _kb(tmp_path)
    cfg = _patch_cfg(monkeypatch, kb)
    write_manifest(tmp_path / "idx", _current(cfg, kb))
    (kb / "stocks" / "a.md").write_text("changed", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert warn_if_stale(tmp_path / "idx") == ["article_hash"]
    assert "article_hash" in caplog.text


def test_warn_if_stale_config_change(tmp_path, monkeypatch, caplog):
    kb = _kb(tmp_path)
    cfg = _patch_cfg(monkeypatch, kb)
    write_manifest(tmp_path / "idx", {**_current(cfg, kb), "chunk_size": 1})
    with caplog.at_level(logging.WARNING):
        assert warn_if_stale(tmp_path / "idx") == ["chunk_size"]


def test_warn_if_stale_missing_manifest(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert warn_if_stale(tmp_path) == ["manifest"]
    assert "manifest" in caplog.text.lower()


def test_warn_if_stale_unreadable_knowledge_base(tmp_path, monkeypatch, caplog):
    kb = _kb(tmp_path)
    cfg = _patch_cfg(monkeypatch, kb)
    write_manifest(tmp_path / "idx", _current(cfg, kb))

    def boom(path):
        raise OSError("disk")

    monkeypatch.setattr(manifest, "compute_article_hash", boom)
    with caplog.at_level(logging.WARNING):
        assert warn_if_stale(tmp_path / "idx") == ["article_hash"]
