import logging

import pytest

from src.rag import vector_store
from src.rag.manifest import write_manifest
from src.rag.vector_store import FaissStore, IndexNotFoundError, VectorStore, get_vector_store
from src.utils.config import ConfigError

A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]
C = [0.6, 0.8, 0.0]


def _store():
    store = FaissStore()
    store.add([A, B, C], [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    return store


@pytest.fixture(autouse=True)
def _clear():
    get_vector_store.cache_clear()
    yield
    get_vector_store.cache_clear()


def test_interface_is_abstract():
    with pytest.raises(TypeError):
        VectorStore()


def test_search_returns_cosine_scores_best_first():
    results = _store().search(A, 2)
    assert [r["id"] for r, _ in results] == ["a", "c"]
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(0.6)


def test_k_larger_than_index_is_clamped():
    assert len(_store().search(A, 50)) == 3


def test_empty_store_and_bad_k_return_nothing():
    assert FaissStore().search(A, 3) == []
    assert _store().search(A, 0) == []


def test_add_validation():
    store = _store()
    with pytest.raises(ValueError):
        store.add([A], [{"id": "x"}, {"id": "y"}])
    with pytest.raises(ValueError):
        store.add([[1.0, 0.0]], [{"id": "x"}])  # wrong dimension
    store.add([], [])  # no-op
    assert len(store) == 3


def test_save_and_load_roundtrip(tmp_path):
    _store().save(tmp_path / "idx")
    loaded = FaissStore()
    loaded.load(tmp_path / "idx")
    assert len(loaded) == 3
    assert loaded.search(B, 1)[0][0] == {"id": "b"}


def test_load_missing_raises(tmp_path):
    with pytest.raises(IndexNotFoundError):
        FaissStore().load(tmp_path)


def test_saving_empty_store_cannot_be_loaded(tmp_path):
    FaissStore().save(tmp_path)
    with pytest.raises(IndexNotFoundError):
        FaissStore().load(tmp_path)


def _cfg(directory, name="faiss"):
    return {"rag": {"vector_store": name, "vector_store_dir": str(directory)}}


def test_get_vector_store_loads_once_and_checks_manifest(tmp_path, monkeypatch):
    _store().save(tmp_path)
    monkeypatch.setattr(vector_store, "get_config", lambda: _cfg(tmp_path))
    checked = []
    monkeypatch.setattr(vector_store, "warn_if_stale", checked.append)
    first = get_vector_store()
    assert first is get_vector_store()
    assert len(first) == 3 and checked == [tmp_path]


def test_get_vector_store_warns_on_stale_manifest(tmp_path, monkeypatch, caplog):
    _store().save(tmp_path)
    write_manifest(tmp_path, {"embedding_model": "other-model"})
    monkeypatch.setattr(vector_store, "get_config", lambda: _cfg(tmp_path))
    with caplog.at_level(logging.WARNING):
        get_vector_store()
    assert "stale" in caplog.text.lower()


def test_unsupported_store_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "get_config", lambda: _cfg(tmp_path, "chroma"))
    with pytest.raises(ConfigError):
        get_vector_store()


def test_missing_index_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "get_config", lambda: _cfg(tmp_path / "none"))
    with pytest.raises(IndexNotFoundError):
        get_vector_store()
