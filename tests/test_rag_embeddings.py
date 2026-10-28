import sys
import types

import numpy as np
import pytest

from src.rag import embeddings
from src.rag.embeddings import SentenceTransformerEmbedder, get_embedder
from src.utils.config import ConfigError


@pytest.fixture(autouse=True)
def _clear():
    get_embedder.cache_clear()
    yield
    get_embedder.cache_clear()


class FakeModel:
    instances = []

    def __init__(self, name):
        self.name = name
        self.encode_calls = []
        self.progress_flags = []
        FakeModel.instances.append(self)

    def encode(self, texts, normalize_embeddings, convert_to_numpy, show_progress_bar=None):
        self.progress_flags.append(show_progress_bar)
        self.encode_calls.append((list(texts), normalize_embeddings, convert_to_numpy))
        return np.array([[1.0, 0.0] for _ in texts])


@pytest.fixture
def fake_st(monkeypatch):
    FakeModel.instances = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_model_is_loaded_lazily(fake_st):
    embedder = SentenceTransformerEmbedder("some/model")
    assert FakeModel.instances == []  # nothing loaded yet
    embedder.embed_query("hello")
    assert len(FakeModel.instances) == 1 and FakeModel.instances[0].name == "some/model"
    embedder.embed_documents(["a", "b"])
    assert len(FakeModel.instances) == 1  # loaded only once


def test_vectors_requested_normalised_and_returned_as_float_lists(fake_st):
    embedder = SentenceTransformerEmbedder("m")
    docs = embedder.embed_documents(["a", "b"])
    assert docs == [[1.0, 0.0], [1.0, 0.0]]
    assert all(isinstance(x, float) for x in docs[0])
    texts, normalise, as_numpy = FakeModel.instances[0].encode_calls[0]
    assert texts == ["a", "b"] and normalise is True and as_numpy is True
    assert embedder.embed_query("q") == [1.0, 0.0]


def test_get_embedder_is_singleton_and_uses_config_model():
    first = get_embedder()
    assert first is get_embedder()
    assert first.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr(
        embeddings, "get_config", lambda: {"embeddings": {"provider": "nope", "model": "x"}}
    )
    with pytest.raises(ConfigError):
        get_embedder()


def test_progress_bar_flag_is_passed_only_when_set(fake_st):
    seen = []

    class Model(FakeModel):
        def encode(self, texts, normalize_embeddings, convert_to_numpy, **kwargs):
            seen.append(kwargs)
            return np.array([[1.0, 0.0] for _ in texts])

    sys.modules["sentence_transformers"].SentenceTransformer = Model
    SentenceTransformerEmbedder("m", show_progress_bar=False).embed_documents(["q"])
    SentenceTransformerEmbedder("m").embed_documents(["q"])
    assert seen == [{"show_progress_bar": False}, {}]


def test_single_queries_never_show_a_progress_bar(fake_st):
    embedder = SentenceTransformerEmbedder("m")   # default: library behaviour for documents
    embedder.embed_documents(["a", "b"])
    embedder.embed_query("q")
    assert FakeModel.instances[0].progress_flags == [None, False]
    verbose = SentenceTransformerEmbedder("m", show_progress_bar=True)
    verbose.embed_query("q")
    assert FakeModel.instances[-1].progress_flags == [False]
