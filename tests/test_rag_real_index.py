"""Retrieval on the saved index with the local embedder (no API, no LLM). Skipped when the index or model is missing.

The unit tests use fakes; this one guards the regressions found with retrieval_check.py, which only the real
embeddings can show (architecture section 5, "Query expansion").
"""
from functools import partial

import pytest

from src.rag.embeddings import get_embedder
from src.rag.retriever import merge_retrieval, retrieve
from src.rag.vector_store import IndexNotFoundError, get_vector_store

TAX_TOPIC = "accounts_and_taxes_india"


@pytest.fixture(scope="module")
def search():
    """A retrieve function bound to the real store and embedder, or a skip when either is unavailable."""
    import huggingface_hub.constants as hub

    # Use the cached model; never call the hub from a test. The env var covers libraries that read it later, the
    # constant covers huggingface_hub, which read the env var when it was first imported.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setattr(hub, "HF_HUB_OFFLINE", True)
        try:
            store, embedder = get_vector_store(), get_embedder()   # get_embedder() is the process-wide Singleton
            embedder.embed_query("warm up")
        except (IndexNotFoundError, OSError) as exc:
            pytest.skip(f"saved index or local embedding model not available: {exc}")
    yield partial(retrieve, embedder=embedder, store=store)


def ids(search, question, topic):
    return [chunk.chunk_id for chunk in merge_retrieval(search, [question], topic)]


def test_how_does_ppf_work_returns_a_ppf_body_chunk_not_only_the_heading_and_intro(search):
    ppf = [i for i in ids(search, "How does PPF work?", TAX_TOPIC) if i.startswith(f"{TAX_TOPIC}/ppf-explained#")]
    assert ppf, "no PPF chunk returned"
    assert any(i not in (f"{TAX_TOPIC}/ppf-explained#0", f"{TAX_TOPIC}/ppf-explained#1") for i in ppf)


@pytest.mark.parametrize("question", ["What is STT?", "What is LTCG?"])
def test_acronym_questions_on_the_tax_topic_return_capital_gains_basics_chunks(search, question):
    assert any(i.startswith(f"{TAX_TOPIC}/capital-gains-basics#") for i in ids(search, question, TAX_TOPIC))
