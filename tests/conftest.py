import pytest

import network_guard
from src.utils.config import get_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Reset the config Singleton so each test starts clean."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture(autouse=True)
def _no_outside_network(request, monkeypatch):
    """No test may open a connection to a non-local host. Opt out only with @pytest.mark.network."""
    if request.node.get_closest_marker("network") is None:
        network_guard.block(monkeypatch)


@pytest.fixture(autouse=True)
def _streamlit_skips_the_component_scan(monkeypatch):
    """Every AppTest run makes Streamlit read the metadata of ALL installed packages (about 0.9 s) to look for custom v2
    components. The app uses none, so tests skip the scan; it changes no element the pages draw."""
    monkeypatch.setattr("streamlit.components.v2.manifest_scanner.scan_component_manifests", lambda *args, **kwargs: [])


@pytest.fixture(autouse=True)
def _conversations_stay_in_memory(monkeypatch):
    """Tests never create the SQLite file: the facade's store is the in-memory one unless a test opens its own (tmp_path)."""
    from src.workflow.persistence import memory_store

    monkeypatch.setattr("src.workflow.facade.open_store", lambda *args, **kwargs: memory_store("Saving conversations is switched off in tests."))


class OfflineLLM:
    """Stands in for a real chat model in workflow tests that did not inject one: every call fails."""

    def invoke(self, *args, **kwargs):
        raise RuntimeError("offline test LLM: inject a fake LLM to script a reply")


@pytest.fixture(autouse=True)
def _workflow_never_reaches_the_network(monkeypatch):
    """The graph's synthesizer / safe_response LLMs are created lazily through get_llm, and advice context comes from
    the real retriever. Neither may run in a test unless the test injects a fake: the LLM stand-in raises and the
    retriever finds nothing (so a merge falls back to the verbatim join and advice to its static text)."""
    monkeypatch.setattr("src.workflow.nodes.get_llm", lambda **kwargs: OfflineLLM())
    monkeypatch.setattr("src.workflow.graph.retrieve", lambda query, topic=None, **kwargs: [])
