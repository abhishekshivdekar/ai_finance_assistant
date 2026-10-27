import sys
import types

import pytest

from src.core import llm_factory
from src.core.usage_tracker import UsageTracker, get_usage_registry, get_usage_tracker
from src.utils.config import ConfigError, get_config


@pytest.fixture(autouse=True)
def _clear_caches():
    get_usage_registry.cache_clear()
    get_usage_tracker.cache_clear()
    yield
    get_usage_registry.cache_clear()
    get_usage_tracker.cache_clear()


class FakeChat:
    """Stands in for ChatOpenAI / ChatGoogleGenerativeAI and records kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_providers(monkeypatch):
    openai_mod = types.ModuleType("langchain_openai")
    openai_mod.ChatOpenAI = FakeChat
    gemini_mod = types.ModuleType("langchain_google_genai")
    gemini_mod.ChatGoogleGenerativeAI = FakeChat
    monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", gemini_mod)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def _use_llm_cfg(monkeypatch, llm_cfg):
    monkeypatch.setattr(llm_factory, "get_config", lambda: {"llm": llm_cfg})


BASE = {
    "provider": "openai",
    "model": "base-model",
    "temperature": None,
    "max_retries": 3,
    "roles": {"router": {"model": "router-model"}},
}


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr(
        llm_factory, "get_config", lambda: {"llm": {"provider": "nonsense", "model": "x"}}
    )
    with pytest.raises(ConfigError):
        llm_factory.get_llm()


def test_role_override_merges_over_default(monkeypatch, fake_providers):
    _use_llm_cfg(monkeypatch, BASE)
    llm = llm_factory.get_llm(role="router")
    assert llm.kwargs["model"] == "router-model"
    assert llm.kwargs["max_retries"] == 3  # inherited from llm.*


@pytest.mark.parametrize("role", [None, "default", "unknown", "synthesizer"])
def test_missing_or_unknown_role_falls_back_to_default(monkeypatch, fake_providers, role):
    _use_llm_cfg(monkeypatch, BASE)
    assert llm_factory.get_llm(role=role).kwargs["model"] == "base-model"


def test_no_roles_section_is_fine(monkeypatch, fake_providers):
    cfg = {k: v for k, v in BASE.items() if k != "roles"}
    _use_llm_cfg(monkeypatch, cfg)
    assert llm_factory.get_llm(role="router").kwargs["model"] == "base-model"


def test_temperature_omitted_when_null(monkeypatch, fake_providers):
    _use_llm_cfg(monkeypatch, BASE)
    assert "temperature" not in llm_factory.get_llm().kwargs


def test_temperature_from_config_and_argument(monkeypatch, fake_providers):
    _use_llm_cfg(monkeypatch, {**BASE, "temperature": 0.2})
    assert llm_factory.get_llm().kwargs["temperature"] == 0.2
    assert llm_factory.get_llm(0.7).kwargs["temperature"] == 0.7  # positional, backwards compatible


def test_role_can_override_temperature(monkeypatch, fake_providers):
    cfg = {**BASE, "temperature": 0.2, "roles": {"router": {"temperature": None}}}
    _use_llm_cfg(monkeypatch, cfg)
    assert "temperature" not in llm_factory.get_llm(role="router").kwargs


def test_gemini_branch(monkeypatch, fake_providers):
    _use_llm_cfg(monkeypatch, {**BASE, "provider": "gemini"})
    llm = llm_factory.get_llm()
    assert llm.kwargs["google_api_key"] == "test-key"


def test_tracker_attached_with_role_and_model(monkeypatch, fake_providers):
    _use_llm_cfg(monkeypatch, BASE)
    tracker = llm_factory.get_llm(role="router").kwargs["callbacks"][0]
    assert isinstance(tracker, UsageTracker)
    assert (tracker.role, tracker.model) == ("router", "router-model")
    default_tracker = llm_factory.get_llm().kwargs["callbacks"][0]
    assert (default_tracker.role, default_tracker.model) == ("default", "base-model")


def test_one_tracker_per_role(monkeypatch, fake_providers):
    _use_llm_cfg(monkeypatch, BASE)
    first = llm_factory.get_llm(role="router").kwargs["callbacks"][0]
    second = llm_factory.get_llm(role="router").kwargs["callbacks"][0]
    assert first is second


def test_real_config_has_roles_and_pricing():
    cfg = get_config()
    assert cfg["llm"]["roles"]["router"]["model"] == "gpt-6-luna"
    assert cfg["llm"]["roles"]["synthesizer"]["model"] == "gpt-6-luna"
    assert cfg["pricing"]["gpt-6-luna"] == {"input": 0.10, "output": 0.50}
    assert cfg["pricing"]["gpt-5.4-mini"] == {"input": 0.75, "output": 4.50}
