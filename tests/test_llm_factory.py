import pytest

from src.core import llm_factory
from src.utils.config import ConfigError


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr(
        llm_factory, "get_config", lambda: {"llm": {"provider": "nonsense", "model": "x"}}
    )
    with pytest.raises(ConfigError):
        llm_factory.get_llm()
