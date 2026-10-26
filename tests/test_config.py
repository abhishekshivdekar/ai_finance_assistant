import pytest

from src.utils.config import ConfigError, get_config, get_secret


def test_config_loads_expected_sections():
    cfg = get_config()
    for section in ["app", "llm", "rag", "market_data", "workflow"]:
        assert section in cfg


def test_config_is_cached_singleton():
    assert get_config() is get_config()


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        get_config(str(tmp_path / "nope.yaml"))


def test_missing_secret_raises(monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST_KEY", raising=False)
    with pytest.raises(ConfigError):
        get_secret("DOES_NOT_EXIST_KEY")


def test_optional_secret_returns_none(monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST_KEY", raising=False)
    assert get_secret("DOES_NOT_EXIST_KEY", required=False) is None
