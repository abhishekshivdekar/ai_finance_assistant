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


def _config_file(tmp_path, windows):
    path = tmp_path / "config.yaml"
    path.write_text(f"market_analysis:\n  sma_windows: {windows}\n", encoding="utf-8")
    return str(path)


def test_valid_sma_windows_load(tmp_path):
    assert get_config(_config_file(tmp_path, "[20, 50]"))["market_analysis"]["sma_windows"] == [20, 50]


@pytest.mark.parametrize("windows", [
    "[20]", "[20, 50, 100]", "[20, 20]", "[50, 20]", "[0, 50]", "[-5, 50]", "[20.5, 50]", "[true, 50]",
    "['20', '50']", "20", "null", "{short: 20, long: 50}",
])
def test_invalid_sma_windows_fail_clearly(tmp_path, windows):
    with pytest.raises(ConfigError, match="sma_windows"):
        get_config(_config_file(tmp_path, windows))


def test_missing_sma_windows_fails_clearly(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("market_analysis:\n  max_tickers: 4\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="sma_windows"):
        get_config(str(path))
