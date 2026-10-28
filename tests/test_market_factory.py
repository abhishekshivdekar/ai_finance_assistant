import json

import pytest

from src.data.market import get_market_provider
from src.data.market import factory
from src.data.market.cache import CachedProvider
from src.data.market.yfinance_provider import YFinanceProvider
from src.utils.config import PROJECT_ROOT, ConfigError, get_config


@pytest.fixture(autouse=True)
def _clear():
    get_market_provider.cache_clear()
    yield
    get_market_provider.cache_clear()


def test_returns_cached_yfinance_provider():
    provider = get_market_provider()
    assert isinstance(provider, CachedProvider)
    assert isinstance(provider._inner, YFinanceProvider)


def test_is_singleton():
    assert get_market_provider() is get_market_provider()


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr(
        factory, "get_config", lambda: {"market_data": {"provider": "alpha_vantage"}}
    )
    with pytest.raises(ConfigError):
        get_market_provider()


def test_snapshot_is_labelled_sample_data():
    path = PROJECT_ROOT / get_config()["market_data"]["snapshot_path"]
    snap = json.loads(path.read_text(encoding="utf-8"))
    meta = snap["_meta"]
    assert meta["label"] == "SAMPLE DATA"
    assert meta["as_of"] and "illustrative" in meta["note"].lower()
    tickers = snap["tickers"]
    for t in ("RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "^NSEI"):
        assert tickers[t]["currency"] == "INR"
    assert tickers["AAPL"]["currency"] == tickers["MSFT"]["currency"] == "USD"
    inr = [t for t in tickers.values() if t["currency"] == "INR"]
    assert len(inr) > len(tickers) - len(inr)  # mostly NSE
