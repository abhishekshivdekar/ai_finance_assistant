import sys
import types
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.market import yfinance_provider as yfp
from src.data.market.provider import PermanentProviderError, TransientProviderError
from src.data.market.yfinance_provider import YFinanceProvider, _classify, _parse_news

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def _provider(monkeypatch, responder):
    monkeypatch.setattr(yfp, "_yf_call", responder)
    return YFinanceProvider(clock=lambda: NOW)


def test_quote_returns_currency_from_source(monkeypatch):
    p = _provider(
        monkeypatch,
        lambda kind, t, period=None: {"price": 110.0, "previous_close": 100.0, "currency": "INR"},
    )
    result = p.get_quote("RELIANCE.NS")
    assert result.currency == "INR"
    assert result.source == "live" and result.as_of == NOW and not result.is_stale
    assert result.data["change"] == pytest.approx(10.0)
    assert result.data["change_pct"] == pytest.approx(10.0)


def test_quote_usd_and_missing_previous_close(monkeypatch):
    p = _provider(
        monkeypatch,
        lambda kind, t, period=None: {"price": 5.0, "previous_close": None, "currency": "USD"},
    )
    result = p.get_quote("AAPL")
    assert result.currency == "USD"
    assert result.data["change"] is None and result.data["change_pct"] is None


def test_currency_is_not_guessed_when_source_omits_it(monkeypatch):
    p = _provider(monkeypatch, lambda kind, t, period=None: {"price": 5.0})
    assert p.get_quote("AAPL").currency is None


def test_quote_without_price_is_permanent(monkeypatch):
    p = _provider(monkeypatch, lambda kind, t, period=None: {"price": float("nan")})
    with pytest.raises(PermanentProviderError):
        p.get_quote("ZZZZ")


def test_history_rows_and_currency(monkeypatch):
    frame = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [2.0, 3.0], "Low": [0.5, 1.5], "Close": [1.5, 2.5], "Volume": [10, 20]},
        index=pd.to_datetime(["2026-10-26", "2026-10-27"]),
    )
    p = _provider(monkeypatch, lambda kind, t, period=None: {"frame": frame, "currency": "USD"})
    result = p.get_history("AAPL", "5d")
    assert result.currency == "USD"
    assert [r["close"] for r in result.data] == [1.5, 2.5]
    assert result.data[0]["date"].startswith("2026-10-26")


def test_history_plain_index_and_nan(monkeypatch):
    frame = pd.DataFrame({"Close": [float("nan")]}, index=["row"])
    p = _provider(monkeypatch, lambda kind, t, period=None: {"frame": frame, "currency": "INR"})
    row = p.get_history("TCS.NS", "1d").data[0]
    assert row["date"] == "row" and row["close"] is None and row["open"] is None


@pytest.mark.parametrize("raw", [{"frame": pd.DataFrame(), "currency": "USD"}, {"frame": None}])
def test_empty_history_is_permanent(monkeypatch, raw):
    p = _provider(monkeypatch, lambda kind, t, period=None: raw)
    with pytest.raises(PermanentProviderError):
        p.get_history("AAPL", "1mo")


def test_news_old_shape():
    items = [{"title": "T", "publisher": "Pub", "link": "http://x", "providerPublishTime": 1_700_000_000}]
    parsed = _parse_news(items)
    assert parsed[0]["title"] == "T" and parsed[0]["publisher"] == "Pub"
    assert parsed[0]["published"].startswith("2023-11-14")


def test_news_new_content_shape():
    items = [
        {
            "content": {
                "title": "New", "pubDate": "2026-10-27T01:00:00Z",
                "provider": {"displayName": "Wire"}, "canonicalUrl": {"url": "http://y"},
            }
        }
    ]
    assert _parse_news(items) == [
        {"title": "New", "publisher": "Wire", "link": "http://y", "published": "2026-10-27T01:00:00Z"}
    ]


@pytest.mark.parametrize("raw", [None, "text", {}, [None, 5, {}, {"content": {}}, {"content": "x"}]])
def test_news_malformed_is_defensive(raw):
    assert _parse_news(raw) == []


def test_news_provider_returns_ok_result(monkeypatch):
    p = _provider(monkeypatch, lambda kind, t, period=None: [{"title": "T"}])
    result = p.get_news("AAPL")
    assert result.data[0]["title"] == "T" and result.currency is None and result.error is None


class YFRateLimitError(Exception):
    pass


@pytest.mark.parametrize(
    "exc, expected",
    [
        (ConnectionError("x"), TransientProviderError),
        (TimeoutError("x"), TransientProviderError),
        (YFRateLimitError("x"), TransientProviderError),
        (ValueError("x"), PermanentProviderError),
        (TransientProviderError("x"), TransientProviderError),
        (PermanentProviderError("x"), PermanentProviderError),
    ],
)
def test_classify(exc, expected):
    assert isinstance(_classify(exc), expected)


def test_fetch_wraps_unknown_exceptions(monkeypatch):
    def boom(kind, t, period=None):
        raise ConnectionError("down")

    with pytest.raises(TransientProviderError):
        _provider(monkeypatch, boom).get_quote("AAPL")


def test_default_clock_is_utc_now(monkeypatch):
    monkeypatch.setattr(yfp, "_yf_call", lambda kind, t, period=None: {"price": 1.0})
    assert YFinanceProvider().get_quote("AAPL").as_of.tzinfo is not None


class FakeFastInfo(dict):
    pass


class FakeTicker:
    def __init__(self, symbol):
        self.fast_info = FakeFastInfo(last_price=10.0, previous_close=9.0, currency="USD")
        self.news = [{"title": "N"}]

    def history(self, period=None):
        return pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-10-27"]))


@pytest.fixture
def fake_yfinance(monkeypatch):
    module = types.ModuleType("yfinance")
    module.Ticker = FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", module)


def test_yf_call_with_fake_module(fake_yfinance):
    assert yfp._yf_call("quote", "AAPL") == {"price": 10.0, "previous_close": 9.0, "currency": "USD"}
    hist = yfp._yf_call("history", "AAPL", "1mo")
    assert hist["currency"] == "USD" and len(hist["frame"]) == 1
    assert yfp._yf_call("news", "AAPL") == [{"title": "N"}]


def test_fast_returns_none_for_missing_key():
    assert yfp._fast({}, "last_price") is None


def test_fast_returns_none_for_attribute_and_type_errors():
    assert yfp._fast(object(), "last_price") is None  # TypeError: not subscriptable
    assert yfp._fast(None, "last_price") is None


class ExplodingFastInfo:
    def __getitem__(self, key):
        raise ConnectionError("network down")


class ConnectionErrorTicker(FakeTicker):
    def __init__(self, symbol):
        super().__init__(symbol)
        self.fast_info = ExplodingFastInfo()


def test_fast_info_connection_error_is_transient_and_retried(monkeypatch):
    from datetime import timedelta

    from src.data.market.cache import CachedProvider
    from src.utils.config import get_config

    module = types.ModuleType("yfinance")
    module.Ticker = ConnectionErrorTicker
    monkeypatch.setitem(sys.modules, "yfinance", module)

    with pytest.raises(TransientProviderError):
        YFinanceProvider(clock=lambda: NOW).get_quote("AAPL")

    attempts = {"n": 0}
    real_call = yfp._yf_call

    def counting(kind, ticker, period=None):
        attempts["n"] += 1
        return real_call(kind, ticker, period)

    monkeypatch.setattr(yfp, "_yf_call", counting)
    sleeps = []
    provider = CachedProvider(YFinanceProvider(clock=lambda: NOW), sleep=sleeps.append)
    result = provider.get_quote("AAPL")  # retried, then falls back (sample data)
    cfg = get_config()["market_data"]
    assert attempts["n"] == cfg["max_retries"]
    assert len(sleeps) == cfg["max_retries"] - 1
    assert result.source == "sample data"


# ---- Pandas4Warning is silenced only when it comes from yfinance ----

import warnings

import pandas.errors


def _emit(category, module):
    warnings.warn_explicit("legacy feature", category, filename=f"{module}.py", lineno=1, module=module, registry={})


def test_pandas4_warning_from_yfinance_is_ignored_but_other_warnings_are_not():
    category = pandas.errors.Pandas4Warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")          # start from "show everything"
        yfp.silence_pandas4_warning()            # then our targeted filter goes on top
        _emit(category, "yfinance.utils")        # ignored
        _emit(category, "yfinance")              # ignored (the package itself)
        _emit(category, "src.data.market.cache")           # our own code: shown
        _emit(category, "yfinance_extras")                 # a different package: shown
        _emit(UserWarning, "yfinance.utils")               # other category from yfinance: shown
    assert [(w.category, w.filename) for w in caught] == [
        (category, "src.data.market.cache.py"), (category, "yfinance_extras.py"),
        (UserWarning, "yfinance.utils.py"),
    ]


def test_filter_is_skipped_when_pandas_has_no_pandas4warning(monkeypatch):
    monkeypatch.delattr(pandas.errors, "Pandas4Warning")
    with warnings.catch_warnings():
        before = list(warnings.filters)
        yfp.silence_pandas4_warning()
        assert list(warnings.filters) == before
