import threading
from datetime import datetime, timedelta, timezone

import pytest

from src.core.models import MarketResult
from src.data.market.cache import CachedProvider
from src.data.market.provider import MarketDataProvider
from src.data.market.ttl_cache import TTLCache
from src.utils.config import get_config

START = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


class FakeInner(MarketDataProvider):
    """Always succeeds; counts calls per method."""

    def __init__(self):
        self.calls = {"quote": 0, "history": 0, "news": 0}

    def _get_quote(self, symbol):
        self.calls["quote"] += 1
        return MarketResult({"price": 1.0}, "INR", START, "live")

    def _get_history(self, symbol, period):
        self.calls["history"] += 1
        return MarketResult([{"close": 1.0}], "INR", START, "live")

    def _get_news(self, symbol):
        self.calls["news"] += 1
        return MarketResult([], None, START, "live")


def _result():
    return MarketResult({"p": 1}, "USD", START, "live")


def test_ttl_cache_miss_hit_and_expiry():
    clock = Clock()
    cache = TTLCache(clock)
    ttl = timedelta(minutes=30)
    assert cache.get_fresh(("AAPL", "quote"), ttl) is None
    cache.put(("AAPL", "quote"), _result())
    assert cache.get_fresh(("AAPL", "quote"), ttl) is not None
    clock.advance(minutes=29)
    assert cache.get_fresh(("AAPL", "quote"), ttl) is not None
    clock.advance(minutes=1)
    assert cache.get_fresh(("AAPL", "quote"), ttl) is None  # expired...
    assert cache.get_any(("AAPL", "quote")) is not None  # ...but kept for stale fallback


def test_ttl_cache_keys_are_separate():
    cache = TTLCache(Clock())
    cache.put(("AAPL", "quote"), _result())
    assert cache.get_any(("AAPL", "history:1mo")) is None
    assert cache.get_any(("MSFT", "quote")) is None


def test_ttl_cache_thread_safety_smoke():
    cache = TTLCache(Clock())

    def work(n):
        for i in range(200):
            cache.put((f"T{n}", "quote"), _result())
            cache.get_any((f"T{n}", "quote"))

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(cache.get_any((f"T{n}", "quote")) for n in range(8))


def _provider():
    clock, inner = Clock(), FakeInner()
    return CachedProvider(inner, clock=clock, sleep=lambda s: None), inner, clock


def test_second_call_is_cache_hit_with_original_as_of():
    provider, inner, clock = _provider()
    first = provider.get_quote("AAPL")
    clock.advance(minutes=5)
    second = provider.get_quote("AAPL")
    assert first.source == "live" and second.source == "cache"
    assert second.as_of == first.as_of and not second.is_stale
    assert inner.calls["quote"] == 1


def test_quote_ttl_expiry_refetches():
    provider, inner, clock = _provider()
    provider.get_quote("AAPL")
    clock.advance(minutes=get_config()["market_data"]["cache_ttl_minutes"])
    assert provider.get_quote("AAPL").source == "live"
    assert inner.calls["quote"] == 2


def test_history_uses_longer_ttl():
    cfg = get_config()["market_data"]
    assert cfg["history_ttl_minutes"] > cfg["cache_ttl_minutes"]
    provider, inner, clock = _provider()
    provider.get_history("AAPL", "1mo")
    clock.advance(minutes=cfg["cache_ttl_minutes"] + 1)
    assert provider.get_history("AAPL", "1mo").source == "cache"
    clock.advance(minutes=cfg["history_ttl_minutes"])
    assert provider.get_history("AAPL", "1mo").source == "live"
    assert inner.calls["history"] == 2


def test_news_with_different_case_and_spacing_hits_one_cache_entry_and_one_fetch():
    provider, inner, _ = _provider()
    fetched = []
    inner._get_news = lambda phrase: fetched.append(phrase) or MarketResult([], None, START, "live")
    provider.get_news("Nifty 50")
    provider.get_news("NIFTY  50")
    assert fetched == ["nifty 50"]


def test_history_cached_per_period_and_news_cached():
    provider, inner, _ = _provider()
    provider.get_history("AAPL", "1mo")
    provider.get_history("AAPL", "5d")
    provider.get_news("AAPL")
    provider.get_news("AAPL")
    assert inner.calls["history"] == 2 and inner.calls["news"] == 1


def test_cached_provider_is_transparent():
    provider, inner, _ = _provider()
    assert isinstance(provider, MarketDataProvider)
    raw = inner.get_quote("AAPL")
    wrapped = provider.get_quote("AAPL")
    assert (wrapped.data, wrapped.currency) == (raw.data, raw.currency)
    assert provider.get_history("AAPL", "1mo").data == inner.get_history("AAPL", "1mo").data
    assert provider.get_news("AAPL").data == inner.get_news("AAPL").data
    for name in ("get_quote", "get_history", "get_news"):
        assert callable(getattr(provider, name))


def test_invalid_ticker_does_not_touch_inner_or_cache():
    provider, inner, _ = _provider()
    result = provider.get_quote("not a ticker")
    assert result.error and inner.calls["quote"] == 0


def test_default_clock_and_sleep_are_usable():
    provider = CachedProvider(FakeInner())
    assert provider.get_quote("AAPL").as_of is not None


def test_config_ttls_are_read():
    cfg = get_config()["market_data"]
    assert cfg["cache_ttl_minutes"] == 30
    assert pytest.approx(cfg["history_ttl_minutes"]) == 240


class TickerInner(MarketDataProvider):
    """Returns data that identifies the ticker, so a mix-up between threads would show."""

    def _get_quote(self, symbol):
        return MarketResult({"price": float(len(symbol)), "ticker": symbol}, "USD", START, "live")

    def _get_history(self, symbol, period):
        return MarketResult([{"ticker": symbol, "period": period}], "USD", START, "live")

    def _get_news(self, symbol):
        return MarketResult([], None, START, "live")


def test_cached_provider_is_correct_under_concurrent_calls():
    """Many threads, same and different tickers (as Market and Portfolio use it): no errors, no mix-ups."""
    provider = CachedProvider(TickerInner(), clock=Clock(), sleep=lambda s: None)
    tickers = ["AAPL", "MSFT", "TCS.NS", "RELIANCE.NS"]
    errors: list[Exception] = []
    barrier = threading.Barrier(16)

    def work(n):
        try:
            barrier.wait()
            for i in range(50):
                ticker = tickers[(n + i) % len(tickers)]
                quote = provider.get_quote(ticker)
                history = provider.get_history(ticker, "1y")
                assert quote.error is None and quote.data["ticker"] == ticker
                assert history.error is None and history.data == [{"ticker": ticker, "period": "1y"}]
                assert quote.source in ("live", "cache") and history.source in ("live", "cache")
        except Exception as exc:                     # collected here so the main thread can assert on it
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(provider.get_quote(t).source == "cache" for t in tickers)
