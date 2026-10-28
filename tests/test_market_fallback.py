from datetime import datetime, timedelta, timezone

import pytest

from src.core.models import MarketResult
from src.data.market import fallback
from src.data.market.cache import CachedProvider
from src.data.market.provider import (
    MarketDataProvider,
    PermanentProviderError,
    TransientProviderError,
)
from src.utils.config import get_config

START = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


class Scripted(MarketDataProvider):
    """Plays back outcomes (exceptions are raised); the last one repeats."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def _play(self):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def _get_quote(self, symbol):
        return self._play()

    def _get_history(self, symbol, period):
        return self._play()

    def _get_news(self, symbol):
        return self._play()


def ok(currency="USD"):
    return MarketResult({"price": 1.0}, currency, START, "live")


def build(inner):
    clock, sleeps = Clock(), []
    return CachedProvider(inner, clock=clock, sleep=sleeps.append), clock, sleeps


def test_retry_count_and_backoff():
    cfg = get_config()["market_data"]
    inner = Scripted(TransientProviderError("down"))
    provider, _, sleeps = build(inner)
    provider.get_quote("ZZZZ")
    assert inner.calls == cfg["max_retries"]
    base = cfg["backoff_seconds"]
    assert sleeps == [base * 2**n for n in range(cfg["max_retries"] - 1)]


def test_transient_then_success_stops_retrying():
    inner = Scripted(TransientProviderError("x"), ok())
    provider, _, sleeps = build(inner)
    result = provider.get_quote("AAPL")
    assert result.source == "live" and result.error is None
    assert inner.calls == 2 and len(sleeps) == 1


def test_no_retry_on_permanent_error():
    inner = Scripted(PermanentProviderError("unknown ticker"))
    provider, _, sleeps = build(inner)
    provider.get_quote("ZZZZ")
    assert inner.calls == 1 and sleeps == []


def test_no_retry_and_no_call_on_bad_ticker():
    inner = Scripted(ok())
    provider, _, sleeps = build(inner)
    assert provider.get_quote("bad ticker!").error
    assert inner.calls == 0 and sleeps == []


def test_unexpected_exception_is_not_retried_and_never_raised():
    inner = Scripted(RuntimeError("weird"))
    provider, _, sleeps = build(inner)
    result = provider.get_quote("ZZZZ")
    assert result.error and inner.calls == 1 and sleeps == []


def test_stale_cache_after_failure():
    inner = Scripted(ok("USD"), TransientProviderError("down"))
    provider, clock, _ = build(inner)
    first = provider.get_quote("AAPL")
    clock.advance(minutes=get_config()["market_data"]["cache_ttl_minutes"] + 1)
    stale = provider.get_quote("AAPL")
    assert stale.source == "stale cache" and stale.is_stale is True
    assert stale.data == first.data and stale.as_of == first.as_of
    assert stale.currency == "USD"


def test_sample_fallback_is_labelled_with_per_ticker_currency():
    provider, _, _ = build(Scripted(TransientProviderError("down")))
    inr = provider.get_quote("TCS.NS")
    usd = provider.get_quote("AAPL")
    assert inr.source == "sample data" and inr.currency == "INR"
    assert usd.source == "sample data" and usd.currency == "USD"
    assert inr.as_of == datetime.fromisoformat("2026-10-27T00:00:00+00:00")
    assert inr.error is None and inr.data["price"] > 0
    assert inr.data["change_pct"] == pytest.approx(
        (inr.data["price"] - inr.data["previous_close"]) / inr.data["previous_close"] * 100
    )


def test_sample_history_and_index():
    provider, _, _ = build(Scripted(TransientProviderError("down")))
    hist = provider.get_history("^NSEI", "1mo")
    assert hist.source == "sample data" and hist.currency == "INR"
    assert len(hist.data) == 5


def test_stale_beats_sample():
    inner = Scripted(ok("USD"), TransientProviderError("down"))
    provider, clock, _ = build(inner)
    provider.get_quote("AAPL")
    clock.advance(hours=5)
    assert provider.get_quote("AAPL").source == "stale cache"


def test_total_failure_gives_error_result():
    provider, _, _ = build(Scripted(TransientProviderError("down")))
    result = provider.get_quote("ZZZZ")  # valid shape, not in snapshot
    assert result.error and result.data is None and result.source == "none"
    assert result.currency is None and not result.is_stale


def test_news_has_no_sample_so_it_errors():
    provider, _, _ = build(Scripted(TransientProviderError("down")))
    assert provider.get_news("AAPL").error


def test_live_result_with_error_falls_through():
    bad = MarketResult(None, None, None, "none", error="provider said no")
    provider, _, _ = build(Scripted(bad))
    assert provider.get_quote("AAPL").source == "sample data"


def test_fresh_result_flags():
    provider, _, _ = build(Scripted(ok("INR")))
    live = provider.get_quote("TCS.NS")
    cached = provider.get_quote("TCS.NS")
    assert (live.source, live.is_stale) == ("live", False)
    assert (cached.source, cached.is_stale) == ("cache", False)
    assert cached.currency == "INR" and cached.as_of == START


def test_missing_snapshot_fails_soft(tmp_path):
    fallback.load_snapshot.cache_clear()
    assert fallback.load_snapshot(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert fallback.load_snapshot(str(bad)) == {}
    fallback.load_snapshot.cache_clear()


def test_chain_without_next_handler_returns_error():
    handler = fallback.FreshCacheHandler(fallback.TTLCache(Clock()))
    request = fallback.Request("quote", "AAPL", None, lambda: ok(), timedelta(minutes=1))
    assert handler.handle(request).error
