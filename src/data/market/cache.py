"""CachedProvider: caching and resilience around any MarketDataProvider.

Design pattern: DECORATOR. CachedProvider implements the same interface as
the provider it wraps and adds a TTL cache plus the fallback chain, without
changing that interface. Example: the Market agent calls get_quote("AAPL")
and cannot tell whether it talks to YFinanceProvider or CachedProvider.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from src.core.models import MarketResult
from src.data.market.fallback import (
    ErrorHandler,
    FreshCacheHandler,
    LiveHandler,
    Request,
    SampleHandler,
    StaleCacheHandler,
)
from src.data.market.provider import MarketDataProvider
from src.data.market.ttl_cache import TTLCache
from src.utils.config import get_config


class CachedProvider(MarketDataProvider):
    """Wraps a provider with a TTL cache and the live -> stale -> sample chain."""

    def __init__(
        self,
        inner: MarketDataProvider,
        *,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = get_config()["market_data"]
        self._inner = inner
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._cache = TTLCache(self._clock)
        self._quote_ttl = timedelta(minutes=cfg["cache_ttl_minutes"])
        self._history_ttl = timedelta(minutes=cfg["history_ttl_minutes"])

        head = FreshCacheHandler(self._cache)
        head.set_next(
            LiveHandler(self._cache, cfg["max_retries"], cfg["backoff_seconds"], sleep)
        ).set_next(StaleCacheHandler(self._cache)).set_next(
            SampleHandler(cfg["snapshot_path"])
        ).set_next(ErrorHandler())
        self._chain = head

    def _get_quote(self, symbol: str) -> MarketResult:
        return self._chain.handle(
            Request("quote", symbol, None, lambda: self._inner.get_quote(symbol), self._quote_ttl)
        )

    def _get_history(self, symbol: str, period: str) -> MarketResult:
        return self._chain.handle(
            Request(
                "history", symbol, period,
                lambda: self._inner.get_history(symbol, period), self._history_ttl,
            )
        )

    def _get_news(self, query: str) -> MarketResult:
        """News is cached per search phrase with the quote TTL; the cache key and the fetch use the same normalised phrase."""
        phrase = " ".join(query.casefold().split())
        return self._chain.handle(
            Request("news", phrase, None, lambda: self._inner.get_news(phrase), self._quote_ttl)
        )
