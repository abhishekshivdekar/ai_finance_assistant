"""Market data fallback chain.

Design pattern: CHAIN OF RESPONSIBILITY. Each handler either returns a
MarketResult or passes the request to the next handler. Example: for AAPL the
live fetch times out three times, so the stale-cache handler answers with the
expired entry (is_stale=True); if there were none, the sample handler would
answer with source="sample data".

Order: fresh cache -> live (with retry) -> stale cache -> sample -> error.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from src.core.models import MarketResult
from src.data.market.provider import (
    PermanentProviderError,
    TransientProviderError,
    error_result,
)
from src.data.market.ttl_cache import Key, TTLCache
from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Request:
    """One market data lookup travelling down the chain."""

    kind: str  # quote | history | news
    ticker: str
    period: str | None
    fetch: Callable[[], MarketResult]
    ttl: timedelta

    @property
    def key(self) -> Key:
        """Cache key (ticker, kind); history is keyed per period."""
        kind = f"history:{self.period}" if self.kind == "history" else self.kind
        return (self.ticker, kind)


class Handler(ABC):
    """Base link of the chain."""

    def __init__(self) -> None:
        self._next: Handler | None = None

    def set_next(self, handler: Handler) -> Handler:
        """Link the next handler and return it (allows chaining)."""
        self._next = handler
        return handler

    def handle(self, request: Request) -> MarketResult:
        """Answer the request, or pass it on."""
        result = self._try(request)
        if result is not None:
            return result
        if self._next is None:
            return error_result(f"Market data for {request.ticker} is currently unavailable.")
        return self._next.handle(request)

    @abstractmethod
    def _try(self, request: Request) -> MarketResult | None:
        """Return a result, or None to pass the request on."""


class FreshCacheHandler(Handler):
    """Returns a cache entry younger than the TTL."""

    def __init__(self, cache: TTLCache) -> None:
        super().__init__()
        self._cache = cache

    def _try(self, request: Request) -> MarketResult | None:
        cached = self._cache.get_fresh(request.key, request.ttl)
        if cached is None:
            return None
        return replace(cached, source="cache", is_stale=False)


class LiveHandler(Handler):
    """Live fetch; retries transient errors with exponential backoff."""

    def __init__(
        self,
        cache: TTLCache,
        max_attempts: int,
        backoff_seconds: float,
        sleep: Callable[[float], None],
    ) -> None:
        super().__init__()
        self._cache = cache
        self._attempts = max(1, int(max_attempts))
        self._backoff = backoff_seconds
        self._sleep = sleep

    def _try(self, request: Request) -> MarketResult | None:
        for attempt in range(self._attempts):
            try:
                result = request.fetch()
            except TransientProviderError as exc:
                log.warning("Transient error for %s (attempt %d): %s", request.ticker, attempt + 1, exc)
                if attempt < self._attempts - 1:
                    self._sleep(self._backoff * 2**attempt)
                continue
            except PermanentProviderError as exc:
                log.warning("Permanent error for %s, not retrying: %s", request.ticker, exc)
                return None
            except Exception:
                log.exception("Unexpected error fetching %s", request.ticker)
                return None
            if result.error:
                return None
            self._cache.put(request.key, result)
            return result
        return None


class StaleCacheHandler(Handler):
    """Returns an expired cache entry, flagged is_stale=True."""

    def __init__(self, cache: TTLCache) -> None:
        super().__init__()
        self._cache = cache

    def _try(self, request: Request) -> MarketResult | None:
        cached = self._cache.get_any(request.key)
        if cached is None:
            return None
        log.warning("Serving stale cache for %s", request.ticker)
        return replace(cached, source="stale cache", is_stale=True)


@lru_cache(maxsize=4)
def load_snapshot(path: str) -> dict[str, Any]:
    """Load the bundled sample snapshot once; an unreadable file gives {}."""
    file = Path(path)
    if not file.is_absolute():
        file = PROJECT_ROOT / file
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("Sample snapshot unavailable at %s", file)
        return {}


class SampleHandler(Handler):
    """Answers from the bundled snapshot, always labelled "sample data"."""

    def __init__(self, snapshot_path: str) -> None:
        super().__init__()
        self._path = snapshot_path

    def _try(self, request: Request) -> MarketResult | None:
        snapshot = load_snapshot(self._path)
        entry = (snapshot.get("tickers") or {}).get(request.ticker)
        if entry is None or request.kind == "news":
            return None
        as_of = datetime.fromisoformat(snapshot["_meta"]["as_of"])
        if request.kind == "history":
            data: Any = entry["history"]
        else:
            price, previous = entry["price"], entry["previous_close"]
            data = {
                "ticker": request.ticker, "price": price, "previous_close": previous,
                "change": price - previous, "change_pct": (price - previous) / previous * 100,
            }
        return MarketResult(
            data=data, currency=entry["currency"], as_of=as_of, source="sample data"
        )


class ErrorHandler(Handler):
    """Terminal handler: total failure becomes an error result."""

    def _try(self, request: Request) -> MarketResult | None:
        return error_result(f"Market data for {request.ticker} is currently unavailable.")
