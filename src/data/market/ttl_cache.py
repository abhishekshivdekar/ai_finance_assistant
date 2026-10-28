"""In-memory TTL cache for MarketResult values.

Used by the fallback chain and CachedProvider. Keys are (ticker, kind), where
kind is "quote", "history:<period>" or "news". Example: ("AAPL", "quote") and
("AAPL", "history:1mo") are separate entries with separate TTLs.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Callable

from src.core.models import MarketResult

Key = tuple[str, str]


class TTLCache:
    """Thread-safe cache with an injectable clock (tests need no real waiting)."""

    def __init__(self, clock: Callable[[], datetime]) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._store: dict[Key, tuple[datetime, MarketResult]] = {}

    def put(self, key: Key, result: MarketResult) -> None:
        """Store a result with the current time."""
        with self._lock:
            self._store[key] = (self._clock(), result)

    def get_fresh(self, key: Key, ttl: timedelta) -> MarketResult | None:
        """Return the entry only if it is younger than `ttl`."""
        with self._lock:
            entry = self._store.get(key)
        if entry is None or self._clock() - entry[0] >= ttl:
            return None
        return entry[1]

    def get_any(self, key: Key) -> MarketResult | None:
        """Return the entry even if expired (for the stale fallback)."""
        with self._lock:
            entry = self._store.get(key)
        return None if entry is None else entry[1]
