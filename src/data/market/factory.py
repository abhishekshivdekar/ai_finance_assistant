"""Market provider factory.

Design pattern: SINGLETON (cached getter) around a FACTORY. The provider named
in `market_data.provider` is built once per process. Example: with
`provider: yfinance`, every agent gets the same CachedProvider(YFinanceProvider())
and therefore shares one cache.
"""
from __future__ import annotations

from functools import lru_cache

from src.data.market.cache import CachedProvider
from src.data.market.provider import MarketDataProvider
from src.data.market.yfinance_provider import YFinanceProvider
from src.utils.config import ConfigError, get_config


@lru_cache(maxsize=1)
def get_market_provider() -> MarketDataProvider:
    """Return the shared market data provider chosen by config."""
    name = str(get_config()["market_data"].get("provider", "")).lower()
    if name == "yfinance":
        return CachedProvider(YFinanceProvider())
    raise ConfigError(f"Unsupported market_data.provider: {name!r}")
