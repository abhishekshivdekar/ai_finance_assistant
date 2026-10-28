"""Market data layer: provider interface, yfinance strategy, cache and fallback chain."""
from src.data.market.factory import get_market_provider

__all__ = ["get_market_provider"]
