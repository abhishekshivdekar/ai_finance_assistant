"""Market data provider interface and symbol validation.

Design patterns:
- STRATEGY: MarketDataProvider is the interface; YFinanceProvider is one
  implementation. Example: an AlphaVantageProvider could replace it via the
  `market_data.provider` config value without any agent changing.
- TEMPLATE METHOD: the public get_quote/get_history/get_news validate the input
  and then call the subclass hooks (_get_quote, ...). Example: "  ???  " is
  rejected here once, so neither the cache nor yfinance ever sees it.

Failure contract for raw providers: fetch problems raise
TransientProviderError (retryable) or PermanentProviderError (not retryable).
CachedProvider turns both into results, so only it never raises.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from src.core.models import MarketResult
from src.data.errors import PermanentProviderError, TransientProviderError  # noqa: F401  (re-exported)
from src.utils.config import get_config

# US: AAPL, BRK-B, BF.B. NSE: RELIANCE.NS, M&M.NS. Index: ^NSEI, ^BSESN, ^GSPC.
_US = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")
_NSE = re.compile(r"^[A-Z0-9&-]{1,20}\.NS$")
_INDEX = re.compile(r"^\^[A-Z0-9]{2,10}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class InvalidSymbolError(ValueError):
    """The ticker or period is empty or malformed."""


def validate_symbol(raw: Any) -> str:
    """Return the normalised symbol or raise InvalidSymbolError."""
    symbol = raw.strip().upper() if isinstance(raw, str) else ""
    if not symbol:
        raise InvalidSymbolError("Ticker is empty.")
    if not (_US.match(symbol) or _NSE.match(symbol) or _INDEX.match(symbol)):
        raise InvalidSymbolError(f"Invalid ticker {raw!r}.")
    return symbol


def validate_period(period: Any) -> str:
    """Return the period if it is listed in market_data.history_periods."""
    allowed = get_config()["market_data"]["history_periods"]
    if period not in allowed:
        raise InvalidSymbolError(f"Invalid history period {period!r}.")
    return period


def validate_news_query(raw: Any) -> str:
    """Return the cleaned search phrase (control characters removed, spaces collapsed) or raise InvalidSymbolError."""
    text = _CONTROL.sub(" ", raw) if isinstance(raw, str) else ""
    query = " ".join(text.split())
    if not query:
        raise InvalidSymbolError("News search is empty.")
    limit = get_config()["news"]["max_query_chars"]
    if len(query) > limit:
        raise InvalidSymbolError(f"News search is longer than {limit} characters.")
    return query


def error_result(message: str) -> MarketResult:
    """A MarketResult that carries only an error (no data, no currency)."""
    return MarketResult(data=None, currency=None, as_of=None, source="none", error=message)


class MarketDataProvider(ABC):
    """Strategy interface for market data (Template Method for validation)."""

    def get_quote(self, ticker: str) -> MarketResult:
        """Validated quote lookup."""
        try:
            symbol = validate_symbol(ticker)
        except InvalidSymbolError as exc:
            return error_result(str(exc))
        return self._get_quote(symbol)

    def get_history(self, ticker: str, period: str) -> MarketResult:
        """Validated price history lookup."""
        try:
            symbol = validate_symbol(ticker)
            period = validate_period(period)
        except InvalidSymbolError as exc:
            return error_result(str(exc))
        return self._get_history(symbol, period)

    def get_news(self, query: str) -> MarketResult:
        """Validated headline lookup for a search phrase (the News agent turns tickers into phrases)."""
        try:
            phrase = validate_news_query(query)
        except InvalidSymbolError as exc:
            return error_result(str(exc))
        return self._get_news(phrase)

    @abstractmethod
    def _get_quote(self, symbol: str) -> MarketResult:
        """Fetch a quote for an already-validated symbol."""

    @abstractmethod
    def _get_history(self, symbol: str, period: str) -> MarketResult:
        """Fetch history for an already-validated symbol and period."""

    @abstractmethod
    def _get_news(self, query: str) -> MarketResult:
        """Fetch headlines for an already-validated search phrase."""
