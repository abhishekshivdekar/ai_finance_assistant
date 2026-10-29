"""yfinance implementation of MarketDataProvider.

Design pattern: STRATEGY. YFinanceProvider is the MVP strategy behind the
MarketDataProvider interface. Example: get_quote("RELIANCE.NS") returns the
price with currency "INR" exactly as yfinance reports it; this class never
decides a currency itself. News comes from a second, injected Strategy
(NewsSource), so the cache and fallback chain wrap both with no extra code.

All yfinance access lives in one function, _yf_call, so tests patch it and
never touch the network.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Callable

import pandas.errors

from src.core.models import MarketResult
from src.data.market.provider import (
    MarketDataProvider,
    PermanentProviderError,
    TransientProviderError,
)
from src.data.news.google_rss import GoogleNewsRssSource
from src.data.news.source import NewsSource
from src.utils.logger import get_logger

log = get_logger(__name__)


def silence_pandas4_warning() -> None:
    """Ignore pandas' Pandas4Warning when it comes from yfinance (and only then).

    yfinance still uses pandas features that pandas 4 will remove, and each call
    would print a warning we cannot fix. The filter is limited to that warning
    category AND to code in the yfinance package, so a Pandas4Warning raised by
    our own code, or any other warning, is still shown.

    Call it AFTER yfinance has been imported: yfinance/__init__.py adds its own
    filter ("default" for DeprecationWarning from ^yfinance) at the front of the
    list, and Pandas4Warning is a DeprecationWarning, so a filter installed
    earlier would be overridden. load_yfinance() does import then silence.
    """
    category = getattr(pandas.errors, "Pandas4Warning", None)   # absent on older pandas
    if category is not None:
        warnings.filterwarnings("ignore", category=category, module=r"yfinance(\.|$)")



def load_yfinance() -> Any:
    """Import yfinance, then (re)apply the Pandas4Warning filter on top of the one yfinance installs.

    Safe to call on every request: warnings.filterwarnings() removes an identical
    existing filter before inserting, so the list does not grow. Every place that
    needs yfinance (scripts, tests, the Streamlit app) goes through here.
    """
    import yfinance as yf

    silence_pandas4_warning()
    return yf


_TRANSIENT_NAMES = {
    "YFRateLimitError", "ReadTimeout", "ConnectTimeout", "Timeout",
    "ConnectionError", "ChunkedEncodingError", "SSLError",
}


def _yf_call(kind: str, ticker: str, period: str | None = None) -> Any:
    """The ONLY place that talks to yfinance. kind: quote | history."""
    yf = load_yfinance()
    handle = yf.Ticker(ticker)
    if kind == "quote":
        info = handle.fast_info
        return {
            "price": _fast(info, "last_price"),
            "previous_close": _fast(info, "previous_close"),
            "currency": _fast(info, "currency"),
        }
    return {
        "frame": handle.history(period=period),
        "currency": _fast(handle.fast_info, "currency"),
    }


def _fast(info: Any, key: str) -> Any:
    """Read a fast_info field, returning None if missing.

    Other errors (ConnectionError, timeouts, rate limits) propagate so that
    _fetch() can classify them as transient and the chain can retry.
    """
    try:
        return info[key]
    except (KeyError, AttributeError, TypeError):
        return None


def _classify(exc: Exception) -> Exception:
    """Map any exception to Transient (retryable) or Permanent."""
    if isinstance(exc, (TransientProviderError, PermanentProviderError)):
        return exc
    names = {cls.__name__ for cls in type(exc).__mro__}
    if isinstance(exc, (ConnectionError, TimeoutError)) or names & _TRANSIENT_NAMES:
        return TransientProviderError(str(exc))
    return PermanentProviderError(str(exc))


def _num(value: Any) -> float | None:
    """float(value), or None for missing/NaN."""
    if value is None or value != value:
        return None
    return float(value)


class YFinanceProvider(MarketDataProvider):
    """Live quotes and history from yfinance (raises Transient/Permanent errors).

    Headlines are not available from yfinance in a usable form, so get_news is
    delegated to an injected NewsSource (default: Google News RSS, architecture section 6).
    """

    def __init__(self, clock: Callable[[], datetime] | None = None, news_source: NewsSource | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._news = news_source if news_source is not None else GoogleNewsRssSource(clock=self._clock)

    def _fetch(self, kind: str, symbol: str, period: str | None = None) -> Any:
        try:
            return _yf_call(kind, symbol, period)
        except Exception as exc:
            raise _classify(exc) from exc

    def _result(self, data: Any, currency: str | None) -> MarketResult:
        return MarketResult(data=data, currency=currency, as_of=self._clock(), source="live")

    def _get_quote(self, symbol: str) -> MarketResult:
        raw = self._fetch("quote", symbol)
        price = _num(raw.get("price"))
        if price is None:
            raise PermanentProviderError(f"No price data for {symbol}.")
        previous = _num(raw.get("previous_close"))
        change = None if previous is None else price - previous
        change_pct = None if not previous else change / previous * 100
        data = {
            "ticker": symbol, "price": price, "previous_close": previous,
            "change": change, "change_pct": change_pct,
        }
        return self._result(data, raw.get("currency"))

    def _get_history(self, symbol: str, period: str) -> MarketResult:
        raw = self._fetch("history", symbol, period)
        frame = raw.get("frame")
        if frame is None or len(frame) == 0:
            raise PermanentProviderError(f"No history for {symbol}.")
        rows = []
        for idx, row in frame.iterrows():
            rows.append(
                {
                    "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "open": _num(row.get("Open")), "high": _num(row.get("High")),
                    "low": _num(row.get("Low")), "close": _num(row.get("Close")),
                    "volume": _num(row.get("Volume")),
                }
            )
        return self._result(rows, raw.get("currency"))

    def _get_news(self, query: str) -> MarketResult:
        return self._result(self._news.fetch(query), None)
