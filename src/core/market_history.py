"""Read the Market agent's request and fetch one price history per ticker.

Design pattern: DEPENDENCY INJECTION. fetch_histories() receives the provider as
an argument. Example: the agent passes the shared CachedProvider (cache, retry,
stale and sample fallback), a test passes a fake that returns fixed rows.

Histories are fetched concurrently (bounded by market_analysis.max_workers);
results keep the input order and one failing ticker never affects the others.
The same one-year history serves every selected period (it is sliced later).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.core.models import MarketResult
from src.data.market.provider import InvalidSymbolError, MarketDataProvider, error_result, validate_symbol
from src.utils.logger import get_logger

log = get_logger(__name__)


def _ticker_list(raw: Any) -> list[Any]:
    """The tickers as a list; a single string may hold several, separated by commas or spaces."""
    if isinstance(raw, str):
        return raw.replace(",", " ").split()
    return list(raw) if isinstance(raw, (list, tuple)) else []


def parse_request(params: Any, cfg: dict[str, Any]) -> tuple[list[str], str, list[str]]:
    """(tickers, period, warnings) from state["market_params"] = {"tickers": [...], "period": "6mo"}.

    Invalid or duplicate tickers and tickers beyond max_tickers become warnings;
    an unknown or missing period falls back to market_analysis.default_period.
    """
    params = params if isinstance(params, dict) else {}
    tickers: list[str] = []
    warnings: list[str] = []
    for raw in _ticker_list(params.get("tickers")):
        try:
            symbol = validate_symbol(raw)
        except InvalidSymbolError:
            warnings.append(f"{str(raw)!r} was skipped: it is not a valid ticker.")
            continue
        if symbol in tickers:
            warnings.append(f"{symbol} appears more than once; it is analysed once.")
        elif len(tickers) >= cfg["max_tickers"]:
            warnings.append(f"Only the first {cfg['max_tickers']} tickers are analysed; {symbol} was left out.")
        else:
            tickers.append(symbol)

    period = params.get("period") or cfg["default_period"]
    if period not in cfg["periods"]:
        warnings.append(f"The period {str(period)!r} is not supported; {cfg['default_period']} is used.")
        period = cfg["default_period"]
    return tickers, period, warnings


def _fetch_history(provider: MarketDataProvider, ticker: str, period: str) -> MarketResult:
    """One history; any exception becomes an error result so one ticker cannot break the rest."""
    try:
        return provider.get_history(ticker, period)
    except Exception as exc:
        log.warning("History lookup for %s raised %s", ticker, type(exc).__name__)
        return error_result(f"History lookup failed for {ticker}.")


def fetch_histories(tickers: list[str], provider: MarketDataProvider, cfg: dict[str, Any]) -> list[MarketResult]:
    """One fetch_period history per ticker, in input order, in at most max_workers threads."""
    if not tickers:
        return []
    workers = max(1, min(int(cfg["max_workers"]), len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda t: _fetch_history(provider, t, cfg["fetch_period"]), tickers))
