"""Pure market metric functions: no I/O, no LLM, no hidden state.

Design pattern: PURE FUNCTIONS (each metric is its own small function).
Example: max_drawdown_pct([100, 110, 99, 108.9]) can be tested with a plain
list, and ticker_metrics() just composes sma -> price_vs_pct -> trend_label.

One year of daily history is fetched per ticker and sliced here:
- the REQUESTED PERIOD slice (slice_period) feeds the period return, the
  volatility and the drawdown;
- the FULL YEAR feeds the moving averages, the 52-week range and the latest
  price / day change.
So the trend label reflects the latest averages, not only the selected window.

Every threshold comes from the `market_analysis:` config block (passed in as
`cfg`). Values are illustrative. A metric that cannot be computed (too few
closes) is None and gets a warning; it never fails the ticker. When the history
is shorter than the requested period the metrics are still computed, but over the
shorter span: the ticker is flagged (period_is_short) and a warning says so.
Prices stay in the ticker's own currency.
"""
from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.core.market_models import AverageMetric, MarketMetrics, Range52w, TickerMetrics

Series = list[tuple[date, float]]          # (day, close), oldest first
NOT_AVAILABLE = "not available"
SOURCE_ORDER = ("live", "cache", "stale cache", "sample data")   # best to worst (MarketResult.source)


def _parse_day(value: Any) -> date | None:
    """The calendar day of an ISO date string, or None if it cannot be read."""
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def price_series(rows: Any) -> Series:
    """(day, close) pairs from provider rows, oldest first. Rows without a usable date or close are dropped."""
    series: Series = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        close, day = row.get("close"), _parse_day(row.get("date"))
        if (isinstance(close, (int, float)) and not isinstance(close, bool)
                and math.isfinite(close) and close > 0 and day is not None):
            series.append((day, float(close)))
    return sorted(series, key=lambda item: item[0])


def slice_period(series: Series, days: int) -> Series:
    """The part of the series within `days` calendar days of its last day (the requested period)."""
    if not series:
        return []
    cutoff = series[-1][0] - timedelta(days=days)
    return [item for item in series if item[0] >= cutoff]


def span_days(series: Series) -> int:
    """Calendar days between the first and the last close (0 for an empty series)."""
    return (series[-1][0] - series[0][0]).days if series else 0


def is_short_history(span: int, period_days: int, tolerance_pct: float) -> bool:
    """True when `span` covers less than the requested period minus the tolerance (in percent)."""
    return span < period_days * (1 - tolerance_pct / 100)


def worst_source(sources: list[str]) -> str:
    """The least live source in SOURCE_ORDER; an unknown label counts as the worst. "" for no sources."""
    if not sources:
        return ""
    return max(sources, key=lambda s: SOURCE_ORDER.index(s) if s in SOURCE_ORDER else len(SOURCE_ORDER))


def latest_and_day_change(closes: list[float]) -> tuple[float, float | None, float | None]:
    """(latest close, change vs the previous close, change in percent); change is None with one close."""
    latest = closes[-1]
    if len(closes) < 2:
        return latest, None, None
    change = latest - closes[-2]
    return latest, change, change / closes[-2] * 100


def period_return_pct(closes: list[float]) -> float | None:
    """(last close / first close - 1) x 100, or None with fewer than two closes."""
    if len(closes) < 2:
        return None
    return (closes[-1] / closes[0] - 1) * 100


def sma(closes: list[float], window: int) -> float | None:
    """Simple moving average of the last `window` closes, or None if there are fewer closes."""
    if window < 1 or len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def price_vs_pct(price: float, reference: float) -> float:
    """How far the price is above (positive) or below (negative) a reference, in percent."""
    return (price / reference - 1) * 100


def daily_returns(closes: list[float]) -> list[float]:
    """Simple day-to-day returns as fractions (0.10 = +10%)."""
    return [b / a - 1 for a, b in zip(closes, closes[1:])]


def annualised_volatility_pct(closes: list[float], trading_days: int) -> float | None:
    """Sample standard deviation of daily returns x sqrt(trading_days) x 100; None with fewer than 3 closes."""
    returns = daily_returns(closes)
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * math.sqrt(trading_days) * 100


def max_drawdown_pct(closes: list[float]) -> float | None:
    """Largest fall from a running peak, in percent (0 or negative); None with fewer than two closes.

    Example: closes 100, 110, 99, 108.9 peak at 110 and fall to 99, so the result is -10.
    """
    if len(closes) < 2:
        return None
    peak, worst = closes[0], 0.0
    for close in closes:
        peak = max(peak, close)
        worst = min(worst, close / peak - 1)
    return worst * 100


def range_52w(series: Series, price: float, min_days: int) -> Range52w | None:
    """52-week high/low from closes and the distance from each; None if the history spans fewer than min_days."""
    if not series or (series[-1][0] - series[0][0]).days < min_days:
        return None
    closes = [close for _, close in series]
    high, low = max(closes), min(closes)
    return Range52w(high=high, low=low, pct_below_high=(1 - price / high) * 100, pct_above_low=price_vs_pct(price, low))


def trend_label(short: float | None, long: float | None, period_return: float | None, cfg: dict[str, Any]) -> str | None:
    """Uptrend / Sideways / Downtrend, or None when an average or the return is missing.

    Uptrend: the short average is above the long one by more than trend.sma_gap_pct
    AND the period return is at least +trend.return_threshold_pct. Downtrend is
    the mirror image. Anything else is Sideways.
    """
    if short is None or long is None or period_return is None:
        return None
    gap = price_vs_pct(short, long)
    limits = cfg["trend"]
    if gap > limits["sma_gap_pct"] and period_return >= limits["return_threshold_pct"]:
        return "Uptrend"
    if gap < -limits["sma_gap_pct"] and period_return <= -limits["return_threshold_pct"]:
        return "Downtrend"
    return "Sideways"


def ticker_metrics(
    ticker: str, series: Series, currency: str, period: str, cfg: dict[str, Any],
    price_source: str = "", price_stale: bool = False, as_of: datetime | None = None,
) -> tuple[TickerMetrics | None, list[str]]:
    """Compose the metrics of ONE ticker; (None, warnings) when there are fewer than two usable closes."""
    if len(series) < 2:
        return None, [f"{ticker} was left out: not enough price history."]
    warnings: list[str] = []
    closes = [close for _, close in series]
    period_days = cfg["period_days"][period]
    window = slice_period(series, period_days)
    window_closes = [close for _, close in window]
    window_span = span_days(window)
    short_history = is_short_history(window_span, period_days, cfg["short_history_tolerance_pct"])
    if short_history:
        warnings.append(
            f"{ticker}: only {window_span} days of data for the requested {period}; "
            "return, volatility and drawdown cover that shorter span."
        )
    latest, change, change_pct = latest_and_day_change(closes)

    averages: dict[int, float] = {}
    for window in sorted(cfg["sma_windows"]):
        value = sma(closes, window)
        if value is None:
            warnings.append(f"{ticker}: the {window}-day average was left out (only {len(closes)} daily prices).")
        else:
            averages[window] = value
    short = averages.get(min(cfg["sma_windows"]))
    long = averages.get(max(cfg["sma_windows"]))

    period_return = period_return_pct(window_closes)
    volatility = annualised_volatility_pct(window_closes, cfg["trading_days"])
    if volatility is None:
        warnings.append(f"{ticker}: volatility was left out (too few prices in the selected period).")
    span = range_52w(series, latest, cfg["week52_min_days"])
    if span is None:
        days = (series[-1][0] - series[0][0]).days
        warnings.append(f"{ticker}: the 52-week range was left out (the history covers only {days} days).")
    trend = trend_label(short, long, period_return, cfg)
    if trend is None:
        warnings.append(f"{ticker}: the trend label was left out because a moving average or the return is missing.")

    return TickerMetrics(
        ticker=ticker, currency=currency, period=period, span_days=window_span,
        period_is_short=short_history, latest_price=latest,
        day_change=change, day_change_pct=change_pct, period_return_pct=period_return,
        averages=[AverageMetric(window=w, value=v, price_vs_pct=price_vs_pct(latest, v)) for w, v in averages.items()],
        volatility_pct=volatility, max_drawdown_pct=max_drawdown_pct(window_closes),
        range_52w=span, trend=trend, price_source=price_source, price_stale=price_stale, as_of=as_of,
    ), warnings


def build_metrics(tickers: list[TickerMetrics], warnings: list[str], period: str) -> MarketMetrics:
    """Collect the tickers (input order kept) and add the stale/sample warning."""
    notes = list(warnings)
    not_live = [f"{t.ticker} ({t.price_source})" for t in tickers if t.price_stale]
    if not_live:
        notes.append("These prices are not live (stale or sample data): " + ", ".join(not_live) + ".")
    dates = [t.as_of for t in tickers if t.as_of is not None]
    return MarketMetrics(
        period=period, tickers=tickers, warnings=notes,
        as_of=min(dates) if dates else None, uses_stale_or_sample=bool(not_live),
    )


def history_points(ticker: str, series: Series, days: int) -> list[dict[str, Any]]:
    """The selected period's closes as flat rows (ticker, date, close) for the UI chart."""
    return [{"ticker": ticker, "date": day.isoformat(), "close": close} for day, close in slice_period(series, days)]


def comparison_rows(metrics: MarketMetrics) -> list[dict[str, Any]]:
    """One table row per ticker in input order: no ranking, no totals, each in its own currency."""
    return [
        {
            "ticker": t.ticker, "currency": t.currency, "period_return_pct": t.period_return_pct,
            "volatility_pct": t.volatility_pct, "max_drawdown_pct": t.max_drawdown_pct, "trend": t.trend,
        }
        for t in metrics.tickers
    ]


def _pct(x: float) -> str:
    return f"{x:.1f}%"


def price_text(t: TickerMetrics, cfg: dict[str, Any]) -> str:
    """The latest price as one sentence part the summary copies: ticker, currency, price and where/when it came from.

    Example: "AAPL last price USD 270.12 (live)"; a price that is not live names its source and time, for example
    "AAPL last price USD 270.12 (stale cache, as of 2026-10-30 09:15 UTC)". Times are shown in UTC: the providers stamp
    UTC-aware times, and a naive time is taken as UTC. Wording comes from market_analysis.price_*.
    """
    if t.price_source in ("", "live"):
        freshness = cfg["price_freshness_live"]
    elif t.as_of is None:
        freshness = t.price_source
    else:
        as_of = t.as_of.astimezone(timezone.utc) if t.as_of.tzinfo else t.as_of
        freshness = cfg["price_freshness_other"].format(source=t.price_source, as_of=as_of.strftime("%Y-%m-%d %H:%M"))
    return cfg["price_text"].format(ticker=t.ticker, currency=t.currency, price=f"{t.latest_price:,.2f}", freshness=freshness)


def _ticker_summary(t: TickerMetrics, cfg: dict[str, Any]) -> dict[str, Any]:
    """One ticker with every number pre-formatted; label numbers (20-day, 52-week) sit inside the values."""
    if t.day_change is None or t.day_change_pct is None:
        day = NOT_AVAILABLE
    else:
        day = f"{t.day_change:+,.2f} ({t.day_change_pct:+.1f}%)"
    averages = [
        f"{a.window}-day average {a.value:,.2f}; the price is {_pct(abs(a.price_vs_pct))} "
        f"{'above' if a.price_vs_pct >= 0 else 'below'} it"
        for a in t.averages
    ]
    if t.range_52w is None:
        span = NOT_AVAILABLE
    else:
        r = t.range_52w
        span = (f"52-week high {r.high:,.2f} (the price is {_pct(r.pct_below_high)} below it); "
                f"52-week low {r.low:,.2f} (the price is {_pct(r.pct_above_low)} above it)")
    if t.max_drawdown_pct is None:
        drawdown = NOT_AVAILABLE
    elif t.max_drawdown_pct == 0:
        drawdown = "no fall from an earlier high during the period"
    else:
        drawdown = f"{_pct(abs(t.max_drawdown_pct))} fall from a peak"
    covered = {"period_covered": f"only {t.span_days} days of data (shorter than the requested period)"} if t.period_is_short else {}
    return {
        "ticker": t.ticker, "currency": t.currency, "latest_price": f"{t.latest_price:,.2f}", "latest_price_text": price_text(t, cfg),
        **covered,
        "day_change": day,
        "return_over_period": NOT_AVAILABLE if t.period_return_pct is None else f"{t.period_return_pct:+.1f}%",
        "moving_averages": averages or [NOT_AVAILABLE],
        "annualised_volatility": NOT_AVAILABLE if t.volatility_pct is None else _pct(t.volatility_pct),
        "max_drawdown_over_period": drawdown,
        "range_52_week": span,
        "trend": t.trend or NOT_AVAILABLE,
        "price_is_stale_or_sample": t.price_stale,
    }


def summary_input(metrics: MarketMetrics, cfg: dict[str, Any]) -> dict[str, Any]:
    """The only thing the LLM sees: the computed metrics with every number pre-formatted as text."""
    return {
        "analysis_period": cfg["period_labels"][metrics.period],
        "tickers": [_ticker_summary(t, cfg) for t in metrics.tickers],
        "warnings": metrics.warnings,
    }
