"""Immutable data models for the Market agent (architecture section 2.3).

Design pattern: VALUE OBJECT. Every model is frozen, so once a TickerMetrics is
built nothing can change it by accident. Example: build_metrics() collects the
finished TickerMetrics of every ticker and never edits them afterwards.

Every price is in the ticker's own currency. Nothing here adds values of
different tickers together.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    """Base: immutable and strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AverageMetric(_Frozen):
    """A simple moving average and where the latest price sits against it."""

    window: int                    # trading days
    value: float                   # in the ticker's currency
    price_vs_pct: float            # (latest price / average - 1) x 100


class Range52w(_Frozen):
    """52-week high and low (from closing prices) and the distance from each."""

    high: float
    low: float
    pct_below_high: float          # (1 - price / high) x 100, 0 at the high
    pct_above_low: float           # (price / low - 1) x 100, 0 at the low


class TickerMetrics(_Frozen):
    """All metrics for ONE ticker. A metric that cannot be computed is None (never guessed)."""

    ticker: str
    currency: str
    period: str                                # requested period, e.g. "6mo"
    span_days: int = 0                         # calendar days the period slice really covers
    period_is_short: bool = False              # history covers noticeably less than the requested period
    latest_price: float
    day_change: float | None
    day_change_pct: float | None
    period_return_pct: float | None
    averages: list[AverageMetric]              # only the windows with enough data
    volatility_pct: float | None               # annualised
    max_drawdown_pct: float | None             # 0 or negative
    range_52w: Range52w | None
    trend: str | None                          # Uptrend | Sideways | Downtrend
    price_source: str = ""
    price_stale: bool = False                  # stale cache or sample data
    as_of: datetime | None = None


class MarketMetrics(_Frozen):
    """Everything the agent computed; the LLM only ever sees a formatted copy of this."""

    period: str
    tickers: list[TickerMetrics]
    warnings: list[str]
    as_of: datetime | None = None              # oldest price time among the tickers
    uses_stale_or_sample: bool = False
