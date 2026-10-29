"""Immutable data models for the Portfolio agent (architecture section 2.2).

Design pattern: VALUE OBJECT. Every model is frozen, so once a Position or a
CurrencyMetrics is built nothing can change it by accident. Example: the agent
computes weights once and builds new Positions with model_copy(update=...)
instead of editing the valued ones in place.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.market.provider import validate_symbol

InstrumentType = Literal["stock", "etf", "fund", "bond", "cash", "other"]   # what the holding IS
AssetClass = Literal["equity", "debt", "gold", "cash", "other"]            # what it behaves like (drives risk)


class _Frozen(BaseModel):
    """Base: immutable and strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Holding(_Frozen):
    """One line of the user's portfolio. The currency comes from the quote, not from here.

    instrument_type says what the holding is; asset_class says what it behaves
    like. Allocation, equity share, risk and the single-asset-class penalty use
    asset_class only. Example: NIFTYBEES.NS is instrument_type "etf" with
    asset_class "equity", so it counts as equity exposure like a stock does.
    """

    ticker: str
    quantity: float = Field(gt=0, allow_inf_nan=False)
    avg_cost: float | None = Field(default=None, gt=0, allow_inf_nan=False)   # per unit, in the quote's currency
    instrument_type: InstrumentType = "stock"
    asset_class: AssetClass = "equity"
    expense_ratio_pct: float | None = Field(default=None, ge=0, allow_inf_nan=False)   # annual fund cost in percent

    @field_validator("ticker")
    @classmethod
    def _valid_ticker(cls, value: str) -> str:
        """Normalise with the market layer's validator (US, .NS and index tickers)."""
        return validate_symbol(value)


class Position(_Frozen):
    """A holding valued at a quote. weight is the share of its own currency's total."""

    ticker: str
    instrument_type: InstrumentType
    asset_class: AssetClass
    quantity: float
    avg_cost: float | None
    expense_ratio_pct: float | None
    currency: str
    price: float
    value: float
    weight: float = 0.0
    pnl: float | None = None        # unrealised, only when avg_cost was given
    pnl_pct: float | None = None
    price_source: str = ""
    price_stale: bool = False       # stale cache or sample data
    as_of: datetime | None = None


class CurrencyMetrics(_Frozen):
    """All metrics for ONE currency. Currencies are never added together."""

    currency: str
    total_value: float
    position_count: int
    allocation_by_asset_class: dict[str, float]   # share of total value, sums to 1
    largest_ticker: str
    largest_weight: float
    top3_weight: float
    hhi: float
    diversification_score: int                    # 0 to 100
    diversification_label: str
    equity_share: float
    risk_label: str                               # Low | Moderate | High
    expense_ratio_weighted_pct: float | None      # None when no holding provides one
    expense_ratio_holdings: int
    expense_ratio_total: int
    expense_ratio_value_share: float              # share of value (0-1) held in holdings that have a ratio
    expense_ratio_coverage: str                   # "covers 47% of value, 1 of 4 holdings" or "not provided"


class PortfolioMetrics(_Frozen):
    """Everything the agent computed; the LLM only ever sees a formatted copy of this."""

    positions: list[Position]
    currencies: dict[str, CurrencyMetrics]
    warnings: list[str]
    as_of: datetime | None = None                 # oldest price time among the positions
    uses_stale_or_sample: bool = False
