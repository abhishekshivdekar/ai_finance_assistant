"""Turn holdings into priced positions using the market provider.

Design pattern: DEPENDENCY INJECTION. value_holdings() receives the provider as
an argument. Example: the agent passes the shared CachedProvider (cache, retry,
stale and sample fallback), a test passes a fake that returns fixed quotes.

Rules: a holding whose price is unavailable is left out of the totals and
reported in the warnings (the rest of the portfolio is still valued); prices
that are stale or sample data are flagged; currencies are never converted or
added together (no FX rate is ever invented).
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import ValidationError

from src.core.portfolio_metrics import unrealised_pnl
from src.core.portfolio_models import Holding, Position
from src.data.market.provider import MarketDataProvider
from src.utils.logger import get_logger

log = get_logger(__name__)

NOT_LIVE_SOURCES = {"stale cache", "sample data"}   # MarketResult.source values that are not live


def parse_holdings(rows: list[Any], max_holdings: int) -> tuple[list[Holding], list[str]]:
    """Validate raw rows (dicts or Holdings). Bad rows, duplicates and extras become warnings."""
    holdings: list[Holding] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        try:
            holding = row if isinstance(row, Holding) else Holding.model_validate(row)
        except ValidationError as exc:
            reason = exc.errors()[0]["msg"]
            warnings.append(f"Row {number} was skipped: {reason}.")
            continue
        if holding.ticker in seen:
            warnings.append(f"{holding.ticker} appears more than once; only the first row is used.")
            continue
        if len(holdings) >= max_holdings:
            warnings.append(f"Only the first {max_holdings} holdings are analysed; {holding.ticker} was left out.")
            continue
        seen.add(holding.ticker)
        holdings.append(holding)
    return holdings, warnings


def _usable_price(data: Any) -> float | None:
    """The quote's price if it is a positive finite number, else None."""
    price = data.get("price") if isinstance(data, dict) else None
    if isinstance(price, (int, float)) and not isinstance(price, bool) and math.isfinite(price) and price > 0:
        return float(price)
    return None


def value_holdings(
    holdings: list[Holding], provider: MarketDataProvider
) -> tuple[list[Position], list[str]]:
    """Price every holding; return (positions, warnings). Never raises for a bad quote."""
    positions: list[Position] = []
    warnings: list[str] = []
    not_live: list[str] = []
    for h in holdings:
        result = provider.get_quote(h.ticker)
        price = _usable_price(result.data)
        if price is None or not result.currency:
            log.warning("No usable quote for %s: %s", h.ticker, result.error)
            warnings.append(f"{h.ticker} was left out: no price is available right now.")
            continue
        stale = bool(result.is_stale or result.source in NOT_LIVE_SOURCES)
        if stale:
            not_live.append(f"{h.ticker} ({result.source})")
        pnl, pnl_pct = unrealised_pnl(price, h.quantity, h.avg_cost)
        positions.append(
            Position(
                ticker=h.ticker, asset_class=h.asset_class, quantity=h.quantity,
                avg_cost=h.avg_cost, expense_ratio_pct=h.expense_ratio_pct,
                currency=result.currency, price=price, value=price * h.quantity,
                pnl=pnl, pnl_pct=pnl_pct, price_source=result.source,
                price_stale=stale, as_of=result.as_of,
            )
        )
    if not_live:
        warnings.append("These prices are not live (stale or sample data): " + ", ".join(not_live) + ".")
    return positions, warnings
