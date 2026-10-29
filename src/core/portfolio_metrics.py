"""Pure portfolio metric functions: no I/O, no LLM, no hidden state.

Design pattern: PURE FUNCTIONS (each metric is its own small function).
Example: risk_label(0.55, 0.20, cfg) can be tested with plain numbers, and the
agent just composes position_weights -> hhi -> diversification_score.

Every threshold comes from the `portfolio:` config block (passed in as `cfg`).
Values are illustrative and are never mixed across currencies: build_metrics()
groups positions by currency first and computes each group on its own.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from src.core.portfolio_models import CurrencyMetrics, PortfolioMetrics, Position
from src.utils.config import get_config

RISK_LEVELS = ["Low", "Moderate", "High"]


def position_weights(values: list[float]) -> list[float]:
    """Each value's share of the total (sums to 1); all zeros if the total is not positive."""
    total = sum(values)
    return [v / total for v in values] if total > 0 else [0.0 for _ in values]


def unrealised_pnl(price: float, quantity: float, avg_cost: float | None) -> tuple[float | None, float | None]:
    """(P&L, P&L %) against avg_cost, or (None, None) when no cost was given."""
    if avg_cost is None:
        return None, None
    return (price - avg_cost) * quantity, (price / avg_cost - 1) * 100


def allocation_by_asset_class(positions: list[Position]) -> dict[str, float]:
    """Share of total value per asset_class (positions are in one currency).

    Only asset_class counts here; instrument_type (stock, etf, ...) is ignored.
    """
    total = sum(p.value for p in positions)
    shares: dict[str, float] = defaultdict(float)
    for p in positions:
        shares[p.asset_class] += p.value / total if total > 0 else 0.0
    return dict(shares)


def concentration(weights: list[float]) -> tuple[float, float]:
    """(largest weight, sum of the three largest weights)."""
    ordered = sorted(weights, reverse=True)
    return (ordered[0] if ordered else 0.0), sum(ordered[:3])


def hhi(weights: list[float]) -> float:
    """Herfindahl-Hirschman index: sum of squared weights (1 = one holding, 1/N = equal split)."""
    return sum(w * w for w in weights)


def diversification_score(weights: list[float], single_asset_class: bool, cfg: dict[str, Any]) -> int:
    """Diversification score from 0 to 100 (an educational indicator, not a rating).

    score = clamp(100 x (1 - HHI) / (1 - 1/N_full), 0, 100), then minus
    single_class_penalty when every position has the same asset_class (instrument
    type is ignored), floored at 0. N_full = portfolio.full_diversification_holdings
    is the number of equal positions that counts as fully spread. Example with
    N_full = 20 and a penalty of 15: one holding scores 0; five equal positions
    (HHI 0.2) score round(100 x 0.8 / 0.95) = 84, and 69 if they are all equity
    asset class; four equal positions, one of them an ETF, all equity, score
    round(100 x 0.75 / 0.95) = 79 - 15 = 64 (Moderate, not Well diversified).
    """
    n_full = max(int(cfg["full_diversification_holdings"]), 2)
    raw = (1 - hhi(weights)) / (1 - 1 / n_full)
    score = round(100 * min(max(raw, 0.0), 1.0))
    if single_asset_class:
        score -= int(cfg["single_class_penalty"])
    return max(score, 0)


def diversification_label(score: int, cfg: dict[str, Any]) -> str:
    """Concentrated / Moderate / Well diversified, from portfolio.diversification_bands."""
    bands = cfg["diversification_bands"]
    if score < bands["concentrated_below"]:
        return "Concentrated"
    if score >= bands["diversified_from"]:
        return "Well diversified"
    return "Moderate"


def equity_share(positions: list[Position], cfg: dict[str, Any]) -> float:
    """Share of value whose asset_class is listed in portfolio.equity_like_classes."""
    like = set(cfg["equity_like_classes"])
    total = sum(p.value for p in positions)
    return sum(p.value for p in positions if p.asset_class in like) / total if total > 0 else 0.0


def _level(value: float, moderate_from: float, high_from: float) -> int:
    """0 = low, 1 = moderate, 2 = high."""
    return 2 if value >= high_from else 1 if value >= moderate_from else 0


def risk_label(equity: float, largest_weight: float, cfg: dict[str, Any]) -> str:
    """Low / Moderate / High: the higher of the equity-share level and the concentration level."""
    risk = cfg["risk"]
    level = max(
        _level(equity, risk["equity_moderate_from"], risk["equity_high_from"]),
        _level(largest_weight, risk["concentration_moderate_from"], risk["concentration_high_from"]),
    )
    return RISK_LEVELS[level]


def weighted_expense_ratio(positions: list[Position]) -> tuple[float | None, int, int]:
    """(weighted average expense ratio in percent, holdings that have one, all holdings).

    Weights are re-normalised to the holdings that provide a ratio; the result is
    None when none does. Coverage is reported by the caller as "N of M holdings".
    """
    with_ratio = [p for p in positions if p.expense_ratio_pct is not None]
    if not with_ratio:
        return None, 0, len(positions)
    weights = position_weights([p.value for p in with_ratio])
    average = sum(w * (p.expense_ratio_pct or 0.0) for w, p in zip(weights, with_ratio))
    return average, len(with_ratio), len(positions)


def expense_ratio_value_share(positions: list[Position]) -> float:
    """Share of total value (0-1) held in positions that provide an expense ratio."""
    total = sum(p.value for p in positions)
    covered = sum(p.value for p in positions if p.expense_ratio_pct is not None)
    return covered / total if total > 0 else 0.0


def expense_ratio_coverage(value_share: float, with_ratio: int, total: int) -> str:
    """"covers 47% of value, 1 of 4 holdings", or "not provided" when no holding has a ratio."""
    if with_ratio == 0:
        return "not provided"
    return f"covers {value_share * 100:.0f}% of value, {with_ratio} of {total} holdings"


def currency_metrics(currency: str, positions: list[Position], cfg: dict[str, Any]) -> CurrencyMetrics:
    """Compose the metrics for the positions of ONE currency."""
    weights = position_weights([p.value for p in positions])
    largest, top3 = concentration(weights)
    largest_ticker = positions[weights.index(largest)].ticker if positions else ""
    score = diversification_score(weights, len({p.asset_class for p in positions}) == 1, cfg)
    equity = equity_share(positions, cfg)
    average, with_ratio, total = weighted_expense_ratio(positions)
    value_share = expense_ratio_value_share(positions)
    return CurrencyMetrics(
        currency=currency,
        total_value=sum(p.value for p in positions),
        position_count=len(positions),
        allocation_by_asset_class=allocation_by_asset_class(positions),
        largest_ticker=largest_ticker,
        largest_weight=largest,
        top3_weight=top3,
        hhi=hhi(weights),
        diversification_score=score,
        diversification_label=diversification_label(score, cfg),
        equity_share=equity,
        risk_label=risk_label(equity, largest, cfg),
        expense_ratio_weighted_pct=average,
        expense_ratio_holdings=with_ratio,
        expense_ratio_total=total,
        expense_ratio_value_share=value_share,
        expense_ratio_coverage=expense_ratio_coverage(value_share, with_ratio, total),
    )


def build_metrics(
    positions: list[Position], warnings: list[str], cfg: dict[str, Any] | None = None
) -> PortfolioMetrics:
    """Group positions by currency, fill in weights and compute each currency on its own."""
    cfg = cfg if cfg is not None else get_config()["portfolio"]
    by_currency: dict[str, list[Position]] = defaultdict(list)
    for p in positions:
        by_currency[p.currency].append(p)

    weighted: list[Position] = []
    currencies: dict[str, CurrencyMetrics] = {}
    notes = list(warnings)
    for currency in sorted(by_currency):
        group = by_currency[currency]
        weights = position_weights([p.value for p in group])
        weighted.extend(p.model_copy(update={"weight": w}) for p, w in zip(group, weights))
        metrics = currency_metrics(currency, group, cfg)
        currencies[currency] = metrics
        if metrics.position_count == 1:
            notes.append(f"The {currency} portfolio has a single holding ({metrics.largest_ticker}).")
        elif metrics.largest_weight >= cfg["concentration_warning"]:
            notes.append(
                f"{metrics.largest_ticker} is {metrics.largest_weight * 100:.1f}% of the {currency} portfolio."
            )

    dates = [p.as_of for p in positions if p.as_of is not None]
    return PortfolioMetrics(
        positions=weighted,
        currencies=currencies,
        warnings=notes,
        as_of=min(dates) if dates else None,
        uses_stale_or_sample=any(p.price_stale for p in positions),
    )


def summary_input(metrics: PortfolioMetrics) -> dict[str, Any]:
    """The only thing the LLM sees: the computed metrics with every number pre-formatted.

    Numbers are strings ("42.0%", "1,234.50") so the model copies them instead of
    calculating anything.
    """
    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    currencies = []
    for cm in metrics.currencies.values():
        provided = cm.expense_ratio_weighted_pct is not None
        ratio = f"{cm.expense_ratio_weighted_pct:.2f}%" if provided else "not provided"
        currencies.append({
            "currency": cm.currency,
            "total_value": f"{cm.total_value:,.2f}",
            "holdings": cm.position_count,
            "largest_position": f"{cm.largest_ticker} {pct(cm.largest_weight)}",
            "top_three_share": pct(cm.top3_weight),
            "allocation_by_asset_class": {k: pct(v) for k, v in cm.allocation_by_asset_class.items()},
            "diversification_score": f"{cm.diversification_score} out of 100",
            "diversification_label": cm.diversification_label,
            "equity_share": pct(cm.equity_share),
            "risk_label": cm.risk_label,
            "weighted_expense_ratio": ratio,
            # coverage is only given when a ratio exists ("not provided" has no coverage fraction)
            **({"expense_ratio_coverage": cm.expense_ratio_coverage} if provided else {}),
        })
    positions = [
        {
            "ticker": p.ticker, "currency": p.currency,
            "instrument_type": p.instrument_type, "asset_class": p.asset_class,
            "weight": pct(p.weight),
            "unrealised_pnl_pct": None if p.pnl_pct is None else f"{p.pnl_pct:+.1f}%",
            "price_is_stale_or_sample": p.price_stale,
        }
        for p in metrics.positions
    ]
    return {"portfolios_by_currency": currencies, "positions": positions, "warnings": metrics.warnings}


_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    """Every number in `text` as a float ("1,234.50" -> 1234.5, "42%" -> 42.0)."""
    return {float(m.replace(",", "")) for m in _NUMBER.findall(text)}


def collect_numbers(value: Any) -> set[float]:
    """Numbers found in the values (not the keys) of a nested dict/list."""
    if isinstance(value, dict):
        return set().union(*(collect_numbers(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(collect_numbers(v) for v in value)) if value else set()
    if isinstance(value, bool) or value is None:
        return set()
    return numbers_in(str(value))
