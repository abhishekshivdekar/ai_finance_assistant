"""Read and validate the Tax calculator request.

Design pattern: GUARD CLAUSES. Each field is checked on its own and every problem is
collected, so the user sees all of them at once. Example: a negative expense and a
missing slab rate give two messages in one needs_input.

Input: state["tax_params"] = {"calculation": "capital_gains", "asset_type": "equity_fund",
"buy_value": 100000, "sell_value": 180000, "expenses": 500, "holding_months": 18,
"ltcg_exemption_already_used": 0, "slab_rate_pct": None} or {"calculation": "small_savings_growth",
"scheme": "ppf", "mode": "annual", "amount": 50000, "years": 15}. All limits and texts come from config.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from src.core.goal_inputs import is_number
from src.core.tax_models import CapitalGainsInputs, SmallSavingsInputs

CALCULATIONS = ("capital_gains", "small_savings_growth")
SAVINGS_MODES = ("annual", "one_time")


def _check_choice(value: Any, field: str, choices: list[str] | tuple[str, ...], cfg: dict[str, Any], errors: list[str]) -> str | None:
    """The value when it is one of the choices; otherwise an error is added and None returned."""
    if isinstance(value, str) and value in choices:
        return value
    errors.append(cfg["errors"]["unknown_choice"].format(field=field, choices=", ".join(choices)))
    return None


def _amount(params: dict[str, Any], key: str, cfg: dict[str, Any], errors: list[str], *, positive: bool = False,
            default: float | None = None, limit: float | None = None) -> float | None:
    """A number between zero (or just above it) and the limit, or None (an error is added).

    The limit is compared BEFORE float() so a huge int never overflows (see is_number).
    """
    msgs = cfg["errors"]
    value = params.get(key)
    if value is None and default is not None:
        return default
    if not is_number(value):
        errors.append(msgs["not_a_number"].format(field=key))
        return None
    limit = cfg["input_limits"]["max_amount"] if limit is None else limit
    if value > limit:
        errors.append(msgs["too_large"].format(field=key, limit=f"{limit:,}"))
        return None
    if value < 0 or (positive and value == 0):
        errors.append(msgs["must_be_positive" if positive else "must_not_be_negative"].format(field=key))
        return None
    return float(value)


def _whole(params: dict[str, Any], key: str, cfg: dict[str, Any], errors: list[str], limit: int, *, minimum: int = 0) -> int | None:
    """A whole number from `minimum` to `limit`, or None (an error is added)."""
    msgs = cfg["errors"]
    value = params.get(key)
    if not is_number(value):
        errors.append(msgs["not_a_number"].format(field=key))
        return None
    if isinstance(value, float) and not value.is_integer():
        errors.append(msgs["whole_number"].format(field=key))
        return None
    if value > limit:
        errors.append(msgs["too_large"].format(field=key, limit=f"{limit:,}"))
        return None
    if value < minimum:
        errors.append(msgs["must_be_positive" if minimum else "must_not_be_negative"].format(field=key))
        return None
    return int(value)


def slab_rate_needed(asset_type: str, holding_months: int, cfg: dict[str, Any], *, has_gain: bool = True) -> bool:
    """True when the gain is taxed at the user's slab rate: debt funds always, other assets when short-term.

    Without a gain (a loss or zero) no tax is computed, so no slab rate is needed whatever the asset.

    Listed equity and equity funds never need it (fixed rates from config). Long-term means MORE than
    the configured number of months.
    """
    if not has_gain or asset_type in cfg["equity_asset_types"]:
        return False
    if asset_type in cfg["slab_asset_types"]:
        return True
    return holding_months <= cfg["figures"]["holding_months_other"]["value"]


def parse_capital_gains(params: dict[str, Any], cfg: dict[str, Any]) -> tuple[CapitalGainsInputs | None, list[str]]:
    """(inputs, errors). inputs is None when any error was found; errors are ready-to-show messages."""
    errors: list[str] = []
    limits = cfg["input_limits"]
    asset = _check_choice(params.get("asset_type"), "asset_type", cfg["asset_types"], cfg, errors)
    buy = _amount(params, "buy_value", cfg, errors)
    sell = _amount(params, "sell_value", cfg, errors)
    expenses = _amount(params, "expenses", cfg, errors, default=0.0)
    used = _amount(params, "ltcg_exemption_already_used", cfg, errors, default=0.0)
    months = _whole(params, "holding_months", cfg, errors, limits["max_months"])
    slab: float | None = None
    has_gain = buy is None or sell is None or expenses is None or Decimal(str(sell)) - Decimal(str(buy)) - Decimal(str(expenses)) > 0
    if asset is not None and months is not None and slab_rate_needed(asset, months, cfg, has_gain=has_gain):
        raw = params.get("slab_rate_pct")
        if raw is None:
            errors.append(cfg["errors"]["slab_rate_needed"])
        else:
            slab = _amount(params, "slab_rate_pct", cfg, errors, limit=limits["max_pct"])
    if errors:
        return None, errors
    assert asset is not None and buy is not None and sell is not None and expenses is not None and used is not None and months is not None
    return CapitalGainsInputs(
        asset_type=asset, buy_value=buy, sell_value=sell, expenses=expenses, holding_months=months,
        ltcg_exemption_already_used=used, slab_rate_pct=slab,
    ), []


def parse_small_savings(params: dict[str, Any], cfg: dict[str, Any]) -> tuple[SmallSavingsInputs | None, list[str]]:
    """(inputs, errors) for a small-savings projection."""
    errors: list[str] = []
    scheme = _check_choice(params.get("scheme"), "scheme", list(cfg["schemes"]), cfg, errors)
    mode = _check_choice(params.get("mode"), "mode", SAVINGS_MODES, cfg, errors)
    amount = _amount(params, "amount", cfg, errors, positive=True)
    years = _whole(params, "years", cfg, errors, cfg["input_limits"]["max_years"], minimum=1)
    if errors:
        return None, errors
    assert scheme is not None and mode is not None and amount is not None and years is not None
    return SmallSavingsInputs(scheme=scheme, mode=mode, amount=amount, years=years), []
