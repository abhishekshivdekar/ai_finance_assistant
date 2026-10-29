"""Read and validate the Goal Planning request.

Design pattern: GUARD CLAUSES. Each field is checked on its own and every problem
is collected, so the user sees all of them at once. Example: a negative
current_savings and an unknown risk profile give two messages in one needs_input.

Input: state["goal_params"] = {"goal_amount": 1_000_000, "horizon_years": 15,
"current_savings": 50_000, "monthly_contribution": 5_000, "risk_profile": "medium",
"annual_step_up_pct": 5, "inflation_adjust": True}. All limits and texts come from config.
"""
from __future__ import annotations

import math
from typing import Any

from src.core.goal_models import GoalInputs


def _number(params: dict[str, Any], key: str, cfg: dict[str, Any], errors: list[str], default: float | None = None) -> float | None:
    """A finite number within max_amount, or None (an error is added). A missing key uses `default` if given."""
    msgs = cfg["errors"]
    value = params.get(key)
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append(msgs["not_a_number"].format(field=key))
        return None
    if value > cfg["max_amount"]:
        errors.append(msgs["too_large"].format(field=key, limit=f"{cfg['max_amount']:,}"))
        return None
    return float(value)


def parse_goal_params(params: Any, cfg: dict[str, Any]) -> tuple[GoalInputs | None, list[str]]:
    """(inputs, errors). inputs is None when any error was found; errors are ready-to-show messages."""
    params = params if isinstance(params, dict) else {}
    msgs = cfg["errors"]
    errors: list[str] = []

    goal = _number(params, "goal_amount", cfg, errors)
    if goal is not None and goal <= 0:
        errors.append(msgs["must_be_positive"].format(field="goal_amount"))
    savings = _number(params, "current_savings", cfg, errors, default=0.0)
    contribution = _number(params, "monthly_contribution", cfg, errors, default=0.0)
    for key, value in (("current_savings", savings), ("monthly_contribution", contribution)):
        if value is not None and value < 0:
            errors.append(msgs["must_not_be_negative"].format(field=key))

    years = params.get("horizon_years")
    low, high = cfg["min_horizon_years"], cfg["max_horizon_years"]
    if isinstance(years, bool) or not isinstance(years, (int, float)) or not math.isfinite(years):
        errors.append(msgs["not_a_number"].format(field="horizon_years"))
        years = None
    elif years != int(years):
        errors.append(msgs["horizon_whole"])
        years = None
    elif not low <= years <= high:
        errors.append(msgs["horizon_range"].format(low=low, high=high))
        years = None

    profile = params.get("risk_profile")
    if profile not in cfg["risk_profiles"]:
        errors.append(msgs["risk_profile"].format(options=", ".join(cfg["risk_profiles"])))

    step_up = _number(params, "annual_step_up_pct", cfg, errors, default=0.0)
    if step_up is not None and not 0 <= step_up <= cfg["max_step_up_pct"]:
        errors.append(msgs["step_up_range"].format(limit=cfg["max_step_up_pct"]))

    adjust = params.get("inflation_adjust")
    if adjust is None:
        adjust = cfg["inflation_adjust_default"]
    elif not isinstance(adjust, bool):
        errors.append(msgs["inflation_flag"])

    if errors:
        return None, errors
    return GoalInputs(
        goal_amount=goal, horizon_years=int(years), current_savings=savings, monthly_contribution=contribution,
        risk_profile=profile, annual_step_up_pct=step_up, inflation_adjust=adjust,
    ), []
