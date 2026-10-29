"""Immutable data models for the Goal Planning agent (architecture section 2.4).

Design pattern: VALUE OBJECT. Every model is frozen, so once a ScenarioResult is
built nothing can change it by accident. Example: build_plan() returns a GoalPlan
and the agent only reads it to build the payload.

Amounts are in one currency (config `goal_planning.currency`) and are never converted.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    """Base: immutable and strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class GoalInputs(_Frozen):
    """A validated goal request."""

    goal_amount: float                 # positive; today's money when inflation_adjust is on
    horizon_years: int
    current_savings: float             # 0 or more
    monthly_contribution: float        # 0 or more
    risk_profile: str
    annual_step_up_pct: float = 0.0
    inflation_adjust: bool = False


class ScenarioResult(_Frozen):
    """The outcome of ONE return scenario (conservative, expected or optimistic)."""

    name: str
    annual_return_pct: float
    final_value: float                           # at the end of the horizon
    final_value_todays_money: float | None       # only when inflation_adjust is on
    shortfall: float                             # target - final_value; negative = surplus
    goal_met: bool
    required_monthly_contribution: float         # starting monthly amount that reaches the target
    required_label: str                          # how to describe that amount (plain, step-up or goal already met)
    months_to_goal: int | None                   # at the current contribution; None = not reached
    months_note: str | None = None


class GoalPlan(_Frozen):
    """Everything the agent computed; the LLM only ever sees a formatted copy."""

    inputs: GoalInputs
    target_amount: float                         # the goal in future money (= goal_amount without inflation)
    scenarios: list[ScenarioResult]
    projection_rows: list[dict[str, float | int | str | None]]   # flat rows for a line chart
