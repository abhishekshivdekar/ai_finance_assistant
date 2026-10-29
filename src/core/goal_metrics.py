"""Pure goal-planning math: no I/O, no LLM, no hidden state.

Design pattern: PURE FUNCTIONS. Example: project(0, 1000, 0, 0, 12)[-1] is 12000.0,
which a test can check by hand; build_plan() only composes these small functions.

Conventions (also written into the payload's assumptions block):
- annual rates are annual EFFECTIVE rates, converted to a monthly rate:
      monthly = (1 + annual_pct / 100) ** (1 / 12) - 1
- deposits are made at the END of each month:
      balance[k] = balance[k-1] * (1 + monthly) + contribution[k]
- a yearly step-up raises the contribution every 12 months (from month 13):
      contribution[k] = start * (1 + step_up_pct / 100) ** ((k - 1) // 12)
- inflation: future_money = x * (1 + i) ** years;  todays_money = x / (1 + i) ** years.
  When inflation_adjust is on, the goal is read as today's money. At the horizon the
  TARGET is the goal in future money; shortfall and required contribution use it.
  Months to goal are different: the goal keeps rising, so month k counts as reached
  when balance_k >= goal * (1 + i) ** (k / 12). This equals the horizon target at
  month = horizon * 12. Only the FIRST crossing is reported: when the return is below
  inflation the balance can cross and later fall back below the rising price, so such
  scenarios are flagged (`below_inflation`).
All rates and limits come from the `goal_planning:` config block; results are
illustrative projections, never predictions.
"""
from __future__ import annotations

from typing import Any

from src.core.goal_models import GoalInputs, GoalPlan, ScenarioResult

MONTHS_PER_YEAR = 12
Row = dict[str, float | int | str | None]


def monthly_rate(annual_pct: float) -> float:
    """(1 + annual/100) ** (1/12) - 1; 0 for a 0% return."""
    return (1 + annual_pct / 100) ** (1 / MONTHS_PER_YEAR) - 1


def contribution_in_month(start: float, step_up_pct: float, month: int) -> float:
    """The contribution of month `month` (1-based): start * (1 + step_up) ** ((month - 1) // 12)."""
    return start * (1 + step_up_pct / 100) ** ((month - 1) // MONTHS_PER_YEAR)


def project(savings: float, contribution: float, step_up_pct: float, annual_pct: float, months: int) -> list[float]:
    """Balance after each month 1..months (month 0 is `savings`), month by month so step-up is exact."""
    rate = monthly_rate(annual_pct)
    balance, balances = savings, []
    for month in range(1, months + 1):
        balance = balance * (1 + rate) + contribution_in_month(contribution, step_up_pct, month)
        balances.append(balance)
    return balances


def final_value(savings: float, contribution: float, step_up_pct: float, annual_pct: float, months: int) -> float:
    """The balance at the end of `months`."""
    return project(savings, contribution, step_up_pct, annual_pct, months)[-1]


def year_end_values(savings: float, balances: list[float]) -> list[float]:
    """[year 0, year 1, ...]: the starting savings, then the balance after every 12th month."""
    return [savings] + balances[MONTHS_PER_YEAR - 1::MONTHS_PER_YEAR]


def required_contribution(goal: float, savings: float, step_up_pct: float, annual_pct: float, months: int) -> float:
    """The starting monthly contribution that reaches `goal` after `months`; 0 if the savings alone get there.

    The final value is LINEAR in the starting contribution c:
        FV(c) = FV(0) + c * FV_unit    (FV_unit = final value of a contribution of 1, no savings)
    so  c = (goal - FV(0)) / FV_unit  is exact for any rate and step-up. With no
    step-up this is the textbook (goal - S(1+m)^n) * m / ((1+m)^n - 1); at 0% it is (goal - S) / n.
    """
    base = final_value(savings, 0.0, step_up_pct, annual_pct, months)
    if base >= goal:
        return 0.0
    return (goal - base) / final_value(0.0, 1.0, step_up_pct, annual_pct, months)


def months_to_goal(goal: float, savings: float, contribution: float, step_up_pct: float, annual_pct: float,
                   max_months: int, inflation_pct: float = 0.0) -> int | None:
    """The first month k whose balance reaches the goal (0 if `savings` already does); None if not within max_months.

    With inflation the goal is a price in today's money that keeps rising, so month k
    is compared with goal * (1 + inflation) ** (k / 12), not with a target fixed at the
    horizon. With inflation_pct 0 the goal is a fixed amount.
    """
    if savings >= goal:
        return 0
    for month, balance in enumerate(project(savings, contribution, step_up_pct, annual_pct, max_months), start=1):
        if balance >= future_money(goal, inflation_pct, month / MONTHS_PER_YEAR):
            return month
    return None


def future_money(amount: float, inflation_pct: float, years: float) -> float:
    """amount * (1 + inflation) ** years: what today's money is worth in future prices."""
    return amount * (1 + inflation_pct / 100) ** years


def todays_money(amount: float, inflation_pct: float, years: float) -> float:
    """amount / (1 + inflation) ** years: what a future amount is worth in today's prices."""
    return amount / (1 + inflation_pct / 100) ** years


def _months_note(reach: int | None, below_inflation: bool, cfg: dict[str, Any]) -> str | None:
    """A note for 'not reached', 'already met' and/or 'below inflation'; None for a normal result."""
    notes = []
    if reach is None:
        notes.append(cfg["not_reached_note"].format(months=cfg["max_months"]))
    elif reach == 0:
        notes.append(cfg["already_met_note"])
    if below_inflation:
        notes.append(cfg["below_inflation_months_note"])
    return " ".join(notes) or None


def required_label(goal_met: bool, step_up_pct: float, cfg: dict[str, Any]) -> str:
    """How to describe the required monthly amount: plain, rising with a step-up, or the minimum when the goal is met."""
    labels = cfg["required_labels"]
    if goal_met:
        key = "goal_met_step_up" if step_up_pct > 0 else "goal_met"
    else:
        key = "step_up" if step_up_pct > 0 else "plain"
    return labels[key].format(pct=f"{step_up_pct:g}")


def _scenario(name: str, annual_pct: float, inputs: GoalInputs, target: float, cfg: dict[str, Any]) -> tuple[ScenarioResult, list[float]]:
    """ScenarioResult plus the year-end values (year 0..horizon) for the chart."""
    months = inputs.horizon_years * MONTHS_PER_YEAR
    args = (inputs.current_savings, inputs.monthly_contribution, inputs.annual_step_up_pct, annual_pct)
    balances = project(*args, months)
    final = balances[-1]
    inflation = cfg["assumptions"]["inflation"]["annual_pct"]
    reach = months_to_goal(inputs.goal_amount, *args, cfg["max_months"], inflation if inputs.inflation_adjust else 0.0)
    below = inputs.inflation_adjust and annual_pct < inflation
    return ScenarioResult(
        name=name, annual_return_pct=annual_pct, final_value=final,
        final_value_todays_money=todays_money(final, inflation, inputs.horizon_years) if inputs.inflation_adjust else None,
        shortfall=target - final, goal_met=final >= target,
        required_monthly_contribution=required_contribution(
            target, inputs.current_savings, inputs.annual_step_up_pct, annual_pct, months),
        months_to_goal=reach, months_note=_months_note(reach, below, cfg), below_inflation=below,
        required_label=required_label(final >= target, inputs.annual_step_up_pct, cfg),
    ), year_end_values(inputs.current_savings, balances)


def build_plan(inputs: GoalInputs, cfg: dict[str, Any]) -> GoalPlan:
    """Compose every scenario for the validated inputs (the one function the agent calls)."""
    inflation = cfg["assumptions"]["inflation"]["annual_pct"]
    adjust = inputs.inflation_adjust
    target = future_money(inputs.goal_amount, inflation, inputs.horizon_years) if adjust else inputs.goal_amount
    returns = cfg["assumptions"]["returns_pct"][inputs.risk_profile]
    scenarios: list[ScenarioResult] = []
    rows: list[Row] = []
    for name in cfg["scenarios"]:
        result, yearly = _scenario(name, returns[name], inputs, target, cfg)
        scenarios.append(result)
        rows += [
            {"scenario": name, "year": year, "value": value,
             "value_todays_money": todays_money(value, inflation, year) if adjust else None}
            for year, value in enumerate(yearly)
        ]
    return GoalPlan(inputs=inputs, target_amount=target, scenarios=scenarios, projection_rows=rows)


def _money(x: float) -> str:
    """Western digit grouping with two decimals (the number guard cannot read lakh/crore grouping)."""
    return f"{x:,.2f}"


def years_months_text(months: int, cfg: dict[str, Any]) -> str:
    """"17 years 4 months" from a month count (units are dropped when zero, singular for 1)."""
    t = cfg["texts"]
    years, rest = divmod(months, MONTHS_PER_YEAR)
    parts = []
    if years:
        parts.append(t["years_one" if years == 1 else "years_many"].format(n=years))
    if rest:
        parts.append(t["months_one" if rest == 1 else "months_many"].format(n=rest))
    return " ".join(parts)


def surplus_or_shortfall_text(shortfall: float, cfg: dict[str, Any]) -> str:
    """"surplus of X" / "shortfall of X" (shortfall is target - final value; negative = surplus)."""
    t = cfg["texts"]
    if shortfall > 0:
        return t["shortfall"].format(amount=_money(shortfall))
    if shortfall < 0:
        return t["surplus"].format(amount=_money(-shortfall))
    return t["exact"]


def _time_text(s: ScenarioResult, horizon_months: int, cfg: dict[str, Any]) -> tuple[str, str]:
    """(time to reach the goal in years and months, how that compares with the horizon)."""
    t = cfg["texts"]
    if s.months_to_goal is None:
        return t["not_within_max"], t["not_within_max"]
    if s.months_to_goal == 0:
        return t["already_met"], t["already_met"]
    compare = t["within_horizon"] if s.months_to_goal <= horizon_months else t["after_horizon"]
    return years_months_text(s.months_to_goal, cfg), compare


def below_inflation_text(s: ScenarioResult, cfg: dict[str, Any]) -> str:
    """Plain sentence for a scenario whose return is below the inflation rate."""
    inflation = cfg["assumptions"]["inflation"]["annual_pct"]
    return cfg["texts"]["below_inflation"].format(rate=f"{s.annual_return_pct:.1f}%", inflation=f"{inflation:.1f}%")


def status_text(s: ScenarioResult, time_text: str, cfg: dict[str, Any]) -> str:
    """One sentence stating both facts: when the balance first covers the goal price, and the outcome at the horizon.

    The template is picked by (goal met at the horizon?, first-crossing month). None of
    the wording says "reached", so a crossing followed by a shortfall is not contradictory.
    """
    when = "now" if s.months_to_goal == 0 else "none" if s.months_to_goal is None else "later"
    key = f"status_{'met' if s.goal_met else 'short'}_{when}"
    return cfg["texts"][key].format(when=time_text, outcome=surplus_or_shortfall_text(s.shortfall, cfg))


def _scenario_summary(s: ScenarioResult, horizon_months: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """One scenario with every number pre-formatted as text."""
    time_text, compare = _time_text(s, horizon_months, cfg)
    item: dict[str, Any] = {
        "scenario": s.name, "assumed_annual_return": f"{s.annual_return_pct:.1f}%",
        "projected_value_at_end": _money(s.final_value),
        "surplus_or_shortfall_text": surplus_or_shortfall_text(s.shortfall, cfg),
        "monthly_amount": _money(s.required_monthly_contribution),
        "monthly_amount_meaning": s.required_label,
        "years_months_to_goal": time_text,
        "time_compared_with_horizon": compare,
        "status_text": status_text(s, time_text, cfg),
    }
    if s.below_inflation:
        item["below_inflation_text"] = below_inflation_text(s, cfg)
    if s.final_value_todays_money is not None:
        item["projected_value_at_end_in_todays_money"] = _money(s.final_value_todays_money)
    return item


def summary_input(plan: GoalPlan, cfg: dict[str, Any]) -> dict[str, Any]:
    """The only thing the LLM sees: the computed results with every number pre-formatted as text."""
    i, a = plan.inputs, cfg["assumptions"]
    t = cfg["texts"]
    inflation: Any = t["inflation_not_applied"]
    if i.inflation_adjust:
        inflation = {
            "goal_in_todays_money_text": t["goal_today"].format(amount=_money(i.goal_amount)),
            "goal_in_future_money_text": t["goal_inflated"].format(
                rate=f"{a['inflation']['annual_pct']:.1f}%", years=years_months_text(i.horizon_years * MONTHS_PER_YEAR, cfg),
                amount=_money(plan.target_amount)),
            "goal_price_at_end_of_horizon_text": t["goal_price_at_end"].format(amount=_money(plan.target_amount)),
            "rising_goal_price_text": t["goal_price_rising"],
        }
    return {
        "currency": cfg["currency"], "risk_profile": i.risk_profile, "horizon": years_months_text(i.horizon_years * MONTHS_PER_YEAR, cfg),
        "current_savings": _money(i.current_savings), "monthly_contribution_now": _money(i.monthly_contribution),
        "yearly_step_up": f"{i.annual_step_up_pct:.1f}%",
        "goal_used_for_comparison": _money(plan.target_amount),
        "inflation": inflation,
        "assumptions": {"label": a["label"], "as_of": a["as_of"], "source": a["source"],
                        "notes": [a["deposit_timing_note"], a["step_up_note"], a["rate_note"]]},
        "scenarios": [_scenario_summary(s, i.horizon_years * MONTHS_PER_YEAR, cfg) for s in plan.scenarios],
    }
