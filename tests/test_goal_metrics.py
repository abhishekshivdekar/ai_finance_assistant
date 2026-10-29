import copy
import math

import pytest

from src.core import goal_metrics as gm
from src.core.goal_models import GoalInputs
from src.utils.config import get_config

CFG = get_config()["goal_planning"]


def zero_cfg():
    """Config with 0% returns and 0% inflation, so results can be checked by hand."""
    cfg = copy.deepcopy(CFG)
    for profile in cfg["assumptions"]["returns_pct"].values():
        for name in profile:
            profile[name] = 0.0
    cfg["assumptions"]["inflation"]["annual_pct"] = 0.0
    return cfg


def inputs(**changes):
    base = dict(goal_amount=12_000.0, horizon_years=1, current_savings=0.0, monthly_contribution=1_000.0,
                risk_profile="low")
    return GoalInputs(**{**base, **changes})


# ---- building blocks ----

def test_monthly_rate_compounds_back_to_the_annual_rate():
    assert gm.monthly_rate(0) == 0.0
    assert (1 + gm.monthly_rate(12)) ** 12 == pytest.approx(1.12)


def test_contribution_steps_up_every_12_months():
    got = [gm.contribution_in_month(100, 10, m) for m in (1, 12, 13, 24, 25)]
    assert got == pytest.approx([100, 100, 110, 110, 121])


def test_zero_return_projection_is_just_the_deposits():
    balances = gm.project(0, 1000, 0, 0, 12)
    assert balances[0] == 1000 and balances[-1] == 12000 and len(balances) == 12


def test_savings_alone_grow_by_the_annual_rate_after_a_year():
    assert gm.final_value(1000, 0, 0, 12, 12) == pytest.approx(1120)


def test_known_future_value_of_monthly_deposits():
    m = gm.monthly_rate(12)
    assert gm.final_value(0, 100, 0, 12, 12) == pytest.approx(100 * ((1 + m) ** 12 - 1) / m)


def test_step_up_is_exact_month_by_month():
    assert gm.final_value(0, 100, 10, 0, 24) == pytest.approx(12 * 100 + 12 * 110)


def test_year_end_values_start_with_the_savings():
    balances = gm.project(500, 100, 0, 0, 24)
    assert gm.year_end_values(500, balances) == [500, 1700, 2900]


# ---- required contribution ----

def test_required_contribution_at_zero_return_is_the_gap_divided_by_months():
    assert gm.required_contribution(12_000, 0, 0, 0, 12) == pytest.approx(1000)
    assert gm.required_contribution(12_000, 6_000, 0, 0, 12) == pytest.approx(500)


def test_required_contribution_is_zero_when_savings_already_reach_the_goal():
    assert gm.required_contribution(1000, 5000, 0, 8, 60) == 0.0


@pytest.mark.parametrize("savings, step_up, rate, months", [
    (0, 0, 9, 180), (50_000, 0, 6, 120), (0, 5, 12, 240), (100_000, 10, 3, 60), (0, 0, 0, 12), (10, 15, 15, 480),
])
def test_required_contribution_round_trips_through_the_projection(savings, step_up, rate, months):
    goal = 2_500_000.0
    needed = gm.required_contribution(goal, savings, step_up, rate, months)
    assert gm.final_value(savings, needed, step_up, rate, months) == pytest.approx(goal)


def test_required_contribution_matches_the_textbook_formula_without_step_up():
    m = gm.monthly_rate(9)
    n, goal, savings = 180, 1_000_000.0, 20_000.0
    textbook = (goal - savings * (1 + m) ** n) * m / ((1 + m) ** n - 1)
    assert gm.required_contribution(goal, savings, 0, 9, n) == pytest.approx(textbook)


# ---- months to goal ----

def test_months_to_goal_at_zero_return():
    assert gm.months_to_goal(5000, 0, 1000, 0, 0, 600) == 5
    assert gm.months_to_goal(5000, 1000, 1000, 0, 0, 600) == 4


def test_months_to_goal_is_zero_when_already_met():
    assert gm.months_to_goal(1000, 1000, 0, 0, 0, 600) == 0


def test_months_to_goal_is_none_when_not_reached_in_time():
    assert gm.months_to_goal(10 ** 9, 0, 100, 0, 5, 60) is None
    assert gm.months_to_goal(1000, 0, 0, 0, 5, 600) is None


def test_months_to_goal_counts_the_step_up():
    assert gm.months_to_goal(2400, 0, 100, 100, 0, 600) == 18      # 12 x 100, then 200 a month: 1200 + 6 x 200 = 2400


# ---- inflation ----

def test_inflation_conversions():
    assert gm.future_money(100, 5, 2) == pytest.approx(110.25)
    assert gm.todays_money(110.25, 5, 2) == pytest.approx(100)


# ---- build_plan ----

def test_one_year_hand_case_is_met_exactly():
    plan = gm.build_plan(inputs(), zero_cfg())
    assert [s.name for s in plan.scenarios] == ["conservative", "expected", "optimistic"]
    for s in plan.scenarios:
        assert (s.final_value, s.shortfall, s.goal_met) == (12000.0, 0.0, True)
        assert s.required_monthly_contribution == pytest.approx(1000) and s.months_to_goal == 12 and s.months_note is None
        assert s.final_value_todays_money is None
    assert plan.target_amount == 12_000.0
    assert len(plan.projection_rows) == 6                                   # years 0 and 1, three scenarios
    assert plan.projection_rows[1] == {"scenario": "conservative", "year": 1, "value": 12000.0, "value_todays_money": None}


def test_scenarios_use_the_configured_returns_and_are_ordered():
    plan = gm.build_plan(inputs(horizon_years=10, goal_amount=1e6, risk_profile="medium"), CFG)
    assert [s.annual_return_pct for s in plan.scenarios] == [5.0, 9.0, 12.0]
    values = [s.final_value for s in plan.scenarios]
    assert values == sorted(values)


def test_goal_already_met_by_savings():
    plan = gm.build_plan(inputs(goal_amount=10_000.0, current_savings=20_000.0, monthly_contribution=0.0), zero_cfg())
    for s in plan.scenarios:
        assert s.months_to_goal == 0 and s.months_note == CFG["already_met_note"]
        assert s.required_monthly_contribution == 0.0 and s.shortfall == -10_000.0


def test_goal_not_reachable_within_the_maximum_months():
    plan = gm.build_plan(inputs(goal_amount=1e12, monthly_contribution=0.0), CFG)
    for s in plan.scenarios:
        assert s.months_to_goal is None and s.months_note == CFG["not_reached_note"].format(months=600)
        assert s.required_monthly_contribution > 0 and not s.goal_met


def test_zero_savings_and_zero_contribution_shows_what_is_needed():
    plan = gm.build_plan(inputs(monthly_contribution=0.0), zero_cfg())
    assert plan.scenarios[0].final_value == 0.0 and plan.scenarios[0].required_monthly_contribution == pytest.approx(1000)


def test_inflation_adjust_raises_the_target_and_adds_todays_money_figures():
    plan = gm.build_plan(inputs(goal_amount=100_000.0, horizon_years=2, inflation_adjust=True), CFG)
    assert plan.target_amount == pytest.approx(110_250)
    for s in plan.scenarios:
        assert s.shortfall == pytest.approx(110_250 - s.final_value)
        assert s.final_value_todays_money == pytest.approx(s.final_value / 1.05 ** 2)
    last = [r for r in plan.projection_rows if r["year"] == 2][0]
    assert last["value_todays_money"] == pytest.approx(last["value"] / 1.05 ** 2)
    assert [r for r in plan.projection_rows if r["year"] == 0][0]["value_todays_money"] == 0.0


def test_plan_required_contribution_reaches_the_target_with_step_up_and_inflation():
    i = inputs(goal_amount=1e6, horizon_years=15, current_savings=50_000.0, risk_profile="high",
               annual_step_up_pct=5.0, inflation_adjust=True)
    plan = gm.build_plan(i, CFG)
    for s in plan.scenarios:
        reached = gm.final_value(50_000.0, s.required_monthly_contribution, 5.0, s.annual_return_pct, 180)
        assert reached == pytest.approx(plan.target_amount)


def test_very_large_numbers_stay_finite():
    i = inputs(goal_amount=1e12, horizon_years=40, current_savings=1e12, monthly_contribution=1e12,
               risk_profile="high", annual_step_up_pct=15.0, inflation_adjust=True)
    plan = gm.build_plan(i, CFG)
    for s in plan.scenarios:
        assert all(math.isfinite(v) for v in (s.final_value, s.shortfall, s.required_monthly_contribution))
    assert all(math.isfinite(r["value"]) for r in plan.projection_rows)


# ---- summary input ----

def test_summary_input_has_only_preformatted_text_and_the_assumptions():
    plan = gm.build_plan(inputs(goal_amount=1_234_567.891, horizon_years=3, monthly_contribution=500.0), CFG)
    data = gm.summary_input(plan, CFG)
    assert data["goal_used_for_comparison"] == "1,234,567.89" and data["horizon"] == "3 years"
    assert data["inflation"] == "not applied"
    assert data["assumptions"]["as_of"] == CFG["assumptions"]["as_of"] and "Illustrative" in data["assumptions"]["label"]
    assert any("end of each month" in n for n in data["assumptions"]["notes"])
    item = data["scenarios"][0]
    assert item["assumed_annual_return"] == "4.0%" and "projected_value_at_end_in_todays_money" not in item
    assert all(isinstance(v, str) for v in item.values())


def test_summary_input_describes_shortfall_surplus_exact_and_timing():
    cfg = zero_cfg()
    short = gm.summary_input(gm.build_plan(inputs(monthly_contribution=10.0), cfg), cfg)["scenarios"][0]
    assert short["surplus_or_shortfall_text"] == "shortfall of 11,880.00"
    assert short["years_months_to_goal"] == "not within the maximum period"
    assert short["time_compared_with_horizon"] == "not within the maximum period"
    exact = gm.summary_input(gm.build_plan(inputs(), cfg), cfg)["scenarios"][0]
    assert exact["surplus_or_shortfall_text"] == "no surplus or shortfall (the goal is met exactly)"
    assert exact["years_months_to_goal"] == "1 year" and exact["time_compared_with_horizon"] == "within the horizon"
    met = inputs(goal_amount=100.0, current_savings=500.0, monthly_contribution=0.0)
    over = gm.summary_input(gm.build_plan(met, cfg), cfg)["scenarios"][0]
    assert over["surplus_or_shortfall_text"] == "surplus of 400.00"
    assert over["years_months_to_goal"] == "already met by the current savings"
    assert over["time_compared_with_horizon"] == "already met by the current savings"


def test_summary_input_gives_plain_words_for_both_goal_figures_when_inflation_is_applied():
    plan = gm.build_plan(inputs(goal_amount=100_000.0, horizon_years=2, inflation_adjust=True), CFG)
    data = gm.summary_input(plan, CFG)
    assert data["inflation"]["goal_in_todays_money_text"] == "The goal of 100,000.00 is stated in today's money."
    assert data["inflation"]["goal_in_future_money_text"] == (
        "With inflation of 5.0% a year over 2 years, the same goal would need about 110,250.00 in future money.")
    assert "labels" not in data["inflation"] and "projected_value_at_end_in_todays_money" in data["scenarios"][0]


@pytest.mark.parametrize("months, text", [
    (1, "1 month"), (5, "5 months"), (12, "1 year"), (24, "2 years"), (208, "17 years 4 months"), (13, "1 year 1 month"),
])
def test_years_months_text(months, text):
    assert gm.years_months_text(months, CFG) == text


def test_time_after_the_horizon_is_compared_with_it():
    cfg = zero_cfg()
    late = gm.summary_input(gm.build_plan(inputs(goal_amount=20_000.0), cfg), cfg)["scenarios"][0]
    assert late["years_months_to_goal"] == "1 year 8 months"
    assert late["time_compared_with_horizon"] == "after the end of the horizon"


# ---- required-amount labels ----

@pytest.mark.parametrize("met, step_up, text", [
    (False, 0, "monthly amount needed to reach the goal"),
    (False, 5, "starting monthly amount (rises 5% a year)"),
    (False, 2.5, "starting monthly amount (rises 2.5% a year)"),
    (True, 0, "minimum monthly amount to still reach the goal"),
    (True, 5, "minimum monthly amount to still reach the goal (starting amount, rises 5% a year)"),
])
def test_required_label(met, step_up, text):
    assert gm.required_label(met, step_up, CFG) == text


def test_plan_labels_follow_step_up_and_whether_the_goal_is_met():
    cfg = zero_cfg()
    plain = gm.build_plan(inputs(goal_amount=1e6), cfg).scenarios[0]
    assert plain.required_label == "monthly amount needed to reach the goal"
    stepped_inputs = inputs(goal_amount=1e6, annual_step_up_pct=5.0)
    assert gm.build_plan(stepped_inputs, cfg).scenarios[0].required_label == "starting monthly amount (rises 5% a year)"
    met = gm.build_plan(inputs(goal_amount=100.0, current_savings=500.0), cfg).scenarios[0]
    assert met.required_label == "minimum monthly amount to still reach the goal"
    data = gm.summary_input(gm.build_plan(stepped_inputs, cfg), cfg)
    assert data["scenarios"][0]["monthly_amount_meaning"] == "starting monthly amount (rises 5% a year)"


def test_summary_input_wording_comes_from_the_config():
    cfg = {**CFG, "texts": {**CFG["texts"], "inflation_not_applied": "no inflation", "years_many": "{n} yrs"}}
    data = gm.summary_input(gm.build_plan(inputs(horizon_years=3), cfg), cfg)
    assert data["inflation"] == "no inflation" and data["horizon"] == "3 yrs"
    assert gm.summary_input(gm.build_plan(inputs(horizon_years=1), CFG), CFG)["horizon"] == "1 year"


# ---- months to goal with a goal that keeps rising with inflation ----

LIVE_GOAL_1 = dict(goal_amount=10_000_000.0, horizon_years=15, current_savings=500_000.0, monthly_contribution=20_000.0,
                   risk_profile="medium", annual_step_up_pct=5.0, inflation_adjust=True)


def test_live_check_goal_1_months_use_the_rising_goal():
    plan = gm.build_plan(GoalInputs(**LIVE_GOAL_1), CFG)
    assert {s.name: s.months_to_goal for s in plan.scenarios} == {"conservative": 488, "expected": 283, "optimistic": 223}
    assert all(not s.goal_met for s in plan.scenarios)             # shortfall at the horizon is unchanged


@pytest.mark.parametrize("changes", [
    {},
    {"risk_profile": "low", "monthly_contribution": 80_000.0, "annual_step_up_pct": 0.0},
    {"risk_profile": "high", "goal_amount": 3_000_000.0, "current_savings": 0.0},
    {"horizon_years": 5, "monthly_contribution": 60_000.0, "annual_step_up_pct": 10.0},
])
def test_real_value_reaches_the_goal_at_the_returned_month_and_not_one_month_earlier(changes):
    i = GoalInputs(**{**LIVE_GOAL_1, **changes})
    inflation = CFG["assumptions"]["inflation"]["annual_pct"]
    for s in gm.build_plan(i, CFG).scenarios:
        k = s.months_to_goal
        assert k is not None and k > 0
        balances = [i.current_savings] + gm.project(i.current_savings, i.monthly_contribution, i.annual_step_up_pct,
                                                    s.annual_return_pct, k)
        real = lambda month: balances[month] / (1 + inflation / 100) ** (month / 12)
        assert real(k) >= i.goal_amount and real(k - 1) < i.goal_amount


def test_rising_goal_takes_longer_than_a_goal_fixed_at_the_horizon():
    fixed_target = gm.future_money(10_000_000.0, 5.0, 15)
    for s in gm.build_plan(GoalInputs(**LIVE_GOAL_1), CFG).scenarios:
        fixed = gm.months_to_goal(fixed_target, 500_000.0, 20_000.0, 5.0, s.annual_return_pct, 600)
        assert fixed is None or s.months_to_goal >= fixed


def test_months_to_goal_with_inflation_hand_case():
    # 0% return, 1,000 a month, goal 12,000 in today's money, 12% inflation:
    # month 12 needs 12,000 x 1.12 = 13,440 > 12,000; month 13 needs 12,000 x 1.12 ** (13/12) ~ 13,567 > 13,000 ...
    assert gm.months_to_goal(12_000, 0, 1_000, 0, 0, 600, 12.0) is not None
    assert gm.months_to_goal(12_000, 0, 1_000, 0, 0, 600, 0.0) == 12
    assert gm.months_to_goal(12_000, 0, 1_000, 0, 0, 600, 12.0) > 12


@pytest.mark.parametrize("inflation_pct", [0.0, 5.0])
def test_inflation_off_still_compares_with_the_fixed_goal(inflation_pct):
    cfg = copy.deepcopy(CFG)
    cfg["assumptions"]["inflation"]["annual_pct"] = inflation_pct       # the configured rate must not matter when off
    i = GoalInputs(**{**LIVE_GOAL_1, "inflation_adjust": False})
    plan = gm.build_plan(i, cfg)
    assert plan.target_amount == 10_000_000.0
    for s in plan.scenarios:
        assert s.months_to_goal == gm.months_to_goal(10_000_000.0, 500_000.0, 20_000.0, 5.0, s.annual_return_pct, 600)
        assert s.final_value_todays_money is None


def test_not_reached_within_the_maximum_months_with_inflation_gives_none_and_the_note():
    i = GoalInputs(**{**LIVE_GOAL_1, "goal_amount": 10.0 ** 12, "monthly_contribution": 100.0})
    for s in gm.build_plan(i, CFG).scenarios:
        assert s.months_to_goal is None
        assert s.months_note == CFG["not_reached_note"].format(months=CFG["max_months"])


def test_required_contribution_and_shortfall_still_use_the_target_at_the_horizon():
    plan = gm.build_plan(GoalInputs(**LIVE_GOAL_1), CFG)
    target = gm.future_money(10_000_000.0, 5.0, 15)
    assert plan.target_amount == pytest.approx(target)
    for s in plan.scenarios:
        assert s.shortfall == pytest.approx(target - s.final_value)
        assert s.required_monthly_contribution == pytest.approx(
            gm.required_contribution(target, 500_000.0, 5.0, s.annual_return_pct, 180))
