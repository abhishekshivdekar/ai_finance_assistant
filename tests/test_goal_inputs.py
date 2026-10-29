import copy

import pytest

from src.core.goal_inputs import parse_goal_params
from src.utils.config import get_config

CFG = get_config()["goal_planning"]
GOOD = {"goal_amount": 1_000_000, "horizon_years": 15, "current_savings": 50_000,
        "monthly_contribution": 5_000, "risk_profile": "medium", "annual_step_up_pct": 5, "inflation_adjust": False}


def parse(**changes):
    return parse_goal_params({**GOOD, **changes}, CFG)


def test_valid_request_becomes_inputs():
    inputs, errors = parse()
    assert errors == []
    assert (inputs.goal_amount, inputs.horizon_years, inputs.risk_profile) == (1_000_000.0, 15, "medium")
    assert inputs.annual_step_up_pct == 5.0 and inputs.inflation_adjust is False


def test_optional_fields_default_and_zero_savings_and_contribution_are_allowed():
    inputs, errors = parse_goal_params({"goal_amount": 5000, "horizon_years": 3, "risk_profile": "low"}, CFG)
    assert errors == []
    assert (inputs.current_savings, inputs.monthly_contribution, inputs.annual_step_up_pct) == (0.0, 0.0, 0.0)
    inputs, _ = parse(current_savings=0, monthly_contribution=0)
    assert inputs.current_savings == 0.0 and inputs.monthly_contribution == 0.0


def test_inflation_adjust_uses_the_config_default_and_the_request_overrides_it():
    without = {k: v for k, v in GOOD.items() if k != "inflation_adjust"}
    assert CFG["inflation_adjust_default"] is True
    assert parse_goal_params(without, CFG)[0].inflation_adjust is True            # default on
    assert parse(inflation_adjust=False)[0].inflation_adjust is False             # request turns it off
    cfg_off = {**copy.deepcopy(CFG), "inflation_adjust_default": False}
    assert parse_goal_params(without, cfg_off)[0].inflation_adjust is False       # default off
    assert parse_goal_params({**GOOD, "inflation_adjust": True}, cfg_off)[0].inflation_adjust is True


def test_a_whole_number_float_horizon_is_accepted():
    assert parse(horizon_years=15.0)[0].horizon_years == 15


@pytest.mark.parametrize("changes, text", [
    ({"goal_amount": 0}, "goal_amount must be greater than zero"),
    ({"goal_amount": -5}, "goal_amount must be greater than zero"),
    ({"goal_amount": None}, "goal_amount must be a number"),
    ({"goal_amount": "lots"}, "goal_amount must be a number"),
    ({"goal_amount": True}, "goal_amount must be a number"),
    ({"goal_amount": float("nan")}, "goal_amount must be a number"),
    ({"goal_amount": float("inf")}, "goal_amount must be a number"),
    ({"goal_amount": 10 ** 13}, "too large"),
    ({"current_savings": -1}, "current_savings cannot be negative"),
    ({"monthly_contribution": -1}, "monthly_contribution cannot be negative"),
    ({"monthly_contribution": "x"}, "monthly_contribution must be a number"),
    ({"horizon_years": 0}, "between 1 and 40"),
    ({"horizon_years": 41}, "between 1 and 40"),
    ({"horizon_years": 2.5}, "whole number of years"),
    ({"horizon_years": None}, "horizon_years must be a number"),
    ({"horizon_years": True}, "horizon_years must be a number"),
    ({"risk_profile": "wild"}, "one of: low, medium, high"),
    ({"risk_profile": None}, "one of: low, medium, high"),
    ({"annual_step_up_pct": -1}, "between 0 and 15 percent"),
    ({"annual_step_up_pct": 16}, "between 0 and 15 percent"),
    ({"inflation_adjust": "yes"}, "true or false"),
])
def test_invalid_values_give_a_clear_message(changes, text):
    inputs, errors = parse(**changes)
    assert inputs is None and any(text in e for e in errors)


def test_all_problems_are_reported_together():
    inputs, errors = parse(goal_amount=-1, risk_profile="wild", horizon_years=0)
    assert inputs is None and len(errors) == 3


@pytest.mark.parametrize("params", [None, [], "x", {}])
def test_missing_or_malformed_params_need_input(params):
    inputs, errors = parse_goal_params(params, CFG)
    assert inputs is None and errors
