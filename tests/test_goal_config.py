import copy

import pytest
import yaml

from src.utils.config import ConfigError, get_config


def _goal_config(tmp_path, change):
    """A config file holding only a (modified) copy of the real goal_planning block."""
    goal = copy.deepcopy(get_config()["goal_planning"])
    get_config.cache_clear()
    change(goal)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"goal_planning": goal}), encoding="utf-8")
    return str(path)


def test_the_real_goal_planning_block_is_valid_and_labelled():
    goal = get_config()["goal_planning"]
    assumptions = goal["assumptions"]
    assert goal["currency"] == "INR" and "Illustrative" in assumptions["label"]
    assert assumptions["as_of"] and assumptions["source"] and assumptions["inflation"]["as_of"]
    assert get_config()["llm"]["roles"]["goal_summary"]["model"]


def test_valid_goal_planning_config_loads(tmp_path):
    assert get_config(_goal_config(tmp_path, lambda g: None))["goal_planning"]["currency"] == "INR"


def test_goal_returns_must_not_decrease_across_scenarios(tmp_path):
    def swap(goal):
        goal["assumptions"]["returns_pct"]["medium"]["expected"] = 2.0

    with pytest.raises(ConfigError, match="medium"):
        get_config(_goal_config(tmp_path, swap))


def test_goal_max_months_must_cover_the_longest_horizon(tmp_path):
    with pytest.raises(ConfigError, match="max_months"):
        get_config(_goal_config(tmp_path, lambda g: g.update(max_months=12)))


@pytest.mark.parametrize("low, high", [(0, 40), (10, 5)])
def test_goal_horizon_limits_must_be_consistent(tmp_path, low, high):
    def limits(goal):
        goal.update(min_horizon_years=low, max_horizon_years=high)

    with pytest.raises(ConfigError, match="horizon limits"):
        get_config(_goal_config(tmp_path, limits))


def test_goal_returns_must_not_be_negative(tmp_path):
    def negative(goal):
        goal["assumptions"]["returns_pct"]["low"]["conservative"] = -1.0

    with pytest.raises(ConfigError, match="negative"):
        get_config(_goal_config(tmp_path, negative))


def test_goal_inflation_must_not_be_negative(tmp_path):
    def negative(goal):
        goal["assumptions"]["inflation"]["annual_pct"] = -0.5

    with pytest.raises(ConfigError, match="inflation"):
        get_config(_goal_config(tmp_path, negative))
