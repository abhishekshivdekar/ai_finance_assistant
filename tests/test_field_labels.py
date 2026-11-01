"""Users see words, never snake_case input keys (UI step 4a: found in the browser as "goal_amount must be a number")."""
from __future__ import annotations

import copy
import re

import pytest
import yaml

from src.core.field_labels import choice_label, choice_list, field_label
from src.core.goal_inputs import parse_goal_params
from src.core.tax_inputs import CALCULATIONS, SAVINGS_MODES, parse_capital_gains, parse_small_savings
from src.core.news_query import resolve_query
from src.utils.config import GOAL_FIELD_KEYS, TAX_FIELD_KEYS, ConfigError, get_config

CFG = get_config()
SNAKE = re.compile(r"(?<![\w{.])[a-z]+(?:_[a-z0-9]+)+(?![\w}])")


def test_a_labelled_key_gives_its_label_and_an_unlabelled_one_a_readable_fallback():
    assert field_label("horizon_years") == "Time horizon (years)" and field_label("annual_step_up_pct") == "Yearly step-up (%)"
    assert field_label("some_new_key") == "Some new key"
    assert choice_label("one_time") == "one-time deposit" and choice_label("brand_new_option") == "brand new option"
    assert choice_list(["annual", "one_time"]) == "yearly deposits, one-time deposit"


def test_every_input_key_the_checks_report_has_a_label_and_every_option_a_choice_label():
    assert all(key in CFG["field_labels"] for key in (*GOAL_FIELD_KEYS, *TAX_FIELD_KEYS, "max_items"))
    options = [*CFG["goal_planning"]["risk_profiles"], *CFG["tax"]["asset_types"], *CFG["tax"]["schemes"], *CALCULATIONS, *SAVINGS_MODES]
    assert all(option in CFG["choice_labels"] for option in options)
    assert not [text for text in (*CFG["field_labels"].values(), *CFG["choice_labels"].values()) if SNAKE.search(text)]


def test_goal_errors_use_labels_for_every_field():
    inputs, errors = parse_goal_params({}, CFG["goal_planning"])
    text = " ".join(errors)
    assert inputs is None and "Goal amount must be a number." in text and "Time horizon (years) must be a number." in text
    assert "The risk profile must be one of: low, medium, high." in text
    assert not SNAKE.findall(text)
    _, errors = parse_goal_params({"goal_amount": 5, "horizon_years": 5, "risk_profile": "low", "inflation_adjust": "yes"}, CFG["goal_planning"])
    assert errors == ["The inflation option must be on or off."]


def test_tax_errors_use_labels_for_fields_and_options():
    _, errors = parse_capital_gains({"asset_type": "nonsense", "buy_value": "x", "sell_value": -1, "holding_months": 2.5}, CFG["tax"])
    text = " ".join(errors)
    assert "Asset type must be one of: listed shares, equity fund, property or unlisted shares, other asset, debt fund bought after March 2023." in text
    assert "Purchase value must be a number." in text and "Holding period (months) must be a whole number." in text
    assert not SNAKE.findall(text)
    _, errors = parse_small_savings({"scheme": "x", "mode": "y"}, CFG["tax"])
    assert "Scheme must be one of: PPF, NSC, SSY." in " ".join(errors) and "Deposit type must be one of: yearly deposits, one-time deposit." in " ".join(errors)
    _, errors = parse_capital_gains({"asset_type": "other", "buy_value": 1, "sell_value": 9, "holding_months": 3}, CFG["tax"])
    assert errors == [CFG["tax"]["errors"]["slab_rate_needed"]] and not SNAKE.findall(errors[0])


def test_the_news_headline_count_warning_uses_the_label_wording():
    warning = resolve_query({"query": "Nifty", "max_items": "many"}, CFG["news"]).warning
    assert warning and "The number of headlines must be a whole number" in warning and not SNAKE.findall(warning)


def _texts(node, path=""):
    """Every user-facing config string: (path, text) for text with more than one word, outside the LLM prompts and the label tables."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _texts(value, f"{path}.{key}")
    elif isinstance(node, list):
        for value in node:
            yield from _texts(value, path)
    elif isinstance(node, str) and len(node.split()) > 1 and "prompt" not in path and not path.startswith((".field_labels", ".choice_labels")):
        yield path, node


def test_no_user_facing_config_message_contains_an_input_key():
    keys = {*CFG["field_labels"], *(k for k in CFG["choice_labels"] if "_" in k)}
    leaks = [(path, token) for path, text in _texts(CFG) for token in SNAKE.findall(text) if token in keys]
    assert leaks == []


def _config(tmp_path, change):
    cfg = copy.deepcopy(dict(get_config()))
    get_config.cache_clear()
    change(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_a_valid_label_block_loads(tmp_path):
    assert get_config(_config(tmp_path, lambda c: None))["field_labels"]["goal_amount"] == "Goal amount"
    get_config.cache_clear()


@pytest.mark.parametrize("change, text", [
    (lambda c: c.update(field_labels="x"), "must be mappings"),
    (lambda c: c.update(choice_labels=["x"]), "must be mappings"),
    (lambda c: c["field_labels"].update(goal_amount="goal_amount"), "snake_case"),
    (lambda c: c["field_labels"].update(goal_amount=" "), "readable label"),
    (lambda c: c["field_labels"].update(goal_amount=5), "readable label"),
    (lambda c: c["choice_labels"].update(low="low_risk"), "snake_case"),
    (lambda c: c["field_labels"].pop("horizon_years"), "missing: horizon_years"),
    (lambda c: c["choice_labels"].pop("one_time"), "missing: one_time"),
    (lambda c: c["choice_labels"].pop("ppf"), "missing: ppf"),
])
def test_a_bad_label_block_is_rejected(tmp_path, change, text):
    path = _config(tmp_path, change)
    with pytest.raises(ConfigError, match=text):
        get_config(path)
    get_config.cache_clear()
