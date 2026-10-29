import pytest

from src.core.tax_inputs import parse_capital_gains, parse_small_savings, slab_rate_needed
from tax_cfg import tax_cfg

CFG = tax_cfg()
ERR = CFG["errors"]
GOOD = {"asset_type": "equity_fund", "buy_value": 1000, "sell_value": 2000, "expenses": 10, "holding_months": 6}
SAVINGS = {"scheme": "ppf", "mode": "annual", "amount": 500, "years": 10}


def cg(**changes):
    return parse_capital_gains({**GOOD, **changes}, CFG)


def test_valid_capital_gains_request_gives_inputs_and_no_errors():
    inputs, errors = cg()
    assert errors == [] and inputs.asset_type == "equity_fund" and inputs.holding_months == 6
    assert inputs.expenses == 10.0 and inputs.ltcg_exemption_already_used == 0.0 and inputs.slab_rate_pct is None


def test_optional_fields_default_to_zero():
    params = {k: v for k, v in GOOD.items() if k != "expenses"}
    assert parse_capital_gains(params, CFG)[0].expenses == 0.0


def test_every_problem_is_reported_at_once():
    inputs, errors = parse_capital_gains({"asset_type": "gold", "buy_value": -1, "sell_value": "x", "holding_months": 1.5}, CFG)
    assert inputs is None and len(errors) == 4
    assert ERR["unknown_choice"].format(field="asset_type", choices=", ".join(CFG["asset_types"])) in errors
    assert ERR["must_not_be_negative"].format(field="buy_value") in errors
    assert ERR["not_a_number"].format(field="sell_value") in errors
    assert ERR["whole_number"].format(field="holding_months") in errors


@pytest.mark.parametrize("bad", [None, "10", True, float("inf"), float("nan"), [1]])
def test_non_numbers_are_rejected_for_amounts(bad):
    inputs, errors = cg(buy_value=bad)
    assert inputs is None and errors == [ERR["not_a_number"].format(field="buy_value")]


def test_huge_and_tiny_numbers_do_not_crash():
    limit = CFG["input_limits"]["max_amount"]
    for value in (10**400, limit + 1):
        inputs, errors = cg(sell_value=value)
        assert inputs is None and errors == [ERR["too_large"].format(field="sell_value", limit=f"{limit:,}")]
    inputs, errors = cg(buy_value=-(10**400))
    assert inputs is None and errors == [ERR["must_not_be_negative"].format(field="buy_value")]
    assert cg(sell_value=limit)[1] == []


def test_holding_months_limits():
    limit = CFG["input_limits"]["max_months"]
    assert cg(holding_months=0)[1] == [] and cg(holding_months=float(limit))[1] == [] and cg(holding_months=limit + 1)[1]
    assert cg(holding_months=-1)[1] == [ERR["must_not_be_negative"].format(field="holding_months")]
    assert cg(holding_months=10**400)[1] == [ERR["too_large"].format(field="holding_months", limit=f"{limit:,}")]


def test_slab_rate_needed_for_debt_funds_and_short_term_other_assets_only():
    assert slab_rate_needed("debt_fund_post_2023", 100, CFG) is True
    assert slab_rate_needed("other", 24, CFG) is True and slab_rate_needed("property_or_unlisted", 25, CFG) is False
    assert slab_rate_needed("listed_equity", 1, CFG) is False and slab_rate_needed("equity_fund", 100, CFG) is False


def test_debt_fund_without_a_slab_rate_is_needs_input():
    inputs, errors = cg(asset_type="debt_fund_post_2023")
    assert inputs is None and errors == [ERR["slab_rate_needed"]]


def test_short_term_other_asset_without_a_slab_rate_is_needs_input():
    assert cg(asset_type="other", holding_months=3)[1] == [ERR["slab_rate_needed"]]
    assert cg(asset_type="other", holding_months=25)[1] == []


def test_slab_rate_is_checked_and_kept_when_given():
    assert cg(asset_type="debt_fund_post_2023", slab_rate_pct=30)[0].slab_rate_pct == 30.0
    limit = CFG["input_limits"]["max_pct"]
    assert cg(asset_type="debt_fund_post_2023", slab_rate_pct=limit + 1)[1] == [ERR["too_large"].format(field="slab_rate_pct", limit=f"{limit:,}")]
    assert cg(asset_type="debt_fund_post_2023", slab_rate_pct=-5)[1] == [ERR["must_not_be_negative"].format(field="slab_rate_pct")]
    assert cg(asset_type="debt_fund_post_2023", slab_rate_pct="30")[1] == [ERR["not_a_number"].format(field="slab_rate_pct")]


def test_a_slab_rate_is_ignored_where_it_is_not_needed():
    assert cg(slab_rate_pct="rubbish")[0].slab_rate_pct is None


def test_missing_asset_type_and_months_skips_the_slab_check_without_crashing():
    inputs, errors = parse_capital_gains({"buy_value": 1, "sell_value": 2}, CFG)
    assert inputs is None and len(errors) == 2


def test_valid_small_savings_request():
    inputs, errors = parse_small_savings(SAVINGS, CFG)
    assert errors == [] and (inputs.scheme, inputs.mode, inputs.amount, inputs.years) == ("ppf", "annual", 500.0, 10)


def test_small_savings_problems_are_all_reported():
    inputs, errors = parse_small_savings({"scheme": "fd", "mode": "monthly", "amount": 0, "years": 0}, CFG)
    assert inputs is None and len(errors) == 4
    assert ERR["unknown_choice"].format(field="scheme", choices="ppf, nsc, ssy") in errors
    assert ERR["unknown_choice"].format(field="mode", choices="annual, one_time") in errors
    assert ERR["must_be_positive"].format(field="amount") in errors
    assert ERR["must_be_positive"].format(field="years") in errors


@pytest.mark.parametrize("years", [51, 10**400, 2.5, None, -3])
def test_small_savings_years_are_limited(years):
    inputs, errors = parse_small_savings({**SAVINGS, "years": years}, CFG)
    assert inputs is None and len(errors) == 1
    assert errors[0].startswith("years")


def test_small_savings_amount_must_be_a_positive_number_within_the_limit():
    limit = CFG["input_limits"]["max_amount"]
    assert parse_small_savings({**SAVINGS, "amount": limit}, CFG)[1] == []
    assert parse_small_savings({**SAVINGS, "amount": limit + 1}, CFG)[1]
    assert parse_small_savings({**SAVINGS, "amount": float("nan")}, CFG)[1] == [ERR["not_a_number"].format(field="amount")]
    assert parse_small_savings({**SAVINGS, "amount": -1}, CFG)[1] == [ERR["must_be_positive"].format(field="amount")]
