import pytest

from src.core.tax_inputs import parse_capital_gains, parse_small_savings
from src.core.tax_metrics import (
    exclusions_for,
    build_facts,
    capital_gains,
    fact_lines,
    figure_entry,
    footer,
    plain,
    small_savings_growth,
    summary_input,
)
from src.core.tax_models import CapitalGainsInputs
from tax_cfg import tax_cfg

CFG = tax_cfg()


def cg(**changes):
    """Run capital_gains through the real input parser with the test figures."""
    params = {"asset_type": "equity_fund", "buy_value": 100000, "sell_value": 250000, "expenses": 0, "holding_months": 6}
    inputs, errors = parse_capital_gains({**params, **changes}, CFG)
    assert errors == []
    return capital_gains(inputs, CFG)


def totals(calc):
    r = calc.results
    return r["gain_or_loss"], r["taxable_gain"], r["tax"], r["cess"], r["total_tax"]


# ---- equity: short and long term, hand computed with the test figures ----

def test_equity_short_term_is_taxed_at_the_short_term_rate_plus_cess():
    calc = cg()        # gain 150000, 15 percent = 22500, cess 10 percent = 2250
    assert totals(calc) == (150000.0, 150000.0, 22500.0, 2250.0, 24750.0)
    assert calc.results["term"] == "short_term" and calc.results["exemption_applied"] == 0.0


def test_equity_long_term_uses_the_exemption_first():
    calc = cg(asset_type="listed_equity", buy_value=200000, sell_value=400000, expenses=5000, holding_months=18)
    # gain 195000, exemption 100000 applied, taxable 95000, 10 percent = 9500, cess 950
    assert totals(calc) == (195000.0, 95000.0, 9500.0, 950.0, 10450.0)
    assert calc.results["term"] == "long_term" and calc.results["exemption_applied"] == 100000.0


def test_holding_period_boundary_twelve_months_is_short_and_thirteen_is_long():
    short = cg(buy_value=0, sell_value=200000, holding_months=12)
    long_ = cg(buy_value=0, sell_value=200000, holding_months=13)
    assert short.results["term"] == "short_term" and short.results["total_tax"] == 33000.0     # 30000 + 3000
    assert long_.results["term"] == "long_term" and long_.results["total_tax"] == 11000.0      # 10000 + 1000


def test_exemption_boundary_gain_equal_to_the_exemption_is_untaxed():
    calc = cg(buy_value=100000, sell_value=200000, holding_months=13)
    assert totals(calc) == (100000.0, 0.0, 0.0, 0.0, 0.0)


def test_exemption_boundary_one_unit_over_is_taxed_on_that_unit_only():
    calc = cg(buy_value=100000, sell_value=200001, holding_months=13)
    assert totals(calc) == (100001.0, 1.0, 0.1, 0.01, 0.11)


def test_exemption_already_used_reduces_what_is_available():
    calc = cg(holding_months=13, ltcg_exemption_already_used=40000)
    # gain 150000, available 60000, taxable 90000, tax 9000, cess 900
    assert totals(calc) == (150000.0, 90000.0, 9000.0, 900.0, 9900.0)
    assert calc.results["exemption_applied"] == 60000.0


def test_exemption_already_used_above_the_exemption_leaves_none():
    calc = cg(holding_months=13, ltcg_exemption_already_used=500000)
    assert totals(calc) == (150000.0, 150000.0, 15000.0, 1500.0, 16500.0)
    assert calc.results["exemption_applied"] == 0.0


def test_exemption_is_not_used_for_a_short_term_gain():
    assert cg(holding_months=12, ltcg_exemption_already_used=0).results["exemption_applied"] == 0.0


# ---- other assets ----

def test_property_long_term_has_no_exemption_and_no_indexation():
    calc = cg(asset_type="property_or_unlisted", buy_value=1000000, sell_value=2000000, expenses=50000, holding_months=25)
    # gain 950000, 20 percent = 190000, cess 19000
    assert totals(calc) == (950000.0, 950000.0, 190000.0, 19000.0, 209000.0)
    assert calc.results["term"] == "long_term" and calc.results["exemption_applied"] == 0.0
    assert "equity_ltcg_exemption" not in calc.config_used and "other_ltcg_pct" in calc.config_used


def test_other_asset_holding_boundary_twenty_four_months_is_short_and_taxed_at_the_slab_rate():
    calc = cg(asset_type="other", buy_value=1000000, sell_value=2000000, expenses=50000, holding_months=24, slab_rate_pct=30)
    # gain 950000, 30 percent = 285000, cess 28500
    assert totals(calc) == (950000.0, 950000.0, 285000.0, 28500.0, 313500.0)
    assert calc.results["term"] == "short_term" and calc.results["rate_pct"] == 30.0
    assert CFG["texts"]["slab_taxed_note"] in calc.notes


def test_debt_fund_is_taxed_at_the_slab_rate_whatever_the_holding_period():
    for months in (3, 36):
        calc = cg(asset_type="debt_fund_post_2023", buy_value=100000, sell_value=150000, holding_months=months, slab_rate_pct=30)
        assert totals(calc) == (50000.0, 50000.0, 15000.0, 1500.0, 16500.0)
        assert calc.results["term"] == "slab_rate"
    assert "holding_months_other" not in calc.config_used and "holding_months_equity" not in calc.config_used


def test_a_zero_slab_rate_means_zero_tax():
    assert cg(asset_type="debt_fund_post_2023", slab_rate_pct=0).results["total_tax"] == 0.0


# ---- losses, rounding, size ----

@pytest.mark.parametrize("sell, expenses", [(90000, 0), (100000, 0), (150000, 50000), (150000, 60000)])
def test_no_gain_gives_zero_tax_and_an_informational_note(sell, expenses):
    calc = cg(sell_value=sell, expenses=expenses, holding_months=13)
    assert calc.results["tax"] == calc.results["cess"] == calc.results["total_tax"] == 0.0
    assert calc.results["taxable_gain"] == 0.0 and calc.notes == [CFG["texts"]["loss_note"]]
    assert calc.steps[-1] == CFG["texts"]["step_loss"]


def test_a_loss_on_a_slab_asset_needs_no_slab_rate_to_run():
    calc = capital_gains(CapitalGainsInputs(asset_type="debt_fund_post_2023", buy_value=10, sell_value=5, holding_months=3), CFG)
    assert calc.results["total_tax"] == 0.0 and calc.results["term"] == "slab_rate"


def test_money_is_rounded_half_up_to_two_places():
    calc = cg(buy_value=0, sell_value=0.05, holding_months=1)      # tax 0.0075 -> 0.01, cess 0.001 -> 0.00
    assert (calc.results["tax"], calc.results["cess"], calc.results["total_tax"]) == (0.01, 0.0, 0.01)


def test_huge_amounts_do_not_crash_and_stay_exact():
    limit = CFG["input_limits"]["max_amount"]
    calc = cg(buy_value=0, sell_value=limit, holding_months=1)
    assert calc.results["tax"] == limit * 15 / 100 and calc.results["total_tax"] == limit * 15 / 100 * 1.1


def test_a_slab_gain_without_a_slab_rate_is_a_programming_error():
    inputs = CapitalGainsInputs(asset_type="debt_fund_post_2023", buy_value=0, sell_value=10, holding_months=1)
    with pytest.raises(ValueError, match="slab_rate_pct"):
        capital_gains(inputs, CFG)


def test_steps_explain_the_calculation_in_order():
    calc = cg(holding_months=13)
    assert len(calc.steps) == 7
    assert calc.steps[0].startswith("Gain or loss =") and calc.steps[-1].startswith("Total =")
    assert "10%" in calc.steps[4] and "10%" in calc.steps[5]       # tax rate and cess rate, both 10 in the test figures


def test_config_used_lists_each_figure_with_its_date_and_source():
    calc = cg(holding_months=13)
    assert set(calc.config_used) == {"cess_pct", "holding_months_equity", "equity_ltcg_pct", "equity_ltcg_exemption"}
    entry = calc.config_used["equity_ltcg_pct"]
    assert entry["value"] == 10 and entry["as_of"] == "2026-10-29" and entry["source"] and entry["verify"] is True


# ---- small savings ----

def savings(**changes):
    params = {"scheme": "ppf", "mode": "annual", "amount": 1000, "years": 2}
    inputs, errors = parse_small_savings({**params, **changes}, CFG)
    assert errors == []
    return small_savings_growth(inputs, CFG)


def test_one_time_deposit_compounds_once_a_year():
    r = savings(mode="one_time").results          # 1000 -> 1100 -> 1210 at 10 percent
    assert (r["total_deposited"], r["final_value"], r["interest_earned"]) == (1000.0, 1210.0, 210.0)


def test_annual_deposits_are_made_at_the_start_of_each_year():
    r = savings().results                          # (1000 x 1.1) -> (1100 + 1000) x 1.1 = 2310
    assert (r["total_deposited"], r["final_value"], r["interest_earned"]) == (2000.0, 2310.0, 310.0)
    assert r["yearly"] == [{"year": 1, "deposited": 1000.0, "balance": 1100.0}, {"year": 2, "deposited": 2000.0, "balance": 2310.0}]


@pytest.mark.parametrize("scheme, rate", [("ppf", 10), ("nsc", 5), ("ssy", 8)])
def test_each_scheme_uses_its_own_configured_rate(scheme, rate):
    calc = savings(scheme=scheme, mode="one_time", years=1, amount=100)
    assert calc.results["rate_pct"] == rate and calc.results["final_value"] == 100 + rate
    assert list(calc.config_used) == [CFG["schemes"][scheme]]


@pytest.mark.parametrize("mode", ["annual", "one_time"])
def test_round_trip_against_a_closed_form(mode):
    amount, years, rate = 12345.67, 30, 0.10
    r = savings(mode=mode, amount=amount, years=years).results
    expected = amount * sum((1 + rate) ** k for k in range(1, years + 1)) if mode == "annual" else amount * (1 + rate) ** years
    assert r["final_value"] == pytest.approx(expected, abs=0.01)
    assert r["interest_earned"] == pytest.approx(r["final_value"] - r["total_deposited"], abs=0.01)


def test_savings_carry_the_approximation_note_and_the_assumptions():
    calc = savings()
    assert calc.notes == [CFG["texts"]["approximate_note"]]
    assert calc.assumptions == [] and calc.exclusions == CFG["exclusions"]["small_savings"]
    assert any("annual compounding" in a and "check the official scheme rules" in a for a in calc.exclusions)


def test_the_largest_projection_does_not_crash():
    limit = CFG["input_limits"]
    r = savings(amount=limit["max_amount"], years=limit["max_years"]).results
    assert r["final_value"] > r["total_deposited"] > 0 and len(r["yearly"]) == limit["max_years"]


# ---- helpers ----

def test_plain_never_uses_an_exponent():
    from decimal import Decimal

    assert (plain(Decimal(20)), plain(Decimal("12.50")), plain(Decimal(100)), plain(Decimal(0))) == ("20", "12.5", "100", "0")


def test_exclusions_for_adds_the_securities_transaction_tax_assumption_first_and_only_for_equity():
    capital = CFG["exclusions"]["capital_gains"]
    assert exclusions_for("capital_gains", CFG, equity=True) == [CFG["stt_assumption"], *capital]
    assert exclusions_for("capital_gains", CFG) == capital and exclusions_for("small_savings", CFG) == CFG["exclusions"]["small_savings"]
    assert "securities transaction tax was paid" in CFG["stt_assumption"]


def test_footer_names_the_label_the_dates_and_the_tax_paid_assumption_for_equity_figures():
    equity = {"equity_ltcg_pct": figure_entry(CFG, "equity_ltcg_pct")}
    text = footer(equity, CFG)
    assert "illustrative, verify before relying" in text and "2026-10-29" in text
    assert text.endswith("Assumes the securities transaction tax was paid on the sale of listed equity and equity funds.")


def test_footer_leaves_out_the_tax_paid_assumption_for_other_figures():
    assert "transaction tax" not in footer({"ppf_pct": figure_entry(CFG, "ppf_pct")}, CFG)


def test_footer_lists_each_date_once_and_sorted():
    cfg = tax_cfg()
    cfg["figures"]["ppf_pct"]["as_of"] = "2026-09-30"
    used = {k: figure_entry(cfg, k) for k in ("cess_pct", "ppf_pct", "nsc_pct")}
    assert "(as of 2026-09-30, 2026-10-29)" in footer(used, cfg)


def test_facts_are_numbered_in_config_order_with_their_conditions():
    facts = build_facts(CFG)
    assert [f["id"] for f in facts] == [f"F{i}" for i in range(1, len(CFG["figures"]) + 1)]
    assert [f["key"] for f in facts] == list(CFG["figures"])
    text = fact_lines(facts, CFG).splitlines()
    assert text[0].startswith("F1. ") and "15%" in text[0] and "Verify: yes" in text[0] and CFG["stt_assumption"] in text[0]
    assert "months" in text[3] and CFG["stt_assumption"] not in text[3] and "INR" in text[2]


def test_a_fact_with_verify_false_says_no():
    cfg = tax_cfg()
    cfg["figures"]["cess_pct"]["verify"] = False
    line = [ln for ln in fact_lines(build_facts(cfg), cfg).splitlines() if "cess" in ln.lower()][0]
    assert "Verify: no" in line


def test_summary_input_has_inputs_results_rates_and_caveats_but_no_dates_or_sources():
    calc = cg(holding_months=13)
    data = summary_input(calc, CFG)
    assert set(data) == {"calculation", "inputs", "results", "rates", "caveats"}
    assert data["rates"]["equity_ltcg_pct"] == "10%" and CFG["stt_assumption"] in data["caveats"]
    assert "2026" not in str(data) and "incometax" not in str(data)


def test_summary_input_leaves_out_the_yearly_table():
    assert "yearly" not in summary_input(savings(), CFG)["results"]


def test_cess_is_computed_on_the_unrounded_tax_and_rounded_once():
    # gain 0.05 at 15 percent = 0.0075 exact, cess 10 percent -> 0.00825 exact, total 0.00825 -> 0.01
    calc = cg(buy_value=0, sell_value=0.05, holding_months=1)
    assert calc.results["tax"] == 0.01 and calc.results["total_tax"] == 0.01 and calc.results["cess"] == 0.0
    # gain 0.30 -> tax 0.045 exact (rounds to 0.05), cess 0.0045 exact, total 0.0495 -> 0.05
    calc = cg(buy_value=0, sell_value=0.30, holding_months=1)
    assert (calc.results["tax"], calc.results["cess"], calc.results["total_tax"]) == (0.05, 0.0, 0.05)
    # rounding the tax first would give 0.05 + round(0.005) = 0.06; one rounding at the end gives 0.05
    r = cg(buy_value=0, sell_value=1234.57, holding_months=1).results      # 185.1855 -> 185.19; total 203.70405 -> 203.70
    assert (r["tax"], r["total_tax"], r["cess"]) == (185.19, 203.70, 18.51)
    assert round(r["tax"] + r["cess"], 2) == r["total_tax"]


def test_a_loss_does_not_list_cess_in_the_figures_used():
    calc = cg(buy_value=500000, sell_value=100000)
    assert "cess_pct" not in calc.config_used and set(calc.config_used) == {"holding_months_equity"}
    gain = cg()
    assert "cess_pct" in gain.config_used


def test_a_debt_fund_loss_through_the_parser_needs_no_slab_rate():
    calc = cg(asset_type="debt_fund_post_2023", buy_value=500000, sell_value=100000)
    assert calc.results["total_tax"] == 0.0 and calc.inputs["slab_rate_pct"] is None
    assert set(calc.config_used) == set()


STT_WORDS = ("transaction tax", "surcharge", "indexation")


@pytest.mark.parametrize("asset, has_stt", [
    ("listed_equity", True), ("equity_fund", True), ("property_or_unlisted", False), ("other", False), ("debt_fund_post_2023", False),
])
def test_the_tax_paid_assumption_is_in_the_exclusions_of_equity_results_only(asset, has_stt):
    calc = cg(asset_type=asset, slab_rate_pct=30, holding_months=36)
    assert (calc.exclusions[0] == CFG["stt_assumption"]) is has_stt
    assert (CFG["stt_assumption"] in calc.exclusions) is has_stt
    assert calc.exclusions[has_stt:] == CFG["exclusions"]["capital_gains"]
    assert (CFG["stt_assumption"] in summary_input(calc, CFG)["caveats"]) is has_stt


def test_the_savings_commentary_input_has_no_capital_gains_caveats():
    calc = savings()
    caveats = " ".join(summary_input(calc, CFG)["caveats"])
    assert not any(word in caveats for word in STT_WORDS) and "income tax slabs" not in caveats and "set-off" not in caveats
    assert "annual compounding" in caveats and "lock-in" in caveats and "check the official scheme rules" in caveats


def test_summary_input_writes_amounts_and_rates_as_text_with_the_currency_sign_and_grouping():
    data = summary_input(cg(buy_value=1234567.5, sell_value=2000000, expenses=0, holding_months=6), CFG)
    assert data["inputs"]["buy_value"] == "₹1,234,567.50" and data["inputs"]["expenses"] == "₹0.00"
    assert data["inputs"]["holding_months"] == 6 and data["inputs"]["asset_type"] == "equity_fund"
    assert data["results"]["gain_or_loss"] == "₹765,432.50" and data["results"]["rate_pct"] == "15%"
    assert data["results"]["term"] == "short_term" and data["inputs"]["slab_rate_pct"] is None
    debt = summary_input(cg(asset_type="debt_fund_post_2023", slab_rate_pct=30.5), CFG)
    assert debt["inputs"]["slab_rate_pct"] == "30.5%" and debt["results"]["rate_pct"] == "30.5%"
    assert data["rates"] == {"cess_pct": "10%", "equity_stcg_pct": "15%", "holding_months_equity": "12 months"} or \
        data["rates"] == {"cess_pct": "10%", "holding_months_equity": "12 months", "equity_stcg_pct": "15%"}


def test_savings_summary_input_is_text_too_and_keeps_the_scheme_and_years():
    data = summary_input(savings(amount=1000.5), CFG)
    assert data["inputs"]["amount"] == "₹1,000.50" and data["inputs"]["years"] == 2 and data["inputs"]["scheme"] == "ppf"
    assert data["results"]["final_value"].startswith("₹") and data["results"]["rate_pct"] == "10%" and data["results"]["scheme"] == "ppf"


def test_the_figure_entry_carries_the_short_source():
    from src.core.tax_metrics import figure_entry
    entry = figure_entry(CFG, "cess_pct")
    assert entry["source_short"] == CFG["figures"]["cess_pct"]["source_short"] and entry["source_short"] != entry["source"]


PLAIN_COUNTS = {"holding_months", "years", "year"}      # whole-number counts that are shown as they are


def _numeric_fields(model):
    """Names of the int/float fields of a pydantic model (also Optional ones)."""
    return {name for name, field in model.model_fields.items() if str(field.annotation) in ("<class 'float'>", "<class 'int'>", "float | None", "int | None")}


def test_every_numeric_model_and_result_field_is_formatted_or_listed_as_a_plain_count():
    from src.core.tax_metrics import MONEY_FIELDS, PERCENT_FIELDS
    from src.core.tax_models import SmallSavingsInputs
    formatted = set(MONEY_FIELDS) | set(PERCENT_FIELDS)
    declared = _numeric_fields(CapitalGainsInputs) | _numeric_fields(SmallSavingsInputs)
    produced = set()
    for calc in (
        capital_gains(CapitalGainsInputs(asset_type="equity_fund", buy_value=1, sell_value=900000, holding_months=24, ltcg_exemption_already_used=1), CFG),
        capital_gains(CapitalGainsInputs(asset_type="debt_fund_post_2023", buy_value=1, sell_value=5, holding_months=1, slab_rate_pct=30), CFG),
        small_savings_growth(SmallSavingsInputs(scheme="ppf", mode="annual", amount=10, years=2), CFG),
    ):
        for part in (calc.inputs, calc.results):
            produced |= {k for k, v in part.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    unformatted = (declared | produced) - formatted - PLAIN_COUNTS
    assert not unformatted, f"numeric field(s) {sorted(unformatted)} are in neither MONEY_FIELDS nor PERCENT_FIELDS: add them or list them as plain counts"


def test_the_field_coverage_check_would_catch_a_new_unformatted_field():
    from pydantic import BaseModel

    class WithNewField(BaseModel):
        cess_pct: float
    assert _numeric_fields(WithNewField) == {"cess_pct"}
