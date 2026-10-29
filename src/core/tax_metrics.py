"""Deterministic, illustrative tax calculators and the helpers around them (architecture section 2.6).

Design pattern: PURE FUNCTIONS. Every function takes its request and the `tax` config block and
returns a value; nothing is read from globals. Example: capital_gains(inputs, cfg) with a test
config holding round made-up rates gives the same answer every time, so a test can compute the
expected tax by hand without touching the real (to-be-verified) rates.

No rate, limit or threshold appears in this file: every figure is read from cfg["figures"]
(value, as_of, source, verify). Amounts use Decimal and are rounded half-up to two places only
for display, so 0.1 + 0.2 style errors cannot reach a result.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from src.core.tax_models import CapitalGainsInputs, SmallSavingsInputs, TaxCalculation

_HUNDRED = Decimal(100)
_CENT = Decimal("0.01")


def _d(value: Any) -> Decimal:
    """A number as an exact Decimal (via str, so 3.25 stays 3.25)."""
    return Decimal(str(value))


def _q(value: Decimal) -> Decimal:
    """Round half-up to two places."""
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def _f(value: Decimal) -> float:
    """A rounded Decimal as a float for the payload."""
    return float(_q(value))


def plain(value: Decimal) -> str:
    """A rate as plain text without an exponent: 7 -> "7", 3.250 -> "3.25"."""
    return format(value.normalize(), "f")


def fmt(value: Decimal) -> str:
    """Money text for the steps, e.g. 1,234.50."""
    return f"{_q(value):,.2f}"


def figure_entry(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """One config figure with everything the reader needs to judge it: label, value, unit, as_of, source, verify."""
    fig = cfg["figures"][key]
    return {
        "label": fig["label"], "value": fig["value"], "unit": fig["unit"], "as_of": str(fig["as_of"]),
        "source": fig["source"], "verify": fig["verify"], "stt_condition": bool(fig.get("stt_condition", False)),
    }


def all_exclusions(cfg: dict[str, Any]) -> list[str]:
    """What every output leaves out; the securities-transaction-tax assumption comes first."""
    return [cfg["stt_assumption"], *cfg["exclusions"]]


def footer(config_used: dict[str, Any], cfg: dict[str, Any]) -> str:
    """The fixed closing line: the label, the as_of dates of the figures used and, for equity figures, the tax-paid assumption."""
    dates = sorted({entry["as_of"] for entry in config_used.values()})
    text = " ".join(cfg["footer"].split()).format(label=cfg["label"], as_of=", ".join(dates))
    if any(entry["stt_condition"] for entry in config_used.values()):
        text += " " + cfg["stt_assumption"][0].upper() + cfg["stt_assumption"][1:] + "."
    return text


def build_facts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every config figure as a numbered fact F1..Fn for the question prompt (same order as the config)."""
    facts = []
    for number, key in enumerate(cfg["figures"], start=1):
        facts.append({"id": f"F{number}", "number": number, "key": key, **figure_entry(cfg, key)})
    return facts


def fact_lines(facts: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    """The facts as prompt lines: label, value, unit, as_of, source, verify and, for equity figures, the tax-paid condition."""
    lines = []
    for fact in facts:
        condition = cfg["fact_condition"].format(stt_assumption=cfg["stt_assumption"]) if fact["stt_condition"] else ""
        lines.append(cfg["fact_line"].format(
            id=fact["id"], label=fact["label"], value=fact["value"], unit=cfg["units"][fact["unit"]],
            as_of=fact["as_of"], source=fact["source"], verify="yes" if fact["verify"] else "no", condition=condition,
        ))
    return "\n".join(lines)


def slab_rate_for(inputs: CapitalGainsInputs) -> Decimal:
    """The user's slab rate; calling the calculator without one for a slab-taxed gain is a programming error."""
    if inputs.slab_rate_pct is None:
        raise ValueError("slab_rate_pct is required for a gain taxed at the slab rate.")
    return _d(inputs.slab_rate_pct)


def capital_gains(inputs: CapitalGainsInputs, cfg: dict[str, Any]) -> TaxCalculation:
    """Illustrative capital gains tax on one sale (all rates and thresholds from config).

    Formulas (D = Decimal, each money result rounded half-up to 2 places):
      gain       = sell - buy - expenses
      long-term  = holding_months > threshold   (threshold: equity or other, from config)
      equity, long-term : applied = min(gain, max(0, exemption - already_used)); taxable = gain - applied
      everything else   : taxable = gain
      rate       = equity short -> equity_stcg_pct; equity long -> equity_ltcg_pct;
                   other long -> other_ltcg_pct; debt fund (post-2023) and other short -> the user's slab rate
      tax = taxable x rate / 100;  cess = tax x cess_pct / 100;  total = tax + cess
    A gain of zero or less gives zero tax plus an informational note; losses are not set off.
    """
    texts, fig = cfg["texts"], cfg["figures"]
    equity = inputs.asset_type in cfg["equity_asset_types"]
    slab_asset = inputs.asset_type in cfg["slab_asset_types"]
    buy, sell, expenses = _d(inputs.buy_value), _d(inputs.sell_value), _d(inputs.expenses)
    gain = sell - buy - expenses
    steps = [texts["step_gain"].format(sell=fmt(sell), buy=fmt(buy), expenses=fmt(expenses), gain=fmt(gain))]
    used = ["cess_pct"]

    threshold_key = "holding_months_equity" if equity else "holding_months_other"
    threshold = fig[threshold_key]["value"]
    if slab_asset:
        term = "slab_rate"
        steps.append(texts["step_term_slab"])
    else:
        used.append(threshold_key)
        term = "long_term" if inputs.holding_months > threshold else "short_term"
        template = texts["step_term_long" if term == "long_term" else "step_term_short"]
        steps.append(template.format(months=inputs.holding_months, threshold=threshold))

    notes: list[str] = []
    rate = Decimal(0)
    applied = Decimal(0)
    taxable = Decimal(0)
    if gain <= 0:
        notes.append(texts["loss_note"])
        steps.append(texts["step_loss"])
    else:
        if term == "slab_rate" or (term == "short_term" and not equity):
            rate = slab_rate_for(inputs)
            notes.append(texts["slab_taxed_note"])
        elif equity:
            rate_key = "equity_stcg_pct" if term == "short_term" else "equity_ltcg_pct"
            used.append(rate_key)
            rate = _d(fig[rate_key]["value"])
        else:
            used.append("other_ltcg_pct")
            rate = _d(fig["other_ltcg_pct"]["value"])
        taxable = gain
        if equity and term == "long_term":
            used.append("equity_ltcg_exemption")
            exemption, already = _d(fig["equity_ltcg_exemption"]["value"]), _d(inputs.ltcg_exemption_already_used)
            available = max(Decimal(0), exemption - already)
            applied = min(gain, available)
            taxable = gain - applied
            steps.append(texts["step_exemption"].format(exemption=fmt(exemption), used=fmt(already), available=fmt(available), applied=fmt(applied)))
            notes.append(texts["exemption_note"])
        steps.append(texts["step_taxable"].format(gain=fmt(gain), applied=fmt(applied), taxable=fmt(taxable)))

    tax = _q(taxable * rate / _HUNDRED)
    cess_rate = _d(fig["cess_pct"]["value"])
    cess = _q(tax * cess_rate / _HUNDRED)
    total = tax + cess
    if gain > 0:
        steps.append(texts["step_tax"].format(taxable=fmt(taxable), rate=plain(rate), tax=fmt(tax)))
        steps.append(texts["step_cess"].format(tax=fmt(tax), cess_rate=plain(cess_rate), cess=fmt(cess)))
        steps.append(texts["step_total"].format(tax=fmt(tax), cess=fmt(cess), total=fmt(total)))
    return TaxCalculation(
        calculation="capital_gains",
        inputs=inputs.model_dump(mode="json"),
        results={
            "gain_or_loss": _f(gain), "term": term, "exemption_applied": _f(applied), "taxable_gain": _f(taxable),
            "rate_pct": float(rate), "tax": _f(tax), "cess": _f(cess), "total_tax": _f(total),
        },
        steps=steps, notes=notes, assumptions=[],
        config_used={key: figure_entry(cfg, key) for key in used},
    )


def small_savings_growth(inputs: SmallSavingsInputs, cfg: dict[str, Any]) -> TaxCalculation:
    """Approximate growth of a small-savings deposit at the configured scheme rate.

    Formula (deposits at the start of each year, annual compounding, rate constant):
      balance_year = (balance_previous_year + deposit_that_year) x (1 + rate / 100)
    `annual` deposits the amount every year; `one_time` deposits it in year one only.
    Scheme limits and lock-ins are NOT checked here (read the official scheme rules).
    """
    key = cfg["schemes"][inputs.scheme]
    rate = _d(cfg["figures"][key]["value"])
    growth = 1 + rate / _HUNDRED
    amount = _d(inputs.amount)
    balance = Decimal(0)
    deposited = Decimal(0)
    rows: list[dict[str, float | int]] = []
    for year in range(1, inputs.years + 1):
        if inputs.mode == "annual" or year == 1:
            balance += amount
            deposited += amount
        balance *= growth
        rows.append({"year": year, "deposited": _f(deposited), "balance": _f(balance)})
    texts = cfg["texts"]
    steps = [
        texts["step_savings_rate"].format(rate=plain(rate)),
        texts["step_savings_formula"],
        texts["step_savings_year"].format(year=inputs.years, deposited=fmt(deposited), balance=fmt(balance)),
    ]
    return TaxCalculation(
        calculation="small_savings_growth",
        inputs=inputs.model_dump(mode="json"),
        results={
            "scheme": inputs.scheme, "rate_pct": float(rate), "total_deposited": _f(deposited),
            "final_value": _f(balance), "interest_earned": _f(balance - deposited), "yearly": rows,
        },
        steps=steps, notes=[texts["approximate_note"]], assumptions=list(cfg["small_savings_assumptions"]),
        config_used={key: figure_entry(cfg, key)},
    )


def summary_input(calc: TaxCalculation, cfg: dict[str, Any]) -> dict[str, Any]:
    """What the commentary LLM sees: inputs, results (without the yearly table), the rates used and the caveats."""
    return {
        "calculation": calc.calculation,
        "inputs": calc.inputs,
        "results": {k: v for k, v in calc.results.items() if k != "yearly"},
        "rates": {key: entry["value"] for key, entry in calc.config_used.items()},
        "caveats": [*calc.notes, *calc.assumptions, *all_exclusions(cfg)],
    }
