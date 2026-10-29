from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.core import portfolio_metrics as pm
from src.core.models import MarketResult
from src.core.portfolio_models import Holding, Position
from src.core.portfolio_valuation import parse_holdings, value_holdings
from src.data.market.provider import error_result
from src.utils.config import get_config

CFG = get_config()["portfolio"]
T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 5, tzinfo=timezone.utc)


def pos(ticker="A.NS", value=100.0, asset_class="equity", currency="INR", er=None, instrument_type="stock", **extra):
    return Position(
        ticker=ticker, instrument_type=instrument_type, asset_class=asset_class, quantity=1, avg_cost=None, expense_ratio_pct=er,
        currency=currency, price=value, value=value, **extra,
    )


# ---- models ----

def test_holding_defaults_and_normalised_ticker():
    h = Holding(ticker=" reliance.ns ", quantity=5)
    assert h.ticker == "RELIANCE.NS" and h.asset_class == "equity" and h.instrument_type == "stock"
    assert h.avg_cost is None and h.expense_ratio_pct is None


@pytest.mark.parametrize("kwargs", [
    {"quantity": 0}, {"quantity": -1}, {"avg_cost": 0}, {"expense_ratio_pct": -0.1},
    {"asset_class": "crypto"}, {"asset_class": "etf"}, {"instrument_type": "crypto"},
    {"instrument_type": "equity"}, {"ticker": "???"}, {"ticker": ""}, {"currency": "INR"},
])
def test_holding_rejects_bad_input(kwargs):
    base = {"ticker": "AAPL", "quantity": 1}
    with pytest.raises(ValidationError):
        Holding(**{**base, **kwargs})


def test_models_are_immutable():
    h = Holding(ticker="AAPL", quantity=1)
    with pytest.raises(ValidationError):
        h.quantity = 2
    with pytest.raises(ValidationError):
        pos().__setattr__("value", 1.0)


# ---- weights, P&L, allocation, concentration ----

def test_position_weights():
    assert pm.position_weights([1, 1, 2]) == [0.25, 0.25, 0.5]
    assert pm.position_weights([]) == [] and pm.position_weights([0, 0]) == [0.0, 0.0]


def test_unrealised_pnl_only_with_avg_cost():
    pnl, pct = pm.unrealised_pnl(110, 10, 100)
    assert pnl == pytest.approx(100) and pct == pytest.approx(10)
    assert pm.unrealised_pnl(90, 10, 100)[1] == pytest.approx(-10)
    assert pm.unrealised_pnl(110, 10, None) == (None, None)


def test_allocation_by_asset_class_sums_to_one():
    alloc = pm.allocation_by_asset_class([pos(value=60), pos("B.NS", 30, "debt"), pos("C.NS", 10, "equity")])
    assert alloc == pytest.approx({"equity": 0.7, "debt": 0.3}) and sum(alloc.values()) == pytest.approx(1)
    assert pm.allocation_by_asset_class([pos(value=0)]) == {"equity": 0.0}


def test_allocation_uses_asset_class_not_instrument_type():
    alloc = pm.allocation_by_asset_class([
        pos("A.NS", 50, "equity", instrument_type="stock"),
        pos("B.NS", 30, "equity", instrument_type="etf"),
        pos("C.NS", 20, "debt", instrument_type="etf"),       # a debt ETF is still debt
    ])
    assert alloc == pytest.approx({"equity": 0.8, "debt": 0.2})


def test_concentration():
    assert pm.concentration([0.5, 0.3, 0.15, 0.05]) == (0.5, pytest.approx(0.95))
    assert pm.concentration([1.0]) == (1.0, 1.0)
    assert pm.concentration([]) == (0.0, 0.0)


def test_hhi():
    assert pm.hhi([1.0]) == 1.0
    assert pm.hhi([0.25] * 4) == pytest.approx(0.25)
    assert pm.hhi([]) == 0.0


# ---- diversification score and label ----

def test_diversification_score_edge_cases():
    assert pm.diversification_score([1.0], True, CFG) == 0           # one holding, floored at 0
    assert pm.diversification_score([1.0], False, CFG) == 0
    assert pm.diversification_score([0.2] * 5, False, CFG) == 84     # round(100 x 0.8 / 0.95)
    assert pm.diversification_score([0.2] * 5, True, CFG) == 69      # minus the single-class penalty
    assert pm.diversification_score([0.05] * 20, False, CFG) == 100
    assert pm.diversification_score([1 / 40] * 40, False, CFG) == 100   # clamped above N_full
    assert pm.diversification_score([1 / 40] * 40, True, CFG) == 100 - CFG["single_class_penalty"]
    assert pm.diversification_score([0.95, 0.05], False, CFG) == 10  # round(100 x 0.095 / 0.95)
    assert pm.diversification_score([0.95, 0.05], True, CFG) == 0    # penalty cannot go below 0


def test_diversification_score_handles_a_tiny_n_full():
    assert pm.diversification_score([0.5, 0.5], False, {**CFG, "full_diversification_holdings": 1}) == 100


@pytest.mark.parametrize("score, label", [
    (0, "Concentrated"), (39, "Concentrated"), (40, "Moderate"), (69, "Moderate"),
    (70, "Well diversified"), (100, "Well diversified"),
])
def test_diversification_label_bands(score, label):
    assert pm.diversification_label(score, CFG) == label


# ---- equity share and risk label ----

def test_equity_share_counts_the_configured_classes():
    positions = [pos(value=40, asset_class="equity"), pos("B.NS", 20, "equity", instrument_type="etf"),
                 pos("C.NS", 30, "debt", instrument_type="etf"), pos("D.NS", 10, "cash")]
    assert pm.equity_share(positions, CFG) == pytest.approx(0.6)   # an ETF counts by its asset class
    assert pm.equity_share([pos(value=10, asset_class="other"), pos("B.NS", 10, "gold")], CFG) == pytest.approx(0.5)
    assert pm.equity_share([pos(value=0)], CFG) == 0.0


@pytest.mark.parametrize("equity, largest, label", [
    (0.0, 0.10, "Low"), (0.39, 0.29, "Low"),
    (0.40, 0.10, "Moderate"), (0.69, 0.29, "Moderate"), (0.10, 0.30, "Moderate"),
    (0.70, 0.10, "High"), (0.10, 0.50, "High"), (0.90, 0.60, "High"), (0.45, 0.55, "High"),
])
def test_risk_label_thresholds(equity, largest, label):
    assert pm.risk_label(equity, largest, CFG) == label


# ---- expense ratio ----

def test_weighted_expense_ratio_uses_only_holdings_that_have_one():
    positions = [pos("A.NS", 300, er=0.2), pos("B.NS", 100, er=1.0), pos("C.NS", 600)]
    average, with_ratio, total = pm.weighted_expense_ratio(positions)
    assert average == pytest.approx(0.75 * 0.2 + 0.25 * 1.0) and (with_ratio, total) == (2, 3)
    assert pm.weighted_expense_ratio([pos("C.NS", 600)]) == (None, 0, 1)


def test_currency_metrics_composes_everything_and_reports_coverage():
    cm = pm.currency_metrics("INR", [pos("A.NS", 300, er=0.2), pos("B.NS", 100, "debt", er=1.0), pos("C.NS", 600)], CFG)
    assert cm.largest_ticker == "C.NS" and cm.largest_weight == pytest.approx(0.6)
    assert cm.top3_weight == pytest.approx(1.0) and cm.position_count == 3
    assert cm.expense_ratio_coverage == "covers 40% of value, 2 of 3 holdings"
    assert cm.expense_ratio_value_share == pytest.approx(0.4)
    assert cm.risk_label == "High"                                   # largest position >= 50%
    assert cm.diversification_score == pm.diversification_score([0.3, 0.1, 0.6], False, CFG)


# ---- build_metrics: per currency, never blended ----

def test_weights_sum_to_one_per_currency_and_currencies_are_not_summed():
    positions = [pos("A.NS", 300), pos("B.NS", 100), pos("AAPL", 50, currency="USD")]
    m = pm.build_metrics(positions, [], CFG)
    for currency in ("INR", "USD"):
        assert sum(p.weight for p in m.positions if p.currency == currency) == pytest.approx(1.0)
    assert m.currencies["INR"].total_value == 400 and m.currencies["USD"].total_value == 50
    assert set(m.currencies) == {"INR", "USD"}
    assert "The USD portfolio has a single holding (AAPL)." in m.warnings


def test_concentration_warning_uses_observation_wording():
    m = pm.build_metrics([pos("A.NS", 300), pos("B.NS", 100)], [], CFG)
    assert "A.NS is 75.0% of the INR portfolio." in m.warnings
    quiet = pm.build_metrics([pos("A.NS", 50), pos("B.NS", 50)], ["kept"], {**CFG, "concentration_warning": 0.6})
    assert quiet.warnings == ["kept"]


def test_build_metrics_flags_stale_and_reports_the_oldest_price_time():
    m = pm.build_metrics([pos("A.NS", 1, price_stale=True, as_of=T1), pos("B.NS", 1, as_of=T0)], [], CFG)
    assert m.uses_stale_or_sample is True and m.as_of == T0
    fresh = pm.build_metrics([pos("A.NS", 1)], [], CFG)
    assert fresh.uses_stale_or_sample is False and fresh.as_of is None


def test_build_metrics_reads_config_when_not_given():
    assert pm.build_metrics([pos()], []).currencies["INR"].diversification_score == 0


def test_single_holding_is_handled():
    cm = pm.build_metrics([pos("A.NS", 500)], [], CFG).currencies["INR"]
    assert cm.diversification_score == 0 and cm.diversification_label == "Concentrated"
    assert cm.risk_label == "High" and cm.largest_weight == 1.0 and cm.hhi == 1.0


# ---- summary input and number helpers ----

def test_summary_input_is_formatted_strings_only_for_numbers():
    positions = [pos("A.NS", 300, er=0.2, pnl_pct=12.34, pnl=1.0), pos("B.NS", 100, "debt", instrument_type="bond")]
    payload = pm.summary_input(pm.build_metrics(positions, [], CFG))
    inr = payload["portfolios_by_currency"][0]
    assert inr["total_value"] == "400.00" and inr["largest_position"] == "A.NS 75.0%"
    assert inr["diversification_score"].endswith("out of 100") and inr["weighted_expense_ratio"] == "0.20%"
    assert inr["expense_ratio_coverage"] == "covers 75% of value, 1 of 2 holdings"
    assert payload["positions"][1]["instrument_type"] == "bond" and payload["positions"][1]["asset_class"] == "debt"
    assert payload["positions"][0]["unrealised_pnl_pct"] == "+12.3%"
    assert payload["positions"][1]["unrealised_pnl_pct"] is None
    none = pm.summary_input(pm.build_metrics([pos()], [], CFG))["portfolios_by_currency"][0]
    assert none["weighted_expense_ratio"] == "not provided"
    assert "expense_ratio_coverage" not in none                  # "not provided" has no coverage fraction


def test_number_helpers():
    assert pm.numbers_in("1,234.50 and 42% of 7") == {1234.5, 42.0, 7.0}
    assert pm.numbers_in("no digits") == set()
    nested = {"a": ["12.5%", {"b": 3}], "n9": None, "flag": True, "empty": {}, "none": []}
    assert pm.collect_numbers(nested) == {12.5, 3.0}                 # keys, None and bools ignored
    assert pm.collect_numbers({}) == set() and pm.collect_numbers([]) == set()


# ---- parse_holdings ----

@pytest.mark.parametrize("field", ["quantity", "avg_cost", "expense_ratio_pct"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 10 ** 400])
def test_parse_holdings_skips_infinite_nan_and_huge_numbers(field, value):
    holdings, warnings = parse_holdings([{"ticker": "AAPL", "quantity": 1, field: value}], 5)
    assert holdings == [] and warnings[0].startswith("Row 1 was skipped")


def test_parse_holdings_skips_bad_rows_duplicates_and_extras():
    rows = [
        {"ticker": "A.NS", "quantity": 1},
        {"ticker": "B.NS", "quantity": 0},                # invalid
        {"ticker": "a.ns", "quantity": 3},                # duplicate of the first
        Holding(ticker="AAPL", quantity=2),               # already a Holding
        {"ticker": "TCS.NS", "quantity": 1},              # over the cap
    ]
    holdings, warnings = parse_holdings(rows, 2)
    assert [h.ticker for h in holdings] == ["A.NS", "AAPL"]
    assert any(w.startswith("Row 2 was skipped") for w in warnings)
    assert "A.NS appears more than once; only the first row is used." in warnings
    assert "Only the first 2 holdings are analysed; TCS.NS was left out." in warnings
    assert parse_holdings([], 5) == ([], [])


# ---- valuation with a fake provider ----

def quote(price=100.0, currency="INR", source="live", stale=False, as_of=T0):
    return MarketResult(data={"ticker": "X", "price": price}, currency=currency, as_of=as_of,
                        source=source, is_stale=stale)


class FakeProvider:
    def __init__(self, quotes):
        self.quotes, self.calls = quotes, []

    def get_quote(self, ticker):
        self.calls.append(ticker)
        return self.quotes.get(ticker, error_result(f"No data for {ticker}"))


def holdings(*rows):
    return parse_holdings(list(rows), 50)[0]


def test_value_holdings_values_positions_in_their_own_currency():
    provider = FakeProvider({"A.NS": quote(200), "AAPL": quote(10, "USD")})
    positions, warnings = value_holdings(holdings(
        {"ticker": "A.NS", "quantity": 3, "avg_cost": 150}, {"ticker": "AAPL", "quantity": 2}), provider)
    assert [(p.ticker, p.currency, p.value) for p in positions] == [("A.NS", "INR", 600), ("AAPL", "USD", 20)]
    assert positions[0].pnl == pytest.approx(150) and positions[0].pnl_pct == pytest.approx(100 / 3)
    assert positions[1].pnl is None and positions[1].pnl_pct is None    # no avg_cost -> no P&L
    assert warnings == []


@pytest.mark.parametrize("bad", [
    error_result("down"),
    MarketResult(data={"price": 0}, currency="INR", as_of=None, source="live"),
    MarketResult(data={"price": float("nan")}, currency="INR", as_of=None, source="live"),
    MarketResult(data={"price": True}, currency="INR", as_of=None, source="live"),
    MarketResult(data={"price": "12"}, currency="INR", as_of=None, source="live"),
    MarketResult(data=None, currency="INR", as_of=None, source="live"),
    MarketResult(data={"price": 50.0}, currency=None, as_of=None, source="live"),
])
def test_unpriced_holding_is_excluded_with_a_warning_and_the_rest_is_kept(bad):
    provider = FakeProvider({"A.NS": quote(200), "B.NS": bad})
    positions, warnings = value_holdings(holdings(
        {"ticker": "A.NS", "quantity": 1}, {"ticker": "B.NS", "quantity": 1}), provider)
    assert [p.ticker for p in positions] == ["A.NS"]
    assert warnings == ["B.NS was left out: no price is available right now."]


def test_stale_and_sample_prices_are_flagged():
    provider = FakeProvider({
        "A.NS": quote(1, source="stale cache", stale=True),
        "B.NS": quote(1, source="sample data"),
        "C.NS": quote(1, source="cache"),
        "D.NS": quote(1, source="live", stale=True),
    })
    positions, warnings = value_holdings(
        holdings(*({"ticker": t, "quantity": 1} for t in ("A.NS", "B.NS", "C.NS", "D.NS"))), provider)
    assert [p.price_stale for p in positions] == [True, True, False, True]
    assert warnings == ["These prices are not live (stale or sample data): "
                        "A.NS (stale cache), B.NS (sample data), D.NS (live)."]


# ---- expense-ratio coverage text ----

def test_expense_ratio_coverage_reports_value_share_and_holding_count():
    positions = [pos("A.NS", 47, er=0.2), pos("B.NS", 53), pos("C.NS", 0.0001), pos("D.NS", 0.0001)]
    assert pm.expense_ratio_value_share(positions) == pytest.approx(0.47, abs=1e-3)
    assert pm.expense_ratio_coverage(0.47, 1, 4) == "covers 47% of value, 1 of 4 holdings"
    assert pm.expense_ratio_coverage(1.0, 2, 2) == "covers 100% of value, 2 of 2 holdings"
    assert pm.expense_ratio_coverage(0.0, 0, 4) == "not provided"      # no coverage fraction at all
    assert pm.expense_ratio_value_share([pos(value=0)]) == 0.0 and pm.expense_ratio_value_share([]) == 0.0
    none = pm.currency_metrics("INR", [pos("A.NS"), pos("B.NS")], CFG)
    assert none.expense_ratio_coverage == "not provided" and none.expense_ratio_weighted_pct is None


# ---- four holdings, all equity, one of them an ETF ----

def test_four_equity_holdings_with_an_etf_get_the_single_class_penalty_and_are_not_well_diversified():
    positions = [
        pos("RELIANCE.NS", 250), pos("TCS.NS", 250),
        pos("NIFTYBEES.NS", 250, instrument_type="etf"), pos("HDFCBANK.NS", 250),
    ]
    cm = pm.build_metrics(positions, [], CFG).currencies["INR"]
    assert cm.allocation_by_asset_class == {"equity": 1.0}                        # the ETF is equity
    without_penalty = pm.diversification_score([0.25] * 4, False, CFG)
    assert without_penalty == 79                                                  # round(100 x 0.75 / 0.95)
    assert cm.diversification_score == without_penalty - CFG["single_class_penalty"] == 64
    assert cm.diversification_label == "Moderate" and cm.diversification_label != "Well diversified"
    assert cm.equity_share == 1.0 and cm.risk_label == "High"                     # all equity


# ---- concurrent quotes ----

import threading
import time


class SlowProvider:
    """Fake provider that records how many quote calls run at the same time."""

    def __init__(self, delays=None, fail=(), price=10.0):
        self.delays, self.fail, self.price = delays or {}, set(fail), price
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def get_quote(self, ticker):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delays.get(ticker, 0.03))
            if ticker in self.fail:
                raise RuntimeError("boom " + ticker)
            return quote(self.price)
        finally:
            with self._lock:
                self.active -= 1


TICKERS = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS", "F.NS"]


def six():
    return holdings(*({"ticker": t, "quantity": 1} for t in TICKERS))


def test_quotes_are_fetched_concurrently_but_bounded_by_max_workers():
    wide = SlowProvider()
    value_holdings(six(), wide, max_workers=6)
    assert wide.max_active > 1                                       # really concurrent
    bounded = SlowProvider()
    value_holdings(six(), bounded, max_workers=2)
    assert 1 < bounded.max_active <= 2
    for workers in (1, 0):                                           # 0 is treated as 1
        serial = SlowProvider()
        value_holdings(six(), serial, max_workers=workers)
        assert serial.max_active == 1


def test_results_keep_input_order_even_when_early_quotes_are_slower():
    delays = {t: 0.02 * (len(TICKERS) - i) for i, t in enumerate(TICKERS)}   # first ticker is the slowest
    positions, warnings = value_holdings(six(), SlowProvider(delays), max_workers=6)
    assert [p.ticker for p in positions] == TICKERS and warnings == []


def test_one_failing_ticker_does_not_affect_the_others():
    provider = SlowProvider(fail={"C.NS", "E.NS"})
    positions, warnings = value_holdings(six(), provider, max_workers=6)
    assert [p.ticker for p in positions] == ["A.NS", "B.NS", "D.NS", "F.NS"]
    assert warnings == ["C.NS was left out: no price is available right now.",
                        "E.NS was left out: no price is available right now."]


def test_default_workers_come_from_config_and_empty_input_is_fine():
    assert get_config()["portfolio"]["max_workers"] >= 1
    positions, warnings = value_holdings(six(), SlowProvider())
    assert len(positions) == 6 and warnings == []
    assert value_holdings([], SlowProvider()) == ([], [])
