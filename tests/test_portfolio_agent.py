import json
from datetime import datetime, timezone

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agents import base_agent, portfolio
from src.agents.portfolio import PortfolioAgent
from src.core.models import MarketResult
from src.core.usage_tracker import UsageTracker, get_usage_registry
from src.data.market.provider import error_result
from src.data.tickers.resolver import build_resolver
from src.utils.config import get_config

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
SUMMARY = "The largest holding is A.NS 50.0% of the INR portfolio, with a risk label of Moderate."
PF = get_config()["portfolio"]


def quote(price, currency="INR", source="live", stale=False):
    return MarketResult(data={"price": price}, currency=currency, as_of=T0, source=source, is_stale=stale)


def prices(currency="INR"):
    """A recent history that passes the resolver's price check."""
    return MarketResult(data=[{"close": 1.0}], currency=currency, as_of=T0, source="live")


class FakeProvider:
    def __init__(self, quotes, histories=None):
        self.quotes, self.histories, self.calls = quotes, histories or {}, []

    def get_history(self, ticker, period):
        return self.histories.get(ticker, error_result("no data"))

    def get_quote(self, ticker):
        self.calls.append(ticker)
        return self.quotes.get(ticker, error_result("no data"))


class FakeLLM:
    def __init__(self, reply=SUMMARY, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return type("Msg", (), {"content": self.reply})()


@pytest.fixture(autouse=True)
def _no_real_llm_or_provider(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the real factory must not be called")

    monkeypatch.setattr(base_agent, "get_llm", fail)
    monkeypatch.setattr(portfolio, "get_market_provider", fail)


TWO = [{"ticker": "A.NS", "quantity": 10}, {"ticker": "B.NS", "quantity": 10}]
PROVIDER_TWO = {"A.NS": quote(100), "B.NS": quote(100)}


class Search:
    """Yahoo's search as a table of quotes (no network); counts its calls."""

    def __init__(self, table=None, error=None):
        self.table, self.error, self.calls = table or {}, error, []

    def __call__(self, query, limit, timeout):
        self.calls.append(query)
        if self.error:
            raise self.error
        return self.table.get(query.casefold(), [])


def run(rows, quotes=None, llm=None, histories=None, search=None):
    provider = FakeProvider(quotes or {}, histories)
    agent = PortfolioAgent(llm=llm or FakeLLM(), provider=provider, resolver=build_resolver(provider, search=search or Search()))
    return agent, agent.run({"turn_id": "t1", "portfolio": rows})


# ---- happy path ----

def test_ok_result_carries_the_summary_and_the_computed_metrics():
    agent, result = run(TWO, PROVIDER_TWO)
    assert result["status"] == "ok" and result["answer"] == SUMMARY and result["sources"] == []
    assert result["agent"] == "portfolio" and result["turn_id"] == "t1"
    inr = result["data"]["metrics"]["currencies"]["INR"]
    assert inr["total_value"] == 2000 and inr["largest_weight"] == 0.5 and inr["position_count"] == 2
    assert [p["weight"] for p in result["data"]["metrics"]["positions"]] == [0.5, 0.5]
    assert len(agent.llm.calls) == 1                                    # exactly one LLM call


def test_summary_prompt_receives_only_the_computed_metrics():
    agent, _ = run(
        [{"ticker": "A.NS", "quantity": 10, "avg_cost": 80, "expense_ratio_pct": 0.5},
         {"ticker": "B.NS", "quantity": 10}], PROVIDER_TWO)
    prompt = agent.llm.calls[0]
    assert PF["summary_prompt"].split("{metrics_json}")[0] in prompt
    sent = json.loads(prompt.split("JSON:\n", 1)[1])
    assert set(sent) == {"portfolios_by_currency", "positions", "warnings"}
    for private in ("quantity", "avg_cost", "expense_ratio_pct", "price_source"):
        assert private not in prompt                                    # raw inputs are not sent
    assert sent["positions"][0]["unrealised_pnl_pct"] == "+25.0%"       # computed in code
    assert sent["portfolios_by_currency"][0]["expense_ratio_coverage"] == "covers 50% of value, 1 of 2 holdings"
    assert sent["portfolios_by_currency"][0]["weighted_expense_ratio"] == "0.50%"


def test_mixed_currencies_are_reported_separately():
    rows = TWO + [{"ticker": "AAPL", "quantity": 1}]
    _, result = run(rows, {**PROVIDER_TWO, "AAPL": quote(200, "USD")})
    currencies = result["data"]["metrics"]["currencies"]
    assert currencies["INR"]["total_value"] == 2000 and currencies["USD"]["total_value"] == 200
    assert set(currencies) == {"INR", "USD"}


def test_missing_price_is_excluded_with_a_warning_but_the_rest_is_analysed():
    _, result = run(TWO + [{"ticker": "C.NS", "quantity": 1}], PROVIDER_TWO)
    assert result["status"] == "ok"
    assert "C.NS was left out: no price is available right now." in result["data"]["warnings"]
    assert result["data"]["metrics"]["currencies"]["INR"]["position_count"] == 2


def test_stale_prices_are_flagged_in_the_payload():
    _, result = run(TWO, {"A.NS": quote(100, source="stale cache", stale=True), "B.NS": quote(100)})
    assert result["data"]["metrics"]["uses_stale_or_sample"] is True
    assert any("not live" in w for w in result["data"]["warnings"])


def test_single_holding_is_handled():
    _, result = run([{"ticker": "A.NS", "quantity": 5}], {"A.NS": quote(100)})
    inr = result["data"]["metrics"]["currencies"]["INR"]
    assert result["status"] == "ok" and inr["diversification_score"] == 0 and inr["risk_label"] == "High"


# ---- summary failure never fails the request ----

@pytest.mark.parametrize("llm, note", [
    (FakeLLM(error=RuntimeError("down")), "unavailable"),
    (FakeLLM(reply="   "), "empty"),
    (FakeLLM(reply="The largest holding is 77.7% of the portfolio."), "contained numbers"),
])
def test_summary_failure_gives_the_fixed_fallback_with_status_ok(llm, note):
    _, result = run(TWO, PROVIDER_TWO, llm=llm)
    assert result["status"] == "ok" and result["answer"] == PF["summary_fallback"].strip()
    assert any(note in w for w in result["data"]["warnings"])
    assert result["data"]["metrics"]["currencies"]["INR"]["total_value"] == 2000   # metrics still returned


def test_summary_using_only_input_numbers_is_accepted():
    # two equal equities: HHI 0.5 -> round(100 x 0.5 / 0.95) = 53, minus the single-class penalty 15 = 38
    reply = "Total value is 2,000.00 INR. The score is 38 out of 100 and the largest holding is 50.0%."
    _, result = run(TWO, PROVIDER_TWO, llm=FakeLLM(reply=reply))
    assert result["answer"] == reply
    assert result["data"]["metrics"]["currencies"]["INR"]["diversification_score"] == 38
    assert not any("numbers" in w for w in result["data"]["warnings"])


# ---- needs_input / no_answer / errors ----

@pytest.mark.parametrize("rows", [None, [], [{"ticker": "A.NS", "quantity": 0}], [None, {"ticker": "", "quantity": 1}, {"ticker": "  ", "quantity": 1}, {"quantity": 1}, 5]])
def test_empty_or_invalid_portfolio_needs_input_and_calls_nothing(rows):
    agent, result = run(rows, PROVIDER_TWO)
    assert result["status"] == "needs_input" and result["answer"] == PF["needs_input_message"].strip()
    assert agent.provider.calls == [] and agent.llm.calls == []


def test_state_without_a_portfolio_key_needs_input():
    agent = PortfolioAgent(llm=FakeLLM(), provider=FakeProvider({}), resolver=build_resolver(FakeProvider({}), search=Search()))
    assert agent.run({"turn_id": "t1"})["status"] == "needs_input"


def test_nothing_priced_returns_no_answer_without_calling_the_llm():
    agent, result = run(TWO, {})
    assert result["status"] == "no_answer" and result["answer"] == PF["unpriced_message"].strip()
    assert len(result["data"]["warnings"]) == 2 and agent.llm.calls == []


def test_provider_exception_for_one_ticker_only_excludes_that_ticker():
    class Boom:
        def get_quote(self, ticker):
            if ticker == "B.NS":
                raise RuntimeError("provider exploded")
            return quote(100)

    agent = PortfolioAgent(llm=FakeLLM(), provider=Boom(), resolver=build_resolver(Boom(), search=Search()))
    result = agent.run({"turn_id": "t1", "portfolio": TWO})
    assert result["status"] == "ok" and result["data"]["metrics"]["currencies"]["INR"]["position_count"] == 1
    assert "B.NS was left out: no price is available right now." in result["data"]["warnings"]


def test_provider_exception_for_every_ticker_gives_no_answer():
    class Boom:
        def get_quote(self, ticker):
            raise RuntimeError("provider exploded")

    result = PortfolioAgent(llm=FakeLLM(), provider=Boom(), resolver=build_resolver(Boom(), search=Search())).run({"turn_id": "t1", "portfolio": TWO})
    assert result["status"] == "no_answer"


def test_unexpected_failure_after_valuation_still_becomes_an_error_result(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("metrics exploded")

    monkeypatch.setattr(portfolio, "build_metrics", broken)
    _, result = run(TWO, PROVIDER_TWO)
    assert result["status"] == "error" and result["answer"] == get_config()["agents"]["error_message"]


# ---- construction and usage ----

def test_default_llm_and_provider_come_from_the_factories(monkeypatch):
    seen = {}
    provider = FakeProvider({})
    monkeypatch.setattr(base_agent, "get_llm", lambda role=None, **kw: seen.setdefault("role", role) and FakeLLM())
    monkeypatch.setattr(portfolio, "get_market_provider", lambda: provider)
    agent = PortfolioAgent()
    assert seen["role"] == "portfolio_summary" and agent.provider is provider
    assert "portfolio_summary" in get_config()["llm"]["roles"]


def test_usage_is_recorded_under_the_portfolio_summary_role():
    registry = get_usage_registry()
    registry.reset()
    message = AIMessage(content=SUMMARY, usage_metadata={"input_tokens": 500, "output_tokens": 60, "total_tokens": 560})
    llm = GenericFakeChatModel(
        messages=iter([message]), callbacks=[UsageTracker("portfolio_summary", get_config()["llm"]["model"])])
    try:
        _, result = run(TWO, PROVIDER_TWO, llm=llm)
        assert result["status"] == "ok"
        assert registry.totals("portfolio_summary")["input_tokens"] == 500
        assert registry.totals("portfolio_summary")["output_tokens"] == 60
    finally:
        registry.reset()


# ---- instrument type vs asset class, end to end ----

def test_sample_style_portfolio_is_all_equity_with_the_penalty_and_not_well_diversified():
    rows = [
        {"ticker": "RELIANCE.NS", "quantity": 10},
        {"ticker": "TCS.NS", "quantity": 10},
        {"ticker": "NIFTYBEES.NS", "quantity": 10, "instrument_type": "etf", "asset_class": "equity",
         "expense_ratio_pct": 0.05},
        {"ticker": "HDFCBANK.NS", "quantity": 10},
    ]
    quotes = {r["ticker"]: quote(100) for r in rows}
    agent, result = run(rows, quotes)
    inr = result["data"]["metrics"]["currencies"]["INR"]
    assert inr["allocation_by_asset_class"] == {"equity": 1.0}
    assert inr["diversification_score"] == 79 - PF["single_class_penalty"]
    assert inr["diversification_label"] == "Moderate"
    assert inr["expense_ratio_coverage"] == "covers 25% of value, 1 of 4 holdings"
    sent = json.loads(agent.llm.calls[0].split("JSON:\n", 1)[1])
    assert sent["portfolios_by_currency"][0]["allocation_by_asset_class"] == {"equity": "100.0%"}
    assert [p["instrument_type"] for p in sent["positions"]] == ["stock", "stock", "etf", "stock"]


def test_old_style_etf_asset_class_is_rejected_with_a_warning():
    rows = [{"ticker": "A.NS", "quantity": 1, "asset_class": "etf"}, {"ticker": "B.NS", "quantity": 1}]
    _, result = run(rows, PROVIDER_TWO)
    assert any(w.startswith("Row 1 was skipped") for w in result["data"]["warnings"])


# ---- expense-ratio numbers pass the number guard; "not provided" has no coverage ----

def test_number_guard_accepts_the_expense_ratio_coverage_numbers():
    rows = [{"ticker": "A.NS", "quantity": 10, "expense_ratio_pct": 0.5}, {"ticker": "B.NS", "quantity": 10}]
    reply = "The expense ratio is 0.50%, and it covers 50% of value, 1 of 2 holdings."
    _, result = run(rows, PROVIDER_TWO, llm=FakeLLM(reply=reply))
    assert result["answer"] == reply and not any("numbers" in w for w in result["data"]["warnings"])
    wrong = "It covers 47% of value, 1 of 4 holdings."
    _, result = run(rows, PROVIDER_TWO, llm=FakeLLM(reply=wrong))
    assert result["answer"] == PF["summary_fallback"].strip()


def test_not_provided_expense_ratio_is_sent_without_any_coverage_fraction():
    agent, _ = run(TWO, PROVIDER_TWO)
    prompt = agent.llm.calls[0]
    sent = json.loads(prompt.split("JSON:\n", 1)[1])["portfolios_by_currency"][0]
    assert sent["weighted_expense_ratio"] == "not provided" and "expense_ratio_coverage" not in sent
    assert "covers" not in prompt.split("JSON:\n", 1)[1]
    instructions = " ".join(PF["summary_prompt"].split())   # ignore the YAML line breaks
    assert "not provided" in instructions and "give no coverage figures" in instructions


# ---- company names in the ticker column (UI step 3a) ----

def q(symbol, name, exchange, display, kind="EQUITY"):
    return {"symbol": symbol, "longname": name, "exchange": exchange, "exchDisp": display, "quoteType": kind}


TABLE = {
    "suzlon": [q("SUZLON.NS", "Suzlon Energy Limited", "NSI", "NSE")],
    "globe corp": [q("GLB", "Globe Corp.", "NMS", "NASDAQ"), q("GLBA", "Globe Corp.", "NMS", "NASDAQ")],
}
QUOTES = {"SUZLON.NS": quote(50), "INFY.NS": quote(100), "A.NS": quote(100), "GLB": quote(20, "USD"), "GLBA": quote(30, "USD")}
HISTORIES = {s: prices("USD" if s.startswith("GLB") else "INR") for s in QUOTES}


def test_a_company_name_is_resolved_and_the_answer_says_which_symbol_it_became():
    search = Search(TABLE)
    agent, result = run([{"ticker": "Suzlon", "quantity": 10}, {"ticker": "A.NS", "quantity": 10}], QUOTES, histories=HISTORIES, search=search)
    assert result["status"] == "ok" and [p["ticker"] for p in result["data"]["metrics"]["positions"]] == ["SUZLON.NS", "A.NS"]
    note = 'Showing SUZLON.NS (Suzlon Energy Limited, NSE, INR) for "Suzlon".'
    assert result["data"]["lookup_notes"] == [note] and result["answer"].endswith(f"\n\n{note}") and result["answer"].count(note) == 1
    assert note not in agent.llm.calls[0] and search.calls == ["Suzlon"]                  # symbols are never looked up


def test_an_alias_name_shows_a_plain_note_when_the_lookup_had_none():
    _, result = run([{"ticker": "Infosys", "quantity": 1}], QUOTES, histories=HISTORIES)
    assert result["data"]["lookup_notes"] == ['Row "Infosys": using INFY.NS.'] and result["answer"].endswith('Row "Infosys": using INFY.NS.')


def test_every_problem_row_is_reported_at_once_and_nothing_is_analysed():
    rows = [{"ticker": "Suzlon", "quantity": 1}, {"ticker": "Globe Corp", "quantity": 1}, {"ticker": "Qzxv Wplk", "quantity": 1}, {"ticker": "???", "quantity": 1},
            {"ticker": "A.NS", "quantity": 1}]
    agent, result = run(rows, QUOTES, histories=HISTORIES, search=Search(TABLE))
    assert result["status"] == "needs_input"
    lines = result["answer"].split("\n\n")
    assert len(lines) == 3 and lines[0].startswith('Row "Globe Corp": I found more than one match for "Globe Corp": GLB (Globe Corp., NASDAQ, USD); GLBA')
    assert lines[1].startswith('Row "Qzxv Wplk": I couldn\'t find a listed company called "Qzxv Wplk"') and lines[2].startswith('Row "???": I couldn\'t read that company name')
    assert agent.provider.calls == [] and agent.llm.calls == [] and "Suzlon" not in result["answer"]       # no quotes asked, no summary written


def test_a_lookup_that_is_down_names_every_name_row_and_never_guesses():
    rows = [{"ticker": "Suzlon", "quantity": 1}, {"ticker": "Zeta Corp", "quantity": 1}, {"ticker": "A.NS", "quantity": 1}]
    agent, result = run(rows, QUOTES, histories=HISTORIES, search=Search(error=ConnectionError("offline")))
    assert result["status"] == "needs_input" and result["answer"].count("can't look up company names") == 2
    assert 'Row "Suzlon":' in result["answer"] and 'Row "Zeta Corp":' in result["answer"] and agent.provider.calls == []


def test_at_most_max_holdings_names_are_looked_up_and_the_rest_are_left_out_with_a_warning(monkeypatch):
    limited = {**get_config(), "portfolio": {**PF, "max_holdings": 2}}
    monkeypatch.setattr(portfolio, "get_config", lambda: limited)
    search = Search({"zetacorp": [q("ZETA.NS", "Zetacorp Limited", "NSI", "NSE")], "etaworks": [q("ETA.NS", "Etaworks Limited", "NSI", "NSE")], "thetacorp": [q("THE.NS", "Thetacorp Limited", "NSI", "NSE")]})
    histories = {s: prices() for s in ("ZETA.NS", "ETA.NS", "THE.NS")}
    quotes = {s: quote(10) for s in histories}
    rows = [{"ticker": "Zetacorp", "quantity": 1}, {"ticker": "Etaworks", "quantity": 1}, {"ticker": "Thetacorp", "quantity": 1}]
    _, result = run(rows, quotes, histories=histories, search=search)
    assert search.calls == ["Zetacorp", "Etaworks"] and result["status"] == "ok"
    assert [p["ticker"] for p in result["data"]["metrics"]["positions"]] == ["ZETA.NS", "ETA.NS"]
    assert 'Only the first 2 rows are looked up; "Thetacorp" was left out.' in result["data"]["warnings"]


def test_resolved_rows_keep_their_order_and_a_name_that_matches_a_symbol_row_is_a_duplicate():
    rows = [{"ticker": "Infosys", "quantity": 1}, {"ticker": "A.NS", "quantity": 2}, {"ticker": "INFY.NS", "quantity": 3}]
    _, result = run(rows, QUOTES, histories=HISTORIES)
    assert [p["ticker"] for p in result["data"]["metrics"]["positions"]] == ["INFY.NS", "A.NS"]
    assert "INFY.NS appears more than once; only the first row is used." in result["data"]["warnings"]


def numbers_in_payload(value):
    if isinstance(value, dict):
        return [n for v in value.values() for n in numbers_in_payload(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in numbers_in_payload(v)]
    return [float(value)] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def test_two_currencies_are_never_added_together():
    rows = [{"ticker": "A.NS", "quantity": 10}, {"ticker": "GLB", "quantity": 5}, {"ticker": "GLBA", "quantity": 5}]
    _, result = run(rows, QUOTES)
    metrics = result["data"]["metrics"]
    assert sorted(metrics["currencies"]) == ["INR", "USD"]
    assert metrics["currencies"]["INR"]["total_value"] == 1000 and metrics["currencies"]["USD"]["total_value"] == 250
    assert 1250.0 not in numbers_in_payload(result["data"])                                           # 1000 INR + 250 USD, never summed
    assert [key for key in metrics if "total" in key] == [] and [key for key in result["data"] if "total" in key] == []
    assert all(p["weight"] <= 1 for p in metrics["positions"]) and sum(c["allocation_by_asset_class"]["equity"] for c in metrics["currencies"].values()) == 2


def test_the_payload_carries_the_freshness_of_the_prices():
    _, live = run(TWO, PROVIDER_TWO)
    assert live["data"]["source"] == "live" and live["data"]["is_stale"] is False and live["data"]["as_of"] == T0.isoformat()
    assert "lookup_notes" not in live["data"]
    _, mixed = run(TWO, {"A.NS": quote(100), "B.NS": quote(100, source="stale cache", stale=True)})
    assert mixed["data"]["source"] == "stale cache" and mixed["data"]["is_stale"] is True
