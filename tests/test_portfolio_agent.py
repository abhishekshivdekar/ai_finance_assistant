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
from src.utils.config import get_config

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
SUMMARY = "The largest holding is A.NS 50.0% of the INR portfolio, with a risk label of Moderate."
PF = get_config()["portfolio"]


def quote(price, currency="INR", source="live", stale=False):
    return MarketResult(data={"price": price}, currency=currency, as_of=T0, source=source, is_stale=stale)


class FakeProvider:
    def __init__(self, quotes):
        self.quotes, self.calls = quotes, []

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


def run(rows, quotes=None, llm=None):
    agent = PortfolioAgent(llm=llm or FakeLLM(), provider=FakeProvider(quotes or {}))
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
    assert sent["portfolios_by_currency"][0]["expense_ratio_coverage"] == "based on 1 of 2 holdings"
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

@pytest.mark.parametrize("rows", [None, [], [{"ticker": "A.NS", "quantity": 0}], [{"ticker": "???", "quantity": 1}]])
def test_empty_or_invalid_portfolio_needs_input_and_calls_nothing(rows):
    agent, result = run(rows, PROVIDER_TWO)
    assert result["status"] == "needs_input" and result["answer"] == PF["needs_input_message"].strip()
    assert agent.provider.calls == [] and agent.llm.calls == []


def test_state_without_a_portfolio_key_needs_input():
    agent = PortfolioAgent(llm=FakeLLM(), provider=FakeProvider({}))
    assert agent.run({"turn_id": "t1"})["status"] == "needs_input"


def test_nothing_priced_returns_no_answer_without_calling_the_llm():
    agent, result = run(TWO, {})
    assert result["status"] == "no_answer" and result["answer"] == PF["unpriced_message"].strip()
    assert len(result["data"]["warnings"]) == 2 and agent.llm.calls == []


def test_provider_exception_becomes_an_error_result():
    class Boom:
        def get_quote(self, ticker):
            raise RuntimeError("provider exploded")

    agent = PortfolioAgent(llm=FakeLLM(), provider=Boom())
    result = agent.run({"turn_id": "t1", "portfolio": TWO})
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
