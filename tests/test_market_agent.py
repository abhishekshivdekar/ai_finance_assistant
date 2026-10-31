import json
from datetime import date, datetime, timedelta, timezone

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agents import base_agent, market
from src.agents.market import MarketAgent
from src.core.models import MarketResult
from src.core.usage_tracker import UsageTracker, get_usage_registry
from src.data.market.provider import error_result
from src.data.tickers.resolver import build_resolver
from src.utils.config import get_config

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
CFG = get_config()["market_analysis"]
SUMMARY = "The latest price is 100.00, above its 20-day average and near its 52-week high."


def rows(closes, start=date(2025, 1, 1)):
    """Provider-style history rows, one per calendar day."""
    return [{"date": (start + timedelta(days=i)).isoformat(), "open": c, "high": c, "low": c, "close": float(c),
             "volume": 1.0} for i, c in enumerate(closes)]


def history(closes, currency="USD", source="live", stale=False):
    return MarketResult(data=rows(closes), currency=currency, as_of=T0, source=source, is_stale=stale)


RISING = [(i + 1) * 100 / 360 for i in range(360)]   # 360 days, rising to exactly 100
FALLING = list(range(460, 100, -1))      # 360 days, ends at 101


class FakeProvider:
    def __init__(self, histories):
        self.histories, self.calls = histories, []

    def get_history(self, ticker, period):
        self.calls.append((ticker, period))
        return self.histories.get(ticker, error_result("no data"))


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
    monkeypatch.setattr(market, "get_market_provider", fail)


def offline(provider, quotes=None):
    """A resolver whose Yahoo search is a table of scripted quotes (no network); the alias table is the real one."""
    table = quotes or {}
    return build_resolver(provider, search=lambda query, limit, timeout: table.get(query.casefold(), []))


def run(tickers, histories=None, llm=None, period="6mo", quotes=None):
    provider = FakeProvider(histories or {})
    agent = MarketAgent(llm=llm or FakeLLM(), provider=provider, resolver=offline(provider, quotes))
    params = {"tickers": tickers, "period": period}
    return agent, agent.run({"turn_id": "t1", "market_params": params})


TWO = {"AAA": history(RISING, "INR"), "BBB": history(FALLING, "USD")}


# ---- happy path ----

def test_ok_result_carries_the_summary_and_the_computed_metrics():
    agent, result = run(["AAA", "BBB"], TWO)
    assert result["status"] == "ok" and result["answer"] == SUMMARY and result["sources"] == []
    assert result["agent"] == "market" and result["turn_id"] == "t1"
    aaa, bbb = result["data"]["metrics"]["tickers"]
    assert (aaa["ticker"], aaa["currency"], aaa["latest_price"], aaa["trend"]) == ("AAA", "INR", 100.0, "Uptrend")
    assert (bbb["ticker"], bbb["currency"], bbb["trend"]) == ("BBB", "USD", "Downtrend")
    assert len(agent.llm.calls) == 1                                    # exactly one LLM call


def test_each_ticker_is_fetched_once_for_the_full_year_whatever_the_period():
    for period in ("1mo", "1y"):
        agent, _ = run(["AAA", "BBB"], TWO, period=period)
        assert sorted(agent.provider.calls) == [("AAA", CFG["fetch_period"]), ("BBB", CFG["fetch_period"])]


def test_the_period_slices_the_returns_but_not_the_averages():
    _, month = run(["AAA"], TWO, period="1mo")
    _, year = run(["AAA"], TWO, period="1y")
    m, y = month["data"]["metrics"]["tickers"][0], year["data"]["metrics"]["tickers"][0]
    assert m["period"] == "1mo" and y["period"] == "1y"
    assert m["period_return_pct"] < y["period_return_pct"]
    assert m["averages"] == y["averages"] and m["range_52w"] == y["range_52w"]
    assert len(month["data"]["history"]) == 31 and len(year["data"]["history"]) == 360


def test_several_tickers_get_a_comparison_table_without_ranking():
    _, result = run(["AAA", "BBB"], TWO)
    table = result["data"]["comparison"]
    assert [(r["ticker"], r["currency"]) for r in table] == [("AAA", "INR"), ("BBB", "USD")]
    assert "best" not in json.dumps(table).lower() and "rank" not in json.dumps(table).lower()
    _, single = run(["AAA"], TWO)
    assert single["data"]["comparison"] == []


def test_currencies_stay_separate_and_nothing_is_added_across_tickers():
    _, result = run(["AAA", "BBB"], TWO)
    assert {t["currency"] for t in result["data"]["metrics"]["tickers"]} == {"INR", "USD"}
    assert "total" not in json.dumps(result["data"]).lower()


# ---- what the LLM sees ----

def test_summary_prompt_receives_only_the_computed_metrics():
    agent, _ = run(["AAA", "BBB"], TWO)
    prompt = agent.llm.calls[0]
    assert CFG["summary_prompt"].split("{metrics_json}")[0] in prompt
    sent = json.loads(prompt.split("JSON:\n", 1)[1])
    assert set(sent) == {"analysis_period", "tickers", "warnings"} and sent["analysis_period"] == "6 months"
    assert [t["ticker"] for t in sent["tickers"]] == ["AAA", "BBB"]
    assert sent["tickers"][0]["latest_price"] == "100.00"
    assert "2025-" not in prompt and '"close"' not in prompt            # no raw history rows or dates


def test_label_numbers_such_as_20_day_and_52_week_pass_the_number_guard():
    _, result = run(["AAA"], TWO)
    assert result["answer"] == SUMMARY and not any("numbers" in w for w in result["data"]["warnings"])


# ---- unavailable data ----

def test_unavailable_ticker_is_a_warning_and_the_other_is_still_analysed():
    agent, result = run(["AAA", "ZZZ"], TWO)
    assert result["status"] == "ok" and [t["ticker"] for t in result["data"]["metrics"]["tickers"]] == ["AAA"]
    assert "ZZZ was left out: no price history is available right now." in result["data"]["warnings"]
    assert len(agent.llm.calls) == 1


def test_all_unavailable_gives_no_answer_without_calling_the_llm():
    agent, result = run(["YYY", "ZZZ"], {})
    assert result["status"] == "no_answer" and result["answer"] == CFG["unavailable_message"].strip()
    assert len(result["data"]["warnings"]) == 2 and agent.llm.calls == []


def test_history_without_currency_or_with_one_price_is_left_out():
    histories = {"AAA": history(RISING, currency=None), "BBB": history([100]), "CCC": history(RISING)}
    agent, result = run(["AAA", "BBB", "CCC"], histories)
    assert [t["ticker"] for t in result["data"]["metrics"]["tickers"]] == ["CCC"]
    assert sum("was left out: no price history" in w for w in result["data"]["warnings"]) == 2


def test_provider_exception_for_every_ticker_gives_no_answer():
    class Boom:
        def get_history(self, ticker, period):
            raise RuntimeError("provider exploded")

    agent = MarketAgent(llm=FakeLLM(), provider=Boom(), resolver=offline(Boom()))
    result = agent.run({"turn_id": "t1", "market_params": {"tickers": ["AAA"]}})
    assert result["status"] == "no_answer" and agent.llm.calls == []


def test_no_valid_ticker_needs_input_and_makes_no_calls():
    agent, result = run([7], TWO)
    assert result["status"] == "needs_input" and result["answer"] == CFG["needs_input_message"].strip()
    assert agent.provider.calls == [] and agent.llm.calls == []
    assert result["data"]["warnings"] == ["'7' was skipped: it is not a valid ticker."]
    empty = MarketAgent(llm=FakeLLM(), provider=FakeProvider({}), resolver=offline(FakeProvider({}))).run({"turn_id": "t1"})
    assert empty["status"] == "needs_input" and empty["answer"] == CFG["needs_input_message"].strip()
    unreadable = run(["???"], TWO)[1]                                   # not a symbol and not a readable name
    assert unreadable["status"] == "needs_input" and unreadable["answer"] == get_config()["ticker_lookup"]["messages"]["invalid"]


# ---- freshness ----

def test_stale_and_sample_prices_are_flagged():
    histories = {"AAA": history(RISING, source="stale cache", stale=True), "BBB": history(FALLING, source="sample data")}
    _, result = run(["AAA", "BBB"], histories)
    assert result["data"]["is_stale"] is True and result["data"]["metrics"]["uses_stale_or_sample"] is True
    assert all(t["price_stale"] for t in result["data"]["metrics"]["tickers"])
    assert any("not live" in w and "AAA (stale cache)" in w and "BBB (sample data)" in w
               for w in result["data"]["warnings"])
    _, live = run(["AAA"], TWO)
    assert live["data"]["is_stale"] is False and live["data"]["as_of"] == T0.isoformat() and live["data"]["as_of"].endswith("+00:00")      # the payload time carries its zone


def test_the_payload_lists_the_price_sentence_of_every_price_that_is_not_live():
    histories = {"AAA": history(RISING, source="cache"), "BBB": history(FALLING, "INR", source="sample data"), "CCC": history(RISING)}
    _, result = run(["AAA", "BBB", "CCC"], histories)
    texts = result["data"]["freshness_texts"]
    assert texts == [
        "AAA last price USD 100.00 (cache, as of 2026-10-01 00:00 UTC)",
        "BBB last price INR 101.00 (sample data, as of 2026-10-01 00:00 UTC)",
    ]                                                                         # CCC is live: nothing to keep
    _, live = run(["AAA"], TWO)
    assert live["data"]["freshness_texts"] == []


def test_short_history_omits_metrics_with_warnings_but_still_answers():
    _, result = run(["AAA"], {"AAA": history(list(range(70, 101)))})        # 31 days
    t = result["data"]["metrics"]["tickers"][0]
    assert result["status"] == "ok" and t["range_52w"] is None and t["trend"] is None
    assert [a["window"] for a in t["averages"]] == [20]
    assert any("52-week range was left out" in w for w in result["data"]["warnings"])


# ---- summary failure never fails the request ----

@pytest.mark.parametrize("llm, note", [
    (FakeLLM(error=RuntimeError("down")), "unavailable"),
    (FakeLLM(reply="   "), "empty"),
    (FakeLLM(reply="The latest price is 77.77."), "contained numbers"),
])
def test_summary_failure_gives_the_fixed_fallback_with_status_ok(llm, note):
    _, result = run(["AAA"], TWO, llm=llm)
    assert result["status"] == "ok" and result["answer"] == CFG["summary_fallback"].strip()
    assert any(note in w for w in result["data"]["warnings"])
    assert result["data"]["metrics"]["tickers"][0]["latest_price"] == 100.0       # metrics still returned


def test_unexpected_failure_after_fetching_becomes_an_error_result(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("metrics exploded")

    monkeypatch.setattr(market, "build_metrics", broken)
    _, result = run(["AAA"], TWO)
    assert result["status"] == "error" and result["answer"] == get_config()["agents"]["error_message"]


# ---- construction and usage ----

def test_default_llm_and_provider_come_from_the_factories(monkeypatch):
    seen = {}
    provider = FakeProvider({})
    monkeypatch.setattr(base_agent, "get_llm", lambda role=None, **kw: seen.setdefault("role", role) and FakeLLM())
    monkeypatch.setattr(market, "get_market_provider", lambda: provider)
    agent = MarketAgent()
    assert seen["role"] == "market_summary" and agent.provider is provider and agent.resolver is not None
    assert "market_summary" in get_config()["llm"]["roles"]


def test_usage_is_recorded_under_the_market_summary_role():
    registry = get_usage_registry()
    registry.reset()
    message = AIMessage(content=SUMMARY, usage_metadata={"input_tokens": 400, "output_tokens": 50, "total_tokens": 450})
    llm = GenericFakeChatModel(
        messages=iter([message]), callbacks=[UsageTracker("market_summary", get_config()["llm"]["model"])])
    try:
        _, result = run(["AAA"], TWO, llm=llm)
        assert result["status"] == "ok"
        assert registry.totals("market_summary")["input_tokens"] == 400
        assert registry.totals("market_summary")["output_tokens"] == 50
    finally:
        registry.reset()


# ---- history shorter than the requested period ----

def sample_snapshot_history(ticker):
    """What SampleHandler returns for a history request: the snapshot rows, labelled "sample data"."""
    entry = json.loads(open(CFG_SNAPSHOT, encoding="utf-8").read())["tickers"][ticker]
    return MarketResult(data=entry["history"], currency=entry["currency"], as_of=T0, source="sample data", is_stale=True)


CFG_SNAPSHOT = get_config()["market_data"]["snapshot_path"]


def test_sample_snapshot_fallback_says_it_covers_a_shorter_span_than_6mo():
    llm = FakeLLM()
    agent, result = run(["AAPL"], {"AAPL": sample_snapshot_history("AAPL")}, llm=llm, period="6mo")
    t = result["data"]["metrics"]["tickers"][0]
    assert result["status"] == "ok" and t["period"] == "6mo"
    assert t["period_is_short"] is True and t["span_days"] == 6          # 5 closes, 2026-10-21 to 2026-10-27
    assert ("AAPL: only 6 days of data for the requested 6mo; "
            "return, volatility and drawdown cover that shorter span.") in result["data"]["warnings"]
    assert "only 6 days of data" in llm.calls[0]                            # the prompt input carries the span


def test_full_length_history_has_no_short_history_warning():
    _, result = run(["AAA"], TWO, period="6mo")
    assert result["data"]["metrics"]["tickers"][0]["period_is_short"] is False
    assert not any("days of data for the requested" in w for w in result["data"]["warnings"])


# ---- payload source ----

def test_payload_source_is_the_least_live_source_among_the_tickers():
    histories = {"AAA": history(RISING), "BBB": history(FALLING, source="stale cache", stale=True)}
    _, mixed = run(["AAA", "BBB"], histories)
    assert mixed["data"]["source"] == "stale cache" and mixed["data"]["is_stale"] is True
    _, live = run(["AAA"], TWO)
    assert live["data"]["source"] == "live" and "quotes" not in live["data"]


# ---- summary prompt format limits ----

def test_summary_prompt_asks_for_short_per_ticker_paragraphs_and_a_limited_set_of_numbers():
    prompt = CFG["summary_prompt"]
    assert "ONE short paragraph per ticker" in prompt
    assert "at most 3 more numbers" in prompt and "at most 4 numbers per ticker" in prompt
    assert '"latest_price_text"' in prompt and "copied exactly" in prompt and "the price is the first number" in prompt
    for allowed in ("return over the period", "20-day and 50-day averages", "52-week high"):
        assert allowed in prompt
    assert "closing sentence" in prompt and "moving average" in prompt and "drawdown" in prompt
    assert "Do not repeat every metric" in prompt and "semicolon" in prompt and "plain beginner language" in prompt
    assert "ONLY numbers that appear in the JSON" in prompt                  # the number-guard requirement stays
    assert "never tell the" in prompt and "Never predict" in prompt          # no advice or prediction wording
    assert prompt.count("{metrics_json}") == 1


# ---- company names (commit 10b): the resolver is injected, Yahoo's search is a table ----

def q(symbol, name, exchange, display, kind="EQUITY"):
    return {"symbol": symbol, "longname": name, "exchange": exchange, "exchDisp": display, "quoteType": kind}


QUOTES = {
    "suzlon": [q("SUZLON.BO", "Suzlon Energy Limited", "BSE", "Bombay"), q("SUZLON.NS", "Suzlon Energy Limited", "NSI", "NSE")],
    "infosys": [q("INFY", "Infosys Limited", "NYQ", "NYSE"), q("INFY.NS", "Infosys Limited", "NSI", "NSE")],
    "globe corp": [q("GLB", "Globe Corp.", "NMS", "NASDAQ"), q("GLBA", "Globe Corp.", "NMS", "NASDAQ")],
    "pinecone": [q("PINE", "Pinecone Systems Inc", "NYQ", "NYSE")],
    "globe": [q("GLB", "Globe Corp.", "NMS", "NASDAQ"), q("GLBA", "Globe Corp.", "NMS", "NASDAQ")],
}
INDIA = {"SUZLON.NS": history(RISING, "INR"), "INFY.NS": history(RISING, "INR"), "INFY": history(FALLING, "USD"), "AAPL": history(RISING, "USD"),
         "GLB": history(RISING, "USD"), "GLBA": history(RISING, "USD"), "PINE": history(RISING, "USD")}


def tickers_of(result):
    return [t["ticker"] for t in result["data"]["metrics"]["tickers"]]


def test_a_company_name_becomes_a_checked_symbol_and_the_answer_says_which_company_it_chose():
    agent, result = run(["Suzlon"], INDIA, quotes=QUOTES)
    assert result["status"] == "ok" and tickers_of(result) == ["SUZLON.NS"]
    note = 'Showing SUZLON.NS (Suzlon Energy Limited, NSE, INR) for "Suzlon".'
    assert result["data"]["lookup_notes"] == [note] and result["answer"] == f"{SUMMARY}\n\n{note}"
    assert "Showing" not in agent.llm.calls[0]                                           # the note is not part of the model's input
    assert "SUZLON.BO" not in [call[0] for call in agent.provider.calls]


def test_an_alias_note_and_a_plain_exact_match_note_only_when_there_is_something_to_say():
    _, tsmc = run(["tsmc"], {"TSM": history(RISING, "USD")})
    assert "NYSE-listed ADR" in tsmc["answer"] and "priced in USD" in tsmc["answer"]
    _, plain = run(["Pinecone Systems"], INDIA, quotes={"pinecone systems": QUOTES["pinecone"]})
    assert plain["data"].get("lookup_notes") is None and plain["answer"] == SUMMARY


def test_an_ambiguous_name_asks_for_the_symbol_and_fetches_nothing():
    agent, result = run(["Globe Corp"], INDIA, quotes=QUOTES)
    assert result["status"] == "needs_input" and agent.llm.calls == []
    assert "GLB (Globe Corp., NASDAQ, USD); GLBA (Globe Corp., NASDAQ, USD)" in result["answer"] and "Please type the ticker symbol" in result["answer"]
    assert all(period == get_config()["ticker_lookup"]["validation_period"] for _, period in agent.provider.calls)      # only price checks


def test_a_name_that_is_not_found_is_a_needs_input_with_the_ticker_hint():
    _, result = run(["Qzxv Wplk"], INDIA, quotes=QUOTES)
    assert result["status"] == "needs_input" and 'couldn\'t find a listed company called "Qzxv Wplk"' in result["answer"] and "AAPL" in result["answer"]


def down(query, limit, timeout):
    raise ConnectionError("offline")


def test_a_lookup_that_is_down_asks_for_the_symbol_instead_of_guessing():
    provider = FakeProvider(INDIA)
    agent = MarketAgent(llm=FakeLLM(), provider=provider, resolver=build_resolver(provider, search=down))
    result = agent.run({"turn_id": "t", "market_params": {"tickers": ["Suzlon"]}})
    assert result["status"] == "needs_input" and "can't look up company names" in result["answer"] and agent.llm.calls == []


def test_a_name_that_needs_the_user_is_a_warning_when_another_ticker_can_be_analysed():
    _, result = run(["AAPL", "Qzxv Wplk", "Globe Corp"], INDIA, quotes=QUOTES)
    assert result["status"] == "ok" and tickers_of(result) == ["AAPL"]
    warnings = result["data"]["warnings"]
    assert any("couldn't find a listed company" in w for w in warnings) and any("more than one match" in w for w in warnings)
    assert "lookup_notes" not in result["data"]


def test_a_symbol_looking_word_without_prices_is_looked_up_once_as_a_name():
    _, result = run(["apple", "AAPL"], INDIA, quotes=QUOTES)                         # APPLE has no prices; "apple" is an alias for AAPL, already in the list
    assert tickers_of(result) == ["AAPL"] and "APPLE was left out: no price history is available right now." in result["data"]["warnings"]
    _, result = run(["apple"], INDIA, quotes=QUOTES)                                 # the alias replaces the symbol that had no prices
    assert tickers_of(result) == ["AAPL"] and result["data"].get("lookup_notes") is None
    _, result = run(["TSMC"], {"TSM": history(RISING, "USD")})                       # 4 letters, a valid symbol shape: its note is kept
    assert tickers_of(result) == ["TSM"] and "NYSE-listed ADR" in result["data"]["lookup_notes"][0]
    _, result = run(["infosys"], INDIA, quotes=QUOTES)                               # a name from the start (7 letters)
    assert tickers_of(result) == ["INFY.NS"]
    _, result = run(["INFY"], INDIA, quotes=QUOTES)                                  # a real symbol with prices is used as typed
    assert tickers_of(result) == ["INFY"] and result["data"].get("lookup_notes") is None


def test_a_symbol_looking_word_with_no_prices_that_is_ambiguous_or_unavailable_adds_the_message():
    _, result = run(["GLOBE", "AAPL"], INDIA, quotes=QUOTES)
    assert any("more than one match" in w for w in result["data"]["warnings"]) and tickers_of(result) == ["AAPL"]
    provider = FakeProvider({"AAPL": history(RISING)})
    agent = MarketAgent(llm=FakeLLM(), provider=provider, resolver=build_resolver(provider, search=down))
    result = agent.run({"turn_id": "t", "market_params": {"tickers": ["AAPL", "ZZZZ"]}})
    assert result["status"] == "ok" and any("can't look up company names" in w for w in result["data"]["warnings"])


def test_market_params_may_be_a_single_text_and_entries_beyond_the_limit_are_warned_about():
    provider = FakeProvider(INDIA)
    agent = MarketAgent(llm=FakeLLM(), provider=provider, resolver=offline(provider, QUOTES))
    result = agent.run({"turn_id": "t", "market_params": {"tickers": "Suzlon, Infosys"}})
    assert tickers_of(result) == ["SUZLON.NS", "INFY.NS"] and len(result["data"]["lookup_notes"]) == 1   # Infosys is an alias: no note
    many = agent.run({"turn_id": "t", "market_params": {"tickers": ["AAPL", "INFY", "SUZLON.NS", "GLB", "GLBA", "PINE", "Suzlon"]}})
    assert any("Only the first" in w and "was left out" in w for w in many["data"]["warnings"])
    assert agent.run({"turn_id": "t", "market_params": "not a dict"})["status"] == "needs_input"


# ---- 10d: bare Indian symbols and suggestions ----

def test_a_capitals_word_without_prices_gets_the_nse_suffix_when_that_has_prices_and_says_so():
    histories = {**INDIA, "HINDUNILVR.NS": history(RISING, "INR")}
    agent, result = run(["HINDUNILVR", "AAPL"], histories, quotes=QUOTES)
    assert tickers_of(result) == ["HINDUNILVR.NS", "AAPL"]
    assert result["data"]["lookup_notes"] == ['Showing HINDUNILVR.NS (NSE, INR); "HINDUNILVR" was not usable on its own.']
    assert result["answer"].endswith('"HINDUNILVR" was not usable on its own.')


def test_a_symbol_shaped_word_without_prices_on_its_own_is_retried_with_the_suffix_but_one_with_prices_stays():
    histories = {**INDIA, "TCSX.NS": history(RISING, "INR"), "TCS": history(FALLING, "USD"), "TCS.NS": history(RISING, "INR")}
    _, result = run(["TCSX"], histories)
    assert tickers_of(result) == ["TCSX.NS"]
    _, result = run(["TCS"], histories)                                                  # valid in the US and has prices: never rewritten
    assert tickers_of(result) == ["TCS"] and result["data"].get("lookup_notes") is None
    assert ("TCS.NS", "5d") not in [call for call in run(["TCS"], histories)[0].provider.calls]


def test_a_name_that_is_not_found_lists_did_you_mean_suggestions_without_choosing_one():
    quotes = {"hindustan uniliver": [q("HUL.NS", "Hindustan Unilever Limited", "NSI", "NSE")]}
    agent, result = run(["Hindustan uniliver"], {"HUL.NS": history(RISING, "INR")}, quotes=quotes)
    assert result["status"] == "needs_input" and "Did you mean: HUL.NS (Hindustan Unilever Limited, NSE, INR)" in result["answer"]
    assert "HUL.NS" not in [call[0] for call in agent.provider.calls if call[1] == CFG["fetch_period"]]      # only a price check, nothing analysed
