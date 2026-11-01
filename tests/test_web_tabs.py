"""The Market and News tabs, driven with AppTest and a fake backend returning real QueryResult objects and payloads."""
from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.utils.config import get_config
from src.web_app import backend
from src.web_app.render_common import plain_text
from web_support import HOSTILE, FakeBackend, answer_result, ask, charts, dataframes, disclaimer, fake, labelled, shown, start, ui  # noqa: F401

LIVE_AT = "2026-10-30T08:32:00+00:00"


def agent_result(agent, status="ok", answer="", data=None):
    return {"agent": agent, "turn_id": "t", "status": status, "answer": answer, "sources": [], "error": None, "data": data}


def market_data(tickers=(("AAPL", "USD"),), **kw):
    data = {
        "history": [{"ticker": t, "date": d, "close": c} for t, _ in tickers for d, c in (("2026-05-01", 100.0), ("2026-05-04", 101.5), ("2026-10-30", 120.25))],
        "as_of": LIVE_AT, "source": "live", "is_stale": False, "freshness_texts": [], "warnings": [],
        "metrics": {"tickers": [{"ticker": t, "currency": c} for t, c in tickers]}, "comparison": [],
    }
    data.update(kw)
    return data


def market_reply(text="AAPL last price USD 120.25 (live). It rose over the period.", data=None, status="ok"):
    return answer_result(text, sources=[], intents=["market"], results=[agent_result("market", status, text, data or market_data())])


def submit_market(at, tickers="AAPL", period="6mo"):
    labelled(at.text_input, ui("market.tickers_label")).input(tickers)
    labelled(at.selectbox, ui("market.period_label")).select(period)
    next(b for b in at.main.button if b.label == ui("market.submit")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def captions(at):
    return [c.value for c in at.main.caption]


def warnings(at):
    return [w.value for w in at.main.warning]


# ============================== Market ==============================

def test_one_ticker_calls_the_facade_once_with_the_hint_and_params_and_shows_answer_freshness_and_a_chart(fake):
    fake.reply = market_reply()
    at = start()
    thread = at.session_state["thread_id"]
    submit_market(at, "AAPL", "6mo")
    assert len(fake.asks) == 1
    args, kwargs = fake.asks[0]
    assert args == ("Market analysis for AAPL over 6 months.", thread, "market")
    assert kwargs == {"market_params": {"tickers": ["AAPL"], "period": "6mo"}}
    texts = shown(at)
    assert any("AAPL last price USD 120.25 (live)." in t for t in texts) and sum(t.count(disclaimer()) for t in texts) == 1
    assert [s.value for s in at.main.success] == [ui("freshness.with_time", source="live", as_of="2026-10-30 08:32 UTC")]
    [chart] = charts(at)
    assert chart["data"][0]["y"] == [100.0, 101.5, 120.25] and chart["data"][0]["x"] == ["2026-05-01", "2026-05-04", "2026-10-30"]
    assert chart["layout"]["title"]["text"] == "AAPL (USD)" and chart["layout"]["yaxis"]["title"]["text"] == "Closing price (USD)"
    assert dataframes(at) == [] and warnings(at) == []                         # a single ticker has no comparison table
    assert ui("market.chart_heading") in texts


def test_two_tickers_in_two_currencies_get_two_charts_and_a_comparison_table_from_the_payload(fake):
    rows = [
        {"ticker": "TCS.NS", "currency": "INR", "period_return_pct": 12.34, "volatility_pct": 20.0, "max_drawdown_pct": -8.25, "trend": "uptrend"},
        {"ticker": "INFY.NS", "currency": "INR", "period_return_pct": None, "volatility_pct": 18.5, "max_drawdown_pct": -5.0, "trend": None},
    ]
    fake.reply = market_reply(data=market_data((("TCS.NS", "INR"), ("INFY.NS", "INR")), comparison=rows))
    at = start()
    submit_market(at, "TCS.NS, INFY.NS", "1y")
    assert fake.asks[0][0][0] == "Market analysis for TCS.NS, INFY.NS over 1 year."
    assert fake.asks[0][1] == {"market_params": {"tickers": ["TCS.NS", "INFY.NS"], "period": "1y"}}
    assert [c["layout"]["title"]["text"] for c in charts(at)] == ["TCS.NS (INR)", "INFY.NS (INR)"]
    frame = dataframes(at)[0].value
    assert list(frame.columns) == [ui(f"market.columns.{k}") for k in ("ticker", "currency", "period_return_pct", "volatility_pct", "max_drawdown_pct", "trend")]
    assert frame.iloc[0].tolist() == ["TCS.NS", "INR", "+12.3%", "+20.0%", "-8.2%", "uptrend"] or frame.iloc[0].tolist() == ["TCS.NS", "INR", "+12.3%", "+20.0%", "-8.3%", "uptrend"]
    assert frame.iloc[1].tolist()[2] == ui("market.not_available") and frame.iloc[1].tolist()[5] == ui("market.not_available")


def test_tickers_are_split_on_commas_and_trimmed_and_the_backend_does_the_validation(fake):
    fake.reply = market_reply()
    at = start()
    submit_market(at, " aapl,, msft , ", "3mo")
    assert fake.asks[0][1]["market_params"] == {"tickers": ["aapl", "msft"], "period": "3mo"}
    assert fake.asks[0][0][0] == "Market analysis for aapl, msft over 3 months."


def test_the_period_choices_and_the_default_come_from_the_config(fake):
    at = start()
    cfg = get_config()["market_analysis"]
    assert list(labelled(at.selectbox, ui("market.period_label")).options) == [cfg["period_labels"][p] for p in cfg["periods"]]
    assert labelled(at.selectbox, ui("market.period_label")).value == cfg["default_period"]


@pytest.mark.parametrize("source, stale, details, level", [
    ("cache", False, ["AAPL last price USD 120.25 (cache, as of 2026-10-30 08:30 UTC)"], "info"),
    ("stale cache", True, ["AAPL last price USD 120.25 (stale cache, as of 2026-10-30 07:00 UTC)"], "warning"),
    ("sample data", True, ["AAPL last price USD 120.25 (sample data, as of 2026-10-01 00:00 UTC)"], "warning"),
])
def test_a_price_that_is_not_live_is_shown_with_its_source_time_and_sentence_from_the_payload(fake, source, stale, details, level):
    fake.reply = market_reply("AAPL moved.", data=market_data(source=source, is_stale=stale, freshness_texts=details))
    at = start()
    submit_market(at)
    line = ui("freshness.with_time", source=source, as_of="2026-10-30 08:32 UTC")
    assert (line in [w.value for w in at.main.warning]) is (level == "warning")
    assert (line in [i.value for i in at.main.info]) is (level == "info")
    assert plain_text(details[0]) in captions(at) and line not in [s.value for s in at.main.success]


def test_the_colour_follows_the_payload_source_even_when_there_are_no_price_sentences(fake):
    fake.reply = market_reply(data=market_data(source="cache"))
    at = start()
    submit_market(at)
    line = ui("freshness.with_time", source="cache", as_of="2026-10-30 08:32 UTC")
    assert line in [i.value for i in at.main.info] and line not in [s.value for s in at.main.success]


def test_a_price_sentence_the_answer_already_carries_is_not_repeated_as_a_caption(fake):
    sentence = "AAPL last price USD 120.25 (cache, as of 2026-10-30 08:30 UTC)"
    fake.reply = market_reply(f"{sentence}. It rose.", data=market_data(source="cache", freshness_texts=[sentence]))
    at = start()
    submit_market(at)
    assert sentence not in captions(at) and any(sentence in t for t in shown(at))           # still in the answer
    assert ui("freshness.with_time", source="cache", as_of="2026-10-30 08:32 UTC") in [i.value for i in at.main.info]


def test_freshness_without_a_time_and_without_a_source(fake):
    fake.reply = market_reply(data=market_data(as_of=""))
    at = start()
    submit_market(at)
    assert ui("freshness.no_time", source="live") in [s.value for s in at.main.success]
    fake.reply = market_reply(data=market_data(source=""))
    submit_market(at)
    assert not [t for t in shown(at) if t.startswith("Data source")]


def test_payload_warnings_are_shown_as_plain_text_and_not_taken_from_the_answer(fake):
    data = market_data(warnings=["'ZZZZ' was skipped: it is not a valid ticker.", f"Odd: {HOSTILE}"])
    fake.reply = market_reply("AAPL moved. No warnings here.", data=data)
    at = start()
    submit_market(at, "AAPL, ZZZZ")
    shown_warnings = warnings(at)
    assert plain_text("'ZZZZ' was skipped: it is not a valid ticker.") in shown_warnings
    odd = [w for w in shown_warnings if w.startswith("Odd:")][0]
    assert "<script" not in odd and "javascript:" not in odd and "&lt;script" in odd


def test_a_ticker_with_no_data_shows_the_payload_warning_and_no_chart(fake):
    data = market_data(history=[], metrics={"tickers": []}, warnings=["No price history was available for NOPE."])
    reply = answer_result("I couldn't get market data for those tickers.", sources=[], results=[agent_result("market", "no_answer", "x", data)])
    fake.reply = reply
    at = start()
    submit_market(at, "NOPE")
    assert plain_text("No price history was available for NOPE.") in warnings(at)
    assert ui("messages.no_answer_hint") in captions(at) and charts(at) == [] and dataframes(at) == []


def test_a_result_without_a_payload_shows_only_the_answer(fake):
    fake.reply = answer_result("Please give a ticker.", sources=[], results=[agent_result("market", "needs_input", "Please give a ticker.", None)])
    at = start()
    submit_market(at)
    assert ui("messages.needs_input_hint") in captions(at) and charts(at) == [] and warnings(at) == []


def test_the_market_tab_rejects_an_empty_ticker_box_without_calling_the_facade(fake):
    at = start()
    submit_market(at, " , ")
    assert fake.asks == [] and ui("market.tickers_required") in warnings(at)


@pytest.mark.parametrize("stopped", ["error", "recursion"])
def test_market_error_and_recursion_stops_are_friendly_warnings(fake, stopped):
    fake.reply = backend.QueryResult(answer=f"Sorry, I couldn't prepare an answer this time.\n\n{disclaimer()}", stopped=stopped)
    at = start()
    submit_market(at)
    assert len(warnings(at)) == 1 and "couldn't prepare an answer" in warnings(at)[0] and charts(at) == []


def test_a_market_budget_stop_shows_the_ui_message(fake):
    fake.reply = backend.QueryResult(answer="backend text", stopped="budget")
    at = start()
    submit_market(at)
    assert warnings(at) == [ui("messages.budget_stopped")] and "backend text" not in " ".join(shown(at))


def test_a_market_rerun_never_calls_the_facade_and_keeps_the_last_result(fake):
    fake.reply = market_reply()
    at = start()
    submit_market(at)
    for _ in range(3):
        at.run()
    at.sidebar.button[0].click().run()
    assert len(fake.asks) == 1 and len(charts(at)) == 1 and any("AAPL last price" in t for t in shown(at))
    submit_market(at, "MSFT")
    assert len(fake.asks) == 2


def test_every_submit_is_disabled_while_a_call_runs_and_the_page_reopens_on_the_tab(fake, monkeypatch):
    seen, tab_defaults = [], []
    real_submit, real_tabs = st.form_submit_button, st.tabs
    monkeypatch.setattr(st, "form_submit_button", lambda label, **kw: seen.append((label, kw.get("disabled"))) or real_submit(label, **kw))
    monkeypatch.setattr(st, "tabs", lambda labels, **kw: tab_defaults.append(kw.get("default")) or real_tabs(labels, **kw))
    fake.reply = market_reply()
    at = start()
    seen.clear(), tab_defaults.clear()
    submit_market(at)
    market, news = ui("market.submit"), ui("news.submit")
    assert [s for s in seen if s[0] == market] == [(market, False), (market, True), (market, False)]
    assert [s for s in seen if s[0] == news] == [(news, True), (news, False)]      # one request at a time: News is drawn disabled while Market's call runs
    assert tab_defaults == [None, ui("tabs.market"), ui("tabs.market")]


def test_the_tab_default_is_passed_only_for_the_renders_after_a_submit(fake, monkeypatch):
    tab_defaults, real_tabs = [], st.tabs
    monkeypatch.setattr(st, "tabs", lambda labels, **kw: tab_defaults.append(kw.get("default")) or real_tabs(labels, **kw))
    fake.reply = market_reply()
    at = start()
    submit_market(at)
    assert tab_defaults[-1] == ui("tabs.market") and at.session_state["active_tab"] is None      # the answer is drawn: the hint is spent
    tab_defaults.clear()
    at.run()
    at.sidebar.button[0].click().run()
    assert tab_defaults == [None, None]
    assert len(fake.asks) == 1


def test_an_invalid_submit_does_not_set_a_tab_default(fake, monkeypatch):
    tab_defaults, real_tabs = [], st.tabs
    monkeypatch.setattr(st, "tabs", lambda labels, **kw: tab_defaults.append(kw.get("default")) or real_tabs(labels, **kw))
    at = start()
    submit_market(at, " , ")
    assert set(tab_defaults) == {None} and not fake.asks


# ============================== News ==============================

ITEMS = [
    {"number": 1, "title": "Nifty ends higher", "publisher": "Example Wire", "published": "2026-10-30T06:15:00+00:00", "link": "https://news.example.com/1"},
    {"number": 2, "title": "Banks lead gains", "publisher": "Daily Example", "published": "2026-10-30T05:00:00+00:00", "link": "https://news.example.com/a(b)"},
    {"number": 3, "title": "Oil slips", "publisher": None, "published": "", "link": ""},
]


def news_data(items=ITEMS, **kw):
    data = {"query": "Nifty 50", "items": items, "as_of": LIVE_AT, "source": "live", "is_stale": False, "warnings": [], "summary_citations": [1, 2]}
    data.update(kw)
    return data


def news_reply(text="- Markets rose [1][2]\n- Banks led [2]", data=None, status="ok"):
    return answer_result(text, sources=[], intents=["news"], results=[agent_result("news", status, text, data or news_data())])


def submit_news(at, text="Nifty 50", mode=None):
    if mode:
        at.radio[0].set_value(mode)
    labelled(at.text_input, ui("news.text_label")).input(text)
    next(b for b in at.main.button if b.label == ui("news.submit")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def answer_markdown(at):
    return [t for t in shown(at) if "Markets rose" in t or "Banks led" in t or "see [" in t][0]


def test_a_topic_search_calls_the_facade_with_the_news_hint_and_a_query_param(fake):
    fake.reply = news_reply()
    at = start()
    thread = at.session_state["thread_id"]
    submit_news(at, "Nifty 50")
    args, kwargs = fake.asks[0]
    assert args == ("Latest news about Nifty 50.", thread, "news") and kwargs == {"news_params": {"query": "Nifty 50"}}
    assert [r.label for r in at.radio] == [ui("news.mode_label")] and at.radio[0].value == "query"


def test_a_ticker_search_sends_the_ticker_param(fake):
    fake.reply = news_reply()
    at = start()
    submit_news(at, " AAPL ", mode="ticker")
    args, kwargs = fake.asks[0]
    assert args[0] == "Latest news for AAPL." and kwargs == {"news_params": {"ticker": "AAPL"}}


def test_markers_become_links_to_their_articles_and_every_headline_is_listed_once(fake):
    fake.reply = news_reply()
    at = start()
    submit_news(at)
    text = answer_markdown(at)
    assert "[\\[1\\]](https://news.example.com/1)[\\[2\\]](https://news.example.com/a%28b%29)" in text
    assert "Banks led [\\[2\\]](https://news.example.com/a%28b%29)" in text
    texts = shown(at)
    assert ui("news.headlines_heading", count=3) in texts
    for number, title in ((1, "Nifty ends higher"), (2, "Banks lead gains"), (3, "Oil slips")):
        assert len([t for t in texts if t.startswith(f"**\\[{number}\\]** {title}")]) == 1
    assert "Example Wire · " + ui("news.published", when="2026-10-30 06:15 UTC") in captions(at)
    assert "Daily Example · " + ui("news.published", when="2026-10-30 05:00 UTC") in captions(at)
    assert [t for t in texts if t == f"[{ui('news.open_article')}](https://news.example.com/1)"]
    assert sum(1 for t in texts if t.startswith(f"[{ui('news.open_article')}]")) == 2        # item 3 has no link
    assert sum(t.count(disclaimer()) for t in texts) == 1


def test_a_headline_cited_many_times_is_still_listed_once(fake):
    fake.reply = news_reply("- One [1] two [1] three [1][1] and [1, 1].")
    at = start()
    submit_news(at)
    texts = shown(at)
    assert len([t for t in texts if t.startswith("**\\[1\\]**")]) == 1
    answer = [t for t in texts if t.startswith("- One")][0]
    assert answer.count("(https://news.example.com/1)") == 6                                  # each marker is linked, none merged


def test_labelled_and_combined_markers_are_resolved_too(fake):
    fake.reply = news_reply("- Themes [News 1][News 2] and [News 1, News 2] and [1, 2].")
    at = start()
    submit_news(at)
    answer = [t for t in shown(at) if t.startswith("- Themes")][0]
    assert "[News" not in answer and answer.count("(https://news.example.com/1)") == 3 and answer.count("a%28b%29)") == 3


def test_a_marker_without_a_matching_or_linkable_item_is_plain_text_not_an_error(fake):
    fake.reply = news_reply("- Missing [9] unlinked [3] fine [1].")
    at = start()
    submit_news(at)
    answer = [t for t in shown(at) if t.startswith("- Missing")][0]
    assert "Missing \\[9\\] unlinked \\[3\\] fine [\\[1\\]](https://news.example.com/1)." in answer


@pytest.mark.parametrize("text", [
    "Results for [FY 2026] and [Q3 2026] and [Table 3] stay.",
    "A [news 1] and [NEWS 1] and [Market 1] and [News1] and [ 1 ] stay.",
    "See [1](https://news.example.com/1) for details.",
])
def test_only_numbered_markers_are_citations_everything_else_in_brackets_is_left_as_written(text):
    from src.web_app.tab_news import resolve_markers

    assert resolve_markers(text, ITEMS) == text


def test_resolve_markers_links_the_three_citation_forms_and_leaves_a_missing_item_plain():
    from src.web_app.tab_news import resolve_markers

    link1, link2 = "https://news.example.com/1", "https://news.example.com/a%28b%29"
    assert resolve_markers("[1] [News 2] [1, 2] [News 1, News 2]", ITEMS) == (
        f"[\\[1\\]]({link1}) [\\[2\\]]({link2}) [\\[1\\]]({link1})[\\[2\\]]({link2}) [\\[1\\]]({link1})[\\[2\\]]({link2})"
    )
    assert resolve_markers("[9] and [FY 2026] and [1, 9]", ITEMS) == f"\\[9\\] and [FY 2026] and [\\[1\\]]({link1})\\[9\\]"


def test_brackets_that_are_not_citations_and_a_ready_made_link_survive_on_the_page(fake):
    fake.reply = news_reply("- see [FY 2026] and [Table 3] and [1](https://news.example.com/1) then [2] and [9].")
    at = start()
    submit_news(at)
    answer = [t for t in shown(at) if t.startswith("- see")][0]
    assert "[FY 2026] and [Table 3] and [1](https://news.example.com/1) then [\\[2\\]](https://news.example.com/a%28b%29) and \\[9\\]." in answer
    assert "((" not in answer and "](https://news.example.com/1)(" not in answer


@pytest.mark.parametrize("source, stale, box", [("live", False, "success"), ("cache", False, "info"), ("stale cache", True, "warning")])
def test_news_freshness_colour_follows_the_payload(fake, source, stale, box):
    fake.reply = news_reply(data=news_data(source=source, is_stale=stale))
    at = start()
    submit_news(at)
    line = ui("freshness.with_time", source=source, as_of="2026-10-30 08:32 UTC")
    for kind in ("success", "info", "warning"):
        assert (line in [e.value for e in getattr(at.main, kind)]) is (kind == box)


def test_hostile_feed_text_is_plain_text_and_its_javascript_link_is_dropped(fake):
    items = [
        {"number": 1, "title": HOSTILE, "publisher": HOSTILE, "published": "2026-10-30T06:15:00+00:00", "link": "javascript:alert(1)"},
        {"number": 2, "title": "Fine", "publisher": "Wire", "published": "", "link": "https://news.example.com/2"},
    ]
    fake.reply = news_reply("- Headline [1] and [2]", data=news_data(items))
    at = start()
    submit_news(at)
    text = "\n".join(shown(at))
    assert "<script" not in text and "javascript:" not in text and "](javascript" not in text
    assert "&lt;script&gt;alert\\(1\\)&lt;/script&gt; x" in text                              # shown literally
    answer = [t for t in shown(at) if t.startswith("- Headline")]
    expected = "- Headline \\[1\\] and [\\[2\\]](https://news.example.com/2)\n\n" + disclaimer().replace("<", "&lt;").replace("$", "\\$")
    assert answer == [expected]                                                      # exact text: item 1 (javascript: link) is plain
    assert "javascript" not in "".join(t for t in shown(at) if "](" in t)             # no link target at all for the hostile URL
    assert not [t for t in shown(at) if t.startswith(f"[{ui('news.open_article')}](javascript")]
    links = [t for t in shown(at) if t.startswith(f"[{ui('news.open_article')}]")]
    assert links == [f"[{ui('news.open_article')}](https://news.example.com/2)"]


def test_stale_news_is_a_warning_with_the_source_and_the_payload_warnings(fake):
    data = news_data(source="stale cache", is_stale=True, warnings=["These headlines were saved earlier (stale cache) and may be out of date."])
    fake.reply = news_reply(data=data)
    at = start()
    submit_news(at)
    assert ui("freshness.with_time", source="stale cache", as_of="2026-10-30 08:32 UTC") in warnings(at)
    assert plain_text("These headlines were saved earlier (stale cache) and may be out of date.") in warnings(at)


def test_a_search_with_no_headlines_shows_the_answer_and_a_hint_and_no_list(fake):
    data = news_data(items=[], source="", as_of="")
    fake.reply = answer_result("No recent headlines were found for this search.", sources=[], results=[agent_result("news", "no_answer", "x", data)])
    at = start()
    submit_news(at, "zzz")
    assert ui("messages.no_answer_hint") in captions(at) and not [t for t in shown(at) if t.startswith("Headlines (")]


def test_news_needs_input_and_stopped_values_are_friendly(fake):
    at = start()
    fake.reply = answer_result("Please give a ticker or a search phrase.", sources=[], results=[agent_result("news", "needs_input", "x", None)])
    submit_news(at)
    assert ui("messages.needs_input_hint") in captions(at)
    for stopped in ("error", "recursion"):
        fake.reply = backend.QueryResult(answer=f"Sorry, couldn't.\n\n{disclaimer()}", stopped=stopped)
        submit_news(at)
        assert len(warnings(at)) == 1 and "Sorry, couldn't." in warnings(at)[0]
    fake.reply = backend.QueryResult(answer="backend text", stopped="budget")
    submit_news(at)
    assert warnings(at) == [ui("messages.budget_stopped")]


def test_an_empty_news_box_does_not_call_the_facade(fake):
    at = start()
    submit_news(at, "   ")
    assert fake.asks == [] and ui("news.text_required") in warnings(at)


def test_a_news_rerun_never_calls_the_facade_and_keeps_the_last_result(fake):
    fake.reply = news_reply()
    at = start()
    submit_news(at)
    for _ in range(3):
        at.run()
    assert len(fake.asks) == 1 and ui("news.headlines_heading", count=3) in shown(at)
    submit_market(at, "AAPL")                                                                  # another tab's submit keeps this result
    assert len(fake.asks) == 2 and ui("news.headlines_heading", count=3) in shown(at)


def test_the_tabs_keep_separate_results_and_new_conversation_keeps_them(fake):
    fake.reply = news_reply()
    at = start()
    submit_news(at)
    fake.reply = market_reply()
    submit_market(at)
    assert ui("news.headlines_heading", count=3) in shown(at) and len(charts(at)) == 1
    at.sidebar.button[0].click().run()
    assert ui("news.headlines_heading", count=3) in shown(at) and len(charts(at)) == 1 and len(fake.asks) == 2
