"""The Portfolio tab. AppTest cannot type into st.data_editor cells, so these tests replace st.data_editor with a function that
returns the table a user would have typed (the same DataFrame type the real editor returns) and check what the tab hands to it and
what it sends on. One test runs the real editor to show that the page builds with it. The real editor must be tried in a browser."""
from __future__ import annotations

from typing import get_args

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.core.portfolio_types import AssetClass, InstrumentType
from src.utils.config import get_config
from src.web_app import backend
from src.web_app.render_common import plain_text
from web_support import HOSTILE, FakeBackend, answer_result, charts, dataframes, disclaimer, fake, shown, start, ui  # noqa: F401

LIVE_AT = "2026-10-30T08:32:00+00:00"
COLUMNS = ["ticker", "quantity", "avg_cost", "instrument_type", "asset_class", "expense_ratio_pct"]


def agent_result(status="ok", answer="", data=None):
    return {"agent": "portfolio", "turn_id": "t", "status": status, "answer": answer, "sources": [], "error": None, "data": data}


def position(ticker, currency, price, quantity, value, weight, **kw):
    row = {"ticker": ticker, "instrument_type": "stock", "asset_class": "equity", "quantity": quantity, "avg_cost": None, "expense_ratio_pct": None,
           "currency": currency, "price": price, "value": value, "weight": weight, "pnl": None, "pnl_pct": None, "price_source": "live",
           "price_stale": False, "as_of": LIVE_AT}
    row.update(kw)
    return row


def block(total, count, allocation, largest, weight, **kw):
    values = {"currency": "", "total_value": total, "position_count": count, "allocation_by_asset_class": allocation, "largest_ticker": largest,
              "largest_weight": weight, "top3_weight": 1.0, "hhi": 0.7, "diversification_score": 31, "diversification_label": "Concentrated",
              "equity_share": 0.8337, "risk_label": "High", "expense_ratio_weighted_pct": None, "expense_ratio_holdings": 0, "expense_ratio_total": count,
              "expense_ratio_value_share": 0.0, "expense_ratio_coverage": "not provided"}
    values.update(kw)
    return values


def mixed_data(**kw):
    positions = [
        position("SUZLON.NS", "INR", 1250.5, 1000.0, 1250500.0, 0.8337, avg_cost=1000.0, pnl=250500.0, pnl_pct=25.05),
        position("GOLDBEES.NS", "INR", 249.5, 1000.0, 249500.0, 0.1663, instrument_type="etf", asset_class="gold", expense_ratio_pct=0.8),
        position("AAPL", "USD", 200.0, 5.0, 1000.0, 1.0),
    ]
    currencies = {
        "INR": block(1500000.0, 2, {"equity": 0.8337, "gold": 0.1663}, "SUZLON.NS", 0.8337, expense_ratio_weighted_pct=0.8, expense_ratio_holdings=1,
                     expense_ratio_coverage="covers 17% of value, 1 of 2 holdings"),
        "USD": block(1000.0, 1, {"equity": 1.0}, "AAPL", 1.0, top3_weight=1.0, equity_share=1.0),
    }
    data = {"metrics": {"positions": positions, "currencies": currencies, "warnings": [], "as_of": LIVE_AT, "uses_stale_or_sample": False},
            "warnings": [], "as_of": LIVE_AT, "source": "live", "is_stale": False}
    data.update(kw)
    return data


def reply(text="The largest holding is SUZLON.NS.", data=None, status="ok"):
    return answer_result(text, sources=[], intents=["portfolio"], results=[agent_result(status, text, data if data is not None else mixed_data())])


class Editor:
    """Stands in for st.data_editor: records how the tab called it and returns the table the user 'typed'."""

    def __init__(self):
        self.calls, self.frame = [], pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})

    def type_rows(self, rows):
        self.frame = pd.DataFrame(rows, columns=COLUMNS)

    def __call__(self, data, **kwargs):
        self.calls.append((data, kwargs))
        return self.frame


@pytest.fixture
def editor(monkeypatch):
    fake_editor = Editor()
    monkeypatch.setattr(st, "data_editor", fake_editor)
    return fake_editor


def submit(at):
    next(b for b in at.main.button if b.label == ui("portfolio.submit")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def warnings(at):
    return [w.value for w in at.main.warning]


def captions(at):
    return [c.value for c in at.main.caption]


def named_frames(at, names):
    """The page's tables in drawing order (measures, then holdings, per currency), named. Fails if there are more or fewer."""
    found = dataframes(at)
    assert len(found) == len(names), [f.value.columns.tolist() for f in found]
    return {name: frame.value for name, frame in zip(names, found)}


def plain_values(value):
    """Every leaf of a nested call argument."""
    if isinstance(value, dict):
        return [leaf for v in value.values() for leaf in plain_values(v)]
    if isinstance(value, (list, tuple)):
        return [leaf for v in value for leaf in plain_values(v)]
    return [value]


# ============================== the page and the table ==============================

def test_the_tabs_are_chat_portfolio_market_news_and_the_real_editor_builds_with_an_empty_table(fake):
    at = start()
    assert [t.label for t in at.tabs] == [ui("tabs.chat"), ui("tabs.portfolio"), ui("tabs.goals"), ui("tabs.market"), ui("tabs.news")]
    assert ui("portfolio.intro") in shown(at)
    editors = [f for f in at.dataframe if f.key == "portfolio_table"]
    assert len(editors) == 1 and len(editors[0].value) == 0 and list(editors[0].value.columns) == COLUMNS         # no example data
    assert [b.label for b in at.main.button] == [ui("portfolio.submit"), ui("goals.submit"), ui("market.submit"), ui("news.submit")] and not any(b.disabled for b in at.main.button)
    assert fake.asks == []


def test_the_editor_gets_an_empty_typed_table_dynamic_rows_and_columns_from_the_config(fake, editor):
    start()
    data, kw = editor.calls[0]
    assert list(data.columns) == COLUMNS and len(data) == 0
    assert kw["num_rows"] == "dynamic" and kw["hide_index"] is True and kw["key"] == "portfolio_table" and kw["width"] == "stretch"
    cfg, pf = get_config()["ui"]["portfolio"], get_config()["portfolio"]
    config = kw["column_config"]
    assert list(config) == COLUMNS
    assert [config[c]["label"] for c in COLUMNS] == [cfg["columns"][c] for c in COLUMNS]
    assert [config[c]["help"] for c in COLUMNS] == [cfg["column_help"][c] for c in COLUMNS]
    assert config["ticker"]["type_config"]["max_chars"] == get_config()["ticker_lookup"]["max_query_chars"]
    kind = config["instrument_type"]["type_config"]
    assert [o["value"] for o in kind["options"]] == pf["instrument_types"] == list(get_args(InstrumentType)) and config["instrument_type"]["default"] == "stock"
    assert [o["label"] for o in kind["options"]] == [cfg["instrument_labels"][k] for k in pf["instrument_types"]]
    cls = config["asset_class"]["type_config"]
    assert [o["value"] for o in cls["options"]] == pf["asset_classes"] == list(get_args(AssetClass)) and config["asset_class"]["default"] == "equity"
    assert [o["label"] for o in cls["options"]] == [cfg["asset_class_labels"][k] for k in pf["asset_classes"]]
    for column in ("quantity", "avg_cost", "expense_ratio_pct"):
        assert config[column]["type_config"]["min_value"] == 0.0


def test_the_dropdown_labels_come_from_the_config_and_an_unknown_choice_is_shown_as_plain_text():
    from src.web_app import tab_portfolio as tab

    config = tab.column_config()
    ui_cfg = get_config()["ui"]["portfolio"]
    assert tab._label("instrument_labels", "etf") == ui_cfg["instrument_labels"]["etf"] and tab._label("asset_class_labels", "gold") == ui_cfg["asset_class_labels"]["gold"]
    assert tab._label("asset_class_labels", "<b>x</b>") == "<b>x</b>"      # raw: table cells are not Markdown
    assert config["instrument_type"] is not None


# ============================== submitting ==============================

def test_submit_sends_the_cleaned_rows_with_the_hint_once_and_shows_the_answer(fake, editor):
    fake.reply = reply()
    at = start()
    thread = at.session_state["thread_id"]
    editor.type_rows([
        ["Suzlon", 1000.0, np.nan, "stock", "equity", np.nan],
        [None, np.nan, np.nan, "stock", "equity", np.nan],                       # a blank row
        ["GOLDBEES.NS", 10, 249.5, "etf", "gold", 0.8],
    ])
    submit(at)
    assert len(fake.asks) == 1
    args, kwargs = fake.asks[0]
    assert args == (ui("portfolio.query_template", count=2), thread, "portfolio")
    assert kwargs == {"portfolio": [
        {"ticker": "Suzlon", "quantity": 1000.0, "instrument_type": "stock", "asset_class": "equity"},
        {"ticker": "GOLDBEES.NS", "quantity": 10, "avg_cost": 249.5, "expense_ratio_pct": 0.8, "instrument_type": "etf", "asset_class": "gold"},
    ]}
    assert all(type(leaf) in (str, int, float) for leaf in plain_values(kwargs))                   # no pandas or numpy object goes to the backend
    assert any("The largest holding is SUZLON.NS." in t for t in shown(at))
    assert sum(t.count(disclaimer()) for t in shown(at)) == 1


def test_a_table_with_only_blank_rows_is_not_sent(fake, editor):
    at = start()
    editor.type_rows([[None, np.nan, np.nan, "stock", "equity", np.nan], ["  ", None, None, None, None, None]])
    submit(at)
    assert fake.asks == [] and ui("portfolio.rows_required") in warnings(at)
    editor.type_rows([])
    submit(at)
    assert fake.asks == []


def test_a_rerun_never_calls_the_facade_and_keeps_the_last_result(fake, editor):
    fake.reply = reply()
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    for _ in range(3):
        at.run()
    at.sidebar.button[0].click().run()
    assert len(fake.asks) == 1 and len(charts(at)) == 2 and any("The largest holding" in t for t in shown(at))


def test_the_page_reopens_on_the_portfolio_tab_after_a_submit_and_not_on_later_reruns(fake, editor, monkeypatch):
    defaults, real_tabs = [], st.tabs
    monkeypatch.setattr(st, "tabs", lambda labels, **kw: defaults.append(kw.get("default")) or real_tabs(labels, **kw))
    fake.reply = reply()
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    defaults.clear()
    submit(at)
    assert defaults == [None, ui("tabs.portfolio"), ui("tabs.portfolio")]            # the click run, the waiting run, the answer run
    defaults.clear()
    at.run()
    assert defaults == [None]


def test_every_submit_is_disabled_while_a_call_runs_on_another_tab(fake, editor, monkeypatch):
    seen, real_submit = [], st.form_submit_button
    monkeypatch.setattr(st, "form_submit_button", lambda label, **kw: seen.append((label, kw.get("disabled"))) or real_submit(label, **kw))
    fake.reply = reply()
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    seen.clear()
    submit(at)
    label = ui("portfolio.submit")
    assert [s for s in seen if s[0] == label] == [(label, False), (label, True), (label, False)]          # draw, draw with the request waiting, answered


def test_a_reached_budget_disables_the_submit_and_shows_the_message(fake, editor):
    fake.usage_value = backend.UsageSummary(tokens=2_000_001, cost=1.0, limit=2_000_000, exceeded=True)
    at = start()
    assert next(b for b in at.main.button if b.label == ui("portfolio.submit")).disabled is True
    assert warnings(at).count(ui("messages.budget_stopped")) == 5 and fake.asks == []


# ============================== showing a result ==============================

def test_two_currencies_get_two_sections_two_pies_and_no_total_across_them(fake, editor):
    fake.reply = reply()
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    texts = shown(at)
    assert ui("portfolio.currency_heading", currency="INR") in texts and ui("portfolio.currency_heading", currency="USD") in texts
    pies = charts(at)
    assert [p["layout"]["title"]["text"] for p in pies] == [ui("portfolio.allocation_title", currency="INR"), ui("portfolio.allocation_title", currency="USD")]
    assert pies[0]["data"][0]["values"] == [0.8337, 0.1663] and pies[0]["data"][0]["labels"] == ["Equity", "Gold"]
    assert pies[1]["data"][0]["values"] == [1.0] and pies[0]["data"][0]["type"] == "pie"
    frames = named_frames(at, ["measures_INR", "positions_INR", "measures_USD", "positions_USD"])
    inr = dict(zip(frames["measures_INR"]["measure"], frames["measures_INR"]["value"]))
    assert inr[ui("portfolio.metrics.total_value")] == "INR 15,00,000.00" and inr[ui("portfolio.metrics.position_count")] == "2"
    assert inr[ui("portfolio.metrics.largest")] == "SUZLON.NS (83.4%)" and inr[ui("portfolio.metrics.top3")] == "100.0%"
    assert inr[ui("portfolio.metrics.diversification")] == "31 out of 100 (Concentrated)" and inr[ui("portfolio.metrics.risk")] == "High"
    assert inr[ui("portfolio.metrics.equity_share")] == "83.4%"
    assert inr[ui("portfolio.metrics.expense_ratio")] == "0.80% (covers 17% of value, 1 of 2 holdings)"
    usd = dict(zip(frames["measures_USD"]["measure"], frames["measures_USD"]["value"]))
    assert usd[ui("portfolio.metrics.total_value")] == "USD 1,000.00" and usd[ui("portfolio.metrics.expense_ratio")] == "not provided"
    everything = " ".join(str(v) for f in frames.values() for v in f.to_numpy().ravel()) + " ".join(texts)
    assert "1,501,000" not in everything and "15,01,000" not in everything and "1501000" not in everything         # INR + USD is never shown


def test_the_positions_table_shows_each_holding_in_its_own_currency_with_display_formatting(fake, editor):
    fake.reply = reply()
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    frames = named_frames(at, ["measures_INR", "positions_INR", "measures_USD", "positions_USD"])
    labels = [ui(f"portfolio.positions.{k}") for k in ("ticker", "instrument_type", "asset_class", "quantity", "avg_cost", "price", "value", "weight", "pnl", "pnl_pct")]
    assert list(frames["positions_INR"].columns) == labels
    first, second = frames["positions_INR"].iloc[0], frames["positions_INR"].iloc[1]
    na = ui("portfolio.not_available")
    assert list(first) == ["SUZLON.NS", "Stock", "Equity", "1,000", "INR 1,000.00", "INR 1,250.50", "INR 12,50,500.00", "83.4%", "INR 2,50,500.00", "+25.1%"]
    assert list(second) == ["GOLDBEES.NS", "ETF", "Gold", "1,000", na, "INR 249.50", "INR 2,49,500.00", "16.6%", na, na]
    assert list(frames["positions_USD"].iloc[0]) == ["AAPL", "Stock", "Equity", "5", na, "USD 200.00", "USD 1,000.00", "100.0%", na, na]


def test_freshness_and_warnings_come_from_the_payload_as_plain_text(fake, editor):
    data = mixed_data(source="stale cache", is_stale=True, warnings=["SUZLON.NS was slow.", f"Odd: {HOSTILE}"])
    fake.reply = reply("Answer.", data=data)
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    line = ui("freshness.with_time", source="stale cache", as_of="2026-10-30 08:32 UTC")
    assert line in warnings(at)
    assert plain_text("SUZLON.NS was slow.") in warnings(at)
    odd = [w for w in warnings(at) if w.startswith("Odd:")][0]
    assert "<script" not in odd and "javascript:" not in odd and "\\<" not in odd.replace("&lt;", "")


@pytest.mark.parametrize("source, stale, kind", [("live", False, "success"), ("cache", False, "info")])
def test_a_live_or_cached_portfolio_price_is_green_or_blue(fake, editor, source, stale, kind):
    fake.reply = reply(data=mixed_data(source=source, is_stale=stale))
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    line = ui("freshness.with_time", source=source, as_of="2026-10-30 08:32 UTC")
    assert line in [e.value for e in getattr(at.main, kind)]


def test_needs_input_names_the_rows_next_to_the_table_and_shows_no_measures(fake, editor):
    text = 'Row "Globe Corp": I found more than one match for "Globe Corp": GLB (Globe Corp., NASDAQ, USD); GLBA (Globe Corp., NASDAQ, USD). Please type the ticker symbol you mean, for example GLB.\n\nRow "Qzxv": I couldn\'t find a listed company called "Qzxv".'
    fake.reply = answer_result(text, sources=[], intents=["portfolio"], results=[agent_result("needs_input", text, {"warnings": []})])
    at = start()
    editor.type_rows([["Globe Corp", 1.0, np.nan, "stock", "equity", np.nan], ["Qzxv", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    page = shown(at)
    assert any('Row "Globe Corp": I found more than one match' in t for t in page) and any('Row "Qzxv"' in t for t in page)
    assert ui("messages.needs_input_hint") in captions(at) and charts(at) == [] and dataframes(at) == []


def test_a_result_without_a_payload_or_without_prices_shows_only_the_answer_and_warnings(fake, editor):
    fake.reply = answer_result("No portfolio data.", sources=[], intents=["portfolio"], results=[])
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    assert charts(at) == [] and dataframes(at) == [] and any("No portfolio data." in t for t in shown(at))
    text = get_config()["portfolio"]["unpriced_message"].strip()
    fake.reply = answer_result(text, sources=[], intents=["portfolio"], results=[agent_result("no_answer", text, {"warnings": ["AAPL was left out: no price is available right now."]})])
    submit(at)
    assert plain_text("AAPL was left out: no price is available right now.") in warnings(at) and charts(at) == []


def test_a_currency_without_shares_has_no_pie_and_odd_text_is_shown_literally(fake, editor):
    weird = mixed_data()
    weird["metrics"]["currencies"] = {"INR": block(1000.0, 1, {}, "<b>X</b>", None, top3_weight=None, equity_share=None,
                                                     diversification_label="<i>odd</i>", risk_label="<u>High</u>")}
    weird["metrics"]["positions"] = [position("<b>X</b>", "INR", 10.0, 100.0, 1000.0, None, instrument_type="weird", asset_class="strange")]
    fake.reply = reply(data=weird)
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    assert charts(at) == []
    frames = named_frames(at, ["measures_INR", "positions_INR"])
    row = frames["positions_INR"].iloc[0]
    na = ui("portfolio.not_available")
    assert row.iloc[0] == "<b>X</b>" and row.iloc[1] == "weird" and row.iloc[2] == "strange" and row.iloc[7] == na       # table cells are data, never HTML
    measures = dict(zip(frames["measures_INR"]["measure"], frames["measures_INR"]["value"]))
    assert measures[ui("portfolio.metrics.largest")] == f"<b>X</b> ({na})" and measures[ui("portfolio.metrics.top3")] == na


@pytest.mark.parametrize("stopped", ["error", "recursion"])
def test_error_and_recursion_stops_are_friendly_warnings(fake, editor, stopped):
    fake.reply = answer_result("Something went wrong. Please try again.", sources=[], stopped=stopped)
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    assert any("Something went wrong" in w for w in warnings(at)) and charts(at) == []


def test_a_budget_stop_shows_the_ui_message(fake, editor):
    fake.reply = answer_result("backend text", sources=[], stopped="budget")
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    assert ui("messages.budget_stopped") in warnings(at) and not any("backend text" in t for t in shown(at))


# ============================== UI step 4c: tables are not Markdown ==============================

def test_table_cells_hold_raw_text_and_the_pie_label_is_html_escaped(fake, editor):
    data = mixed_data()
    data["metrics"]["positions"][1] = {**data["metrics"]["positions"][1], "instrument_type": "fund (new) - x", "asset_class": "<b>odd</b> (x)"}
    data["metrics"]["currencies"]["INR"]["allocation_by_asset_class"] = {"equity": 0.8337, "<b>odd</b> (x)": 0.1663}
    fake.reply = reply(data=data)
    at = start()
    editor.type_rows([["AAPL", 1.0, np.nan, "stock", "equity", np.nan]])
    submit(at)
    cells = [str(v) for frame in dataframes(at) for v in frame.value.to_numpy().ravel()]
    assert "fund (new) - x" in cells and "<b>odd</b> (x)" in cells and not [c for c in cells if "\\" in c]
    labels = [label for c in charts(at) if c["data"][0]["type"] == "pie" for label in c["data"][0]["labels"]]
    assert "&lt;b&gt;odd&lt;/b&gt; (x)" in labels and not [label for label in labels if "<b>" in label]
