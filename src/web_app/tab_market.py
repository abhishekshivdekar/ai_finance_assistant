"""The Market tab: tickers and a period go to the Market agent with intent_hint="market" (the router is skipped).

Everything shown comes from the MarketPayload of the result: freshness and warnings are the payload's, the history
chart and the comparison table are its `history` and `comparison` rows. The UI only formats; it calculates nothing.
Each ticker has its own chart because the tickers may be in different currencies and their prices must not share an axis.
"""
from __future__ import annotations

import html
from typing import Any, Callable

import plotly.graph_objects as go
import streamlit as st

from src.utils.config import get_config
from src.web_app import backend, session
from src.web_app.calls import run_waiting
from src.web_app.format import format_percent
from src.web_app.render_common import payload_of, render_freshness, render_result, render_warnings
from src.web_app.text import ui_text

TAB, HINT = "market", "market"
COLUMN_KEYS = ("ticker", "currency", "period_return_pct", "volatility_pct", "max_drawdown_pct", "trend")
PERCENT_KEYS = ("period_return_pct", "volatility_pct", "max_drawdown_pct")


def split_tickers(text: str) -> list[str]:
    """The comma-separated entries of the input box, trimmed, without empty ones. The backend validates them."""
    return [part.strip() for part in text.split(",") if part.strip()]


def _charts(data: dict[str, Any]) -> None:
    """One line chart of closing prices per ticker, titled with its currency, straight from `history`."""
    currencies = {t.get("ticker"): t.get("currency") for t in (data.get("metrics") or {}).get("tickers", [])}
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in data.get("history") or []:
        by_ticker.setdefault(str(row["ticker"]), []).append(row)
    if not by_ticker:
        return
    st.subheader(ui_text("market.chart_heading"))
    for ticker, rows in by_ticker.items():
        currency = currencies.get(ticker)
        figure = go.Figure(go.Scatter(x=[r["date"] for r in rows], y=[r["close"] for r in rows], mode="lines", name=ticker))
        figure.update_layout(
            title=html.escape(ui_text("market.chart_title", ticker=ticker, currency=currency) if currency else ticker),
            yaxis_title=html.escape(ui_text("market.chart_y", currency=currency) if currency else ui_text("market.chart_y_plain")),
            showlegend=False, height=320, margin={"l": 10, "r": 10, "t": 50, "b": 10},
        )
        st.plotly_chart(figure, key=f"price_{ticker}")


def _comparison(data: dict[str, Any]) -> None:
    """The comparison rows of the payload as a table with display-formatted percentages."""
    rows = data.get("comparison") or []
    if not rows:
        return
    st.subheader(ui_text("market.table_heading"))
    missing = ui_text("market.not_available")
    table = [
        {ui_text(f"market.columns.{key}"): format_percent(row.get(key), missing) if key in PERCENT_KEYS else str(row.get(key) or missing)
         for key in COLUMN_KEYS}
        for row in rows
    ]
    st.dataframe(table, hide_index=True)


def _render_result(result: backend.QueryResult) -> None:
    """The answer, then freshness and warnings of the payload, then the charts and the table."""
    render_result(result)
    data = payload_of(result, "market")
    if data is None:
        return
    render_freshness(str(data.get("source") or ""), str(data.get("as_of") or ""), bool(data.get("is_stale")), list(data.get("freshness_texts") or []))
    render_warnings(data)
    _charts(data)
    _comparison(data)


def render(budget_exceeded: bool) -> Callable[[], None] | None:
    """Draw the form and the last result; returns the deferred call when a request is waiting (made after every tab is drawn)."""
    busy = session.is_busy()
    cfg = get_config()["market_analysis"]
    periods = list(cfg["periods"])
    st.markdown(ui_text("market.intro", max_tickers=cfg["max_tickers"]))
    if budget_exceeded:
        st.warning(ui_text("messages.budget_stopped"))
    with st.form("market_form"):
        text = st.text_input(
            ui_text("market.tickers_label"), placeholder=ui_text("market.tickers_placeholder"),
            help=ui_text("market.tickers_help"), disabled=busy,
        )
        period = st.selectbox(
            ui_text("market.period_label"), periods, index=periods.index(cfg["default_period"]),
            format_func=lambda p: cfg["period_labels"][p], disabled=busy,
        )
        submitted = st.form_submit_button(ui_text("market.submit"), disabled=busy or budget_exceeded)
    if submitted:
        tickers = split_tickers(text)
        if not tickers:
            st.warning(ui_text("market.tickers_required"))
        else:
            query = ui_text("market.query_template", tickers=", ".join(tickers), period=cfg["period_labels"][period])
            session.submit(TAB, query, HINT, market_params={"tickers": tickers, "period": period})
            st.rerun()
    slot = st.container()                                  # the spinner of the waiting call is drawn here
    last = session.result_for(TAB)
    if last is not None:
        _render_result(last)
    return (lambda: run_waiting(TAB, slot, "market.working")) if session.has_pending(TAB) else None
