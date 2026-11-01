"""The Portfolio tab: an editable table of holdings goes to the Portfolio agent with intent_hint="portfolio" (the router is skipped).

Everything shown comes from the PortfolioPayload of the result: the pie is `allocation_by_asset_class`, the measures are
`metrics.currencies`, the table is `metrics.positions`, freshness and warnings are the payload's. The UI only formats; it
calculates nothing and it never adds amounts of different currencies: each currency has its own section.

Design pattern: FORM around a DATA EDITOR. The table lives inside `st.form`, so edits are sent to the page only when the user
presses the submit button (one rerun per analysis, not one per cell). Example: typing "Suzlon" in the first cell and "10" in the
quantity cell, then pressing Analyse, sends [{"ticker": "Suzlon", "quantity": 10.0, "instrument_type": "stock", ...}] to the
backend; the Portfolio agent turns the name into SUZLON.NS or asks which listing is meant.
"""
from __future__ import annotations

import html
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.config import get_config
from src.web_app import backend, session
from src.web_app.calls import run_waiting
from src.web_app.format import (
    clean_portfolio_rows, format_amount, format_percent, format_quantity, format_ratio_percent, format_share,
)
from src.web_app.render_common import payload_of, plain_text, render_freshness, render_result, render_warnings
from src.web_app.text import ui_text

TAB, HINT = "portfolio", "portfolio"
EDITOR_KEY = "portfolio_table"
COLUMNS = ("ticker", "quantity", "avg_cost", "instrument_type", "asset_class", "expense_ratio_pct")
TEXT_COLUMNS = ("ticker", "instrument_type", "asset_class")


def empty_table() -> pd.DataFrame:
    """The table the user starts from: the six columns, no rows (no example data)."""
    return pd.DataFrame({column: pd.Series(dtype="object" if column in TEXT_COLUMNS else "float64") for column in COLUMNS})


def _label(group: str, key: str) -> str:
    """The UI label of an instrument type or asset class; an unknown key is shown as it is (plain text)."""
    return str(get_config()["ui"]["portfolio"][group].get(key, plain_text(str(key))))


def column_config() -> dict[str, Any]:
    """The editor's columns: labels and help from config.ui, dropdown choices and defaults from config.portfolio."""
    cfg = get_config()
    pf, lookup = cfg["portfolio"], cfg["ticker_lookup"]
    columns = st.column_config
    return {
        "ticker": columns.TextColumn(ui_text("portfolio.columns.ticker"), help=ui_text("portfolio.column_help.ticker"), max_chars=lookup["max_query_chars"]),
        "quantity": columns.NumberColumn(ui_text("portfolio.columns.quantity"), help=ui_text("portfolio.column_help.quantity"), min_value=0.0),
        "avg_cost": columns.NumberColumn(ui_text("portfolio.columns.avg_cost"), help=ui_text("portfolio.column_help.avg_cost"), min_value=0.0),
        "instrument_type": columns.SelectboxColumn(
            ui_text("portfolio.columns.instrument_type"), help=ui_text("portfolio.column_help.instrument_type"), options=list(pf["instrument_types"]),
            default=pf["defaults"]["instrument_type"], format_func=lambda key: _label("instrument_labels", key),
        ),
        "asset_class": columns.SelectboxColumn(
            ui_text("portfolio.columns.asset_class"), help=ui_text("portfolio.column_help.asset_class"), options=list(pf["asset_classes"]),
            default=pf["defaults"]["asset_class"], format_func=lambda key: _label("asset_class_labels", key),
        ),
        "expense_ratio_pct": columns.NumberColumn(ui_text("portfolio.columns.expense_ratio_pct"), help=ui_text("portfolio.column_help.expense_ratio_pct"), min_value=0.0),
    }


def _money(value: Any, currency: str, missing: str) -> str:
    """An amount of one currency, or the "not available" text."""
    return missing if value is None else format_amount(float(value), currency)


def _allocation_chart(currency: str, shares: dict[str, float]) -> None:
    """One pie per currency from the payload's shares (they sum to 1 inside that currency)."""
    labels = [_label("asset_class_labels", key) for key in shares]
    figure = go.Figure(go.Pie(labels=labels, values=list(shares.values()), textinfo="label+percent", sort=False))
    figure.update_layout(title=html.escape(ui_text("portfolio.allocation_title", currency=currency)), height=320, margin={"l": 10, "r": 10, "t": 50, "b": 10})
    st.plotly_chart(figure, key=f"allocation_{currency}")


def _measures(currency: str, block: dict[str, Any]) -> None:
    """The currency's measures as a two-column table of display texts."""
    missing = ui_text("portfolio.not_available")
    weighted = block.get("expense_ratio_weighted_pct")
    coverage = str(block.get("expense_ratio_coverage") or missing)
    rows = {
        "total_value": format_amount(float(block["total_value"]), currency),
        "position_count": str(block["position_count"]),
        "largest": ui_text("portfolio.largest_value", ticker=block["largest_ticker"], share=format_share(block.get("largest_weight"), missing)),
        "top3": format_share(block.get("top3_weight"), missing),
        "diversification": ui_text("portfolio.diversification_value", score=block["diversification_score"], label=block["diversification_label"]),
        "risk": str(block["risk_label"]),
        "equity_share": format_share(block.get("equity_share"), missing),
        "expense_ratio": coverage if weighted is None else ui_text("portfolio.expense_value", percent=format_ratio_percent(weighted, missing), coverage=coverage),
    }
    st.dataframe([{"measure": ui_text(f"portfolio.metrics.{key}"), "value": value} for key, value in rows.items()], hide_index=True, key=f"measures_{currency}")


def _positions(currency: str, positions: list[dict[str, Any]]) -> None:
    """The holdings of one currency; every amount is in that currency."""
    missing = ui_text("portfolio.not_available")
    table = [
        {
            ui_text("portfolio.positions.ticker"): str(p["ticker"]),
            ui_text("portfolio.positions.instrument_type"): _label("instrument_labels", p["instrument_type"]),
            ui_text("portfolio.positions.asset_class"): _label("asset_class_labels", p["asset_class"]),
            ui_text("portfolio.positions.quantity"): format_quantity(p["quantity"]),
            ui_text("portfolio.positions.avg_cost"): _money(p.get("avg_cost"), currency, missing),
            ui_text("portfolio.positions.price"): format_amount(p["price"], currency),
            ui_text("portfolio.positions.value"): format_amount(p["value"], currency),
            ui_text("portfolio.positions.weight"): format_share(p.get("weight"), missing),
            ui_text("portfolio.positions.pnl"): _money(p.get("pnl"), currency, missing),
            ui_text("portfolio.positions.pnl_pct"): format_percent(p.get("pnl_pct"), missing),
        }
        for p in positions
    ]
    st.dataframe(table, hide_index=True, key=f"positions_{currency}")


def _render_result(result: backend.QueryResult) -> None:
    """The answer, then freshness and warnings of the payload, then one section per currency (never added together)."""
    render_result(result)
    data = payload_of(result, "portfolio")
    if data is None:
        return
    render_freshness(str(data.get("source") or ""), str(data.get("as_of") or ""), bool(data.get("is_stale")), [])
    render_warnings(data)
    metrics = data.get("metrics") or {}
    positions = metrics.get("positions") or []
    for currency, block in (metrics.get("currencies") or {}).items():
        st.subheader(ui_text("portfolio.currency_heading", currency=plain_text(str(currency))))
        shares = block.get("allocation_by_asset_class") or {}
        if shares:
            st.markdown(f"**{ui_text('portfolio.allocation_heading')}**")
            _allocation_chart(str(currency), shares)
        st.markdown(f"**{ui_text('portfolio.metrics_heading')}**")
        _measures(str(currency), block)
        st.markdown(f"**{ui_text('portfolio.positions_heading')}**")
        _positions(str(currency), [p for p in positions if p.get("currency") == currency])


def render(budget_exceeded: bool) -> Callable[[], None] | None:
    """Draw the table form and the last result; returns the deferred call when a request is waiting (made after every tab is drawn)."""
    busy = session.is_busy()
    st.markdown(ui_text("portfolio.intro"))
    if budget_exceeded:
        st.warning(ui_text("messages.budget_stopped"))
    with st.form("portfolio_form"):
        edited = st.data_editor(
            empty_table(), key=EDITOR_KEY, num_rows="dynamic", hide_index=True, column_config=column_config(), width="stretch",
        )
        submitted = st.form_submit_button(ui_text("portfolio.submit"), disabled=busy or budget_exceeded)
    if submitted:
        rows = clean_portfolio_rows(edited.to_dict("records"))           # plain dicts only: nothing from pandas or numpy goes on
        if not rows:
            st.warning(ui_text("portfolio.rows_required"))
        else:
            session.submit(TAB, ui_text("portfolio.query_template", count=len(rows)), HINT, portfolio=rows)
            st.rerun()
    slot = st.container()                                  # the spinner of the waiting call is drawn here
    last = session.result_for(TAB)
    if last is not None:
        _render_result(last)
    return (lambda: run_waiting(TAB, slot, "portfolio.working")) if session.has_pending(TAB) else None
