"""The Goals tab: a form goes to the Goal Planning agent with intent_hint="goal_planning" (the router is skipped).

Everything shown comes from the GoalPayload of the result: the lines are `projections`, the table is `scenarios`, the inflation
lines are `inflation`, the assumptions are `assumptions`, warnings are the payload's. The UI only formats; it calculates nothing.
Amounts are in the one currency of `goal_planning.currency` and are never converted.

Design pattern: FORM-AS-REQUEST (the same shape as the Market tab). Example: typing 2500000 in "Goal amount" shows the caption
"That is INR 25,00,000.00." at once (display only); pressing "Plan my goal" sends goal_params={"goal_amount": 2500000, ...} to
the backend, and the agent either plans it or answers with the labelled problems ("Time horizon (years) must be a number.").
The money boxes are text boxes that accept digits only, so the page can show the caption as typed and an empty goal amount or
horizon reaches the backend, which says in words what is missing. The widgets are not inside st.form, because a form would hold
back the caption until the button is pressed.
"""
from __future__ import annotations

import html
from typing import Any, Callable

import plotly.graph_objects as go
import streamlit as st

from src.utils.config import get_config
from src.web_app import backend, session
from src.web_app.calls import run_waiting
from src.web_app.format import format_amount, format_ratio_percent
from src.web_app.render_common import payload_of, plain_text, render_result, render_warnings
from src.web_app.text import ui_text

TAB, HINT, AGENT = "goals", "goal_planning", "goal_planning"
MONEY_KEYS = ("goal_amount", "current_savings", "monthly_contribution")
DIGIT_KEYS = (*MONEY_KEYS, "horizon_years")
EMPTY, BAD = "empty", "bad"


def read_digits(text: str) -> int | str:
    """The whole number typed in a box, EMPTY for a blank box, or BAD for anything but ASCII digits. No limits are checked here."""
    text = text.strip()
    if not text:
        return EMPTY
    return int(text) if text.isascii() and text.isdigit() else BAD


def _scenario(name: Any, where: str = "markdown") -> str:
    """The UI label of a scenario. An unknown name is untrusted text and is made safe for where it is shown:
    "chart" = HTML-escaped, "table" = raw (a table cell is not Markdown, so escaping would show backslashes), "markdown" = plain_text."""
    safe = {"chart": lambda text: html.escape(text), "table": lambda text: text, "markdown": plain_text}[where]
    return str(get_config()["ui"]["goals"]["scenario_labels"].get(name, safe(str(name))))


def _profile(name: Any) -> str:
    """The UI label of a risk profile; an unknown name is shown as plain text."""
    return str(get_config()["ui"]["goals"]["risk_labels"].get(name, plain_text(str(name))))


def _chart(data: dict[str, Any], currency: str) -> None:
    """One line per scenario from `projections` (today's money when inflation is applied), with the goal as a horizontal line."""
    rows = data.get("projections") or []
    if not rows:
        return
    inflation = data.get("inflation") or {}
    todays = bool(inflation)
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    figure = go.Figure()
    for name, points in by_scenario.items():
        values = [p["value_todays_money"] if todays and p.get("value_todays_money") is not None else p["value"] for p in points]
        colour = get_config()["ui"]["goals"]["line_colors"].get(name)
        figure.add_trace(go.Scatter(x=[p["year"] for p in points], y=values, mode="lines", name=_scenario(name, "chart"), line={"color": colour} if colour else None))
    goal = inflation.get("goal_in_todays_money") if todays else data.get("target_amount")
    if goal is not None:
        figure.add_hline(y=goal, line_dash="dash", line_color=get_config()["ui"]["goals"]["goal_line_color"], annotation_text=html.escape(ui_text("goals.goal_line_todays" if todays else "goals.goal_line"), quote=False))
    figure.update_layout(
        title=html.escape(ui_text("goals.chart_title_todays" if todays else "goals.chart_title", currency=currency), quote=False),
        xaxis_title=html.escape(ui_text("goals.chart_x"), quote=False), yaxis_title=html.escape(ui_text("goals.chart_y", currency=currency), quote=False),
        height=360, margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.subheader(ui_text("goals.chart_heading"))
    st.plotly_chart(figure, key="goal_projection")


def _table(data: dict[str, Any], currency: str) -> None:
    """The scenarios as a table of display texts, then the distinct notes of the payload about the time to reach the goal."""
    scenarios = data.get("scenarios") or []
    if not scenarios:
        return
    todays = bool(data.get("inflation"))
    missing = ui_text("goals.not_available")
    table = []
    for s in scenarios:
        gap = format_amount(abs(float(s["shortfall"])), currency)
        months = s.get("months_to_goal")
        row = {
            ui_text("goals.columns.scenario"): _scenario(s["name"], "table"),
            ui_text("goals.columns.annual_return"): format_ratio_percent(s.get("annual_return_pct"), missing),
            ui_text("goals.columns.final_value"): format_amount(float(s["final_value"]), currency),
        }
        if todays:
            value = s.get("final_value_todays_money")
            row[ui_text("goals.columns.final_value_todays")] = missing if value is None else format_amount(float(value), currency)
        row[ui_text("goals.columns.outcome")] = ui_text("goals.surplus" if s.get("goal_met") else "goals.shortfall", amount=gap)
        row[ui_text("goals.columns.required")] = format_amount(float(s["required_monthly_contribution"]), currency)
        row[ui_text("goals.columns.months")] = ui_text("goals.months_value", months=months) if months is not None else str(s.get("months_note") or missing)
        table.append(row)
    st.subheader(ui_text("goals.table_heading"))
    st.dataframe(table, hide_index=True, key="goal_scenarios")
    _required_notes(data)
    for note in dict.fromkeys(str(s["months_note"]) for s in scenarios if s.get("months_note")):
        st.caption(plain_text(note))


def _required_notes(data: dict[str, Any]) -> None:
    """What the "Monthly amount needed" column means, once, from the payload's `required_labels` (unchanged text).

    One caption when every scenario has the same label; otherwise one caption per label, led by the scenarios it belongs to.
    """
    groups: dict[str, list[str]] = {}
    for name, label in (data.get("required_labels") or {}).items():
        groups.setdefault(str(label), []).append(_scenario(name))
    for label, names in groups.items():
        text = plain_text(label)
        st.caption(text if len(groups) == 1 else ui_text("goals.required_note", scenarios=", ".join(names), label=text))


def _inflation(data: dict[str, Any], currency: str) -> None:
    """The goal in today's and in future money, from the payload's `inflation` block (empty when inflation is not applied)."""
    inflation = data.get("inflation") or {}
    if not inflation:
        return
    st.markdown(f"**{ui_text('goals.inflation_heading')}**")
    st.markdown(ui_text("goals.inflation_todays", amount=format_amount(float(inflation["goal_in_todays_money"]), currency)))
    st.markdown(ui_text("goals.inflation_future", amount=format_amount(float(inflation["goal_in_future_money"]), currency)))
    st.markdown(ui_text("goals.inflation_rate", rate=format_ratio_percent(inflation["annual_pct"], "")))


def _assumptions(data: dict[str, Any]) -> None:
    """The assumptions block: its label is always visible; the details (date, source, rates, timing notes) are in an expander."""
    block = data.get("assumptions") or {}
    if not block:
        return
    st.caption(plain_text(str(block.get("label") or "")))
    with st.expander(ui_text("goals.assumptions_title")):
        st.markdown(plain_text(ui_text("goals.assumptions_asof", as_of=str(block.get("as_of") or ""), source=str(block.get("source") or ""))))
        returns = ", ".join(f"{_scenario(name)} {format_ratio_percent(value, '')}" for name, value in (block.get("annual_returns_pct") or {}).items())
        st.markdown(ui_text("goals.assumptions_returns", profile=_profile(block.get("risk_profile")), returns=returns))
        inflation = block.get("inflation") or {}
        if inflation:
            applied = ui_text("goals.assumptions_applied" if inflation.get("applied") else "goals.assumptions_not_applied")
            st.markdown(ui_text("goals.assumptions_inflation", rate=format_ratio_percent(inflation.get("annual_pct"), ""), applied=applied))
        for key in ("deposit_timing", "step_up_timing", "return_convention"):
            if block.get(key):
                st.caption(plain_text(str(block[key])))


def _goal_status(result: backend.QueryResult) -> str | None:
    """The status of the Goal Planning agent's result in this turn, or None."""
    return next((item["status"] for item in result.results if item["agent"] == AGENT), None)


def _render_result(result: backend.QueryResult) -> None:
    """The answer, the warnings, then the chart, the table, the inflation lines and the assumptions (only for a planned goal)."""
    render_result(result)
    data = payload_of(result, AGENT)
    if data is None or _goal_status(result) == "needs_input":          # a needs_input answer already lists its problems
        return
    currency = get_config()["goal_planning"]["currency"]
    render_warnings(data)
    _chart(data, currency)
    _table(data, currency)
    _inflation(data, currency)
    _assumptions(data)


def _amount_box(key: str, busy: bool, max_chars: int, currency: str, placeholder_key: str = "goals.money_placeholder") -> str:
    """A digits-only text box with its caption "That is INR ..." under it (display only). Returns the typed text."""
    text = st.text_input(
        ui_text(f"goals.{key}"), key=f"goals_{key}", help=ui_text(f"goals.{key}_help"), placeholder=ui_text(placeholder_key),
        max_chars=max_chars, disabled=busy,
    )
    typed = read_digits(text)
    if isinstance(typed, int):
        st.caption(ui_text("goals.amount_caption", amount=format_amount(float(typed), currency)))
    return text


def render(budget_exceeded: bool) -> Callable[[], None] | None:
    """Draw the form and the last result; returns the deferred call when a request is waiting (made after every tab is drawn)."""
    busy = session.is_busy()
    cfg = get_config()["goal_planning"]
    currency, chars = cfg["currency"], len(str(cfg["max_amount"]))
    st.markdown(ui_text("goals.intro"))
    if budget_exceeded:
        st.warning(ui_text("messages.budget_stopped"))
    texts = {"goal_amount": _amount_box("goal_amount", busy, chars, currency)}
    texts["horizon_years"] = st.text_input(
        ui_text("goals.horizon_years"), key="goals_horizon_years", help=ui_text("goals.horizon_years_help"),
        placeholder=ui_text("goals.horizon_years_placeholder"), max_chars=len(str(cfg["max_horizon_years"])) + 2, disabled=busy,
    )
    texts["current_savings"] = _amount_box("current_savings", busy, chars, currency)
    texts["monthly_contribution"] = _amount_box("monthly_contribution", busy, chars, currency)
    profiles = list(cfg["risk_profiles"])
    profile = st.selectbox(
        ui_text("goals.risk_profile"), profiles, index=profiles.index(get_config()["ui"]["goals"]["default_risk_profile"]),
        format_func=_profile, help=ui_text("goals.risk_profile_help"), key="goals_risk_profile", disabled=busy,
    )
    step_up = st.number_input(
        ui_text("goals.annual_step_up_pct"), min_value=0.0, max_value=float(cfg["max_step_up_pct"]), value=0.0, step=1.0,
        help=ui_text("goals.annual_step_up_pct_help"), key="goals_step_up", disabled=busy,
    )
    inflation = st.checkbox(
        ui_text("goals.inflation_adjust"), value=bool(cfg["inflation_adjust_default"]), help=ui_text("goals.inflation_adjust_help"),
        key="goals_inflation", disabled=busy,
    )
    if st.button(ui_text("goals.submit"), key="goals_submit", disabled=busy or budget_exceeded):
        typed = {key: read_digits(texts[key]) for key in DIGIT_KEYS}
        wrong = [key for key, value in typed.items() if value == BAD]
        for key in wrong:
            st.warning(ui_text("goals.digits_only", field=ui_text(f"goals.{key}")))
        if not wrong:
            params: dict[str, Any] = {key: value for key, value in typed.items() if isinstance(value, int)}
            params.update(risk_profile=str(profile), annual_step_up_pct=float(step_up), inflation_adjust=bool(inflation))
            session.submit(TAB, ui_text("goals.query_template"), HINT, goal_params=params)
            st.rerun()
    slot = st.container()                                  # the spinner of the waiting call is drawn here
    last = session.result_for(TAB)
    if last is not None:
        _render_result(last)
    return (lambda: run_waiting(TAB, slot, "goals.working")) if session.has_pending(TAB) else None
