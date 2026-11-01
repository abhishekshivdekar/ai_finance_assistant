"""The Goals tab (UI step 4b). The payloads are REAL ones: the Goal Planning agent is run with a fake LLM and the result is handed
to the page through the fake backend, so the tests check the mapping from the real payload fields to the widgets."""
from __future__ import annotations

import copy

import pytest
import streamlit as st

from src.agents.goal_planning import GoalPlanningAgent
from src.utils.config import get_config
from src.web_app import backend
from src.web_app.format import format_amount, format_ratio_percent
from src.web_app.render_common import plain_text
from src.web_app.tab_goals import BAD, EMPTY, read_digits
from web_support import HOSTILE, FakeBackend, answer_result, charts, dataframes, disclaimer, fake, labelled, shown, start, ui  # noqa: F401

PARAMS = {"goal_amount": 2_500_000, "horizon_years": 10, "current_savings": 200_000, "monthly_contribution": 10_000,
          "risk_profile": "medium", "annual_step_up_pct": 0.0, "inflation_adjust": True}
COMMENTARY = "These are illustrative assumptions, not predictions."


class Llm:
    """A model that writes a short commentary without numbers."""

    def invoke(self, prompt):
        return COMMENTARY


def planned(**changes):
    """The real AgentResult of the Goal Planning agent for the sample goal with some inputs changed (None removes an input)."""
    params = {key: value for key, value in {**PARAMS, **changes}.items() if value is not None}
    return GoalPlanningAgent(llm=Llm()).run({"turn_id": "t", "goal_params": params})


def reply(result, text=None):
    """What the facade returns for a hinted goal turn."""
    return answer_result(text or result["answer"], sources=[], intents=["goal_planning"], results=[result])


def fill(at, goal="", years="", savings="", monthly="", risk=None, step=None, inflation=None):
    """Type into the form (the values are applied on the next run)."""
    labelled(at.text_input, ui("goals.goal_amount")).input(goal)
    labelled(at.text_input, ui("goals.horizon_years")).input(years)
    labelled(at.text_input, ui("goals.current_savings")).input(savings)
    labelled(at.text_input, ui("goals.monthly_contribution")).input(monthly)
    if risk is not None:
        labelled(at.selectbox, ui("goals.risk_profile")).select(risk)
    if step is not None:
        labelled(at.number_input, ui("goals.annual_step_up_pct")).set_value(step)
    if inflation is not None:
        labelled(at.checkbox, ui("goals.inflation_adjust")).set_value(inflation)
    return at


def plan_button(at):
    return next(b for b in at.main.button if b.label == ui("goals.submit"))


def submit(at, **form):
    fill(at, **form)
    plan_button(at).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def sample(at, **form):
    """Scenario 1 of the browser checks: 25,00,000 in 10 years, savings 2,00,000, monthly 10,000, medium, step-up 0, inflation on."""
    return submit(at, **{"goal": "2500000", "years": "10", "savings": "200000", "monthly": "10000", **form})


def warnings(at):
    return [w.value for w in at.main.warning]


def captions(at):
    return [c.value for c in at.main.caption]


# ============================== the form ==============================

def test_the_tab_sits_between_portfolio_and_market_and_the_form_starts_empty_with_the_defaults_from_the_config(fake):
    at = start()
    assert [t.label for t in at.tabs] == [ui("tabs.chat"), ui("tabs.portfolio"), ui("tabs.goals"), ui("tabs.market"), ui("tabs.news")]
    assert labelled(at.text_input, ui("goals.goal_amount")).value == "" and labelled(at.text_input, ui("goals.horizon_years")).value == ""
    assert labelled(at.text_input, ui("goals.current_savings")).value == "" and labelled(at.text_input, ui("goals.monthly_contribution")).value == ""
    risk = labelled(at.selectbox, ui("goals.risk_profile"))
    assert risk.value == "medium" and list(risk.options) == [get_config()["ui"]["goals"]["risk_labels"][p] for p in get_config()["goal_planning"]["risk_profiles"]]
    step = labelled(at.number_input, ui("goals.annual_step_up_pct"))
    assert step.value == 0.0 and step.min == 0.0 and step.max == float(get_config()["goal_planning"]["max_step_up_pct"])
    assert labelled(at.checkbox, ui("goals.inflation_adjust")).value is get_config()["goal_planning"]["inflation_adjust_default"]
    assert not plan_button(at).disabled and fake.asks == []
    assert not [c for c in captions(at) if c.startswith("That is")]


def test_the_labels_and_help_texts_come_from_the_config_and_say_digits_only(fake):
    at = start()
    for key in ("goal_amount", "horizon_years", "current_savings", "monthly_contribution"):
        assert labelled(at.text_input, ui(f"goals.{key}")).help == ui(f"goals.{key}_help") and "igits only" in ui(f"goals.{key}_help")
    assert ui("goals.intro") in shown(at)
    assert not [text for text in shown(at) if "_" in text and text.split()[0] in get_config()["field_labels"]]


@pytest.mark.parametrize("typed, shown_amount", [("2500000", "INR 25,00,000.00"), ("0", "INR 0.00"), ("1000000000000", "INR 10,00,00,00,00,000.00")])
def test_a_caption_under_each_money_box_shows_the_amount_as_typed_in_lakh_and_crore(fake, typed, shown_amount):
    at = start()
    fill(at, goal=typed, savings=typed, monthly=typed).run()
    assert captions(at).count(ui("goals.amount_caption", amount=shown_amount)) == 3 and shown_amount == format_amount(float(typed), "INR")


@pytest.mark.parametrize("typed", ["", "abc", "25,00,000", "12.5", "-5", "²", " "])
def test_no_caption_is_shown_for_empty_or_non_digit_text(fake, typed):
    at = start()
    fill(at, goal=typed).run()
    assert not [c for c in captions(at) if c.startswith("That is")]


def test_read_digits_accepts_ascii_digits_only():
    assert read_digits(" 42 ") == 42 and read_digits("007") == 7 and read_digits("") == EMPTY and read_digits("  ") == EMPTY
    assert all(read_digits(text) == BAD for text in ("4 2", "4.2", "4,2", "-4", "²", "٣", "1e5", "ten"))


# ============================== a planned goal ==============================

def test_the_sample_goal_calls_the_facade_once_with_plain_params_and_every_payload_field_has_its_widget(fake):
    result = planned()
    data = result["data"]
    fake.reply = reply(result)
    at = start()
    sample(at)
    [args, kwargs] = fake.asks[0]
    assert args[0] == ui("goals.query_template") and args[2] == "goal_planning" and len(fake.asks) == 1
    assert kwargs == {"goal_params": {**PARAMS, "annual_step_up_pct": 0.0}} and kwargs["goal_params"]["goal_amount"] == 2_500_000
    assert all(type(v) in (int, float, str, bool) for v in kwargs["goal_params"].values())
    texts = shown(at)
    assert COMMENTARY in " ".join(texts) and sum(t.count(disclaimer()) for t in texts) == 1
    assert warnings(at) == [plain_text(w) for w in data["warnings"]]
    # chart: one line per scenario, in today's money because inflation is on, plus the goal line
    [chart] = charts(at)
    names = [trace["name"] for trace in chart["data"]]
    assert names == ["Conservative", "Expected", "Optimistic"]
    for trace, name in zip(chart["data"], ("conservative", "expected", "optimistic")):
        rows = [r for r in data["projections"] if r["scenario"] == name]
        assert trace["x"] == [r["year"] for r in rows] and trace["y"] == pytest.approx([r["value_todays_money"] for r in rows])
    assert chart["layout"]["title"]["text"] == ui("goals.chart_title_todays", currency="INR")
    assert chart["layout"]["shapes"][0]["y0"] == data["inflation"]["goal_in_todays_money"] and ui("goals.chart_heading") in texts
    # table: one row per scenario, the payload's numbers formatted for display
    [table] = dataframes(at)
    frame = table.value
    assert list(frame.columns) == [ui(f"goals.columns.{key}") for key in ("scenario", "annual_return", "final_value", "final_value_todays", "outcome", "required", "months")]
    for (_, row), s in zip(frame.iterrows(), data["scenarios"]):
        assert row[ui("goals.columns.annual_return")] == format_ratio_percent(s["annual_return_pct"], "")
        assert row[ui("goals.columns.final_value")] == format_amount(s["final_value"], "INR")
        assert row[ui("goals.columns.final_value_todays")] == format_amount(s["final_value_todays_money"], "INR")
        assert row[ui("goals.columns.required")].startswith(format_amount(s["required_monthly_contribution"], "INR"))
        assert row[ui("goals.columns.outcome")].startswith("Surplus" if s["goal_met"] else "Shortfall")
    # inflation lines and assumptions
    assert ui("goals.inflation_todays", amount=format_amount(data["inflation"]["goal_in_todays_money"], "INR")) in texts
    assert ui("goals.inflation_future", amount=format_amount(data["inflation"]["goal_in_future_money"], "INR")) in texts
    assert plain_text(data["assumptions"]["label"]) in captions(at)
    assert [e.label for e in at.expander] == [ui("goals.assumptions_title")]
    assert plain_text(ui("goals.assumptions_asof", as_of=data["assumptions"]["as_of"], source=data["assumptions"]["source"])) in texts
    assert ui("goals.assumptions_returns", profile="Medium", returns="Conservative 5.00%, Expected 9.00%, Optimistic 12.00%") in texts
    for key in ("deposit_timing", "step_up_timing", "return_convention"):
        assert plain_text(data["assumptions"][key]) in captions(at)


def test_without_inflation_the_chart_uses_the_plain_values_and_the_inflation_lines_and_column_are_absent(fake):
    result = planned(inflation_adjust=False)
    fake.reply = reply(result)
    at = start()
    sample(at, inflation=False)
    assert fake.asks[0][1]["goal_params"]["inflation_adjust"] is False
    [chart] = charts(at)
    rows = [r for r in result["data"]["projections"] if r["scenario"] == "expected"]
    assert chart["data"][1]["y"] == pytest.approx([r["value"] for r in rows]) and chart["layout"]["title"]["text"] == ui("goals.chart_title", currency="INR")
    assert chart["layout"]["shapes"][0]["y0"] == result["data"]["target_amount"]
    assert ui("goals.columns.final_value_todays") not in dataframes(at)[0].value.columns and ui("goals.inflation_heading") not in " ".join(shown(at))
    assert ui("goals.assumptions_inflation", rate="5.00%", applied=ui("goals.assumptions_not_applied")) in shown(at)


def test_a_goal_that_is_not_reached_shows_the_payload_note_and_a_goal_already_met_shows_a_surplus(fake):
    hard = planned(goal_amount=1_000_000_000, horizon_years=5, current_savings=0, monthly_contribution=5_000, risk_profile="high", annual_step_up_pct=10.0)
    fake.reply = reply(hard)
    at = start()
    submit(at, goal="1000000000", years="5", savings="", monthly="5000", risk="high", step=10.0)
    assert fake.asks[0][1]["goal_params"] == {"goal_amount": 1_000_000_000, "horizon_years": 5, "monthly_contribution": 5000, "risk_profile": "high",
                                              "annual_step_up_pct": 10.0, "inflation_adjust": True}
    frame = dataframes(at)[0].value
    for (_, row), s in zip(frame.iterrows(), hard["data"]["scenarios"]):
        assert s["months_to_goal"] is None and row[ui("goals.columns.months")] == s["months_note"] and row[ui("goals.columns.outcome")].startswith("Shortfall")
    assert plain_text(hard["data"]["scenarios"][0]["months_note"]) in captions(at)
    easy = planned(goal_amount=500_000, horizon_years=3, current_savings=600_000, monthly_contribution=0, risk_profile="low", inflation_adjust=False)
    fake.reply = reply(easy)
    submit(at, goal="500000", years="3", savings="600000", monthly="0", risk="low", inflation=False)
    frame = dataframes(at)[0].value
    assert all(text.startswith("Surplus of") for text in frame[ui("goals.columns.outcome")]) and len(fake.asks) == 2
    assert all(row == ui("goals.months_value", months=s["months_to_goal"]) for row, s in zip(frame[ui("goals.columns.months")], easy["data"]["scenarios"]))


def test_a_payload_warning_is_shown_as_plain_text(fake):
    result = planned(risk_profile="low")
    result["data"]["warnings"] = ["Low scenario: grows slower than prices.", f"Odd: {HOSTILE}"]
    fake.reply = reply(result)
    at = start()
    sample(at, risk="low")
    shown_warnings = warnings(at)
    assert plain_text("Low scenario: grows slower than prices.") in shown_warnings
    odd = next(w for w in shown_warnings if w.startswith("Odd:"))
    assert "<script" not in odd and "javascript:" not in odd and "&lt;script" in odd


# ============================== hostile text ==============================

def test_hostile_text_in_the_answer_assumptions_notes_and_labels_is_neutralised(fake):
    result = planned()
    result["data"]["assumptions"] = {**result["data"]["assumptions"], "source": HOSTILE, "label": HOSTILE, "deposit_timing": HOSTILE}
    result["data"]["scenarios"][0]["months_note"] = HOSTILE
    result["data"]["scenarios"][1]["name"] = HOSTILE
    result["data"]["scenarios"][2]["required_label"] = HOSTILE
    result["data"]["projections"] = [{**r, "scenario": HOSTILE if r["scenario"] == "expected" else r["scenario"]} for r in result["data"]["projections"]]
    fake.reply = reply(result, text=f"Plan: {HOSTILE}")
    at = start()
    sample(at)
    cells = [str(v) for v in dataframes(at)[0].value.to_numpy().ravel()]
    page = [t for t in shown(at) if ("script" in t or "javascript" in t) and t not in cells]          # table cells are not Markdown or HTML: raw text is inert there
    assert page and all("<script" not in t and "](javascript:" not in t for t in page)
    [chart] = charts(at)
    assert all("<script" not in trace["name"] for trace in chart["data"]) and "&lt;script" in " ".join(trace["name"] for trace in chart["data"])
    assert HOSTILE in cells and not any("\\" in c for c in cells if c.startswith(("INR", "+", "-")))


# ============================== needs_input and stopped values ==============================

def test_an_empty_goal_amount_reaches_the_backend_and_the_answer_names_it_by_its_label(fake):
    result = planned(goal_amount=None)
    assert result["status"] == "needs_input"
    fake.reply = reply(result)
    at = start()
    sample(at, goal="")
    assert "goal_amount" not in fake.asks[0][1]["goal_params"] and len(fake.asks) == 1
    page = " ".join(shown(at))
    assert "Goal amount must be a number." in page and "goal_amount" not in page and "horizon_years" not in page
    assert charts(at) == [] and dataframes(at) == [] and warnings(at) == [] and [e.label for e in at.expander] == []
    assert ui("messages.needs_input_hint") in captions(at)


def test_a_horizon_of_41_years_is_sent_and_the_range_message_is_shown(fake):
    result = planned(horizon_years=41)
    fake.reply = reply(result)
    at = start()
    sample(at, years="41")
    assert fake.asks[0][1]["goal_params"]["horizon_years"] == 41
    assert "The horizon must be between 1 and 40 years." in " ".join(shown(at)) and charts(at) == []


@pytest.mark.parametrize("field, bad", [("goal", "25,00,000"), ("years", "10.5"), ("savings", "2 lakh"), ("monthly", "-5")])
def test_text_that_is_not_digits_is_refused_in_the_page_and_nothing_is_sent(fake, field, bad):
    at = start()
    sample(at, **{field: bad})
    key = {"goal": "goal_amount", "years": "horizon_years", "savings": "current_savings", "monthly": "monthly_contribution"}[field]
    assert warnings(at) == [ui("goals.digits_only", field=ui(f"goals.{key}"))] and fake.asks == []
    assert at.session_state["active_tab"] is None and at.session_state["pending_request"] is None


def test_every_wrong_box_is_reported_at_once(fake):
    at = start()
    sample(at, goal="a", years="b")
    assert len(warnings(at)) == 2 and fake.asks == []


@pytest.mark.parametrize("stopped", ["error", "recursion"])
def test_error_and_recursion_stops_are_friendly_warnings(fake, stopped):
    fake.reply = backend.QueryResult(answer=f"Sorry, I couldn't prepare an answer this time.\n\n{disclaimer()}", stopped=stopped)
    at = start()
    sample(at)
    assert len(warnings(at)) == 1 and "couldn't prepare an answer" in warnings(at)[0] and charts(at) == [] and dataframes(at) == []


def test_a_budget_stop_shows_the_ui_message_not_the_backend_text(fake):
    fake.reply = backend.QueryResult(answer="backend text", stopped="budget")
    at = start()
    sample(at)
    assert warnings(at) == [ui("messages.budget_stopped")] and "backend text" not in " ".join(shown(at))


def test_a_reached_budget_disables_the_button_and_shows_the_message_in_the_tab(fake):
    fake.usage_value = backend.UsageSummary(tokens=2_000_001, cost=1.0, limit=2_000_000, exceeded=True)
    at = start()
    assert plan_button(at).disabled is True and warnings(at).count(ui("messages.budget_stopped")) == 5 and fake.asks == []


def test_a_result_without_a_payload_shows_only_the_answer(fake):
    fake.reply = answer_result("Please check the goal details.", sources=[], results=[{"agent": "goal_planning", "turn_id": "t", "status": "needs_input", "answer": "x", "sources": [], "error": None, "data": None}])
    at = start()
    sample(at)
    assert charts(at) == [] and dataframes(at) == [] and ui("messages.needs_input_hint") in captions(at)


def test_a_payload_with_no_projection_rows_or_scenarios_draws_no_chart_or_table(fake):
    result = planned()
    result["data"] = {"warnings": [], "inflation": {}, "assumptions": {}}
    fake.reply = reply(result)
    at = start()
    sample(at)
    assert charts(at) == [] and dataframes(at) == [] and [e.label for e in at.expander] == []


def test_a_payload_without_a_goal_value_draws_the_chart_without_a_goal_line(fake):
    result = planned(inflation_adjust=False)
    del result["data"]["target_amount"]
    fake.reply = reply(result)
    at = start()
    sample(at, inflation=False)
    assert "shapes" not in charts(at)[0]["layout"] or not charts(at)[0]["layout"]["shapes"]


def test_a_payload_with_missing_today_values_falls_back_to_the_plain_value_and_the_not_available_text(fake):
    result = planned()
    for row in result["data"]["projections"]:
        row["value_todays_money"] = None
    for s in result["data"]["scenarios"]:
        s["final_value_todays_money"] = None
        s["months_note"] = None
        s["months_to_goal"] = None
    fake.reply = reply(result)
    at = start()
    sample(at)
    frame = dataframes(at)[0].value
    assert set(frame[ui("goals.columns.final_value_todays")]) == {ui("goals.not_available")} and set(frame[ui("goals.columns.months")]) == {ui("goals.not_available")}
    assert charts(at)[0]["data"][0]["y"][0] == pytest.approx(result["data"]["projections"][0]["value"])


def test_a_payload_without_the_inflation_rate_in_the_assumptions_omits_that_line(fake):
    result = planned()
    result["data"]["assumptions"] = {key: value for key, value in result["data"]["assumptions"].items() if key != "inflation"}
    fake.reply = reply(result)
    at = start()
    sample(at)
    assert not [t for t in shown(at) if t.startswith("Illustrative inflation rate") and "(" in t]


# ============================== one request at a time ==============================

def test_reruns_never_call_the_facade_again_and_the_last_result_stays(fake):
    fake.reply = reply(planned())
    at = start()
    sample(at)
    for _ in range(3):
        at.run()
    at.sidebar.button[0].click().run()                                    # New conversation keeps the Goals result
    assert len(fake.asks) == 1 and len(charts(at)) == 1 and len(dataframes(at)) == 1
    sample(at, years="12")
    assert len(fake.asks) == 2


def test_the_button_is_drawn_disabled_while_a_call_runs_and_the_page_reopens_on_the_tab(fake, monkeypatch):
    seen, tab_defaults = [], []
    real_button, real_tabs = st.button, st.tabs
    monkeypatch.setattr(st, "button", lambda label, *a, **kw: seen.append((label, kw.get("disabled"))) or real_button(label, *a, **kw))
    monkeypatch.setattr(st, "tabs", lambda labels, **kw: tab_defaults.append(kw.get("default")) or real_tabs(labels, **kw))
    fake.reply = reply(planned())
    at = start()
    seen.clear(), tab_defaults.clear()
    sample(at)
    goals = ui("goals.submit")
    assert [s for s in seen if s[0] == goals] == [(goals, False), (goals, True), (goals, False)]
    assert tab_defaults == [None, ui("tabs.goals"), ui("tabs.goals")]
    tab_defaults.clear()
    at.run()
    assert tab_defaults == [None] and len(fake.asks) == 1



# ============================== the config ==============================

def _config(tmp_path, change):
    import yaml

    cfg = copy.deepcopy(dict(get_config()))
    get_config.cache_clear()
    change(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_a_valid_goals_block_loads(tmp_path):
    assert get_config(_config(tmp_path, lambda c: None))["ui"]["goals"]["default_risk_profile"] == "medium"
    get_config.cache_clear()


@pytest.mark.parametrize("change, text", [
    (lambda c: c["ui"]["goals"]["risk_labels"].pop("high"), "risk_labels"),
    (lambda c: c["ui"]["goals"]["risk_labels"].update(extra="Extra"), "risk_labels"),
    (lambda c: c["ui"]["goals"]["risk_labels"].update(low=" "), "risk_labels"),
    (lambda c: c["ui"]["goals"].pop("scenario_labels"), "scenario_labels"),
    (lambda c: c["ui"]["goals"]["scenario_labels"].pop("expected"), "scenario_labels"),
    (lambda c: c["ui"]["goals"].update(default_risk_profile="wild"), "default_risk_profile"),
    (lambda c: c["ui"]["goals"].pop("submit"), "ui.goals.submit"),
    (lambda c: c["ui"]["tabs"].pop("goals"), "ui.tabs.goals"),
])
def test_a_bad_goals_block_is_rejected(tmp_path, change, text):
    from src.utils.config import ConfigError

    path = _config(tmp_path, change)
    with pytest.raises(ConfigError, match=text):
        get_config(path)
    get_config.cache_clear()


# ============================== UI step 4c: browser fixes ==============================

def test_table_cells_hold_raw_text_so_no_backslash_is_shown(fake):
    result = planned(annual_step_up_pct=10.0)
    notes = [s["required_label"] for s in result["data"]["scenarios"]]
    result["data"]["scenarios"][0]["months_to_goal"] = None
    result["data"]["scenarios"][0]["months_note"] = "Not reached (in 600 months) - try again, 10% a year."
    fake.reply = reply(result)
    at = start()
    sample(at, step=10.0)
    cells = [str(v) for v in dataframes(at)[0].value.to_numpy().ravel()]
    assert not [c for c in cells if "\\" in c] and "Not reached (in 600 months) - try again, 10% a year." in cells
    assert any("rises 10%" in n for n in notes)          # the real label has parentheses and a percent sign


def test_the_monthly_amount_cell_holds_only_the_amount_and_the_payload_label_is_shown_once_as_a_caption(fake):
    result = planned()
    label = result["data"]["required_labels"]["expected"]
    fake.reply = reply(result)
    at = start()
    sample(at)
    column = dataframes(at)[0].value[ui("goals.columns.required")]
    assert list(column) == [format_amount(s["required_monthly_contribution"], "INR") for s in result["data"]["scenarios"]]
    assert captions(at).count(plain_text(label)) == 1


def test_labels_that_differ_between_scenarios_are_captioned_with_their_scenario_names(fake):
    result = planned(annual_step_up_pct=10.0)
    result["data"]["required_labels"] = {"conservative": "amount A (rises 10% a year)", "expected": "amount B", "optimistic": "amount B"}
    fake.reply = reply(result)
    at = start()
    sample(at, step=10.0)
    assert ui("goals.required_note", scenarios="Conservative", label=plain_text("amount A (rises 10% a year)")) in captions(at)
    assert ui("goals.required_note", scenarios="Expected, Optimistic", label="amount B") in captions(at)
    assert "amount B" not in captions(at)


def test_the_outcome_column_says_it_compares_future_money_and_the_other_column_names_are_unchanged(fake):
    fake.reply = reply(planned())
    at = start()
    sample(at)
    assert list(dataframes(at)[0].value.columns) == ["Scenario", "Assumed yearly return", "Savings at the end", "Savings at the end (today's money)",
                                                      "Against the goal (future money)", "Monthly amount needed", "Time to reach the goal"]


def test_the_three_scenario_lines_and_the_goal_line_have_four_different_colours_from_the_config(fake):
    fake.reply = reply(planned())
    at = start()
    sample(at)
    [chart] = charts(at)
    ui_cfg = get_config()["ui"]["goals"]
    colours = [trace["line"]["color"] for trace in chart["data"]]
    assert colours == [ui_cfg["line_colors"][n] for n in ("conservative", "expected", "optimistic")]
    goal = chart["layout"]["shapes"][0]["line"]
    assert goal["dash"] == "dash" and goal["color"] == ui_cfg["goal_line_color"]
    assert len({*colours, goal["color"]}) == 4


def test_a_scenario_without_a_configured_colour_gets_the_default(fake):
    result = planned()
    result["data"]["projections"] = [{**r, "scenario": "mystery"} for r in result["data"]["projections"] if r["scenario"] == "expected"]
    fake.reply = reply(result)
    at = start()
    sample(at)
    assert charts(at)[0]["data"][0]["name"] == "mystery" and "color" not in (charts(at)[0]["data"][0].get("line") or {})


def test_payload_warnings_start_with_a_capital_letter_but_the_payload_text_is_not_changed(fake):
    result = planned()
    result["data"]["warnings"] = ["low scenario: grows slower than prices.", "  already Capital.", "", "(odd) start"]
    original = list(result["data"]["warnings"])
    fake.reply = reply(result)
    at = start()
    sample(at)
    assert warnings(at)[:2] == [plain_text("Low scenario: grows slower than prices."), plain_text("Already Capital.")]
    assert result["data"]["warnings"] == original


@pytest.mark.parametrize("change, text", [
    (lambda c: c["ui"]["goals"].pop("line_colors"), "line_colors"),
    (lambda c: c["ui"]["goals"]["line_colors"].pop("expected"), "line_colors"),
    (lambda c: c["ui"]["goals"]["line_colors"].update(expected="red"), "different #rrggbb"),
    (lambda c: c["ui"]["goals"]["line_colors"].update(expected="#3B82F6"), "different #rrggbb"),
    (lambda c: c["ui"]["goals"].update(goal_line_color="#10b981"), "different #rrggbb"),
    (lambda c: c["ui"]["goals"].pop("goal_line_color"), "different #rrggbb"),
])
def test_a_bad_colour_block_is_rejected(tmp_path, change, text):
    from src.utils.config import ConfigError

    path = _config(tmp_path, change)
    with pytest.raises(ConfigError, match=text):
        get_config(path)
    get_config.cache_clear()
