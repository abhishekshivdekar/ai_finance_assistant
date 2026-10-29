import json

import pytest

from src.agents import base_agent
from src.agents.goal_planning import GoalPlanningAgent
from src.core.goal_inputs import parse_goal_params
from src.core.goal_metrics import build_plan, summary_input
from src.utils.config import get_config

CFG = get_config()["goal_planning"]
COMMENT = "In the expected scenario the savings reach 2,622,695.31. These are illustrative assumptions, not predictions."
PARAMS = {"goal_amount": 1_000_000, "horizon_years": 15, "current_savings": 50_000, "monthly_contribution": 5_000,
          "risk_profile": "medium", "annual_step_up_pct": 5, "inflation_adjust": False}


class FakeLLM:
    def __init__(self, reply=COMMENT, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return type("Msg", (), {"content": self.reply})()


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the real factory must not be called")

    monkeypatch.setattr(base_agent, "get_llm", fail)


def run(params=PARAMS, llm=None):
    llm = llm or FakeLLM()
    return llm, GoalPlanningAgent(llm=llm).run({"turn_id": "t1", "goal_params": params})


def sent_json(llm):
    """The JSON object the prompt ended with."""
    return json.loads(llm.calls[0].split("JSON:", 1)[1])


# ---- happy path and payload ----

def test_ok_result_carries_the_commentary_and_a_full_payload():
    llm, result = run()
    data = result["data"]
    assert result["status"] == "ok" and result["agent"] == "goal_planning" and result["turn_id"] == "t1"
    assert result["answer"] == COMMENT and len(llm.calls) == 1
    assert [s["name"] for s in data["scenarios"]] == ["conservative", "expected", "optimistic"]
    assert set(data["required_monthly_contribution"]) == set(data["months_to_goal"]) == set(data["shortfall"]) == \
        {"conservative", "expected", "optimistic"}
    assert data["warnings"] == [] and data["inputs"]["risk_profile"] == "medium"
    assert {"scenario", "year", "value", "value_todays_money"} == set(data["projections"][0])
    assert len(data["projections"]) == 3 * 16                       # years 0..15 for three scenarios
    assert data["inflation"] == {}


def test_payload_numbers_match_the_pure_functions():
    _, result = run()
    inputs, _ = parse_goal_params(PARAMS, CFG)
    plan = build_plan(inputs, CFG)
    assert result["data"]["target_amount"] == plan.target_amount
    assert result["data"]["required_monthly_contribution"]["expected"] == plan.scenarios[1].required_monthly_contribution


def test_inflation_block_shows_both_goal_figures_when_applied():
    _, result = run({**PARAMS, "inflation_adjust": True})
    block = result["data"]["inflation"]
    assert block["goal_in_todays_money"] == 1_000_000 and block["goal_in_future_money"] == pytest.approx(1_000_000 * 1.05 ** 15)
    assert "future money" in block["labels"]["goal_future_money"] and "today" in block["labels"]["projection_todays_money"]
    assert result["data"]["projections"][0]["value_todays_money"] is not None


def test_inflation_default_from_config_and_request_override():
    without = {k: v for k, v in PARAMS.items() if k != "inflation_adjust"}
    assert run(without)[1]["data"]["inflation"] != {}                  # config default is on
    assert run({**without, "inflation_adjust": False})[1]["data"]["inflation"] == {}


def test_zero_savings_and_zero_contribution_still_give_a_plan():
    llm, result = run({"goal_amount": 100_000, "horizon_years": 5, "risk_profile": "low"})
    assert result["status"] == "ok" and result["data"]["required_monthly_contribution"]["expected"] > 0
    assert result["data"]["months_to_goal"]["expected"] is None


# ---- assumptions are always in the payload ----

def assert_assumptions(data):
    a = data["assumptions"]
    assert a["illustrative"] is True and "not predictions" in a["label"]
    assert a["as_of"] == CFG["assumptions"]["as_of"] and a["source"] == CFG["assumptions"]["source"]
    assert a["currency"] == "INR" and a["annual_returns_pct"] == {"conservative": 5.0, "expected": 9.0, "optimistic": 12.0}
    assert a["inflation"]["annual_pct"] == 5.0 and a["inflation"]["as_of"] and a["inflation"]["applied"] is False
    assert "end of each month" in a["deposit_timing"] and "12 months" in a["step_up_timing"] and a["return_convention"]


@pytest.mark.parametrize("llm", [FakeLLM(), FakeLLM(error=RuntimeError("boom")), FakeLLM(reply="  "), FakeLLM(reply="99,999")])
def test_assumptions_are_present_whatever_the_llm_does(llm):
    assert_assumptions(run(llm=llm)[1]["data"])


# ---- prompt receives only computed JSON ----

def test_the_prompt_receives_only_the_computed_results():
    llm, _ = run()
    inputs, _ = parse_goal_params(PARAMS, CFG)
    expected = json.loads(json.dumps(summary_input(build_plan(inputs, CFG), CFG)))
    assert sent_json(llm) == expected
    text = llm.calls[0]
    for raw_key in ("goal_amount", "horizon_years", "annual_step_up_pct", "risk_profile\": \"wild"):
        assert raw_key not in text
    assert "{results_json}" not in text


def test_the_prompt_forbids_advice_predictions_and_number_conversion():
    prompt = CFG["summary_prompt"]
    assert "ONLY numbers that appear in the JSON" in prompt and "copied exactly as written" in prompt
    assert "lakh" in prompt and "crore" in prompt and "western grouping" in prompt
    assert "illustrative assumptions" in prompt and "not predictions or guaranteed returns" in prompt
    assert "never recommend a product or fund" in prompt and 'never say "you should"' in prompt
    assert prompt.count("{results_json}") == 1


def test_the_prompt_asks_for_plain_words_per_scenario_results_and_a_closing_sentence():
    prompt = " ".join(CFG["summary_prompt"].split())
    assert "Do not quote field names or labels" in prompt and "plain words" in prompt
    assert "ONE short paragraph, then one bullet line per scenario" in prompt and "then ONE closing sentence" in prompt
    assert "state the surplus or shortfall for that scenario" in prompt
    assert "the time to reach the goal at the current monthly amount" in prompt
    assert "how the time needed to reach the goal compares with the horizon" in prompt
    assert "the monthly amount, the time horizon, the risk profile and the yearly step-up" in prompt
    assert "Use observation wording only" in prompt


def test_the_inflation_part_of_the_json_is_two_plain_sentences():
    llm, _ = run({**PARAMS, "inflation_adjust": True})
    assert set(sent_json(llm)["inflation"]) == {"goal_in_todays_money_text", "goal_in_future_money_text"}


# ---- the new precomputed text passes the number guard ----

def test_text_copied_from_the_new_fields_passes_the_number_guard():
    probe, _ = run({**PARAMS, "inflation_adjust": True})
    sent = sent_json(probe)
    parts = [sent["inflation"]["goal_in_todays_money_text"], sent["inflation"]["goal_in_future_money_text"]]
    for s in sent["scenarios"]:
        parts.append(f"{s['scenario']}: {s['surplus_or_shortfall_text']}, {s['years_months_to_goal']}, "
                     f"{s['monthly_amount']} ({s['monthly_amount_meaning']}).")
    reply = " ".join(parts)
    assert "years" in reply and "month" in reply
    _, result = run({**PARAMS, "inflation_adjust": True}, llm=FakeLLM(reply=reply))
    assert result["answer"] == reply and result["data"]["warnings"] == []


def test_a_years_and_months_figure_that_was_not_calculated_is_rejected():
    _, result = run(llm=FakeLLM(reply="It would take 97 years 11 months."))
    assert result["answer"] == CFG["summary_fallback"].strip()


def test_payload_labels_follow_step_up_and_goal_met():
    hard = {**PARAMS, "goal_amount": 50_000_000}
    _, stepped = run(hard)
    assert stepped["data"]["required_labels"]["expected"] == "starting monthly amount (rises 5% a year)"
    assert stepped["data"]["scenarios"][1]["required_label"] == stepped["data"]["required_labels"]["expected"]
    _, plain = run({**hard, "annual_step_up_pct": 0})
    assert plain["data"]["required_labels"]["expected"] == "monthly amount needed to reach the goal"
    _, met_step = run()                                  # the expected scenario already exceeds the goal
    assert met_step["data"]["required_labels"]["expected"].startswith("minimum monthly amount to still reach the goal")
    _, met = run({**PARAMS, "current_savings": 5_000_000, "annual_step_up_pct": 0})
    assert set(met["data"]["required_labels"].values()) == {"minimum monthly amount to still reach the goal"}


# ---- guarded commentary ----

def test_fallback_on_llm_error_keeps_status_ok_and_adds_a_warning():
    _, result = run(llm=FakeLLM(error=RuntimeError("boom")))
    assert result["status"] == "ok" and result["answer"] == CFG["summary_fallback"].strip()
    assert any("unavailable" in w for w in result["data"]["warnings"])


def test_fallback_on_empty_reply():
    _, result = run(llm=FakeLLM(reply="   "))
    assert result["answer"] == CFG["summary_fallback"].strip() and any("empty" in w for w in result["data"]["warnings"])


def test_fallback_when_the_reply_contains_a_number_that_was_not_calculated():
    _, result = run(llm=FakeLLM(reply="The savings reach 4,321,987.00 in the expected scenario."))
    assert result["answer"] == CFG["summary_fallback"].strip() and any("not calculated" in w for w in result["data"]["warnings"])


def test_the_fallback_text_contains_no_numbers_and_the_disclaimer_words():
    text = CFG["summary_fallback"]
    assert not any(ch.isdigit() for ch in text)
    assert "illustrative" in text and "not predictions" in text and "not personal advice" in text


# ---- validation ----

@pytest.mark.parametrize("changes, text", [
    ({"risk_profile": "wild"}, "one of: low, medium, high"),
    ({"goal_amount": -1}, "goal_amount must be greater than zero"),
    ({"horizon_years": 99}, "between 1 and 40 years"),
])
def test_invalid_input_needs_input_and_the_llm_is_not_called(changes, text):
    llm, result = run({**PARAMS, **changes})
    assert result["status"] == "needs_input" and text in result["answer"] and llm.calls == []
    assert result["answer"].startswith(CFG["needs_input_message"].strip())
    assert result["data"]["warnings"]


def test_missing_goal_params_need_input():
    agent = GoalPlanningAgent(llm=FakeLLM())
    assert agent.run({"turn_id": "t1"})["status"] == "needs_input"
