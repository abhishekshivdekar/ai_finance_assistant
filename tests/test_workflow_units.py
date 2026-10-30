"""Pure helpers of the workflow: routing, the verbatim merge, notes, the node wrappers and the config block."""
import copy

import pytest
import yaml

from src.utils.config import ConfigError, get_config
from src.workflow import nodes, routing
from src.workflow.answers import add_disclaimer, merge_verbatim, notes_for
from src.workflow.routing import next_agent, plan_agents, reroute_target, route_at_end

CFG = get_config()
WF = CFG["workflow"]
SRC1 = {"title": "One", "url": None, "chunk_id": "a/a#0", "score": 0.5}
SRC2 = {"title": "Two", "url": None, "chunk_id": "b/b#0", "score": 0.4}
SRC3 = {"title": "Three", "url": None, "chunk_id": "c/c#0", "score": 0.3}


def result(agent, status="ok", answer="text", sources=None, turn_id="t1", error=None):
    return {"agent": agent, "turn_id": turn_id, "status": status, "answer": answer,
            "sources": sources or [], "error": error, "data": None}


# ---- routing helpers ----

def test_plan_agents_keeps_agent_intents_in_order_without_duplicates_up_to_the_cap():
    assert plan_agents(["news", "market", "news", "clarify"], WF) == ["news", "market"]
    assert plan_agents(["market", "news", "finance_qa", "portfolio"], WF) == ["market", "news", "finance_qa"]
    assert plan_agents(["out_of_scope"], WF) == [] and plan_agents([], WF) == []


def test_specific_advice_empties_the_queue_so_no_agent_can_run():
    assert plan_agents(["market", "specific_advice"], WF) == []


def test_next_agent_skips_agents_that_already_ran_this_turn_only():
    state = {"turn_id": "t2", "agents_to_run": ["market", "news"],
             "agent_outputs": [result("market", turn_id="t1"), result("market", turn_id="t2")]}
    assert next_agent(state) == "news"
    assert next_agent({**state, "agent_outputs": []}) == "market"       # a result of another turn does not count
    assert next_agent({"turn_id": "t2"}) is None


def test_reroute_target_needs_the_params_and_a_candidate_that_has_not_run():
    state = {"turn_id": "t1", "agents_to_run": ["finance_qa"], "market_params": {"tickers": ["AAPL"]},
             "agent_outputs": [result("finance_qa", "no_answer")]}
    assert reroute_target(state, WF) == "market"
    assert reroute_target({**state, "market_params": None}, WF) is None
    assert reroute_target({**state, "market_params": None, "news_params": {"ticker": "AAPL"}}, WF) == "news"
    assert reroute_target({**state, "agent_outputs": []}, WF) is None
    assert reroute_target({**state, "agents_to_run": ["finance_qa", "market"]}, WF) is None
    assert reroute_target({**state, "agent_outputs": [result("market", "no_answer")]}, WF) is None   # no table entry for market


@pytest.mark.parametrize("statuses, expected", [
    (["ok"], "synthesizer"), (["error", "ok"], "synthesizer"), (["needs_input", "ok"], "synthesizer"),
    (["needs_input"], "clarify"), (["needs_input", "no_answer"], "clarify"),
    (["no_answer"], "synthesizer"), (["no_answer", "error"], "synthesizer"),
    (["error"], "fallback"), (["wrong_agent"], "fallback"), ([], "fallback"),
])
def test_route_at_end(statuses, expected):
    state = {"turn_id": "t1", "agent_outputs": [result(f"a{i}", s) for i, s in enumerate(statuses)]}
    assert route_at_end(state) == expected


# ---- answers ----

def test_add_disclaimer_appends_the_config_text_once_after_the_trimmed_answer():
    text = add_disclaimer("  An answer.  ")
    assert text == f"An answer.\n\n{' '.join(CFG['app']['disclaimer'].split())}"


def test_merge_verbatim_shifts_markers_and_joins_the_sources_in_order():
    first = result("finance_qa", answer="A bond is a loan [1][2].", sources=[SRC1, SRC2])
    second = result("tax_education", answer="Gains are taxed [1, 2] and [F3].", sources=[SRC3, SRC1])
    text, sources = merge_verbatim([first, second])
    assert sources == [SRC1, SRC2, SRC3, SRC1]
    assert text == ("**Concepts**\nA bond is a loan [1][2].\n\n"
                    "**Tax**\nGains are taxed [3, 4] and [F3].")


def test_merge_verbatim_labels_the_markers_of_an_answer_without_sources():
    news = result("news", answer="- Shares rose [1, 2]. Another [3].")
    qa = result("finance_qa", answer="A concept [1].", sources=[SRC1])
    text, sources = merge_verbatim([qa, news])
    assert sources == [SRC1]
    assert "[News 1, News 2]" in text and "[News 3]" in text and "A concept [1]." in text


def test_merge_verbatim_leaves_text_without_markers_alone():
    text, sources = merge_verbatim([result("market", answer="Up 3 percent."), result("portfolio", answer="Balanced.")])
    assert text == "**Market**\nUp 3 percent.\n\n**Portfolio**\nBalanced." and sources == []


def test_notes_cover_needs_input_error_and_no_answer_only():
    notes = notes_for([
        result("market", "needs_input", "Which ticker?"), result("news", "error", "boom", error="boom"),
        result("finance_qa", "no_answer"), result("portfolio"),
    ])
    assert notes == [
        WF["messages"]["needs_input_note"].format(label="Market", message="Which ticker?"),
        " ".join(WF["messages"]["error_note"].format(label="News").split()),
        " ".join(WF["messages"]["no_answer_note"].format(label="Concepts", message="text").split()),
    ]


def test_notes_use_the_agent_name_when_it_has_no_label():
    assert "mystery" in notes_for([result("mystery", "error", error="x")])[0]


# ---- nodes used directly ----

def test_the_synthesizer_with_nothing_to_present_falls_back():
    update = nodes.make_synthesizer_node(nodes.LazyLLM("synthesizer", object()))({"turn_id": "t1", "agent_outputs": [result("market", "error", error="x")]})
    assert " ".join(WF["messages"]["fallback"].split()) in update["final_answer"]


def test_the_agent_node_returns_only_its_result_and_tags_an_error_with_the_turn():
    class Boom:
        def run(self, state):
            return result("market", "error", "sorry", turn_id=state["turn_id"], error="market failed")

    update = nodes.make_agent_node(Boom())({"turn_id": "t9"})
    assert set(update) == {"agent_outputs", "errors"} and update["errors"] == [{"turn_id": "t9", "message": "market failed"}]
    ok = nodes.make_agent_node(type("Ok", (), {"run": lambda self, s: result("market")})())({"turn_id": "t1"})
    assert set(ok) == {"agent_outputs"}


# ---- the workflow config block ----

def _write(tmp_path, change):
    wf = copy.deepcopy(CFG["workflow"])
    change(wf)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"workflow": wf}), encoding="utf-8")
    get_config.cache_clear()
    return str(path)


def test_the_real_workflow_block_is_valid_and_the_roles_exist():
    assert WF["hint_nodes"] == WF["agent_intents"] and WF["max_route_attempts"] == 2
    for role in ("router", "synthesizer", "safe_response"):
        assert CFG["llm"]["roles"][role]["model"]


def test_a_valid_workflow_block_loads(tmp_path):
    assert get_config(_write(tmp_path, lambda w: None))["workflow"]["recursion_limit"] > 0


@pytest.mark.parametrize("change", [
    lambda w: w.update(router_min_confidence=1.5),
    lambda w: w.update(router_min_confidence=True),
    lambda w: w.update(router_min_confidence="high"),
    lambda w: w.update(max_route_attempts=-1),
    lambda w: w.update(recursion_limit=0),
    lambda w: w.update(max_agents_per_turn=1.5),
    lambda w: w.update(intents=["finance_qa", "market"]),
    lambda w: w.update(agent_intents=["finance_qa", "not_an_intent"]),
    lambda w: w.update(agent_intents=[]),
    lambda w: w.update(hint_nodes=["clarify"]),
    lambda w: w["agent_labels"].pop("tax_education"),
    lambda w: w.update(reroute={"clarify": []}),
    lambda w: w.update(reroute={"finance_qa": [{"agent": "nobody", "needs": "market_params"}]}),
    lambda w: w.update(reroute={"finance_qa": [{"agent": "market", "needs": ""}]}),
    lambda w: w["messages"].pop("fallback"),
    lambda w: w["messages"].update(out_of_scope="  "),
    lambda w: w["messages"].pop("no_answer_note"),
    lambda w: w.update(router_history_max_chars=0),
    lambda w: w.update(part_max_chars=0),
    lambda w: w.update(part_max_chars=True),
    lambda w: w["messages"].pop("partial_note"),
    lambda w: w.update(router_history_max_chars=True),
    lambda w: w.update(router_prompt="  "),
    lambda w: w.update(synthesizer_prompt=""),
    lambda w: w.pop("safe_response_prompt"),
    lambda w: w.update(advice_patterns=["(unclosed"]),
    lambda w: w.update(ticker_aliases=["apple"]),
    lambda w: w["ticker_aliases"].update(broken=""),
    lambda w: w["ticker_aliases"].update({"": "AAPL"}),
    lambda w: w.pop("merge_banned_phrases"),
    lambda w: w.update(merge_banned_phrases=[]),
    lambda w: w.update(merge_banned_phrases="json"),
    lambda w: w.update(merge_banned_phrases=["json", " "]),
    lambda w: w.update(merge_banned_phrases=["(unclosed"]),
    lambda w: w.pop("advice_opening_pattern"),
    lambda w: w.update(advice_opening_pattern="  "),
    lambda w: w.update(advice_opening_pattern="(unclosed"),
    lambda w: w["messages"].pop("advice_opening"),
])
def test_a_broken_workflow_block_is_rejected(tmp_path, change):
    with pytest.raises(ConfigError):
        get_config(_write(tmp_path, change))


def test_workflow_cfg_reads_the_config(monkeypatch):
    monkeypatch.setattr(routing, "get_config", lambda: {"workflow": {"x": 1}})
    assert routing.workflow_cfg() == {"x": 1}
