import logging

import pytest

from src.agents import base_agent
from src.agents.base_agent import BaseAgent
from src.utils.config import get_config


class FakeLLM:
    def __init__(self, reply="Investors often study concentration risk."):
        self.reply = reply

    def invoke(self, prompt):
        return type("Msg", (), {"content": self.reply})()


def _result(**overrides):
    base = {
        "agent": "echo", "turn_id": "", "status": "ok", "answer": "Plain answer.",
        "sources": [], "error": None, "data": None,
    }
    return {**base, **overrides}


class EchoAgent(BaseAgent):
    name = "echo"
    llm_role = "router"

    def __init__(self, llm=None, result=None):
        super().__init__(llm=llm)
        self._res = result or _result()

    def _execute(self, state):
        return dict(self._res)


class BoomAgent(BaseAgent):
    name = "boom"

    def _execute(self, state):
        raise ValueError("secret technical detail")


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    def fail(**kwargs):
        raise AssertionError("get_llm must not be called")

    monkeypatch.setattr(base_agent, "get_llm", fail)


STATE = {"turn_id": "t1"}


def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError):
        BaseAgent(llm=FakeLLM())


def test_injected_llm_is_used_without_factory():
    llm = FakeLLM()
    assert EchoAgent(llm=llm).llm is llm


def test_factory_called_with_role_when_not_injected(monkeypatch):
    seen = {}
    sentinel = FakeLLM()

    def fake_get_llm(role):
        seen["role"] = role
        return sentinel

    monkeypatch.setattr(base_agent, "get_llm", fake_get_llm)
    agent = EchoAgent()
    assert seen["role"] == "router"
    assert agent.llm is sentinel


def test_run_returns_plain_answer_without_disclaimer():
    result = EchoAgent(llm=FakeLLM()).run(STATE)
    assert result["status"] == "ok"
    assert result["answer"] == "Plain answer."
    assert get_config()["app"]["disclaimer"].strip() not in result["answer"]


def test_turn_id_filled_when_missing_and_kept_when_present():
    assert EchoAgent(llm=FakeLLM()).run(STATE)["turn_id"] == "t1"
    kept = EchoAgent(llm=FakeLLM(), result=_result(turn_id="other")).run(STATE)
    assert kept["turn_id"] == "other"


def test_agent_name_filled_when_missing():
    result = EchoAgent(llm=FakeLLM(), result=_result(agent="")).run(STATE)
    assert result["agent"] == "echo"


def test_exception_becomes_error_result_and_detail_only_in_log(caplog):
    with caplog.at_level(logging.ERROR):
        result = BoomAgent(llm=FakeLLM()).run(STATE)
    assert result["status"] == "error"
    assert result["turn_id"] == "t1"
    assert result["agent"] == "boom"
    assert result["answer"] == get_config()["agents"]["error_message"]
    assert "secret technical detail" not in result["answer"]
    assert "secret technical detail" not in (result["error"] or "")
    assert "secret technical detail" in caplog.text


def test_guard_replaces_advice_on_ok_answer():
    advice = _result(answer="With one stock, you should sell half.")
    result = EchoAgent(llm=FakeLLM("Investors often study concentration risk."), result=advice).run(STATE)
    assert result["answer"] == "Investors often study concentration risk."


def test_guard_falls_back_to_safe_message():
    advice = _result(answer="You should sell.")
    result = EchoAgent(llm=FakeLLM("You should sell."), result=advice).run(STATE)
    assert result["answer"] == get_config()["output_guard"]["safe_message"].strip()


@pytest.mark.parametrize("status", ["error", "no_answer", "wrong_agent", "needs_input"])
def test_guard_applies_to_ok_only(status):
    res = _result(status=status, answer="you should sell")
    assert EchoAgent(llm=FakeLLM(), result=res).run(STATE)["answer"] == "you should sell"
