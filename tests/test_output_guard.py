import pytest

from src.core.output_guard import find_violation, guard_answer
from src.utils.config import get_config


class FakeLLM:
    def __init__(self, reply="", raises=False):
        self.reply = reply
        self.raises = raises
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.raises:
            raise RuntimeError("llm down")
        return type("Msg", (), {"content": self.reply})()


@pytest.mark.parametrize(
    "text",
    [
        "You should sell this stock now.",
        "I recommend buying more shares.",
        "Sell your holdings.",
        "BUY MORE of it",
        "Honestly, you should HOLD.",
    ],
)
def test_flagged_phrases_detected(text):
    assert find_violation(text) is not None


def test_a_reworded_text_with_a_new_number_gets_the_safe_message():
    safe = get_config()["output_guard"]["safe_message"].strip()
    assert guard_answer("You should sell AAPL at 100.", FakeLLM("AAPL fell 7 percent at 100.")) == safe


def test_a_reworded_text_with_the_same_numbers_is_used():
    assert guard_answer("You should sell AAPL at 100 [1].", FakeLLM("AAPL is at 100 [1].")) == "AAPL is at 100 [1]."


def test_a_caller_can_do_its_own_number_check():
    assert guard_answer("You should sell AAPL at 100.", FakeLLM("AAPL fell 7 percent."), check_numbers=False) == "AAPL fell 7 percent."


def test_the_base_agent_guard_rejects_a_reworded_answer_with_an_invented_number():
    from src.agents.base_agent import BaseAgent

    class Dummy(BaseAgent):
        name = "dummy"

        def _execute(self, state):
            return {"agent": "dummy", "turn_id": "", "status": "ok", "answer": "You should sell AAPL at 100.", "sources": [], "error": None, "data": None}

    result = Dummy(llm=FakeLLM("AAPL fell 7 percent at 100.")).run({"turn_id": "t"})
    assert result["answer"] == get_config()["output_guard"]["safe_message"].strip()


def test_clean_text_passes_unchanged_without_llm_call():
    llm = FakeLLM()
    text = "Diversification spreads risk across many assets."
    assert find_violation(text) is None
    assert guard_answer(text, llm) == text
    assert llm.calls == []


def test_flagged_then_clean_rewording_is_returned():
    llm = FakeLLM(reply="Investors often review concentration risk.")
    result = guard_answer("You should sell AAPL.", llm)
    assert result == "Investors often review concentration risk."
    assert len(llm.calls) == 1
    assert "You should sell AAPL." in llm.calls[0]


def test_still_flagged_gives_safe_message():
    llm = FakeLLM(reply="Still, you should sell.")
    result = guard_answer("You should sell AAPL.", llm)
    assert result == get_config()["output_guard"]["safe_message"].strip()
    assert len(llm.calls) == 1  # one retry only


def test_retry_failure_gives_safe_message():
    result = guard_answer("I recommend buying X", FakeLLM(raises=True))
    assert result == get_config()["output_guard"]["safe_message"].strip()
