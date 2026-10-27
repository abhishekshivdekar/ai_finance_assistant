import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.core import usage_tracker
from src.core.usage_tracker import (
    UsageTracker,
    get_usage_registry,
    get_usage_tracker,
    is_budget_exceeded,
)

PRICING = {
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
}


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    get_usage_registry.cache_clear()
    get_usage_tracker.cache_clear()
    monkeypatch.setattr(
        usage_tracker, "get_config", lambda: {"pricing": PRICING, "llm": {}}
    )
    yield
    get_usage_registry.cache_clear()
    get_usage_tracker.cache_clear()


def _result(input_tokens=None, output_tokens=None) -> LLMResult:
    usage = (
        None
        if input_tokens is None
        else {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    )
    msg = AIMessage(content="hi", usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_registry_is_singleton():
    assert get_usage_registry() is get_usage_registry()


def test_tracker_math_per_role():
    router = UsageTracker("router", "gpt-6-luna")
    default = UsageTracker("default", "gpt-5.4-mini")
    router.on_llm_end(_result(1_000_000, 2_000_000))
    default.on_llm_end(_result(1_000_000, 1_000_000))

    reg = get_usage_registry()
    r = reg.totals("router")
    assert (r["input_tokens"], r["output_tokens"]) == (1_000_000, 2_000_000)
    assert r["cost"] == pytest.approx(0.10 + 2 * 0.50)
    d = reg.totals("default")
    assert d["cost"] == pytest.approx(0.75 + 4.50)
    assert reg.session_tokens() == 5_000_000


def test_calls_accumulate():
    tracker = UsageTracker("router", "gpt-6-luna")
    tracker.on_llm_end(_result(100, 50))
    tracker.on_llm_end(_result(200, 25))
    totals = get_usage_registry().totals("router")
    assert (totals["input_tokens"], totals["output_tokens"]) == (300, 75)
    assert totals["cost"] == pytest.approx((300 * 0.10 + 75 * 0.50) / 1_000_000)


def test_unknown_model_gives_tokens_but_no_cost():
    UsageTracker("router", "mystery-model").on_llm_end(_result(10, 5))
    totals = get_usage_registry().totals("router")
    assert (totals["input_tokens"], totals["output_tokens"]) == (10, 5)
    assert totals["cost"] is None


def test_cost_stays_none_once_unpriced_call_recorded():
    reg = get_usage_registry()
    reg.record("router", "mystery-model", 10, 5)
    reg.record("router", "gpt-6-luna", 10, 5)
    assert reg.totals("router")["cost"] is None


def test_missing_usage_metadata_ignored():
    UsageTracker("router", "gpt-6-luna").on_llm_end(_result())
    assert get_usage_registry().session_tokens() == 0


def test_generation_without_message_ignored():
    from langchain_core.outputs import Generation

    result = LLMResult(generations=[[Generation(text="plain")]])
    UsageTracker("router", "gpt-6-luna").on_llm_end(result)
    assert get_usage_registry().session_tokens() == 0


def test_totals_unknown_role_and_all_roles():
    reg = get_usage_registry()
    assert reg.totals("nobody")["input_tokens"] == 0
    reg.record("a", "gpt-6-luna", 1, 1)
    reg.record("b", "gpt-6-luna", 2, 2)
    assert set(reg.totals()) == {"a", "b"}


def test_reset():
    reg = get_usage_registry()
    reg.record("a", "gpt-6-luna", 1, 1)
    reg.reset()
    assert reg.session_tokens() == 0


def test_get_usage_tracker_cached_per_role_model():
    assert get_usage_tracker("router", "m") is get_usage_tracker("router", "m")
    assert get_usage_tracker("router", "m") is not get_usage_tracker("default", "m")


def test_missing_pricing_table_gives_no_cost(monkeypatch):
    monkeypatch.setattr(usage_tracker, "get_config", lambda: {})
    get_usage_registry().record("router", "gpt-6-luna", 1, 1)
    assert get_usage_registry().totals("router")["cost"] is None


@pytest.mark.parametrize(
    "llm_cfg, tokens, expected",
    [
        ({}, 10_000, False),                                  # no limit configured
        ({"max_session_tokens": None}, 10_000, False),        # null = no limit
        ({"max_session_tokens": 100}, 99, False),             # under
        ({"max_session_tokens": 100}, 100, True),             # at limit
        ({"max_session_tokens": 100}, 150, True),             # over
    ],
)
def test_budget_guard(monkeypatch, llm_cfg, tokens, expected):
    monkeypatch.setattr(
        usage_tracker, "get_config", lambda: {"pricing": PRICING, "llm": llm_cfg}
    )
    get_usage_registry().record("router", "gpt-6-luna", tokens, 0)
    assert is_budget_exceeded() is expected
