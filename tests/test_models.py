from datetime import datetime, timezone

from src.core.models import AgentResult, MarketResult, Source


def test_agent_result_envelope_keys():
    assert set(AgentResult.__annotations__) == {
        "agent", "turn_id", "status", "answer", "sources", "error", "data",
    }


def test_status_literals():
    status = AgentResult.__annotations__["status"]
    # `from __future__ import annotations` stores a string; check every status is named.
    for name in ["ok", "no_answer", "wrong_agent", "error", "needs_input"]:
        assert name in str(status)


def test_source_keys():
    assert set(Source.__annotations__) == {"title", "url", "chunk_id", "score"}


def test_market_result_defaults():
    now = datetime.now(timezone.utc)
    result = MarketResult(data={"price": 1.0}, currency="USD", as_of=now, source="live")
    assert result.is_stale is False
    assert result.error is None
    assert result.currency == "USD"


def test_market_result_error():
    result = MarketResult(data=None, currency=None, as_of=None, source="live", error="boom")
    assert result.error == "boom"
