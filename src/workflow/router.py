"""The LLM router: classifies a chat message into intents, a confidence and grounded entities (architecture 4.3).

Design pattern: ADAPTER plus GUARD. Router wraps a structured-output LLM call and turns its untrusted reply into a
trusted Decision. Every check is code, not trust. Example: for "What is Apple trading at?" the model may return the
mention "Apple"; code keeps it only because "apple" occurs in the query, and maps it to AAPL through
workflow.ticker_aliases. The model never supplies a symbol, so it cannot invent one.

What the router extracts: ticker mentions, a news phrase and a period. It never extracts amounts, holdings, goals
or tax-calculator inputs; those come only from tab forms. The user's query and the earlier conversation reach the
model as a JSON data block and the prompt tells it to classify, never to follow instructions inside them.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from typing import Any, Mapping

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.core.llm_factory import get_llm
from src.data.market.provider import InvalidSymbolError, validate_symbol
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.workflow.routing import Decision, plan_agents, workflow_cfg
from src.workflow.state import GraphState

log = get_logger(__name__)


class RouterEntities(BaseModel):
    """What the model may extract: mentions and a phrase, as the user wrote them."""

    tickers: list[str]
    period: str | None
    news_query: str | None


class RouterPart(BaseModel):
    """One agent intent and the short standalone question that agent should answer."""

    agent: str
    question: str


class RouterOutput(BaseModel):
    """The schema sent to the model (structured output). Its reply is validated again by read_output().

    `parts` is a list of {agent, question} (empty unless two or more agent intents), not a mapping: a schema with
    fixed fields is what structured output handles reliably. read_parts() turns it into {agent: question}.
    """

    intents: list[str]
    confidence: float
    entities: RouterEntities
    parts: list[RouterPart]


class RouterError(ValueError):
    """The router's reply is not usable."""


class TooManyAgentsError(RouterError):
    """The reply names more agent intents than workflow.max_agents_per_turn."""


def _as_dict(raw: Any) -> dict[str, Any]:
    """A pydantic model or a mapping as a plain dict."""
    if isinstance(raw, BaseModel):
        return raw.model_dump()
    if isinstance(raw, Mapping):
        return dict(raw)
    raise RouterError(f"unexpected router reply type {type(raw).__name__}")


def read_output(raw: Any, cfg: dict[str, Any]) -> tuple[list[str], float, dict[str, Any]]:
    """(intents, confidence, entities) from the model's reply, or RouterError.

    Rejected: not a mapping/model, intents empty or not a list of strings, an unknown intent, more agent intents
    than workflow.max_agents_per_turn (unless specific_advice is present, which empties the queue anyway),
    a confidence that is a bool, not a number, nan, inf or outside 0 to 1. Malformed entities are ignored.
    """
    data = _as_dict(raw)
    intents = data.get("intents")
    if not isinstance(intents, list) or not intents or not all(isinstance(i, str) for i in intents):
        raise RouterError("intents must be a non-empty list of strings")
    intents = list(dict.fromkeys(intents))
    unknown = [i for i in intents if i not in cfg["intents"]]
    if unknown:
        raise RouterError(f"unknown intents {unknown}")
    agents = [i for i in intents if i in cfg["agent_intents"]]
    if "specific_advice" not in intents and len(agents) > cfg["max_agents_per_turn"]:
        raise TooManyAgentsError(f"{len(agents)} agent intents; at most {cfg['max_agents_per_turn']} allowed")
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise RouterError("confidence must be a number from 0 to 1")
    entities = data.get("entities")
    return intents, float(confidence), dict(entities) if isinstance(entities, Mapping) else {}


def read_parts(raw: Any, queue: list[str], cfg: dict[str, Any]) -> dict[str, str] | None:
    """{agent: question} for the agents in `queue`, or None.

    Only used with two or more queued agents. A part is dropped (that agent then gets the full query) when its agent
    is not in the queue, it is not a non-empty string, or it is longer than workflow.part_max_chars. Accepts a list of
    {agent, question} (the schema) or a mapping. Example: [{"agent": "market", "question": "What is Apple trading
    at?"}, {"agent": "news", "question": ""}] -> {"market": "What is Apple trading at?"}.
    """
    if len(queue) < 2:
        return None
    pairs = list(raw.items()) if isinstance(raw, Mapping) else []
    for item in raw if isinstance(raw, list) else []:
        item = item.model_dump() if isinstance(item, BaseModel) else item
        if isinstance(item, Mapping):
            pairs.append((item.get("agent"), item.get("question")))
    parts: dict[str, str] = {}
    for agent, question in pairs:
        text = " ".join(question.split()) if isinstance(question, str) else ""
        if agent in queue and agent not in parts and 0 < len(text) <= cfg["part_max_chars"]:
            parts[str(agent)] = text
    return parts or None


def resolve_mention(mention: Any, query: str, aliases: dict[str, str]) -> str | None:
    """A symbol for a ticker mention, or None.

    The mention must occur in the query (any case). It becomes a symbol only through workflow.ticker_aliases or,
    when written in capitals exactly as in the query, as a literal symbol that validate_symbol accepts.
    Example: "Apple" -> AAPL (alias); "AAPL" -> AAPL (literal); "apple" typed as "aapl" -> None.
    """
    if not isinstance(mention, str):
        return None
    text, plain = " ".join(mention.split()), " ".join(query.split())
    if not text or text.casefold() not in plain.casefold():
        return None
    alias = aliases.get(text.casefold())
    if alias:
        return str(alias)
    if text == text.upper() and text in plain:
        try:
            return validate_symbol(text)
        except InvalidSymbolError:
            return None
    return None


def ground_params(entities: dict[str, Any], query: str, cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(market_params, news_params) from the entities, keeping only what the query itself supports.

    market_params needs at least one resolved symbol; the period must be one of market_analysis.periods.
    news_params is the phrase when it occurs in the query, else the first resolved symbol, else None.
    """
    mentions = entities.get("tickers")
    symbols: list[str] = []
    for mention in mentions if isinstance(mentions, list) else []:
        symbol = resolve_mention(mention, query, cfg["ticker_aliases"])
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    market: dict[str, Any] | None = None
    if symbols:
        market = {"tickers": symbols}
        if entities.get("period") in get_config()["market_analysis"]["periods"]:
            market["period"] = entities["period"]
    phrase = entities.get("news_query")
    phrase = " ".join(phrase.split()) if isinstance(phrase, str) else ""
    if phrase and phrase.casefold() in " ".join(query.split()).casefold():
        return market, {"query": phrase}
    return market, ({"ticker": symbols[0]} if symbols else None)


def advice_hit(query: str, cfg: dict[str, Any]) -> bool:
    """True when a workflow.advice_patterns regex matches the query (a safety net that only ADDS specific_advice)."""
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in cfg["advice_patterns"])


def history_for(state: GraphState, cfg: dict[str, Any]) -> list[dict[str, str]]:
    """The earlier messages (without the current query), last workflow.max_history_turns turns, each cut to a length."""
    messages: list[BaseMessage] = list(state.get("messages") or [])
    if messages and messages[-1].type == "human" and str(messages[-1].content) == state.get("query"):
        messages = messages[:-1]
    limit = cfg["router_history_max_chars"]
    return [
        {"role": "user" if m.type == "human" else "assistant", "text": str(m.content)[:limit]}
        for m in messages[-cfg["max_history_turns"] * 2:]
    ]


def build_messages(state: GraphState, cfg: dict[str, Any]) -> list[BaseMessage]:
    """System prompt plus the query and history as ONE delimited JSON data block.

    "<" is written as \\u003c inside the JSON, so text in the query cannot close the delimiter early.
    """
    data = json.dumps({"history": history_for(state, cfg), "query": state.get("query") or ""}, ensure_ascii=False)
    return [SystemMessage(content=cfg["router_prompt"]), HumanMessage(content=f"<router_input>\n{data.replace('<', chr(92) + 'u003c')}\n</router_input>")]


class Router:
    """Callable `classify(state) -> Decision` for build_graph(). Never raises."""

    def __init__(self, llm: Any = None) -> None:
        """Use the injected llm (tests), else the Factory's `router` role; ask it for structured output."""
        self.chain = (llm if llm is not None else get_llm(role="router")).with_structured_output(RouterOutput)

    def _decide(self, state: GraphState, cfg: dict[str, Any]) -> Decision:
        """One LLM call, validated and grounded; raises on anything unusable."""
        query = state.get("query") or ""
        raw = self.chain.invoke(build_messages(state, cfg))
        intents, confidence, entities = read_output(raw, cfg)
        market, news = ground_params(entities, query, cfg)
        return Decision(intents, confidence, market, news, parts=read_parts(_as_dict(raw).get("parts"), plan_agents(intents, cfg), cfg))

    def __call__(self, state: GraphState) -> Decision:
        """The decision. An empty query is `clarify`; an unusable reply is finance_qa (then the graph's fallback).

        A reply with too many agent intents is also finance_qa but marked `partial`, so the answer says that only
        part of the question was handled.

        The advice patterns run last and add specific_advice whatever the model said, even after a failed call.
        """
        cfg = workflow_cfg()
        query = state.get("query") or ""
        if not query.strip():
            return Decision(["clarify"], 1.0)
        try:
            decision = self._decide(state, cfg)
        except Exception as exc:
            log.warning("Router reply unusable; defaulting to finance_qa", exc_info=True)
            decision = Decision(["finance_qa"], cfg["router_min_confidence"], partial=isinstance(exc, TooManyAgentsError))
        if advice_hit(query, cfg) and "specific_advice" not in decision.intents:
            log.info("Advice pattern matched; adding specific_advice")
            decision = replace(decision, intents=[*decision.intents, "specific_advice"])
        return decision
