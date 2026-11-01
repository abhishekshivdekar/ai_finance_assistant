import re

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agents import base_agent, finance_qa
from src.agents.finance_qa import FinanceQAAgent, apply_citations, clean_rewrite
from src.core.usage_tracker import UsageTracker, get_usage_registry
from src.rag.retriever import RetrievedChunk
from src.utils.config import get_config

REFS = [{"name": "SEBI", "url": "https://www.sebi.gov.in"}]
STATE = {"turn_id": "t1", "query": "how will my pile last after my salary stops"}
REWRITE = "retirement corpus withdrawal how long savings last"


def chunk(chunk_id, score=0.6, topic=None, title=None):
    topic = topic or chunk_id.split("/")[0]
    return RetrievedChunk(f"text of {chunk_id}", title or f"Title {chunk_id}", topic, REFS, chunk_id, score)


class FakeLLM:
    """Records every prompt; returns `reply` or raises `error`."""

    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return type("Msg", (), {"content": self.reply})()


class FakeRetrieve:
    """retrieve_fn stand-in: results per query, calls recorded."""

    def __init__(self, by_query):
        self.by_query, self.calls = by_query, []

    def __call__(self, query, topic=None):
        self.calls.append((query, topic))
        return list(self.by_query.get(query, []))


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    def fail(**kwargs):
        raise AssertionError("get_llm must not be called")

    monkeypatch.setattr(base_agent, "get_llm", fail)
    monkeypatch.setattr(finance_qa, "get_llm", fail)


def make(by_query=None, answer="", rewrite=REWRITE, answer_error=None, rewrite_error=None):
    retrieve = FakeRetrieve(by_query if by_query is not None else {})
    agent = FinanceQAAgent(
        llm=FakeLLM(answer, answer_error),
        rewrite_llm=FakeLLM(rewrite, rewrite_error),
        retrieve_fn=retrieve,
    )
    return agent, retrieve


A, B, C = chunk("stocks/a#0", 0.7), chunk("bonds/b#0", 0.6), chunk("etfs_and_funds/c#0", 0.5)


# ---- rewriting ----

def test_rewrite_is_used_and_both_queries_are_retrieved():
    agent, retrieve = make({STATE["query"]: [A], REWRITE: [B]}, answer="Answer [1][2].")
    result = agent.run(STATE)
    assert [q for q, _ in retrieve.calls] == [STATE["query"], REWRITE]
    assert result["data"]["rewritten_query"] == REWRITE
    assert "pile last" in agent.rewrite_llm.calls[0]


@pytest.mark.parametrize("kwargs", [
    {"rewrite_error": RuntimeError("down")},
    {"rewrite": ""},
    {"rewrite": "   \n  "},
    {"rewrite": '""'},
    {"rewrite": "x" * 400},
])
def test_rewrite_failure_or_malformed_falls_back_to_the_original(kwargs):
    agent, retrieve = make({STATE["query"]: [A]}, answer="Fine [1].", **kwargs)
    result = agent.run(STATE)
    assert result["status"] == "ok"
    assert result["data"]["rewritten_query"] == STATE["query"]
    assert [q for q, _ in retrieve.calls] == [STATE["query"]]   # no second retrieval


def test_rewrite_equal_to_the_question_retrieves_once():
    agent, retrieve = make({STATE["query"]: [A]}, answer="Fine [1].", rewrite=STATE["query"])
    agent.run(STATE)
    assert len(retrieve.calls) == 1


def test_rewrite_prompt_keeps_non_finance_questions_unchanged():
    prompt = get_config()["finance_qa"]["rewrite_prompt"]
    assert "If the question is not about money, finance or investing, return it unchanged." in prompt
    assert "Never add finance words to a non-finance question." in prompt
    assert "{question}" in prompt


def test_clean_rewrite():
    assert clean_rewrite('"bond coupon yield"', 120) == "bond coupon yield"
    assert clean_rewrite("\n\n  first line\nsecond line", 120) == "first line"
    assert clean_rewrite("", 120) == "" and clean_rewrite("?!", 120) == ""
    assert clean_rewrite("alpha beta gamma delta", 12) == "alpha beta"   # cut at a word boundary
    assert clean_rewrite("alpha beta gamma delta", 11) == "alpha beta"   # still within twice the limit
    assert clean_rewrite("supercalifragilistic", 10) == ""                # nothing to cut at
    assert clean_rewrite("word " * 30, 20) == ""                          # more than twice the limit: malformed


# ---- retrieval and merge ----

def test_merge_dedupes_by_chunk_and_keeps_the_best_score():
    low, high = chunk("stocks/a#0", 0.4), chunk("stocks/a#0", 0.8)
    agent, _ = make({STATE["query"]: [low, B], REWRITE: [high, C]}, answer="x [1] [2] [3]")
    result = agent.run(STATE)
    assert result["data"]["retrieved"] == 3
    context = agent.llm.calls[0][1].content
    assert context.index("Title stocks/a#0") < context.index("Title bonds/b#0") < context.index("Title etfs_and_funds/c#0")
    assert [s["score"] for s in result["sources"]] == [0.8, 0.6, 0.5]


def test_merged_results_respect_the_article_cap_and_top_k():
    rag = get_config()["rag"]
    first = [chunk("stocks/x#0", 0.9), chunk("stocks/x#1", 0.8)]
    second = [chunk("stocks/x#2", 0.7)] + [chunk(f"bonds/b{i}#0", 0.6 - i / 100) for i in range(8)]
    agent, _ = make({STATE["query"]: first, REWRITE: second}, answer="ok [1]")
    agent.run(STATE)
    context = agent.llm.calls[0][1].content
    assert "stocks/x#2" not in context                      # third chunk of one article is capped
    assert context.count("\n[") == rag["top_k"]             # at most top_k numbered blocks


def test_topic_filter_from_state_is_passed_to_both_retrievals():
    agent, retrieve = make({STATE["query"]: [A]}, answer="ok [1]")
    agent.run({**STATE, "topic_filter": "bonds"})
    assert retrieve.calls == [(STATE["query"], "bonds"), (REWRITE, "bonds")]
    agent2, retrieve2 = make({STATE["query"]: [A]}, answer="ok [1]")
    agent2.run(STATE)
    assert {t for _, t in retrieve2.calls} == {None}


# ---- no_answer paths ----

def test_empty_retrieval_returns_no_answer_without_calling_the_answer_llm():
    agent, _ = make({}, answer_error=AssertionError("answer LLM must not be called"))
    result = agent.run(STATE)
    assert result["status"] == "no_answer" and result["sources"] == []
    assert result["answer"] == get_config()["finance_qa"]["no_answer_message"].strip()
    assert agent.llm.calls == []
    assert len(agent.rewrite_llm.calls) == 1     # the rewrite call is the only LLM call


@pytest.mark.parametrize("reply", ["NO_ANSWER", "NO_ANSWER.", "  `NO_ANSWER`\n", '"no_answer"'])
def test_sentinel_gives_no_answer_with_the_fixed_message(reply):
    agent, _ = make({STATE["query"]: [A]}, answer=reply)
    result = agent.run(STATE)
    assert result["status"] == "no_answer" and result["sources"] == []
    assert result["answer"] == get_config()["finance_qa"]["no_answer_message"].strip()


def test_text_that_merely_mentions_the_sentinel_is_an_answer():
    agent, _ = make({STATE["query"]: [A]}, answer="This is not a NO_ANSWER case [1].")
    assert agent.run(STATE)["status"] == "ok"


def test_empty_query_needs_input_and_calls_nothing():
    agent, retrieve = make({})
    result = agent.run({"turn_id": "t1", "query": "  "})
    assert result["status"] == "needs_input" and retrieve.calls == []
    assert agent.rewrite_llm.calls == [] and agent.llm.calls == []


# ---- answer, citations, sources ----

def test_prompt_is_grounded_and_uses_the_original_question():
    agent, _ = make({STATE["query"]: [A, B]}, answer="Answer [1].")
    agent.run(STATE)
    system, human = agent.llm.calls[0]
    for needle in ("ONLY from the numbered context", "[1], [2]", "NO_ANSWER", "loans, crypto",
                   "Educational only", "Never tell the reader what to buy"):
        assert needle in system.content
    assert "tax rates, limits or deadlines" in system.content and "India-first" in system.content
    assert "[1] Title stocks/a#0\ntext of stocks/a#0" in human.content and "[2] Title bonds/b#0" in human.content
    assert human.content.endswith("Question: " + STATE["query"])


def test_sources_are_only_the_cited_chunks_and_renumbered():
    agent, _ = make({STATE["query"]: [A, B, C]}, answer="Loans differ [2]. Prices move [2][3].")
    result = agent.run(STATE)
    assert result["status"] == "ok"
    assert result["answer"] == "Loans differ [1]. Prices move [1][2]."
    assert [s["chunk_id"] for s in result["sources"]] == ["bonds/b#0", "etfs_and_funds/c#0"]
    assert result["sources"][0]["url"] == "https://www.sebi.gov.in"
    assert result["sources"][0]["title"] == "Title bonds/b#0"


@pytest.mark.parametrize("text, expected", [
    ("A [1, 2].", "A [1, 2]."),
    ("A [1,2].", "A [1, 2]."),
    ("A [1][2].", "A [1][2]."),
    ("A [2][1].", "A [1][2]."),
    ("A [3, 1] and B [3].", "A [1, 2] and B [1]."),
    ("A [1, 1, 2].", "A [1, 2]."),
])
def test_citation_formats(text, expected):
    answer, cited = apply_citations(text, [A, B, C])
    assert answer == expected and len(cited) == 2


def test_two_chunks_of_one_article_are_two_numbered_sources():
    same_a = chunk("stocks/dividends#0", 0.8, title="Dividends Explained")
    same_b = chunk("stocks/dividends#3", 0.7, title="Dividends Explained")
    agent, _ = make({STATE["query"]: [same_a, same_b, B]}, answer="Paid from profit [2]. Dates matter [1][3].")
    result = agent.run(STATE)
    # cited in order 2, 1, 3 -> renumbered 1, 2, 3; every [n] is the n-th entry of sources, one-to-one
    assert result["answer"] == "Paid from profit [1]. Dates matter [2][3]."
    assert [s["chunk_id"] for s in result["sources"]] == ["stocks/dividends#3", "stocks/dividends#0", "bonds/b#0"]
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])})
    assert used == list(range(1, len(result["sources"]) + 1))
    assert len({s["chunk_id"] for s in result["sources"]}) == len(result["sources"])


def test_citation_formats_cited_chunks():
    assert [c.chunk_id for c in apply_citations("A [1, 2].", [A, B, C])[1]] == ["stocks/a#0", "bonds/b#0"]
    assert [c.chunk_id for c in apply_citations("A [1,2].", [A, B, C])[1]] == ["stocks/a#0", "bonds/b#0"]
    assert [c.chunk_id for c in apply_citations("A [3][1].", [A, B, C])[1]] == ["etfs_and_funds/c#0", "stocks/a#0"]


def test_invalid_citation_numbers_are_stripped():
    answer, cited = apply_citations("Rates matter [9]. Prices fall [1, 7] and [0].", [A, B])
    assert answer == "Rates matter. Prices fall [1] and."
    assert [c.chunk_id for c in cited] == ["stocks/a#0"]


def test_answer_with_only_invalid_citations_has_no_sources(caplog):
    agent, _ = make({STATE["query"]: [A]}, answer="Plain answer [5].")
    with caplog.at_level("WARNING"):
        result = agent.run(STATE)
    assert result["status"] == "ok" and result["sources"] == [] and result["answer"] == "Plain answer."
    assert "no valid citation" in caplog.text


# ---- errors ----

def test_answer_llm_exception_becomes_an_error_result():
    agent, _ = make({STATE["query"]: [A]}, answer_error=RuntimeError("boom"))
    result = agent.run(STATE)
    assert result["status"] == "error" and result["agent"] == "finance_qa" and result["turn_id"] == "t1"
    assert result["answer"] == get_config()["agents"]["error_message"]
    assert "boom" not in result["answer"] and result["sources"] == []


def test_retrieval_exception_becomes_an_error_result():
    def broken(query, topic=None):
        raise FileNotFoundError("no index")

    agent = FinanceQAAgent(llm=FakeLLM("x"), rewrite_llm=FakeLLM(REWRITE), retrieve_fn=broken)
    assert agent.run(STATE)["status"] == "error"


# ---- construction and usage ----

def test_default_llms_come_from_the_factory_with_agent_roles(monkeypatch):
    roles = []

    def fake_get_llm(role=None, **kwargs):
        roles.append(role)
        return FakeLLM("x")

    monkeypatch.setattr(base_agent, "get_llm", fake_get_llm)
    monkeypatch.setattr(finance_qa, "get_llm", fake_get_llm)
    agent = FinanceQAAgent(retrieve_fn=FakeRetrieve({}))
    assert roles == ["finance_qa", "finance_qa_rewrite"] and agent.name == "finance_qa"
    assert set(roles) <= set(get_config()["llm"]["roles"])      # both roles are configured


def test_default_retrieve_function_is_the_shared_retriever():
    from src.rag.retriever import retrieve

    assert FinanceQAAgent(llm=FakeLLM(), rewrite_llm=FakeLLM()).retrieve_fn is retrieve


def test_usage_of_both_calls_is_recorded_under_the_agent_roles():
    registry = get_usage_registry()
    registry.reset()

    def model(role, text, tokens_in, tokens_out):
        message = AIMessage(content=text, usage_metadata={
            "input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out})
        return GenericFakeChatModel(
            messages=iter([message]), callbacks=[UsageTracker(role, get_config()["llm"]["model"])])

    agent = FinanceQAAgent(
        llm=model("finance_qa", "Bonds are loans [1].", 300, 40),
        rewrite_llm=model("finance_qa_rewrite", REWRITE, 60, 12),
        retrieve_fn=FakeRetrieve({STATE["query"]: [A]}),
    )
    try:
        result = agent.run(STATE)
        assert result["status"] == "ok"
        assert registry.totals("finance_qa")["input_tokens"] == 300
        assert registry.totals("finance_qa")["output_tokens"] == 40
        assert registry.totals("finance_qa_rewrite")["input_tokens"] == 60
        assert registry.session_tokens() == 412
    finally:
        registry.reset()


def test_an_acronym_in_the_question_is_also_searched_in_plain_words():
    agent, retrieve = make({"What is long-term capital gains tax?": [A]}, answer="Gains are taxed [1].", rewrite="what is LTCG")
    result = agent.run({"turn_id": "t1", "query": "What is LTCG?"})
    assert [q for q, _ in retrieve.calls] == [
        "What is LTCG?", "What is long-term capital gains tax?", "what is LTCG", "what is long-term capital gains tax"]
    assert result["status"] == "ok" and result["sources"]


def test_an_answer_that_mentions_the_data_format_gives_the_fixed_no_answer_message():
    agent, _ = make({STATE["query"]: [A]}, answer="A bond is a loan [1]. The JSON says it pays interest.")
    result = agent.run(STATE)
    assert result["status"] == "no_answer" and result["answer"] == get_config()["finance_qa"]["no_answer_message"].strip() and result["sources"] == []
    ok, _ = make({STATE["query"]: [A]}, answer="A bond is a loan [1]. Analysts use data and inputs to a DCF.")
    assert ok.run(STATE)["status"] == "ok"
