"""Live check for the WHOLE workflow through run_query (uses the REAL LLM and the real market/news/RAG sources, so it
costs tokens and uses the network). Not run by the tests; run it yourself from the project root:  python live_check_workflow.py

Each question gets its own conversation thread, except the last two, which share one thread to show that the history
is used ("And what about Apple?" needs the Microsoft turn). Per turn it prints: intents, confidence, nodes visited,
merge mode (llm / verbatim / none), the final answer, the sources and the tokens/cost of that turn; at the end the totals
by role. It reads the Market freshness and warnings from the result payloads, as the UI must.
"""
import uuid

from src.core.usage_tracker import get_usage_registry
from src.workflow.facade import run_query
from live_checks.transcript import run_with_transcript

# (label, query); a label starting with "thread:" shares the conversation thread with the other turns of that name.
TURNS: list[tuple[str, str]] = [
    ("finance_qa", "What is a mutual fund?"),
    ("two agents", "What is Apple trading at, and what is a P/E ratio?"),
    ("ambiguous", "How is it doing?"),
    ("advice", "Should I buy Tesla?"),
    ("advice, Indian stock", "Is Reliance a good buy right now?"),
    ("advice, prediction", "Will TCS go up?"),
    ("advice, fund", "Which mutual fund should I invest in?"),
    ("off-topic", "Best pasta recipe?"),
    ("tax question", "What is LTCG?"),
    ("news", "Latest news on Nifty 50"),
    ("three parts", "What is Apple trading at, what is a P/E ratio, and what is the latest news on Apple?"),
    ("portfolio chat ask", "Can you analyse my portfolio?"),
    ("thread:follow-up 1", "How has Microsoft done over the last 6 months?"),
    ("thread:follow-up 2", "And what about Apple?"),
]


def totals() -> tuple[int, int, float | None]:
    """(input tokens, output tokens, cost) recorded so far across all roles; cost is None once any call is unpriced."""
    rows = list(get_usage_registry().totals().values())
    costs = [r["cost"] for r in rows]
    return (sum(r["input_tokens"] for r in rows), sum(r["output_tokens"] for r in rows),
            None if any(c is None for c in costs) else sum(costs))


def payload_lines(result) -> list[str]:
    """Market source and every agent's payload warnings, read from the result payloads (never from the answer text)."""
    lines = []
    for item in result.results:
        data = item.get("data") or {}
        if data.get("source"):
            lines.append(f"{item['agent']} source: {data['source']}")
        lines += [f"{item['agent']} warning: {w}" for w in data.get("warnings") or []]
    return lines


def main() -> None:
    threads: dict[str, str] = {}
    for label, query in TURNS:
        key = label.split(":")[0] if label.startswith("thread:") else label
        thread = threads.setdefault(key, f"live-{uuid.uuid4().hex[:8]}")
        before = totals()
        result = run_query(query, thread)
        after = totals()
        print(f"\n[{label}] {query}")
        print(f"  intents:     {result.intents}")
        print(f"  confidence:  {result.confidence}")
        print(f"  nodes:       {result.nodes}")
        print(f"  merge:       {result.merge_mode or 'none'}")
        print(f"  stopped:     {result.stopped}")
        print(f"  answer:      {result.answer}")
        print(f"  sources:     {[(s.get('title'), s.get('url')) for s in result.sources]}")
        print(f"  payloads:    {payload_lines(result)}")
        cost = "n/a" if None in (before[2], after[2]) else f"${after[2] - before[2]:.6f}"
        print(f"  tokens:      in={after[0] - before[0]} out={after[1] - before[1]} cost={cost}")
    print("\nTotals by role:")
    for role, row in get_usage_registry().totals().items():
        print(f"  {role}: in={row['input_tokens']} out={row['output_tokens']} cost={row.get('cost')}")


if __name__ == "__main__":
    run_with_transcript(main, __file__)
