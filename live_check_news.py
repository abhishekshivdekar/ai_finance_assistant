"""Live check for the News agent (REAL Google News RSS and REAL LLM, so it uses network and tokens).

Run from the project root:  python live_check_news.py
For each request it prints the search phrase used, the headlines (publisher, time), the summary with
its citations, the warnings and, at the end, the token cost.
"""
from src.agents.news import NewsAgent
from src.core.usage_tracker import get_usage_registry
from live_checks.transcript import run_with_transcript

REQUESTS = [
    {"ticker": "AAPL"},
    {"ticker": "RELIANCE.NS"},        # uses the query override in config
    {"query": "Nifty 50"},            # free search phrase
    {"query": "qzxv wplk jjrt zzqy"},  # gibberish: expected to find nothing
]


def show(agent: NewsAgent, params: dict) -> None:
    """Run one request and print everything a reviewer needs to judge it."""
    result = agent.run({"turn_id": "live", "news_params": params})
    data = result["data"] or {}
    print(f"\n=== {params} ===")
    print(f"status: {result['status']}  query used: {data.get('query')!r}  source: {data.get('source')}  "
          f"stale: {data.get('is_stale')}  as_of: {data.get('as_of')}")
    for item in data.get("items", []):
        print(f"  [{item['number']}] {item['publisher'] or '-'} | {item['published']} | {item['title']}")
    print("summary:")
    print(result["answer"])
    print("citations:", data.get("summary_citations") or "none")
    print("warnings:", data.get("warnings") or "none")


def main() -> None:
    agent = NewsAgent()
    for params in REQUESTS:
        show(agent, params)
    print("\n=== Token cost ===")
    for role, t in sorted(get_usage_registry().totals().items()):
        cost = "unpriced model" if t["cost"] is None else f"${t['cost']:.6f}"
        print(f"  {role:20s} in={t['input_tokens']:>6} out={t['output_tokens']:>6} cost={cost}")


if __name__ == "__main__":
    run_with_transcript(main, __file__)
