"""Smoke test: config, both LLM roles, latency and the usage tracker.
In VS Code: open this file and click the Run (play) button at the top right.
"""
import time

from src.core.llm_factory import get_llm
from src.core.usage_tracker import get_usage_registry
from src.utils.config import get_config


def main() -> None:
    cfg = get_config()
    print("App:", cfg["app"]["name"])
    print("Model:", cfg["llm"]["model"])
    for role in ("default", "router"):
        llm = get_llm(role=role)
        start = time.perf_counter()
        reply = llm.invoke("In one short sentence, what is diversification in investing?")
        print(f"\n[{role}] {time.perf_counter() - start:.1f}s: {reply.content}")
    print("\nUsage:", get_usage_registry().totals())
    print("\nSUCCESS: smoke test passed.")


if __name__ == "__main__":
    main()