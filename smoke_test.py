"""Day 1 check: confirms config, logging and your Gemini key all work.
In VS Code: open this file and click the Run (play) button at the top right.
"""
from src.core.llm_factory import get_llm
from src.utils.config import get_config


def main() -> None:
    cfg = get_config()
    print("App:", cfg["app"]["name"])
    print("Model:", cfg["llm"]["model"])
    llm = get_llm()
    reply = llm.invoke("In one short sentence, what is diversification in investing?")
    print("\nLLM reply:\n", reply.content)
    print("\nSUCCESS: Day 1 setup works.")


if __name__ == "__main__":
    main()
