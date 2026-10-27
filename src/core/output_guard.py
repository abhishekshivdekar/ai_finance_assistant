"""Rule-based output guard (architecture section 3.3).

Design pattern: TEMPLATE METHOD (as used by BaseAgent.run()). The guard lives
in the shared run() step, so every agent gets it for free. Example: a
one-stock portfolio makes the commentary say "you should sell"; run() catches
it, asks the LLM once for an educational rewording, and falls back to a safe
message if the phrase is still there.
"""
from __future__ import annotations

import re
from typing import Any

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def find_violation(text: str) -> str | None:
    """Return the first configured pattern that matches `text`, else None."""
    for pattern in get_config()["output_guard"]["patterns"]:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return None


def guard_answer(answer: str, llm: Any) -> str:
    """Return `answer` if clean; otherwise one rewording retry, then a safe message."""
    pattern = find_violation(answer)
    if pattern is None:
        return answer

    cfg = get_config()["output_guard"]
    log.warning("Output guard matched %r; asking for a rewording", pattern)
    try:
        reply = llm.invoke(cfg["rewrite_prompt"].format(answer=answer))
        reworded = str(getattr(reply, "content", reply))
    except Exception:
        log.exception("Output guard rewording failed")
        return cfg["safe_message"].strip()

    if find_violation(reworded) is None:
        return reworded
    log.warning("Reworded answer still matched the guard; using the safe message")
    return cfg["safe_message"].strip()
