"""Rejects LLM-written text that talks about the internal data format ("the JSON says ...").

Design pattern: GUARD (one shared rule, called by every agent next to its number guard). Example: the Market summary
"... and the JSON says it is 0.7% above the 20-day average" matches `\\bjson\\b`, so the agent uses its code-built fallback text
instead and adds the warning `output_guard.format_warning`. The patterns are in config (`output_guard.format_patterns`),
case-insensitive, whole words, and there is NO exemption for words that appear in the input: a user never needs to be told about
JSON, a payload or a schema in an answer about money. Ordinary words such as "data" or "inputs to a DCF" do not match.
"""
from __future__ import annotations

import re

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def mentions_data_format(text: str) -> str | None:
    """The first configured pattern (output_guard.format_patterns) that matches `text` as a whole word, else None."""
    for pattern in get_config()["output_guard"]["format_patterns"]:
        if re.search(pattern, text, re.IGNORECASE):
            log.warning("Text mentions the internal data format (pattern %r); the fallback text will be used", pattern)
            return str(pattern)
    return None


def format_warning() -> str:
    """The warning shown with a fallback text that replaced a summary which mentioned the data format."""
    return " ".join(get_config()["output_guard"]["format_warning"].split())
