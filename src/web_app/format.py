"""Display formatting for the UI: the ONE place that turns numbers into text (no calculation happens here)."""
from __future__ import annotations


def format_count(value: int) -> str:
    """A whole number with thousands separators. Example: 1234567 -> "1,234,567"."""
    return f"{value:,}"


def format_usd(value: float) -> str:
    """A small USD amount with four decimals. Example: 0.00597 -> "0.0060"."""
    return f"{value:,.4f}"


def display_snippet(snippet: str) -> str:
    """The source snippet, with "…" added when it does not end like a sentence (it was cut from a longer chunk)."""
    text = snippet.strip()
    return text if not text or text[-1] in ".!?" else f"{text}…"
