"""Display formatting for the UI: the ONE place that turns numbers and times into text (no calculation happens here)."""
from __future__ import annotations

from datetime import datetime, timezone


def format_count(value: int) -> str:
    """A whole number with thousands separators. Example: 1234567 -> "1,234,567"."""
    return f"{value:,}"


def format_usd(value: float) -> str:
    """A small USD amount with four decimals. Example: 0.00597 -> "0.0060"."""
    return f"{value:,.4f}"


def format_percent(value: float | None, missing: str) -> str:
    """A percentage with one decimal and a sign for gains. Example: 12.34 -> "+12.3%", -4.5 -> "-4.5%"; None -> `missing`."""
    return missing if value is None else f"{value:+.1f}%"


def format_timestamp(iso: str) -> str:
    """An ISO time as "2026-10-30 08:32 UTC" (an aware time is converted to UTC, a naive one is taken as UTC).

    Text that is not an ISO time is returned unchanged, so a payload's odd value is shown rather than hidden.
    """
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    moment = moment.astimezone(timezone.utc) if moment.tzinfo else moment
    return f"{moment:%Y-%m-%d %H:%M} UTC"


def display_snippet(snippet: str) -> str:
    """The source snippet, with "…" added when it does not end like a sentence (it was cut from a longer chunk)."""
    text = snippet.strip()
    return text if not text or text[-1] in ".!?" else f"{text}…"
