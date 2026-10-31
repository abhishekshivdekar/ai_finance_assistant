"""Display formatting for the UI: the ONE place that turns numbers and times into text (no calculation happens here)."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


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


def _indian_grouping(digits: str) -> str:
    """Digits grouped as 12,34,567 (last three, then pairs). Example: "1234567" -> "12,34,567"."""
    head, tail = digits[:-3], digits[-3:]
    pairs = []
    while len(head) > 2:
        pairs.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + pairs + [tail]) if len(digits) > 3 else digits


def format_amount(value: float, currency: str) -> str:
    """An amount with its currency and two decimals; INR uses lakh/crore grouping, other currencies thousands.

    Example: (1234567.891, "INR") -> "INR 12,34,567.89"; (1234567.891, "USD") -> "USD 1,234,567.89"; (-50, "INR") -> "INR -50.00".
    """
    whole, _, decimals = f"{abs(value):.2f}".partition(".")
    grouped = _indian_grouping(whole) if currency.upper() == "INR" else f"{int(whole):,}"
    sign = "-" if value < 0 and float(f"{abs(value):.2f}") != 0 else ""
    return f"{currency} {sign}{grouped}.{decimals}"


def format_quantity(value: float) -> str:
    """A quantity with up to four decimals and no trailing zeros. Example: 10.0 -> "10", 2.50 -> "2.5", 1234.5 -> "1,234.5"."""
    return f"{value:,.4f}".rstrip("0").rstrip(".")


PORTFOLIO_NUMBER_COLUMNS = ("quantity", "avg_cost", "expense_ratio_pct")
PORTFOLIO_TEXT_COLUMNS = ("instrument_type", "asset_class")


def _plain(value: Any) -> Any:
    """A table cell as a plain Python value, or None when it is missing (None, NaN, pandas NA, blank text).

    numpy scalars are turned into Python numbers with .item(); nothing from pandas or numpy leaves this function.
    """
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


def clean_portfolio_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows of the Portfolio table as plain dicts for the backend: blank rows skipped, missing cells left out.

    `records` is the table's `to_dict("records")`. A row is blank when ticker, quantity, average cost and expense
    ratio are all empty (the dropdown defaults do not count). Numbers stay numbers (a numeric text becomes a float; other
    text is kept, and the backend reports the row), the ticker and the dropdown values are trimmed text. No validation
    happens here. Example: {"ticker": " Suzlon ", "quantity": 10.0, "avg_cost": nan} -> {"ticker": "Suzlon", "quantity": 10.0}.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        ticker = _plain(record.get("ticker"))
        if ticker is not None:
            row["ticker"] = ticker if isinstance(ticker, str) else str(ticker)
        for column in PORTFOLIO_NUMBER_COLUMNS:
            value = _plain(record.get(column))
            if isinstance(value, str):
                try:
                    number = float(value.replace(",", ""))
                except ValueError:
                    number = None
                value = value if number is None else (None if math.isnan(number) else number)
            if value is not None:
                row[column] = value
        if not row:
            continue
        for column in PORTFOLIO_TEXT_COLUMNS:
            value = _plain(record.get(column))
            if value is not None:
                row[column] = value if isinstance(value, str) else str(value)
        rows.append(row)
    return rows
