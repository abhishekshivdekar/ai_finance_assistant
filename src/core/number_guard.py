"""The number guard: which numbers a text contains.

Design pattern: GUARD (a pure check shared by every agent). An agent builds the set of numbers its input
holds (collect_numbers) and rejects a model text with any number outside it (numbers_in).
Example: a summary that says "12.5%" passes only if 12.5 appears in the data the model was given.
Moved here from portfolio_metrics.py so Portfolio, Market, Goal Planning, News, Tax and the workflow
synthesizer share one module.
"""
from __future__ import annotations

import re
from typing import Any

# Western grouping (1,234,567), Indian grouping (12,34,567) or a plain number, each with an optional decimal part.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,3}(?:,\d{2})+,\d{3}(?:\.\d+)?|\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    """Every number in `text` as a float ("1,234.50" -> 1234.5, "1,25,000" -> 125000.0, "42%" -> 42.0).

    "1.25 lakh" and "2 crore" give 1.25 and 2 exactly as written (the unit word is not multiplied in).
    """
    return {float(m.replace(",", "")) for m in _NUMBER.findall(text)}


def collect_numbers(value: Any) -> set[float]:
    """Numbers found in the values (not the keys) of a nested dict/list."""
    if isinstance(value, dict):
        return set().union(*(collect_numbers(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(collect_numbers(v) for v in value)) if value else set()
    if isinstance(value, bool) or value is None:
        return set()
    return numbers_in(str(value))
