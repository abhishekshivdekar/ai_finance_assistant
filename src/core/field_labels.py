"""The words a user sees for an input field or an option, instead of its snake_case key.

Design pattern: LOOKUP TABLE kept in config (config.yaml `field_labels` and `choice_labels`), with a safe fallback.
Example: the Goal Planning check reports `field_label("horizon_years")` = "Time horizon (years)", so Chat and the Goals
tab both show "Time horizon (years) must be a number." and never "horizon_years must be a number.".
"""
from __future__ import annotations

from src.utils.config import get_config


def _fallback(key: str) -> str:
    """A key nobody labelled, made readable: "some_new_key" -> "Some new key"."""
    return key.replace("_", " ").strip().capitalize()


def field_label(key: str) -> str:
    """The label of an input field (sentence case), or a readable form of the key when it has no label."""
    return str(get_config().get("field_labels", {}).get(key) or _fallback(key))


def choice_label(value: str) -> str:
    """The label of an option value (lower case, usable inside a sentence), or the value made readable."""
    return str(get_config().get("choice_labels", {}).get(value) or value.replace("_", " "))


def choice_list(values: list[str] | tuple[str, ...]) -> str:
    """Option values as a comma-separated list of labels. Example: ["annual", "one_time"] -> "yearly deposits, one-time deposit"."""
    return ", ".join(choice_label(value) for value in values)
