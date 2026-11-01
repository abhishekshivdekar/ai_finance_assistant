"""The shared check that keeps "the JSON says ..." out of user answers (UI-visible defect found in live_check_workflow 2026-10-31)."""
from __future__ import annotations

import copy

import pytest
import yaml

from src.core.format_guard import format_warning, mentions_data_format
from src.utils.config import ConfigError, get_config


@pytest.mark.parametrize("text", [
    "The price is above its average, and the JSON says it is 0.7% above the 20-day average.",
    "the json says so", "Here is the JSON: a bond is a loan.", "JSON", "(json)", "A JSON-like list of rates.",
    "The payload shows a loss.", "Payload: none.", "According to the schema, fees are low.", "SCHEMA",
])
def test_text_that_mentions_the_internal_data_format_is_caught_whatever_the_case(text):
    assert mentions_data_format(text)


@pytest.mark.parametrize("text", [
    "Analysts gather data and use inputs to a DCF model to estimate value.",
    "The data show a 5% return; the inputs to a DCF are cash flows and a discount rate.",
    "A schemata of risk, a payloads list, jsonify and unjsoned are not whole words.",
    "The price is 100.00, above its 20-day average, and 1.2% below its 52-week high.",
    "",
])
def test_ordinary_finance_sentences_pass(text):
    assert mentions_data_format(text) is None


def test_the_patterns_and_the_warning_come_from_the_config():
    guard = get_config()["output_guard"]
    assert guard["format_patterns"] == [r"\bjson\b", r"\bpayload\b", r"\bschema\b"]
    assert mentions_data_format("see the json") == r"\bjson\b" and mentions_data_format("the schema") == r"\bschema\b"
    assert format_warning() == " ".join(guard["format_warning"].split()) and "data format" in format_warning()


def _config(tmp_path, change):
    guard = copy.deepcopy(get_config()["output_guard"])
    get_config.cache_clear()
    change(guard)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"output_guard": guard}), encoding="utf-8")
    return str(path)


def test_a_valid_output_guard_block_loads(tmp_path):
    assert get_config(_config(tmp_path, lambda g: None))["output_guard"]["format_patterns"]


@pytest.mark.parametrize("change, text", [
    (lambda g: g.pop("format_patterns"), "format_patterns"),
    (lambda g: g.update(format_patterns=[]), "format_patterns"),
    (lambda g: g.update(format_patterns="json"), "format_patterns"),
    (lambda g: g.update(format_patterns=["json", " "]), "format_patterns"),
    (lambda g: g.update(format_patterns=["(unclosed"]), "invalid regex"),
    (lambda g: g.update(format_warning=" "), "format_warning"),
    (lambda g: g.pop("format_warning"), "format_warning"),
])
def test_a_bad_output_guard_block_is_rejected(tmp_path, change, text):
    with pytest.raises(ConfigError, match=text):
        get_config(_config(tmp_path, change))
