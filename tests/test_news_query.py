import pytest

from src.core.news_query import resolve_query, ticker_query
from src.utils.config import get_config

CFG = get_config()["news"]


def test_ticker_with_an_override_uses_the_configured_phrase():
    assert resolve_query({"ticker": "RELIANCE.NS"}, CFG).query == "Reliance Industries stock"
    assert resolve_query({"ticker": " aapl "}, CFG).query == "Apple Inc AAPL stock"
    assert resolve_query({"ticker": "^NSEI"}, CFG).query == "Nifty 50 index"


@pytest.mark.parametrize("symbol, expected", [
    ("MARUTI.NS", "MARUTI NSE stock"),
    ("TSLA", "TSLA stock"),
    ("^GSPC", "GSPC index"),
    ("BRK-B", "BRK-B stock"),
])
def test_ticker_without_an_override_uses_a_template(symbol, expected):
    assert ticker_query(symbol, CFG) == expected
    assert resolve_query({"ticker": symbol}, CFG).query == expected


def test_free_query_is_cleaned_and_used_as_typed():
    request = resolve_query({"query": "  Nifty \n 50  "}, CFG)
    assert request.query == "Nifty 50" and request.message is None


def test_query_wins_when_both_are_given():
    assert resolve_query({"ticker": "AAPL", "query": "Nifty 50"}, CFG).query == "Nifty 50"


@pytest.mark.parametrize("params", [None, {}, "AAPL", 5, {"ticker": ""}, {"query": "   "}, {"ticker": None, "query": None}])
def test_missing_input_asks_for_a_ticker_or_phrase(params):
    request = resolve_query(params, CFG)
    assert request.query is None and request.message == CFG["messages"]["needs_input"]


def test_invalid_ticker_gets_a_message_with_the_detail():
    request = resolve_query({"ticker": "???"}, CFG)
    assert request.query is None and "not valid" in request.message and "Invalid ticker" in request.message


def test_too_long_query_is_rejected():
    request = resolve_query({"query": "x" * (CFG["max_query_chars"] + 1)}, CFG)
    assert request.query is None and "longer than" in request.message


@pytest.mark.parametrize("raw, expected", [
    (3, 3), (1, 1), (0, 1), (-4, 1), (999, CFG["max_items"]), (None, CFG["max_items"]),
    ("5", CFG["max_items"]), (True, CFG["max_items"]), (2.5, CFG["max_items"]),
])
def test_max_items_is_clamped_to_the_config_range(raw, expected):
    assert resolve_query({"ticker": "AAPL", "max_items": raw}, CFG).max_items == expected


@pytest.mark.parametrize("raw", ["5", True, 2.5, [3]])
def test_a_non_int_max_items_uses_the_default_and_warns(raw):
    request = resolve_query({"ticker": "AAPL", "max_items": raw}, CFG)
    assert request.max_items == CFG["max_items"]
    assert request.warning == CFG["messages"]["max_items_ignored"].format(default=CFG["max_items"])


@pytest.mark.parametrize("params", [{"ticker": "AAPL"}, {"ticker": "AAPL", "max_items": None}, {"ticker": "AAPL", "max_items": 3}])
def test_a_missing_or_int_max_items_gives_no_warning(params):
    assert resolve_query(params, CFG).warning is None


def test_the_warning_is_kept_when_the_input_itself_needs_fixing():
    assert resolve_query({"ticker": "???", "max_items": "x"}, CFG).warning
    assert resolve_query({"max_items": "x"}, CFG).warning


def test_limit_is_also_returned_with_a_message():
    assert resolve_query({"max_items": 2}, CFG).max_items == 2
