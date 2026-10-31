"""Sorting a ticker box into symbols and names, and resolving the names (fake lookup, no network)."""
from __future__ import annotations

import pytest

from src.data.errors import TransientProviderError
from src.data.tickers import resolver as r
from src.data.tickers.entries import resolve_entries, split_entries
from src.data.tickers.fake import FakeLookup
from src.data.tickers.models import Candidate, Validation
from src.utils.config import get_config

CFG = get_config()["ticker_lookup"]
SUZLON = [Candidate("SUZLON.NS", "Suzlon Energy Limited", "NSI", "NSE", "EQUITY")]
GOOGS = [Candidate("GOOG", "Alphabet Inc.", "NMS", "NASDAQ", "EQUITY"), Candidate("GOOGL", "Alphabet Inc.", "NMS", "NASDAQ", "EQUITY")]


def resolver(table=None):
    return r.TickerResolver([FakeLookup(table or {})], lambda symbol: Validation(True, "INR"))


@pytest.mark.parametrize("raw, expected", [
    (["AAPL", "Suzlon"], ["AAPL", "Suzlon"]),
    (("AAPL",), ["AAPL"]),
    ("AAPL MSFT", ["AAPL", "MSFT"]),
    ("aapl,  Taiwan Semiconductor ,, ", ["aapl", "Taiwan Semiconductor"]),
    ("tata consultancy", ["tata consultancy"]),
    ("  Suzlon   ", ["Suzlon"]),
    ("   ", []),
    (None, []),
    (5, []),
])
def test_split_entries(raw, expected):
    assert split_entries(raw) == expected


def test_symbols_stay_names_are_resolved_and_the_text_typed_is_remembered_for_symbols():
    out = resolve_entries(["aapl", "Suzlon", "TCS.NS"], resolver({"suzlon": SUZLON}), 5)
    assert out.tickers == ["AAPL", "SUZLON.NS", "TCS.NS"]
    assert out.names == {"AAPL": "aapl", "TCS.NS": "TCS.NS"} and out.asks == [] and out.warnings == []
    assert out.notes == ['Showing SUZLON.NS (Suzlon Energy Limited, NSE, INR) for "Suzlon".']


def test_a_name_that_needs_the_user_is_an_ask_and_adds_no_symbol():
    out = resolve_entries(["Alphabet", "qzx wplk", "AAPL"], resolver({"alphabet": GOOGS}), 5)
    assert out.tickers == ["AAPL"] and len(out.asks) == 2
    assert "GOOG (Alphabet Inc., NASDAQ, INR)" in out.asks[0] and "couldn't find a listed company called \"qzx wplk\"" in out.asks[1]


def test_a_lookup_that_is_down_is_an_ask_not_a_guess():
    out = resolve_entries(["Suzlon"], resolver({"suzlon": TransientProviderError("t")}), 5)
    assert out.tickers == [] and "can't look up company names" in out.asks[0]


def test_entries_beyond_the_limit_are_reported_and_not_looked_up_and_non_text_entries_pass_through():
    lookup = FakeLookup({})
    out = resolve_entries(["AAPL", "MSFT", "Some Name", 7], r.TickerResolver([lookup], lambda s: Validation(True)), 2)
    assert out.tickers == ["AAPL", "MSFT"] and lookup.calls == []
    assert out.warnings == ["Only the first 2 tickers are analysed; 'Some Name' was left out.", "Only the first 2 tickers are analysed; '7' was left out."]
    out = resolve_entries([7, None], resolver(), 5)
    assert out.tickers == [7, None] and out.asks == []
