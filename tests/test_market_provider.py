import pytest

from src.core.models import MarketResult
from src.data.market.provider import (
    InvalidSymbolError,
    MarketDataProvider,
    error_result,
    validate_period,
    validate_symbol,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("AAPL", "AAPL"),
        (" msft ", "MSFT"),
        ("BRK-B", "BRK-B"),
        ("BF.B", "BF.B"),
        ("RELIANCE.NS", "RELIANCE.NS"),
        ("m&m.ns", "M&M.NS"),
        ("BAJAJ-AUTO.NS", "BAJAJ-AUTO.NS"),
        ("^NSEI", "^NSEI"),
        ("^BSESN", "^BSESN"),
        ("^GSPC", "^GSPC"),
    ],
)
def test_valid_symbols(raw, expected):
    assert validate_symbol(raw) == expected


@pytest.mark.parametrize(
    "raw", ["", "   ", None, 123, "^", "AAPL.XYZ", "RELIANCE.BO", "A B", "TOOLONGTICKER", ".NS", "AAPL;DROP"]
)
def test_invalid_symbols(raw):
    with pytest.raises(InvalidSymbolError):
        validate_symbol(raw)


def test_period_validation():
    assert validate_period("1mo") == "1mo"
    with pytest.raises(InvalidSymbolError):
        validate_period("7 years")


def test_error_result_has_no_data_or_currency():
    result = error_result("boom")
    assert result.data is None and result.currency is None and result.as_of is None
    assert result.error == "boom"


class Recorder(MarketDataProvider):
    """Records hook calls so we can prove invalid input never reaches them."""

    def __init__(self):
        self.calls = []

    def _get_quote(self, symbol):
        self.calls.append(("quote", symbol))
        return MarketResult(data={}, currency="USD", as_of=None, source="live")

    def _get_history(self, symbol, period):
        self.calls.append(("history", symbol, period))
        return MarketResult(data=[], currency="USD", as_of=None, source="live")

    def _get_news(self, symbol):
        self.calls.append(("news", symbol))
        return MarketResult(data=[], currency=None, as_of=None, source="live")


def test_valid_input_reaches_hooks_normalised():
    rec = Recorder()
    rec.get_quote(" aapl ")
    rec.get_history("^nsei", "1mo")
    rec.get_news("  Nifty \x00 50  ")
    assert rec.calls == [("quote", "AAPL"), ("history", "^NSEI", "1mo"), ("news", "Nifty 50")]


def test_invalid_input_returns_error_result_without_hook_call():
    rec = Recorder()
    for result in (rec.get_quote(""), rec.get_history("???", "1mo"),
                   rec.get_history("AAPL", "bad"), rec.get_news("   "), rec.get_news(None),
                   rec.get_news("x" * 500)):
        assert result.error
        assert result.data is None and result.source == "none"
    assert rec.calls == []


def test_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        MarketDataProvider()
