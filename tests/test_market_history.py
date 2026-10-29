import threading
import time

from src.core.market_history import fetch_histories, parse_request
from src.core.models import MarketResult
from src.utils.config import get_config

CFG = get_config()["market_analysis"]
TICKERS = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS", "F.NS"]


# ---- reading the request ----

def test_no_params_gives_no_tickers_and_the_default_period():
    assert parse_request(None, CFG) == ([], CFG["default_period"], [])
    assert parse_request({"tickers": 5}, CFG)[0] == []


def test_huge_infinite_and_odd_values_never_raise():
    odd = [10 ** 400, float("inf"), float("nan"), None, {}]
    for value in odd:
        assert parse_request({"tickers": [value], "period": value}, CFG)[:2] == ([], CFG["default_period"])
        assert parse_request({"tickers": value, "period": value}, CFG)[:2] == ([], CFG["default_period"])


def test_tickers_are_normalised_deduplicated_and_capped():
    cap = CFG["max_tickers"]
    names = ["A", "B", "C", "D", "E", "F"][: cap + 1]
    tickers, period, warnings = parse_request({"tickers": [" aapl", "bad!!", "AAPL", *names[1:]], "period": "3mo"}, CFG)
    assert tickers == (["AAPL"] + names[1:])[:cap] and period == "3mo"
    assert warnings[0] == "'bad!!' was skipped: it is not a valid ticker."
    assert warnings[1] == "AAPL appears more than once; it is analysed once."
    assert warnings[2] == f"Only the first {cap} tickers are analysed; {names[-1]} was left out."


def test_a_string_of_tickers_is_split_on_commas_and_spaces():
    assert parse_request({"tickers": "aapl, TCS.NS msft"}, CFG)[0] == ["AAPL", "TCS.NS", "MSFT"]


def test_a_non_text_ticker_is_skipped_with_a_warning():
    tickers, _, warnings = parse_request({"tickers": [7, "AAPL"]}, CFG)
    assert tickers == ["AAPL"] and warnings == ["'7' was skipped: it is not a valid ticker."]


def test_unknown_period_falls_back_to_the_default_with_a_warning():
    _, period, warnings = parse_request({"tickers": ["AAPL"], "period": "5y"}, CFG)
    assert period == CFG["default_period"]
    assert warnings == [f"The period '5y' is not supported; {CFG['default_period']} is used."]
    assert parse_request({"tickers": ["AAPL"], "period": None}, CFG)[1] == CFG["default_period"]


# ---- concurrent fetching ----

def result(ticker):
    return MarketResult(data=[{"ticker": ticker}], currency="INR", as_of=None, source="live")


class SlowProvider:
    """Fake provider that records the requested period and how many calls run at the same time."""

    def __init__(self, delays=None, fail=()):
        self.delays, self.fail, self.periods = delays or {}, set(fail), []
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def get_history(self, ticker, period):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.periods.append(period)
        try:
            time.sleep(self.delays.get(ticker, 0.03))
            if ticker in self.fail:
                raise RuntimeError("boom " + ticker)
            return result(ticker)
        finally:
            with self._lock:
                self.active -= 1


def test_every_ticker_is_fetched_once_for_the_configured_fetch_period():
    provider = SlowProvider()
    fetch_histories(TICKERS, provider, CFG)
    assert provider.periods == [CFG["fetch_period"]] * len(TICKERS)


def test_histories_are_fetched_concurrently_but_bounded_by_max_workers():
    wide = SlowProvider()
    fetch_histories(TICKERS, wide, {**CFG, "max_workers": 6})
    assert wide.max_active > 1
    bounded = SlowProvider()
    fetch_histories(TICKERS, bounded, {**CFG, "max_workers": 2})
    assert 1 < bounded.max_active <= 2
    for workers in (1, 0):                                           # 0 is treated as 1
        serial = SlowProvider()
        fetch_histories(TICKERS, serial, {**CFG, "max_workers": workers})
        assert serial.max_active == 1


def test_results_keep_input_order_even_when_early_histories_are_slower():
    delays = {t: 0.02 * (len(TICKERS) - i) for i, t in enumerate(TICKERS)}
    results = fetch_histories(TICKERS, SlowProvider(delays), {**CFG, "max_workers": 6})
    assert [r.data[0]["ticker"] for r in results] == TICKERS


def test_one_failing_ticker_does_not_affect_the_others():
    results = fetch_histories(TICKERS, SlowProvider(fail={"C.NS"}), {**CFG, "max_workers": 6})
    assert [r.error for r in results] == [None, None, "History lookup failed for C.NS.", None, None, None]
    assert results[0].data and results[3].data


def test_no_tickers_means_no_calls():
    provider = SlowProvider()
    assert fetch_histories([], provider, CFG) == [] and provider.periods == []
