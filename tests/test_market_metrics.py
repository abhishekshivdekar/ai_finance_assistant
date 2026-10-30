from datetime import date, datetime, timedelta, timezone

import pytest

from src.core import market_metrics as mm
from src.core.market_models import AverageMetric, MarketMetrics, TickerMetrics
from src.utils.config import get_config

CFG = get_config()["market_analysis"]
T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def series(closes, start=date(2025, 1, 1)):
    """Consecutive calendar days, oldest first."""
    return [(start + timedelta(days=i), float(c)) for i, c in enumerate(closes)]


def metrics_for(closes, period="6mo", **kwargs):
    return mm.ticker_metrics("AAA", series(closes), "USD", period, CFG, **kwargs)


# ---- reading provider rows ----

def test_price_series_keeps_usable_rows_sorted_by_date():
    rows = [
        {"date": "2026-01-03T00:00:00+05:30", "close": 3.0},
        {"date": "2026-01-01", "close": 1.0},
        {"date": "2026-01-02", "close": None},          # no close
        {"date": "2026-01-04", "close": float("nan")},  # not finite
        {"date": "2026-01-05", "close": -2.0},          # not positive
        {"date": "2026-01-06", "close": True},          # a bool is not a price
        {"date": "not a date", "close": 5.0},           # no usable date
        "junk",
    ]
    assert mm.price_series(rows) == [(date(2026, 1, 1), 1.0), (date(2026, 1, 3), 3.0)]
    assert mm.price_series(None) == [] and mm.price_series("x") == []


def test_slice_period_keeps_days_within_the_window_of_the_last_day():
    data = series(range(1, 11))                     # 10 consecutive days
    assert [c for _, c in mm.slice_period(data, 3)] == [7.0, 8.0, 9.0, 10.0]
    assert mm.slice_period(data, 365) == data
    assert mm.slice_period([], 30) == []


# ---- single metrics against hand-computed numbers ----

def test_latest_price_and_day_change():
    assert mm.latest_and_day_change([100, 110]) == (110, 10, pytest.approx(10.0))
    assert mm.latest_and_day_change([100]) == (100, None, None)


def test_period_return_pct():
    assert mm.period_return_pct([100, 110, 99, 108.9]) == pytest.approx(8.9)
    assert mm.period_return_pct([100]) is None


def test_sma_uses_the_last_window_closes_and_needs_enough_data():
    closes = [1, 2, 3, 4, 5]
    assert mm.sma(closes, 3) == 4 and mm.sma(closes, 5) == 3
    assert mm.sma(closes, 6) is None and mm.sma(closes, 0) is None


def test_price_vs_pct_is_signed():
    assert mm.price_vs_pct(110, 100) == pytest.approx(10.0)
    assert mm.price_vs_pct(90, 100) == pytest.approx(-10.0)


def test_daily_returns():
    assert mm.daily_returns([100, 110, 99, 108.9]) == pytest.approx([0.10, -0.10, 0.10])


def test_annualised_volatility_matches_hand_calculation():
    # returns +0.1, -0.1, +0.1: mean 0.03333, squared deviations sum 0.026667,
    # sample variance 0.013333, stdev 0.11547; x sqrt(4) x 100 = 23.094
    assert mm.annualised_volatility_pct([100, 110, 99, 108.9], 4) == pytest.approx(23.094, abs=1e-3)
    assert mm.annualised_volatility_pct([100, 100, 100], 252) == 0.0
    assert mm.annualised_volatility_pct([100, 110], 252) is None     # a single return has no spread


def test_max_drawdown_is_the_largest_fall_from_a_running_peak():
    assert mm.max_drawdown_pct([100, 110, 99, 108.9]) == pytest.approx(-10.0)
    assert mm.max_drawdown_pct([100, 120, 90, 130, 117]) == pytest.approx(-25.0)   # 120 -> 90 beats 130 -> 117
    assert mm.max_drawdown_pct([1, 2, 3]) == 0.0
    assert mm.max_drawdown_pct([5]) is None


def test_range_52w_needs_enough_history_and_reports_distance_from_each_end():
    data = [(date(2025, 1, 1), 50.0), (date(2025, 6, 1), 100.0), (date(2026, 1, 5), 80.0)]   # spans 369 days
    r = mm.range_52w(data, 80.0, 350)
    assert (r.high, r.low) == (100.0, 50.0)
    assert r.pct_below_high == pytest.approx(20.0) and r.pct_above_low == pytest.approx(60.0)
    assert mm.range_52w(data, 80.0, 400) is None                                  # history too short
    assert mm.range_52w([], 80.0, 350) is None


# ---- trend thresholds (config: gap 0.5, return 3.0) ----

@pytest.mark.parametrize("short, long, ret, expected", [
    (101, 100, 3.0, "Uptrend"),         # gap 1.0 > 0.5 and return exactly at the threshold
    (101, 100, 2.99, "Sideways"),       # return just below the threshold
    (100.2, 100, 10.0, "Sideways"),     # gap inside the 0.5 band
    (99, 100, -3.0, "Downtrend"),
    (99, 100, -2.99, "Sideways"),
    (99.8, 100, -10.0, "Sideways"),      # gap inside the band
    (101, 100, -10.0, "Sideways"),      # averages and return disagree
    (99, 100, 10.0, "Sideways"),
    (None, 100, 5.0, None),
    (101, None, 5.0, None),
    (101, 100, None, None),
])
def test_trend_label_thresholds(short, long, ret, expected):
    assert mm.trend_label(short, long, ret, CFG) == expected


# ---- composing one ticker ----

def test_ticker_metrics_slices_the_period_but_averages_and_range_use_the_full_year():
    closes = [50] * 300 + [100] * 60                 # 360 days: low half, then high half
    found, warnings = metrics_for(closes, period="1mo")
    assert warnings == []
    # 1mo slice is 31 flat days at 100: no return, no volatility, no drawdown
    assert found.period_return_pct == pytest.approx(0.0) and found.volatility_pct == 0.0
    assert found.max_drawdown_pct == 0.0
    # full year: both averages are 100, high 100, low 50
    assert [(a.window, a.value) for a in found.averages] == [(20, 100.0), (50, 100.0)]
    assert (found.range_52w.high, found.range_52w.low) == (100.0, 50.0)
    assert found.range_52w.pct_above_low == pytest.approx(100.0) and found.trend == "Sideways"
    # the whole year as the period sees the jump
    whole, _ = metrics_for(closes, period="1y")
    assert whole.period_return_pct == pytest.approx(100.0)


def test_ticker_metrics_latest_price_and_day_change_come_from_the_last_two_closes():
    found, _ = metrics_for(range(1, 61), price_source="live", price_stale=False, as_of=T0)
    assert found.latest_price == 60 and found.day_change == 1
    assert found.day_change_pct == pytest.approx(1 / 59 * 100)
    assert found.currency == "USD" and found.price_source == "live" and found.as_of == T0


def test_rising_series_is_an_uptrend_and_falling_is_a_downtrend():
    up, _ = metrics_for(range(100, 460), period="1y")
    down, _ = metrics_for(range(460, 100, -1), period="1y")
    assert up.trend == "Uptrend" and down.trend == "Downtrend"
    assert up.max_drawdown_pct == 0.0 and down.max_drawdown_pct < 0


def test_short_history_omits_metrics_with_warnings_and_does_not_fail():
    found, warnings = metrics_for([100 + i for i in range(30)])        # 30 days only
    assert [a.window for a in found.averages] == [20]                  # 50-day average omitted
    assert found.range_52w is None and found.trend is None
    assert found.period_return_pct is not None and found.volatility_pct is not None
    assert warnings == [
        "AAA: only 29 days of data for the requested 6mo; return, volatility and drawdown cover that shorter span.",
        "AAA: the 50-day average was left out (only 30 daily prices).",
        "AAA: the 52-week range was left out (the history covers only 29 days).",
        "AAA: the trend label was left out because a moving average or the return is missing.",
    ]


def test_two_prices_give_return_and_drawdown_but_no_volatility():
    found, warnings = metrics_for([100, 110])
    assert found.period_return_pct == pytest.approx(10.0) and found.max_drawdown_pct == 0.0
    assert found.volatility_pct is None and found.averages == []
    assert any("volatility was left out" in w for w in warnings)


def test_fewer_than_two_prices_gives_no_metrics():
    assert mm.ticker_metrics("AAA", series([100]), "USD", "6mo", CFG) == (
        None, ["AAA was left out: not enough price history."])
    assert mm.ticker_metrics("AAA", [], "USD", "6mo", CFG)[0] is None


# ---- collecting tickers ----

def tm(ticker="AAA", currency="USD", stale=False, source="live", as_of=T0, trend="Uptrend"):
    return TickerMetrics(
        ticker=ticker, currency=currency, period="6mo", latest_price=100.0, day_change=1.0, day_change_pct=1.0,
        period_return_pct=5.0, averages=[], volatility_pct=20.0, max_drawdown_pct=-4.0, range_52w=None,
        trend=trend, price_source=source, price_stale=stale, as_of=as_of,
    )


def test_build_metrics_flags_stale_and_sample_prices_and_keeps_the_oldest_time():
    older = datetime(2026, 9, 1, tzinfo=timezone.utc)
    built = mm.build_metrics([tm("A"), tm("B", stale=True, source="sample data", as_of=older)], ["note"], "6mo")
    assert built.uses_stale_or_sample and built.as_of == older
    assert built.warnings == ["note", "These prices are not live (stale or sample data): B (sample data)."]
    clean = mm.build_metrics([tm("A", as_of=None)], [], "6mo")
    assert not clean.uses_stale_or_sample and clean.as_of is None and clean.warnings == []


def test_comparison_rows_keep_input_order_and_each_currency_without_totals_or_ranking():
    built = mm.build_metrics([tm("B.NS", "INR"), tm("A", "USD", trend="Sideways")], [], "6mo")
    rows = mm.comparison_rows(built)
    assert [r["ticker"] for r in rows] == ["B.NS", "A"] and [r["currency"] for r in rows] == ["INR", "USD"]
    assert set(rows[0]) == {"ticker", "currency", "period_return_pct", "volatility_pct", "max_drawdown_pct", "trend"}
    assert "rank" not in str(rows).lower() and "best" not in str(rows).lower()


def test_history_points_are_the_selected_period_only():
    points = mm.history_points("AAA", series(range(1, 41)), 30)
    assert len(points) == 31 and points[0] == {"ticker": "AAA", "date": "2025-01-10", "close": 10.0}


# ---- what the LLM sees ----

def test_summary_input_formats_every_number_and_puts_label_numbers_in_the_values():
    closes = [50] * 300 + [100] * 60
    found, _ = metrics_for(closes, period="1mo")                     # the flat last month
    sent = mm.summary_input(mm.build_metrics([found], ["w"], "1mo"), CFG)
    assert sent["analysis_period"] == "1 month" and sent["warnings"] == ["w"]
    t = sent["tickers"][0]
    assert t["latest_price"] == "100.00" and t["day_change"] == "+0.00 (+0.0%)"
    assert t["moving_averages"] == [
        "20-day average 100.00; the price is 0.0% above it", "50-day average 100.00; the price is 0.0% above it"]
    assert t["range_52_week"] == (
        "52-week high 100.00 (the price is 0.0% below it); 52-week low 50.00 (the price is 100.0% above it)")
    assert t["max_drawdown_over_period"] == "no fall from an earlier high during the period"
    assert t["annualised_volatility"] == "0.0%" and t["trend"] == "Sideways"
    assert t["price_is_stale_or_sample"] is False


def test_the_price_text_names_the_price_the_currency_and_where_it_came_from():
    live = tm(ticker="AAPL", currency="USD").model_copy(update={"latest_price": 270.12, "price_source": "live"})
    assert mm.price_text(live, CFG) == "AAPL last price USD 270.12 (live)"
    when = datetime(2026, 10, 30, 9, 15)
    stale = live.model_copy(update={"price_source": "stale cache", "as_of": when})
    assert mm.price_text(stale, CFG) == "AAPL last price USD 270.12 (stale cache, as of 2026-10-30 09:15 UTC)"
    ist = timezone(timedelta(hours=5, minutes=30))
    aware = stale.model_copy(update={"as_of": datetime(2026, 10, 30, 14, 45, tzinfo=ist)})        # 14:45 IST is 09:15 UTC
    assert mm.price_text(aware, CFG) == "AAPL last price USD 270.12 (stale cache, as of 2026-10-30 09:15 UTC)"
    assert mm.price_text(stale.model_copy(update={"as_of": None}), CFG) == "AAPL last price USD 270.12 (stale cache)"
    assert mm.price_text(live.model_copy(update={"price_source": ""}), CFG) == "AAPL last price USD 270.12 (live)"


def test_the_summary_input_carries_the_price_text_and_its_numbers_pass_the_number_guard():
    from src.core.number_guard import collect_numbers, numbers_in

    found = tm(ticker="AAPL", source="cache", as_of=datetime(2026, 10, 30, 9, 15)).model_copy(update={"latest_price": 1270.5})
    sent = mm.summary_input(MarketMetrics(period="6mo", tickers=[found], warnings=[]), CFG)
    text = sent["tickers"][0]["latest_price_text"]
    assert text == "AAPL last price USD 1,270.50 (cache, as of 2026-10-30 09:15 UTC)"
    assert numbers_in(text) - collect_numbers(sent) == set()


def test_summary_input_describes_a_fall_and_marks_missing_values_not_available():
    full = tm()
    sparse = TickerMetrics(
        ticker="B", currency="INR", period="6mo", latest_price=1234.5, day_change=None, day_change_pct=None,
        period_return_pct=None, averages=[AverageMetric(window=20, value=1300.0, price_vs_pct=-5.0)],
        volatility_pct=None, max_drawdown_pct=None, range_52w=None, trend=None, price_stale=True,
    )
    sent = mm.summary_input(MarketMetrics(period="6mo", tickers=[full, sparse], warnings=[]), CFG)
    assert sent["tickers"][0]["max_drawdown_over_period"] == "4.0% fall from a peak"
    b = sent["tickers"][1]
    assert b["latest_price"] == "1,234.50" and b["currency"] == "INR"
    assert b["moving_averages"] == ["20-day average 1,300.00; the price is 5.0% below it"]
    assert [b[k] for k in ("day_change", "return_over_period", "annualised_volatility",
                           "max_drawdown_over_period", "range_52_week", "trend")] == ["not available"] * 6
    assert b["price_is_stale_or_sample"] is True
    assert mm.summary_input(MarketMetrics(period="1y", tickers=[sparse.model_copy(update={"averages": []})],
                                          warnings=[]), CFG)["tickers"][0]["moving_averages"] == ["not available"]


# ---- history shorter than the requested period ----

def test_span_days_and_short_history_rule():
    assert mm.span_days([]) == 0 and mm.span_days(series([1, 2, 3])) == 2
    # 182-day period, 15% tolerance: spans under 154.7 days are short
    assert mm.is_short_history(154, 182, 15) is True
    assert mm.is_short_history(155, 182, 15) is False
    assert mm.is_short_history(0, 182, 0) is True and mm.is_short_history(182, 182, 0) is False


def test_short_history_is_flagged_with_the_actual_span():
    found, warnings = metrics_for([100 + i for i in range(30)])             # 29 days, 6mo asked
    assert found.period_is_short is True and found.span_days == 29
    assert warnings[0] == ("AAA: only 29 days of data for the requested 6mo; "
                           "return, volatility and drawdown cover that shorter span.")
    assert found.period_return_pct == pytest.approx((129 / 100 - 1) * 100)   # computed over those 29 days


def test_full_history_is_not_flagged():
    found, warnings = metrics_for([100 + i * 0.1 for i in range(300)], period="6mo")
    assert found.period_is_short is False and found.span_days == 182
    assert not any("days of data for the requested" in w for w in warnings)


def test_summary_input_tells_the_model_the_shorter_span_only_when_short():
    short, _ = metrics_for([100 + i for i in range(30)])
    long_, _ = metrics_for([100 + i * 0.1 for i in range(300)])
    short_input = mm.summary_input(mm.build_metrics([short], [], "6mo"), CFG)["tickers"][0]
    long_input = mm.summary_input(mm.build_metrics([long_], [], "6mo"), CFG)["tickers"][0]
    assert short_input["period_covered"] == "only 29 days of data (shorter than the requested period)"
    assert "period_covered" not in long_input


# ---- worst source ----

@pytest.mark.parametrize("sources, expected", [
    ([], ""),
    (["live"], "live"),
    (["live", "cache"], "cache"),
    (["cache", "stale cache", "live"], "stale cache"),
    (["stale cache", "sample data", "live"], "sample data"),
    (["live", "something new"], "something new"),          # an unknown label is treated as the worst
])
def test_worst_source_follows_the_freshness_order(sources, expected):
    assert mm.worst_source(sources) == expected
