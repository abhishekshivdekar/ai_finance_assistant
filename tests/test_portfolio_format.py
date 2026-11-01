"""Display formatting and table cleaning for the Portfolio tab (pure functions, no Streamlit)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.web_app.format import clean_portfolio_rows, format_amount, format_quantity, format_ratio_percent, format_share


@pytest.mark.parametrize("value, currency, expected", [
    (1234567.891, "INR", "INR 12,34,567.89"),
    (12345678.9, "INR", "INR 1,23,45,678.90"),
    (100, "INR", "INR 100.00"),
    (1000, "INR", "INR 1,000.00"),
    (99999.5, "INR", "INR 99,999.50"),
    (100000, "inr", "inr 1,00,000.00"),
    (0, "INR", "INR 0.00"),
    (-50, "INR", "INR -50.00"),
    (-0.001, "INR", "INR 0.00"),
    (-1234567.891, "INR", "INR -12,34,567.89"),
    (1234567.891, "USD", "USD 1,234,567.89"),
    (-1234.5, "USD", "USD -1,234.50"),
    (12.3, "EUR", "EUR 12.30"),
])
def test_amounts_use_lakh_crore_grouping_for_inr_and_thousands_for_everything_else(value, currency, expected):
    assert format_amount(value, currency) == expected


@pytest.mark.parametrize("value, expected", [(10.0, "10"), (2.5, "2.5"), (1234.5, "1,234.5"), (0.12345, "0.1235"), (3, "3"), (1000000, "1,000,000")])
def test_quantities_drop_trailing_zeros(value, expected):
    assert format_quantity(value) == expected


def test_a_blank_table_gives_no_rows_and_blank_rows_are_skipped_even_with_dropdown_defaults():
    assert clean_portfolio_rows([]) == []
    blank = {"ticker": None, "quantity": float("nan"), "avg_cost": None, "expense_ratio_pct": "", "instrument_type": "stock", "asset_class": "equity"}
    assert clean_portfolio_rows([blank, {"ticker": "   ", "quantity": None}, {}]) == []


def test_missing_cells_are_left_out_and_text_is_trimmed():
    row = {"ticker": " Suzlon ", "quantity": 10.0, "avg_cost": float("nan"), "expense_ratio_pct": None, "instrument_type": " etf ", "asset_class": None}
    assert clean_portfolio_rows([row]) == [{"ticker": "Suzlon", "quantity": 10.0, "instrument_type": "etf"}]
    full = {"ticker": "AAPL", "quantity": 2, "avg_cost": 150.5, "expense_ratio_pct": 0.2, "instrument_type": "stock", "asset_class": "equity", "extra": "dropped"}
    assert clean_portfolio_rows([full]) == [{"ticker": "AAPL", "quantity": 2, "avg_cost": 150.5, "expense_ratio_pct": 0.2, "instrument_type": "stock", "asset_class": "equity"}]


def test_a_row_without_a_ticker_but_with_a_number_is_kept_for_the_backend_to_report():
    assert clean_portfolio_rows([{"ticker": None, "quantity": 5.0, "instrument_type": "stock"}]) == [{"quantity": 5.0, "instrument_type": "stock"}]


def test_numbers_stay_numbers_numeric_text_becomes_a_float_and_other_text_is_kept():
    rows = clean_portfolio_rows([{"ticker": "A", "quantity": "1,250.5", "avg_cost": "abc", "expense_ratio_pct": "nan"}, {"ticker": "B", "quantity": True}])
    assert rows == [{"ticker": "A", "quantity": 1250.5, "avg_cost": "abc"}, {"ticker": "B", "quantity": True}]
    assert clean_portfolio_rows([{"ticker": 2330, "quantity": 1}]) == [{"ticker": "2330", "quantity": 1}]


def test_infinity_is_passed_on_for_the_backend_to_reject():
    assert math.isinf(clean_portfolio_rows([{"ticker": "A", "quantity": float("inf")}])[0]["quantity"])


def test_no_pandas_or_numpy_object_leaves_the_cleaner():
    frame = pd.DataFrame({
        "ticker": ["Suzlon", None, "AAPL", ""], "quantity": [10, np.nan, 2.5, np.nan], "avg_cost": [np.nan, np.nan, 150.0, np.nan],
        "expense_ratio_pct": [pd.NA, np.nan, 0.2, np.nan], "instrument_type": ["stock", "stock", "etf", "stock"], "asset_class": ["equity", None, "equity", "equity"],
    })
    rows = clean_portfolio_rows(frame.to_dict("records"))
    assert rows == [{"ticker": "Suzlon", "quantity": 10, "instrument_type": "stock", "asset_class": "equity"},
                    {"ticker": "AAPL", "quantity": 2.5, "avg_cost": 150.0, "expense_ratio_pct": 0.2, "instrument_type": "etf", "asset_class": "equity"}]
    for row in rows:
        for value in row.values():
            assert type(value) in (str, int, float), (value, type(value))
    scalar = clean_portfolio_rows([{"ticker": "A", "quantity": np.int64(3), "avg_cost": np.float64(1.5), "expense_ratio_pct": pd.NaT}])
    assert scalar == [{"ticker": "A", "quantity": 3, "avg_cost": 1.5}] and type(scalar[0]["quantity"]) is int and type(scalar[0]["avg_cost"]) is float


def test_a_cell_whose_item_cannot_be_read_is_treated_as_missing():
    class Broken:
        def item(self):
            raise ValueError("no scalar")

    assert clean_portfolio_rows([{"ticker": "A", "quantity": 1, "avg_cost": Broken()}]) == [{"ticker": "A", "quantity": 1}]


@pytest.mark.parametrize("fraction, expected", [(0.4237, "42.4%"), (1.0, "100.0%"), (0, "0.0%"), (0.00049, "0.0%"), (None, "n/a")])
def test_a_share_is_shown_as_a_percentage(fraction, expected):
    assert format_share(fraction, "n/a") == expected


@pytest.mark.parametrize("percent, expected", [(0.5, "0.50%"), (0, "0.00%"), (1.234, "1.23%"), (None, "n/a")])
def test_an_expense_ratio_has_two_decimals_and_no_sign(percent, expected):
    assert format_ratio_percent(percent, "n/a") == expected
