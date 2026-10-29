import pytest

from src.core import number_guard as ng


def test_number_helpers():
    assert ng.numbers_in("1,234.50 and 42% of 7") == {1234.5, 42.0, 7.0}
    assert ng.numbers_in("no digits") == set()
    nested = {"a": ["12.5%", {"b": 3}], "n9": None, "flag": True, "empty": {}, "none": []}
    assert ng.collect_numbers(nested) == {12.5, 3.0}                 # keys, None and bools ignored
    assert ng.collect_numbers({}) == set() and ng.collect_numbers([]) == set()


@pytest.mark.parametrize("text, expected", [
    ("1,234,567", {1234567.0}), ("12,34,567", {1234567.0}), ("1,25,000", {125000.0}),
    ("10,00,000.50", {1000000.5}), ("12,34,56,789", {123456789.0}), ("1,234.50", {1234.5}),
    ("1.25 lakh", {1.25}), ("2 crore", {2.0}), ("INR 1.25lakh and 3 crore", {1.25, 3.0}),
    ("INR 1,25,000 (1.25 lakh) per year", {125000.0, 1.25}), ("12.5% and 20%", {12.5, 20.0}),
])
def test_numbers_in_reads_western_and_indian_grouping_and_lakh_crore_as_written(text, expected):
    assert ng.numbers_in(text) == expected
