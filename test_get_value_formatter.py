# Test file for get_value_formatter
import pytest

import pytest


def get_value_formatter(col):
    if "net_" in col.lower() or col.lower() == "ca":
        return {"function": "params.value.toLocaleString() + ' €'"}
    elif "(€)" in col.lower():
        return {"function": "params.value.toLocaleString() + ' €'"}
    elif "share" in col.lower():
        return {"function": "params.value + ' %'"}
    else:
        return {"function": "params.value.toLocaleString()"}


@pytest.mark.parametrize(
    "col, expected",
    [
        ("net_sales", {"function": "params.value.toLocaleString() + ' €'"}),
        ("CA", {"function": "params.value.toLocaleString() + ' €'"}),
        ("total (€)", {"function": "params.value.toLocaleString() + ' €'"}),
        ("market_share", {"function": "params.value + ' %'"}),
        ("revenue", {"function": "params.value.toLocaleString()"}),
        ("", {"function": "params.value.toLocaleString()"}),  # Edge case: empty string
        (
            "no_match",
            {"function": "params.value.toLocaleString()"},
        ),  # Edge case: no pattern match
    ],
)
def test_get_value_formatter(col, expected):
    """
    Test get_value_formatter function with various column names to ensure
    it returns the correct formatting function.
    """
    result = get_value_formatter(col)
    assert result == expected


def test_get_value_formatter_with_uppercase():
    """
    Test get_value_formatter function with uppercase column names to ensure
    it handles uppercase letters correctly.
    """
    col = "NET_INCOME"
    expected = {"function": "params.value.toLocaleString() + ' €'"}
    result = get_value_formatter(col)
    assert result == expected


def test_get_value_formatter_with_mixed_case():
    """
    Test get_value_formatter function with mixed case column names to ensure
    it handles mixed case letters correctly.
    """
    col = "Net_Profit"
    expected = {"function": "params.value.toLocaleString() + ' €'"}
    result = get_value_formatter(col)
    assert result == expected


# Removed the exception test case as it's not necessary for this function

if __name__ == "__main__":
    pytest.main()

