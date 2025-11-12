import pandas as pd
import pytest

from backtesting.portfolio import (
    aggregate_weighted_returns,
    normalize_weights,
)


def test_normalize_weights_default_equal():
    weights = normalize_weights(["AAPL", "MSFT"])
    assert pytest.approx(0.5) == weights["AAPL"]
    assert pytest.approx(0.5) == weights["MSFT"]


def test_normalize_weights_dict_keeps_order():
    weights_cfg = {"AAPL": 0.2, "MSFT": 0.8}
    weights = normalize_weights(["AAPL", "MSFT"], weights_cfg)
    assert pytest.approx(0.2) == weights["AAPL"]
    assert pytest.approx(0.8) == weights["MSFT"]


def test_aggregate_weighted_returns_intersection():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    returns_map = {
        "AAPL": pd.Series([0.01, -0.02], index=dates),
        "MSFT": pd.Series([0.02, 0.03], index=dates),
    }
    weights = normalize_weights(["AAPL", "MSFT"], [0.3, 0.7])

    result = aggregate_weighted_returns(returns_map, weights, alignment="intersection")

    expected = pd.Series(
        [
            0.01 * 0.3 + 0.02 * 0.7,
            -0.02 * 0.3 + 0.03 * 0.7,
        ],
        index=dates,
    )
    pd.testing.assert_series_equal(result, expected, check_freq=False)


def test_aggregate_weighted_returns_union_fills_missing_with_zero():
    intersection_dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    msft_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    returns_map = {
        "AAPL": pd.Series([0.01, -0.02], index=intersection_dates),
        "MSFT": pd.Series([0.02, 0.03, 0.01], index=msft_dates),
    }
    weights = normalize_weights(["AAPL", "MSFT"], [0.5, 0.5])

    result = aggregate_weighted_returns(returns_map, weights, alignment="union")

    expected_index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    expected_values = [
        0.5 * 0.01 + 0.5 * 0.02,
        0.5 * -0.02 + 0.5 * 0.03,
        0.5 * 0.0 + 0.5 * 0.01,
    ]
    expected = pd.Series(expected_values, index=expected_index)
    pd.testing.assert_series_equal(result, expected, check_freq=False)
