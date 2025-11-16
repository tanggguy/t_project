import numpy as np
import pandas as pd
import pytest

import backtesting.portfolio as portfolio
from backtesting.portfolio import (
    aggregate_weighted_returns,
    compute_portfolio_metrics,
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


def test_normalize_weights_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="tickers valides"):
        normalize_weights(["", None])

    with pytest.raises(ValueError, match="nombre de poids"):
        normalize_weights(["AAPL", "MSFT"], [0.3])

    with pytest.raises(ValueError, match="Poids manquants"):
        normalize_weights(["AAPL", "MSFT"], {"AAPL": 0.3})

    with pytest.raises(ValueError, match="somme des poids"):
        normalize_weights(["AAPL", "MSFT"], [0.5, -0.5])


def test_aggregate_weighted_returns_ignores_bad_series():
    dates = pd.to_datetime(["2024-01-01"])
    returns_map = {
        "AAPL": pd.Series([0.01], index=dates),
        "MSFT": pd.Series([0.02], index=dates),
        "EMPTY": pd.Series(dtype=float),
        "NONE": None,
    }
    weights = normalize_weights(["AAPL", "MSFT", "GOOGL"], [0.4, 0.6, 1.0])

    result = aggregate_weighted_returns(returns_map, weights, alignment="intersection")

    expected = pd.Series([0.4 * 0.01 + 0.6 * 0.02], index=dates)
    pd.testing.assert_series_equal(result, expected, check_freq=False)


def test_aggregate_weighted_returns_returns_empty_when_no_returns():
    weights = normalize_weights(["AAPL"], [1.0])
    result = aggregate_weighted_returns({}, weights)
    assert result.empty


def test_compute_portfolio_metrics_rejects_empty_returns():
    with pytest.raises(ValueError, match="série de rendements"):
        compute_portfolio_metrics(pd.Series(dtype=float), 100.0)


def test_compute_portfolio_metrics_applies_log_returns(monkeypatch):
    raw_returns = pd.Series(
        [0.01, -0.02], index=pd.to_datetime(["2024-01-01", "2024-01-02"])
    )
    expected_equity = (1.0 + raw_returns).cumprod() * 100.0

    def fake_analyze(equity):
        pd.testing.assert_series_equal(equity, expected_equity)
        return (
            {"max_drawdown": 0.1, "ulcer_index": 0.02},
            pd.Series([-0.01, -0.03], index=raw_returns.index),
        )

    def fake_compute(*, equity, returns, trades, periods_per_year, risk_free_rate_annual, mar_annual):
        pd.testing.assert_series_equal(returns, np.log1p(raw_returns))
        assert trades is None
        assert periods_per_year == 252
        assert risk_free_rate_annual == pytest.approx(0.0)
        assert mar_annual == pytest.approx(0.0)
        return {"cagr": 0.4}

    monkeypatch.setattr(portfolio.dd_analyzer, "analyze", fake_analyze)
    monkeypatch.setattr(portfolio.perf_analyzer, "compute", fake_compute)

    metrics, equity, working_returns, underwater = compute_portfolio_metrics(
        raw_returns, 100.0
    )

    pd.testing.assert_series_equal(equity, expected_equity)
    pd.testing.assert_series_equal(working_returns, np.log1p(raw_returns))
    pd.testing.assert_series_equal(
        underwater, pd.Series([-0.01, -0.03], index=raw_returns.index), check_freq=False
    )

    final_value = float(expected_equity.iloc[-1])
    pnl_value = final_value - 100.0

    assert metrics["final_value"] == pytest.approx(final_value)
    assert metrics["pnl"] == pytest.approx(pnl_value)
    assert metrics["pnl_pct"] == pytest.approx((pnl_value / 100.0) * 100)
    assert metrics["max_drawdown"] == pytest.approx(0.1)
    assert metrics["ulcer_index"] == pytest.approx(0.02)
    assert metrics["calmar_ratio"] == pytest.approx(0.4 / 0.1)
    assert metrics["expectancy"] == pytest.approx(0.0)


def test_compute_portfolio_metrics_honors_simple_returns_flag(monkeypatch):
    raw_returns = pd.Series(
        [0.015, 0.005], index=pd.to_datetime(["2024-01-01", "2024-01-02"])
    )
    analytics_settings = {
        "returns": "simple",
        "periods_per_year": 12,
        "risk_free_rate": 0.03,
        "mar": 0.02,
    }

    def fake_analyze(equity):
        return (
            {"max_drawdown": 0.1, "ulcer_index": 0.01},
            pd.Series([-0.005, -0.01], index=raw_returns.index),
        )

    captured = {}

    def fake_compute(*, returns, periods_per_year, risk_free_rate_annual, mar_annual, **kwargs):
        captured["returns"] = returns
        captured["periods_per_year"] = periods_per_year
        captured["risk_free_rate"] = risk_free_rate_annual
        captured["mar"] = mar_annual
        return {"cagr": 0.2}

    monkeypatch.setattr(portfolio.dd_analyzer, "analyze", fake_analyze)
    monkeypatch.setattr(portfolio.perf_analyzer, "compute", fake_compute)

    compute_portfolio_metrics(raw_returns, 50.0, analytics_settings=analytics_settings)

    pd.testing.assert_series_equal(captured["returns"], raw_returns)
    assert captured["periods_per_year"] == 12
    assert captured["risk_free_rate"] == pytest.approx(0.03)
    assert captured["mar"] == pytest.approx(0.02)


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
