import numpy as np
import pandas as pd
import pytest

from backtesting.analyzers.performance import (
    _as_series,
    compute,
    compute_cagr,
    compute_calmar,
    compute_sharpe,
    compute_sortino,
    compute_trade_stats,
)


def _sample_trade_dataframe():
    return pd.DataFrame(
        {
            "entry_dt": pd.to_datetime(
                ["2023-01-03", "2023-01-01", "2023-01-02"]
            ),
            "exit_dt": pd.to_datetime(
                ["2023-01-05", "2023-01-02", "2023-01-04"]
            ),
            "net_pnl": [10.0, -5.0, 20.0],
            "pnl": [10.0, -5.0, 20.0],
            "size": ["1", "-2", "2"],
            "duration_days": [1, 2, 3],
        }
    )


def test_as_series_handles_dict_and_datetime_index():
    sample = {"2023-01-02": 0.2, "2023-01-01": 0.1}

    result = _as_series(sample)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert list(result.index) == [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-02"),
    ]
    assert list(result.values) == [0.1, 0.2]


def test_compute_cagr_handles_invalid_input():
    assert compute_cagr(None) == 0.0
    assert compute_cagr(pd.Series([100.0])) == 0.0
    assert compute_cagr(pd.Series([100.0, -10.0])) == 0.0


def test_compute_cagr_calculates_expected_growth():
    equity = pd.Series([100.0, 200.0])
    expected = np.sqrt(2.0) - 1.0

    assert compute_cagr(equity, periods_per_year=1) == pytest.approx(
        expected, rel=1e-9
    )


def test_compute_sharpe_handles_constant_returns_and_positive_case():
    constant_returns = pd.Series([0.01, 0.01, 0.01])
    assert compute_sharpe(constant_returns) == 0.0

    positive_returns = pd.Series([0.01, 0.02])
    assert compute_sharpe(positive_returns, periods_per_year=4) == pytest.approx(
        6.0
    )


def test_compute_sharpe_applies_risk_free_rate_adjustment():
    returns = pd.Series([0.02, 0.01])
    result = compute_sharpe(
        returns, periods_per_year=4, risk_free_rate_annual=0.04
    )
    assert result == pytest.approx(2.0)


def test_compute_sortino_handles_no_downside_and_downside_case():
    rising_returns = pd.Series([0.01, 0.02])
    assert compute_sortino(rising_returns, periods_per_year=4) == 0.0

    mixed_returns = pd.Series([0.01, -0.005])
    assert compute_sortino(mixed_returns, periods_per_year=4) == pytest.approx(
        1.0
    )


def test_compute_sortino_respects_mar_threshold():
    returns = pd.Series([0.01, -0.01])
    result = compute_sortino(
        returns, periods_per_year=4, mar_annual=0.04
    )
    assert result == pytest.approx(-1.0)


def test_compute_calmar_normalizes_by_max_drawdown():
    assert compute_calmar(0.15, -0.05) == pytest.approx(3.0)
    assert compute_calmar(0.15, 0.0) == 0.0


def test_compute_trade_stats_defaults_when_no_pnl_column():
    trades = pd.DataFrame({"entry_dt": pd.to_datetime(["2023-01-01"])})

    result = compute_trade_stats(trades)

    assert result["total_trades"] == 1
    assert result["won_trades"] == 0
    assert result["profit_factor"] == np.inf
    assert result["win_rate"] == 0.0


def test_compute_trade_stats_calculations_with_net_pnl():
    trades = _sample_trade_dataframe()

    result = compute_trade_stats(trades)

    assert result["total_trades"] == 3
    assert result["won_trades"] == 2
    assert result["lost_trades"] == 1
    assert result["win_rate"] == pytest.approx(66.6666, rel=1e-3)
    assert result["profit_factor"] == pytest.approx(6.0)
    assert result["avg_win"] == pytest.approx(15.0)
    assert result["avg_loss"] == pytest.approx(-5.0)
    assert result["best_trade"] == pytest.approx(20.0)
    assert result["worst_trade"] == pytest.approx(-5.0)
    assert result["payoff_ratio"] == pytest.approx(3.0)
    assert result["expectancy"] == pytest.approx(8.3333333333, rel=1e-6)
    assert result["max_consecutive_losses"] == 1
    assert result["avg_trade_size"] == pytest.approx(1.6666666667, rel=1e-6)
    assert result["avg_trade_duration_days"] == pytest.approx(2.0)


def test_compute_trade_stats_uses_pnl_when_net_pnl_missing():
    trades = pd.DataFrame(
        {
            "entry_dt": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "pnl": [5.0, -3.0],
        }
    )

    result = compute_trade_stats(trades)

    assert result["total_trades"] == 2
    assert result["best_trade"] == pytest.approx(5.0)
    assert result["worst_trade"] == pytest.approx(-3.0)
    assert result["profit_factor"] == pytest.approx(5.0 / 3.0)
    assert result["win_rate"] == pytest.approx(50.0)


def test_compute_trade_stats_tracks_consecutive_losses():
    trades = pd.DataFrame(
        {
            "entry_dt": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
            "net_pnl": [10.0, -5.0, -2.0, 20.0],
        }
    )

    result = compute_trade_stats(trades)

    assert result["max_consecutive_losses"] == 2


def test_compute_combines_ratios_and_trade_stats():
    trades = _sample_trade_dataframe()
    returns = {"2023-01-02": -0.005, "2023-01-01": 0.01}

    metrics = compute(
        equity=None,
        returns=returns,
        trades=trades,
    )

    expected_returns = _as_series(returns).astype(float)
    expected_ann_vol = expected_returns.std(ddof=0) * np.sqrt(252)

    assert metrics["sharpe_ratio"] == pytest.approx(
        compute_sharpe(expected_returns), rel=1e-6
    )
    assert metrics["sortino_ratio"] == pytest.approx(
        compute_sortino(expected_returns), rel=1e-6
    )
    assert metrics["ann_vol"] == pytest.approx(expected_ann_vol)
    assert metrics["cagr"] == 0.0

    trade_stats = compute_trade_stats(trades)
    for key in trade_stats:
        assert metrics[key] == trade_stats[key]


def test_compute_returns_defaults_with_empty_series():
    equity = pd.Series([100.0, 110.0])
    metrics = compute(
        equity=equity,
        returns=pd.Series(dtype=float),
        trades=None,
    )

    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["sortino_ratio"] == 0.0
    assert metrics["ann_vol"] == 0.0
    assert metrics["cagr"] == pytest.approx(compute_cagr(equity))
    assert metrics["total_trades"] == 0


def test_compute_handles_missing_equity():
    metrics = compute(
        equity=None,
        returns={"2023-01-01": 0.01},
        trades=None,
    )

    assert metrics["cagr"] == 0.0
