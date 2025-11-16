import pandas as pd
import numpy as np
import pytest
from backtesting.analyzers.drawdown import analyze

def test_analyze_empty_equity():
    """
    Test with an empty equity series.
    """
    equity = pd.Series([], dtype=float)
    metrics, underwater = analyze(equity)
    
    assert metrics["max_drawdown"] == 0.0
    assert metrics["max_drawdown_start"] is None
    assert metrics["max_drawdown_trough"] is None
    assert metrics["max_drawdown_recovery"] is None
    assert metrics["max_drawdown_duration"] == 0
    assert metrics["recovery_bars"] is None
    assert metrics["ulcer_index"] == 0.0
    assert underwater.empty

def test_analyze_none_equity():
    """
    Test with None as equity series.
    """
    metrics, underwater = analyze(None)
    
    assert metrics["max_drawdown"] == 0.0
    assert metrics["max_drawdown_start"] is None
    assert metrics["max_drawdown_trough"] is None
    assert metrics["max_drawdown_recovery"] is None
    assert metrics["max_drawdown_duration"] == 0
    assert metrics["recovery_bars"] is None
    assert metrics["ulcer_index"] == 0.0
    assert underwater.empty

def test_analyze_no_drawdown():
    """
    Test with an equity series that is monotonically increasing.
    """
    equity = pd.Series([100, 110, 120, 130, 140])
    metrics, underwater = analyze(equity)
    
    assert metrics["max_drawdown"] == 0.0
    assert all(underwater >= 0)

def test_simple_drawdown_and_recovery():
    """
    Test a simple case with one drawdown and full recovery.
    """
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    equity = pd.Series([100, 120, 100, 80, 120], index=dates)
    metrics, underwater = analyze(equity)
    
    assert pytest.approx(metrics["max_drawdown"]) == 1/3 # (120 - 80) / 120
    assert metrics["max_drawdown_start"] == dates[1]
    assert metrics["max_drawdown_trough"] == dates[3]
    assert metrics["max_drawdown_recovery"] == dates[4]
    assert metrics["max_drawdown_duration"] == 2 # from index 1 to 3
    assert metrics["recovery_bars"] == 1 # from index 3 to 4

def test_drawdown_no_recovery():
    """
    Test a drawdown that does not recover by the end of the series.
    """
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    equity = pd.Series([100, 120, 100, 80], index=dates)
    metrics, underwater = analyze(equity)
    
    assert pytest.approx(metrics["max_drawdown"]) == 1/3
    assert metrics["max_drawdown_start"] == dates[1]
    assert metrics["max_drawdown_trough"] == dates[3]
    assert metrics["max_drawdown_recovery"] is None
    assert metrics["max_drawdown_duration"] == 2
    assert metrics["recovery_bars"] is None

def test_multiple_drawdowns():
    """
    Test with multiple drawdowns to ensure the max is identified.
    """
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07'])
    equity = pd.Series([100, 110, 105, 120, 90, 100, 125], index=dates)
    # First DD: (110-105)/110 = 4.5%
    # Second DD: (120-90)/120 = 25%
    metrics, underwater = analyze(equity)

    assert pytest.approx(metrics["max_drawdown"]) == 0.25
    assert metrics["max_drawdown_start"] == dates[3]
    assert metrics["max_drawdown_trough"] == dates[4]
    assert metrics["max_drawdown_recovery"] == dates[6]
    assert metrics["max_drawdown_duration"] == 1
    assert metrics["recovery_bars"] == 2

def test_equity_with_nans():
    """
    Test that NaNs are handled correctly.
    """
    equity = pd.Series([100, 110, np.nan, 120, 100])
    metrics, underwater = analyze(equity)
    
    # The NaN is dropped, so the series is [100, 110, 120, 100]
    # Peak is 120, trough is 100. DD = (120-100)/120 = 1/6
    assert pytest.approx(metrics["max_drawdown"]) == 1/6

def test_constant_equity():
    """
    Test with a flat equity curve.
    """
    equity = pd.Series([100, 100, 100, 100])
    metrics, underwater = analyze(equity)
    
    assert metrics["max_drawdown"] == 0.0
    assert all(underwater == 0)

def test_ulcer_index_calculation():
    """
    Test the Ulcer Index calculation with a known example.
    """
    equity = pd.Series([100, 90, 80, 90, 100, 95, 90])
    # Underwater values: 0, -0.1, -0.2, -0.1, 0, -0.05, -0.1
    # Negative underwater values: -0.1, -0.2, -0.1, -0.05, -0.1
    # Squared: 0.01, 0.04, 0.01, 0.0025, 0.01
    # Mean of squares: (0.01 + 0.04 + 0.01 + 0.0025 + 0.01) / 5 = 0.0725 / 5 = 0.0145
    # Ulcer Index: sqrt(0.0145) approx 0.1204
    
    # The implementation calculates mean over all values, not just negative ones.
    # Let's re-calculate based on the implementation logic.
    # dd_sq = underwater[underwater < 0].pow(2)
    # dd_sq is [0.01, 0.04, 0.01, 0.0025, 0.01]
    # mean is 0.0145
    # sqrt is approx 0.1204
    
    metrics, _ = analyze(equity)
    
    # Let's trace the code's logic
    # underwater: [0, -0.1, -0.2, -0.1, 0, -0.05, -0.1]
    # dd_sq (before mean): [0.01, 0.04, 0.01, 0.0025, 0.01]
    # mean of dd_sq: 0.0725 / 5 = 0.0145
    # sqrt(0.0145) = 0.1204159...
    
    # The code does dd_sq.mean(). The series has 5 elements.
    # The original series has 7. Let's see what pandas does.
    # It should divide by 5.
    
    expected_ulcer_index = np.sqrt(((0.1**2) + (0.2**2) + (0.1**2) + (0.05**2) + (0.1**2)) / 5)
    
    # The code is `dd_sq.mean()`. `dd_sq` is a filtered series.
    # So it should be divided by the number of elements in `dd_sq`.
    
    assert pytest.approx(metrics["ulcer_index"], 0.001) == np.sqrt(pd.Series([0.01, 0.04, 0.01, 0.0025, 0.01]).mean())

