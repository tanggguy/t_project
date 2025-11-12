import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


def _ensure_optimization_stub() -> None:
    """Provide a light package stub so importing overfitting_check skips heavy deps."""

    if "optimization" in sys.modules:
        return

    package = types.ModuleType("optimization")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "optimization")]
    sys.modules["optimization"] = package

    optuna_optimizer_stub = types.ModuleType("optimization.optuna_optimizer")
    optuna_optimizer_stub.ParameterSpec = Any
    sys.modules["optimization.optuna_optimizer"] = optuna_optimizer_stub


def _ensure_backtrader_stub() -> None:
    """Provide a tiny backtrader stub for modules importing it at import-time."""

    if "backtrader" in sys.modules:
        return

    backtrader_stub = types.ModuleType("backtrader")

    class _Strategy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

    class _Order:
        Submitted = 0
        Accepted = 1
        Completed = 2
        Canceled = 3
        Margin = 4
        Rejected = 5
        Status = {
            Submitted: "Submitted",
            Accepted: "Accepted",
            Completed: "Completed",
            Canceled: "Canceled",
            Margin: "Margin",
            Rejected: "Rejected",
        }

    class _Broker:
        def setcash(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

        def setcommission(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

        def set_slippage_perc(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

        def getvalue(self) -> float:  # pragma: no cover
            return 0.0

    class _Analyzer:
        pass

    class _Cerebro:
        def __init__(self) -> None:
            self.broker = _Broker()

        def addanalyzer(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

    class _Sizer:
        pass

    analyzers = types.SimpleNamespace(
        SharpeRatio_A=object,
        Returns=object,
        DrawDown=object,
        TradeAnalyzer=object,
        TimeReturn=object,
    )
    timeframe = types.SimpleNamespace(Days=1)

    backtrader_stub.Strategy = _Strategy
    backtrader_stub.Order = _Order
    backtrader_stub.Cerebro = _Cerebro
    backtrader_stub.Analyzer = _Analyzer
    backtrader_stub.Sizer = _Sizer
    backtrader_stub.analyzers = analyzers
    backtrader_stub.TimeFrame = timeframe

    sys.modules["backtrader"] = backtrader_stub


def _ensure_coloredlogs_stub() -> None:
    """Patch coloredlogs import used by utils.logger."""

    if "coloredlogs" in sys.modules:
        return

    module = types.ModuleType("coloredlogs")

    def _install(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        pass

    class _ColoredFormatter(logging.Formatter):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.pop("level_styles", None)
            kwargs.pop("field_styles", None)
            super().__init__(*args, **kwargs)

    module.install = _install
    module.ColoredFormatter = _ColoredFormatter
    module.DEFAULT_LEVEL_STYLES = {}
    module.DEFAULT_FIELD_STYLES = {}
    sys.modules["coloredlogs"] = module


def _ensure_pandas_ta_stub() -> None:
    if "pandas_ta" not in sys.modules:
        sys.modules["pandas_ta"] = types.ModuleType("pandas_ta")


def _ensure_yfinance_stub() -> None:
    if "yfinance" in sys.modules:
        return

    yf_module = types.ModuleType("yfinance")

    class _Ticker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

    yf_module.Ticker = _Ticker
    sys.modules["yfinance"] = yf_module


def _ensure_optuna_stub() -> None:
    """Create a lightweight optuna stub so we can import the checker in isolation."""

    if "optuna" in sys.modules:
        return

    optuna_stub = types.ModuleType("optuna")

    class _DummyTrial:  # noqa: D401 - simple placeholder
        """Minimal stand-in for optuna.Trial."""

    class _DummyStudy:
        """Minimal stand-in for optuna.Study."""

    class _DummySampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

    class _DummyPruner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

    samplers_module = types.ModuleType("optuna.samplers")
    samplers_module.TPESampler = _DummySampler

    pruners_module = types.ModuleType("optuna.pruners")
    pruners_module.MedianPruner = _DummyPruner

    def _create_study(*args: Any, **kwargs: Any) -> _DummyStudy:
        return _DummyStudy()

    optuna_stub.Trial = _DummyTrial
    optuna_stub.Study = _DummyStudy
    optuna_stub.samplers = samplers_module
    optuna_stub.pruners = pruners_module
    optuna_stub.create_study = _create_study

    sys.modules["optuna"] = optuna_stub
    sys.modules["optuna.samplers"] = samplers_module
    sys.modules["optuna.pruners"] = pruners_module


def _load_overfitting_module() -> types.ModuleType:
    """Load optimization.overfitting_check without executing package __init__."""

    module_name = "optimization.overfitting_check"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().parents[3] / "optimization" / "overfitting_check.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to load module spec for {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_optimization_stub()
_ensure_backtrader_stub()
_ensure_coloredlogs_stub()
_ensure_pandas_ta_stub()
_ensure_yfinance_stub()
_ensure_optuna_stub()

OverfittingChecker = _load_overfitting_module().OverfittingChecker


class DummyStrategy:
    """Minimal placeholder used to satisfy the checker interface."""

    __name__ = "DummyStrategy"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return daily OHLCV data over ~2 years to cover all helper scenarios."""

    index = pd.date_range("2020-01-01", periods=720, freq="D")
    values = np.linspace(100.0, 200.0, len(index))
    df = pd.DataFrame(
        {
            "open": values,
            "high": values + 1,
            "low": values - 1,
            "close": values + 0.5,
            "volume": np.full(len(index), 10_000.0),
        },
        index=index,
    )
    return df


@pytest.fixture
def checker(sample_df: pd.DataFrame, tmp_path: Path) -> OverfittingChecker:
    """Checker configured with single-ticker in-memory data."""

    return OverfittingChecker(
        strategy_class=DummyStrategy,
        param_space={"fast_period": (5, 12)},
        data=sample_df,
        output_dir=tmp_path / "overfitting-check",
    )


@pytest.fixture
def multi_checker(sample_df: pd.DataFrame, tmp_path: Path) -> OverfittingChecker:
    """Checker configured with two tickers, one much shorter than the other."""

    short_df = sample_df.iloc[:30].copy()
    multi_data = {"AAA": sample_df, "BBB": short_df}
    return OverfittingChecker(
        strategy_class=DummyStrategy,
        param_space={"fast_period": (5, 12)},
        data=multi_data,
        output_dir=tmp_path / "overfitting-multi",
    )


def test_validate_dataframe_rejects_non_datetime_index(checker: OverfittingChecker) -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=[0, 1])

    with pytest.raises(ValueError):
        checker._validate_dataframe(df, label="invalid")


def test_validate_dataframe_sorts_index(checker: OverfittingChecker) -> None:
    unsorted = pd.DataFrame(
        {"close": [2.0, 1.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-01"]),
    )

    validated = checker._validate_dataframe(unsorted, label="sorted")

    assert list(validated.index) == sorted(unsorted.index.tolist())


def test_slice_dataset_omits_empty_tickers(multi_checker: OverfittingChecker) -> None:
    subset = multi_checker._slice_dataset(
        start="2020-03-01",
        end="2020-03-10",
    )

    assert "AAA" in subset
    assert "BBB" not in subset
    assert subset["AAA"].index.min() == pd.Timestamp("2020-03-01")
    assert subset["AAA"].index.max() == pd.Timestamp("2020-03-10")


def test_build_anchored_folds_produces_non_overlapping_windows(
    checker: OverfittingChecker,
) -> None:
    folds = checker._build_anchored_folds(
        checker.data.index,
        train_years=1,
        test_months=3,
        step_months=6,
    )

    assert len(folds) >= 1
    for train_df, test_df in folds:
        assert len(train_df) >= 30
        assert len(test_df) >= 5
        assert train_df.index.max() < test_df.index.min()


def test_prepare_oos_windows_defaults_to_last_year(checker: OverfittingChecker) -> None:
    windows = checker._prepare_oos_windows(windows=None, years=2)

    assert len(windows) == 1
    start, end = windows[0]
    expected_start = end - pd.DateOffset(years=2)

    assert end == checker.data.index.max()
    assert start == expected_start


def test_block_bootstrap_trades_converts_pnl_to_returns(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    trades = pd.DataFrame({"net_pnl": [100.0, -50.0, 25.0]})
    mocked_series = pd.Series([0.01, -0.02, 0.03], index=pd.RangeIndex(3))
    patched = MagicMock(return_value=mocked_series)
    monkeypatch.setattr(checker, "_block_bootstrap_series", patched)

    rng = np.random.default_rng(42)
    result = checker._block_bootstrap_trades(
        trades,
        block_size=2,
        rng=rng,
        capital=2000.0,
    )

    pd.testing.assert_series_equal(
        patched.call_args.args[0],
        trades["net_pnl"].astype(float) / 2000.0,
    )
    assert patched.call_args.kwargs["block_size"] == 2
    assert patched.call_args.kwargs["rng"] is rng
    assert result is mocked_series
