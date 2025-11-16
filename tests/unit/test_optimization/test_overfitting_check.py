import importlib.util
import json
import logging
import sys
import types
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def test_walk_forward_analysis_requires_sufficient_data(checker: OverfittingChecker) -> None:
    with pytest.raises(ValueError):
        checker.walk_forward_analysis(
            start_date="2020-01-01",
            end_date="2020-01-01",
        )


def test_walk_forward_analysis_raises_when_no_folds(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    def _fake_build(
        index: pd.DatetimeIndex,
        *,
        train_years: int,
        test_months: int,
        step_months: int,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        return []

    monkeypatch.setattr(checker, "_build_anchored_folds", _fake_build)

    with pytest.raises(ValueError):
        checker.walk_forward_analysis(
            start_date="2020-01-01",
            end_date="2020-12-31",
        )


def test_out_of_sample_test_requires_windows(checker: OverfittingChecker, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        checker,
        "_prepare_oos_windows",
        lambda _windows, years: [],
    )

    with pytest.raises(ValueError):
        checker.out_of_sample_test(params={"fast_period": 8})


def test_out_of_sample_test_requires_nonempty_results(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    window = (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-30"))

    def _fake_prepare(windows: Any, *, years: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        return [window]

    monkeypatch.setattr(checker, "_prepare_oos_windows", _fake_prepare)
    monkeypatch.setattr(
        checker,
        "_slice_dataset",
        lambda start, end, include_warmup=False: {},
    )
    monkeypatch.setattr(checker, "_is_dataset_empty", lambda dataset: True)

    with pytest.raises(ValueError):
        checker.out_of_sample_test(params={"fast_period": 8})


def test_init_data_frames_rejects_invalid_type(checker: OverfittingChecker) -> None:
    with pytest.raises(TypeError):
        checker._init_data_frames("invalid")  # type: ignore[arg-type]


def test_load_data_frames_requires_ticker(checker: OverfittingChecker, monkeypatch: Any) -> None:
    monkeypatch.setattr(checker, "data_config", {}, raising=False)
    with pytest.raises(ValueError):
        checker._load_data_frames_from_config()


def test_load_data_frames_multiticker_requires_dates(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        checker,
        "data_config",
        {"tickers": ["AAA", "BBB"]},
        raising=False,
    )
    with pytest.raises(ValueError):
        checker._load_data_frames_from_config()


def test_determine_warmup_bars_handles_invalid_override(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        checker,
        "optimization_config",
        {"warmup_bars": "oops"},  # type: ignore[assignment]
        raising=False,
    )
    assert checker._determine_warmup_bars() == 0


def test_determine_warmup_bars_uses_settings_when_needed(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    monkeypatch.setattr(checker, "optimization_config", {}, raising=False)
    monkeypatch.setattr(checker, "overfitting_settings", {"warmup_bars": "7"}, raising=False)
    monkeypatch.setattr(checker, "_infer_warmup_from_space", lambda: 0)

    assert checker._determine_warmup_bars() == 7


def test_checker_requires_param_space(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        OverfittingChecker(
            strategy_class=DummyStrategy,
            param_space={},
            data=sample_df,
            output_dir=tmp_path / "invalid-param-space",
        )


def test_checker_requires_data_or_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        OverfittingChecker(
            strategy_class=DummyStrategy,
            param_space={"fast_period": (5, 12)},
            data=None,
            data_config={},
            output_dir=tmp_path / "missing-data",
        )


def test_walk_forward_analysis_populates_robustness_metrics(
    checker: OverfittingChecker,
    monkeypatch: Any,
) -> None:
    folds = [
        (checker.data.iloc[0:60], checker.data.iloc[60:90]),
        (checker.data.iloc[90:150], checker.data.iloc[150:180]),
    ]

    def _fake_build_anchored_folds(
        index: pd.DatetimeIndex,
        *,
        train_years: int,
        test_months: int,
        step_months: int,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        return folds

    class _DummyStudy:
        def __init__(self) -> None:
            self.best_params = {"fast_period": 8}
            self.best_trial = types.SimpleNamespace(
                user_attrs={"strategy_params": self.best_params}
            )

    sharpe_sequence = {"values": [1.0, 0.5, 1.0, -0.5], "index": 0}

    def _fake_run_backtest(*args: Any, **kwargs: Any) -> Any:
        value = sharpe_sequence["values"][sharpe_sequence["index"]]
        sharpe_sequence["index"] += 1
        return types.SimpleNamespace(metrics={"sharpe_ratio": value})

    monkeypatch.setattr(checker, "_build_anchored_folds", _fake_build_anchored_folds)
    monkeypatch.setattr(checker, "_run_optuna", lambda *_, **__: _DummyStudy())
    monkeypatch.setattr(checker, "_run_backtest", _fake_run_backtest)
    monkeypatch.setattr(checker, "_export_wfa_results", MagicMock())

    summary = checker.walk_forward_analysis(
        start_date="2020-01-01",
        end_date="2021-12-31",
        fold_config={"train_years": 1, "test_months": 1, "step_months": 1},
        param_space={"fast_period": (5, 12)},
        label="wfa-test",
    )

    assert summary["degradation_ratio"] == pytest.approx(0.0)
    assert summary["test_vs_train_gap"] == pytest.approx(-1.0)
    assert summary["frac_test_sharpe_lt_0"] == pytest.approx(0.5)
    assert summary["frac_test_sharpe_lt_alpha_train"] == pytest.approx(0.5)
    assert summary["alpha"] == pytest.approx(0.5)
    assert summary["robustness_label"] == "overfitted"


def test_prepare_oos_windows_returns_explicit_pairs(checker: OverfittingChecker) -> None:
    windows = [("2020-01-01", "2020-03-01")]

    result = checker._prepare_oos_windows(windows=windows, years=2)

    assert result == [(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01"))]


def test_normalize_timestamp_aligns_timezone(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    tz_df = sample_df.copy()
    tz_df.index = tz_df.index.tz_localize("UTC")
    tz_checker = OverfittingChecker(
        strategy_class=DummyStrategy,
        param_space={"fast_period": (5, 12)},
        data=tz_df,
        output_dir=tmp_path / "tz-checker",
    )

    normalized = tz_checker._normalize_timestamp(pd.Timestamp("2020-06-01"))

    assert normalized.tz == tz_df.index.tz
    assert str(normalized.tz) == "UTC"


def test_block_bootstrap_series_rejects_empty(checker: OverfittingChecker) -> None:
    empty = pd.Series(dtype=float)
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        checker._block_bootstrap_series(empty, block_size=4, rng=rng)


def test_block_bootstrap_trades_requires_pnl_column(checker: OverfittingChecker) -> None:
    trades = pd.DataFrame({"foo": [1.0, 2.0]})
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        checker._block_bootstrap_trades(trades, block_size=1, rng=rng, capital=1000.0)


def test_block_bootstrap_trades_normalizes_invalid_capital(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    trades = pd.DataFrame({"net_pnl": [10.0, -5.0]})
    captured: Dict[str, pd.Series] = {}

    def _fake_series(series: pd.Series, *, block_size: int, rng: np.random.Generator) -> pd.Series:
        captured["series"] = series
        return pd.Series([0.0], index=series.index[:1])

    monkeypatch.setattr(checker, "_block_bootstrap_series", _fake_series)
    rng = np.random.default_rng(0)
    result = checker._block_bootstrap_trades(trades, block_size=1, rng=rng, capital=0.0)

    pd.testing.assert_series_equal(
        captured["series"],
        trades["net_pnl"].astype(float),
    )
    assert isinstance(result, pd.Series)


def test_out_of_sample_test_computes_summary(checker: OverfittingChecker, monkeypatch: Any) -> None:
    windows = [
        (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-30")),
        (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-07-31")),
    ]
    monkeypatch.setattr(
        checker,
        "_prepare_oos_windows",
        lambda _windows, years: windows,
    )
    monkeypatch.setattr(
        checker,
        "_slice_dataset",
        lambda start, end, include_warmup=False: {
            "start": start,
            "end": end,
            "warmup": include_warmup,
        },
    )
    monkeypatch.setattr(checker, "_is_dataset_empty", lambda dataset: False)

    metrics_iter = iter(
        [
            {"sharpe_ratio": 0.8, "max_drawdown": 0.1, "total_trades": 12},
            {"sharpe_ratio": -0.2, "max_drawdown": 0.4, "total_trades": 8},
            {"sharpe_ratio": 0.6, "max_drawdown": 0.2, "total_trades": 20},
        ]
    )

    def _fake_run_backtest(*args: Any, **kwargs: Any) -> Any:
        metrics = next(metrics_iter)
        return types.SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(checker, "_run_backtest", _fake_run_backtest)
    monkeypatch.setattr(
        checker,
        "_compute_oos_robustness",
        lambda summary, results: {
            "badge": "robust",
            "metrics": {"frac_oos_sharpe_lt_0": 0.5},
        },
    )

    export_mock = MagicMock()
    monkeypatch.setattr(checker, "_export_oos_results", export_mock)

    summary = checker.out_of_sample_test(
        params={"fast_period": 8},
        windows=None,
        years=1,
        label="oos-test",
    )

    assert summary["oos_sharpe_mean"] == pytest.approx(0.3)
    assert summary["oos_sharpe_std"] == pytest.approx(0.70710678, rel=1e-6)
    assert summary["oos_sharpe_min"] == pytest.approx(-0.2)
    assert summary["oos_sharpe_median"] == pytest.approx(0.3)
    assert summary["train_sharpe_reference"] == pytest.approx(0.6)
    assert summary["oos_degradation_ratio"] == pytest.approx(0.5)
    assert summary["robustness_summary"]["badge"] == "robust"
    export_mock.assert_called_once()


def test_register_report_section_handles_render_errors(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    module = sys.modules["optimization.overfitting_check"]
    monkeypatch.setattr(
        module.overfitting_report,
        "render_overfitting_index",
        lambda *args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    checker._register_report_section(
        name="Demo",
        description="desc",
        relative_path="demo.html",
        badge="robust",
    )

    assert any(section["name"] == "Demo" for section in checker._report_sections)


def test_summarize_simulations_probability_fields(checker: OverfittingChecker) -> None:
    simulations = [
        {"sharpe_ratio": 1.2, "cagr": 0.10, "max_drawdown": 0.25, "prob_negative": 0.0},
        {"sharpe_ratio": -0.4, "cagr": -0.05, "max_drawdown": 0.50, "prob_negative": 0.5},
        {"sharpe_ratio": 0.3, "cagr": 0.02, "max_drawdown": 0.40, "prob_negative": 0.25},
    ]

    summary = checker._summarize_simulations(
        simulations,
        max_dd_threshold=0.35,
    )

    assert summary["p_sharpe_lt_0"] == pytest.approx(1 / 3)
    assert summary["p_cagr_lt_0"] == pytest.approx(1 / 3)
    assert summary["p_max_dd_gt_threshold"] == pytest.approx(2 / 3)
    assert summary["prob_negative"] == pytest.approx((0.0 + 0.5 + 0.25) / 3)


def test_resolve_badge_prioritizes_robustness(checker: OverfittingChecker) -> None:
    assert checker._resolve_badge(robust=True, overfit=True) == "robust"
    assert checker._resolve_badge(robust=False, overfit=True) == "overfitted"
    assert checker._resolve_badge(robust=False, overfit=False) == "borderline"


def test_monte_carlo_simulation_rejects_invalid_source(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    base_result = types.SimpleNamespace(
        metrics={"initial_capital": 1000.0},
        returns=pd.Series([0.01, 0.02]),
        trades=pd.DataFrame({"net_pnl": [10.0]}),
    )
    monkeypatch.setattr(checker, "_run_backtest", lambda *args, **kwargs: base_result)

    with pytest.raises(ValueError):
        checker.monte_carlo_simulation(
            params={"fast_period": 8},
            data=checker.data,
            source="invalid",
            n_simulations=1,
        )


def test_monte_carlo_simulation_requires_returns(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    base_result = types.SimpleNamespace(
        metrics={"initial_capital": 1000.0},
        returns=pd.Series(dtype=float),
        trades=pd.DataFrame({"net_pnl": [10.0]}),
    )
    monkeypatch.setattr(checker, "_run_backtest", lambda *args, **kwargs: base_result)

    with pytest.raises(ValueError):
        checker.monte_carlo_simulation(
            params={"fast_period": 8},
            data=checker.data,
            source="returns",
            n_simulations=1,
        )


def test_monte_carlo_simulation_aggregates_bootstrap_returns(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    base_result = types.SimpleNamespace(
        metrics={"initial_capital": 1000.0},
        returns=pd.Series([0.01, 0.02], index=pd.RangeIndex(2)),
        trades=pd.DataFrame({"net_pnl": [10.0]}),
    )
    monkeypatch.setattr(checker, "_run_backtest", lambda *args, **kwargs: base_result)

    bootstrapped = pd.Series([0.03, -0.01], index=pd.RangeIndex(2))
    monkeypatch.setattr(
        checker,
        "_block_bootstrap_series",
        lambda series, block_size, rng: bootstrapped,
    )

    metrics_iter = iter(
        [
            {"sharpe_ratio": 0.8, "cagr": 0.02, "max_drawdown": 0.10, "final_value": 1.1},
            {"sharpe_ratio": -0.4, "cagr": -0.03, "max_drawdown": 0.50, "final_value": 0.8},
        ]
    )
    monkeypatch.setattr(
        checker,
        "_compute_metrics_from_returns",
        lambda returns: next(metrics_iter),
    )

    def _fake_get_overfitting_value(path: Any, default: Any = None) -> Any:
        if tuple(path) == ("monte_carlo", "max_drawdown"):
            return {"threshold": 0.3}
        return {}

    monkeypatch.setattr(checker, "_get_overfitting_value", _fake_get_overfitting_value)
    monkeypatch.setattr(
        checker,
        "_compute_monte_carlo_robustness",
        lambda summary: {"badge": "borderline", "metrics": {"dummy": 1}},
    )
    export_mock = MagicMock()
    monkeypatch.setattr(checker, "_export_monte_carlo", export_mock)

    result = checker.monte_carlo_simulation(
        params={"fast_period": 8},
        data=checker.data,
        source="returns",
        n_simulations=2,
        block_size=1,
        seed=123,
        label="mc-test",
    )

    summary = result["summary"]
    assert summary["p_sharpe_lt_0"] == pytest.approx(0.5)
    assert summary["p_cagr_lt_0"] == pytest.approx(0.5)
    assert summary["p_max_dd_gt_threshold"] == pytest.approx(0.5)
    assert summary["prob_negative"] == pytest.approx(0.5)
    assert summary["robustness_summary"]["badge"] == "borderline"
    export_mock.assert_called_once()


def test_export_global_summary_returns_none_when_no_inputs(
    checker: OverfittingChecker,
) -> None:
    assert checker.export_global_summary() is None


def test_export_global_summary_writes_payload(
    checker: OverfittingChecker,
) -> None:
    summary_path = checker.export_global_summary(
        wfa={"metric": 1.0, "timestamp": pd.Timestamp("2020-01-02")},
    )

    assert summary_path is not None
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    assert data["wfa"]["metric"] == 1.0
    assert data["wfa"]["timestamp"].startswith("2020-01-02")


def test_generate_param_variations_skips_non_numeric(checker: OverfittingChecker) -> None:
    variations = checker._generate_param_variations(
        {"int_param": 10, "float_param": 2.5, "name": "foo"},
        perturbation=0.1,
        steps=3,
    )

    assert all("__param_name__" in entry for entry in variations)
    assert not any(entry["__param_name__"] == "name" for entry in variations)


def test_slice_dataset_applies_warmup_offset(checker: OverfittingChecker) -> None:
    checker.warmup_bars = 5
    subset = checker._slice_dataset(
        start="2020-06-10",
        end="2020-06-20",
        include_warmup=True,
    )

    assert subset.index.min() == pd.Timestamp("2020-06-10") - pd.DateOffset(days=5)
    assert subset.index.max() == pd.Timestamp("2020-06-20")


def test_normalize_timestamp_aligns_timezones(
    checker: OverfittingChecker, sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    tz_df = sample_df.copy()
    tz_df.index = tz_df.index.tz_localize("UTC")
    tz_checker = OverfittingChecker(
        strategy_class=DummyStrategy,
        param_space={"fast_period": (5, 12)},
        data=tz_df,
        output_dir=tmp_path / "tz-checker",
    )

    aligned = tz_checker._normalize_timestamp("2020-01-05")
    assert aligned.tzinfo == tz_df.index.tz

    aware_input = pd.Timestamp("2020-02-01", tz="Europe/Paris")
    converted = tz_checker._normalize_timestamp(aware_input)
    assert converted.tzinfo == tz_df.index.tz
    assert converted.tz_convert("UTC") == aware_input.tz_convert("UTC")

    naive_target = pd.Timestamp("2020-03-01", tz="UTC")
    naive_result = checker._normalize_timestamp(naive_target)
    assert naive_result.tzinfo is None


def test_is_dataset_empty_handles_mixed_dict(checker: OverfittingChecker) -> None:
    empty = {"AAA": pd.DataFrame(), "BBB": pd.DataFrame()}
    assert checker._is_dataset_empty(empty) is True

    mixed = {"AAA": checker.data.iloc[:1], "BBB": pd.DataFrame()}
    assert checker._is_dataset_empty(mixed) is False


def test_build_anchored_folds_skips_small_windows(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    tiny_df = sample_df.iloc[:20].copy()
    small_checker = OverfittingChecker(
        strategy_class=DummyStrategy,
        param_space={"fast_period": (5, 12)},
        data=tiny_df,
        output_dir=tmp_path / "tiny-folds",
    )
    folds = small_checker._build_anchored_folds(
        tiny_df.index,
        train_years=0,
        test_months=1,
        step_months=1,
    )

    assert folds == []


def test_run_optuna_handles_low_trades_and_success(
    checker: OverfittingChecker, monkeypatch: Any
) -> None:
    checker.min_trades = 5

    metrics_iter = iter(
        [
            {"sharpe_ratio": 0.9, "total_trades": 2},
            {"sharpe_ratio": 1.2, "total_trades": 10},
        ]
    )

    monkeypatch.setattr(
        checker,
        "_run_backtest",
        lambda *args, **kwargs: types.SimpleNamespace(metrics=next(metrics_iter)),
    )
    monkeypatch.setattr(
        checker,
        "_build_params",
        lambda trial, space: {"fast_period": 8},
    )

    class _DummyTrial:
        def __init__(self, number: int) -> None:
            self.number = number
            self.attrs: Dict[str, Any] = {}

        def set_user_attr(self, key: str, value: Any) -> None:
            self.attrs[key] = value

    class _DummyStudy:
        def __init__(self) -> None:
            self.trials: List[Tuple[_DummyTrial, float]] = []
            self.best_params = {"fast_period": 8}
            self.best_trial = _DummyTrial(0)

        def optimize(
            self,
            objective: Any,
            n_trials: int,
            timeout: Any,
            n_jobs: int,
            show_progress_bar: bool,
        ) -> None:
            for number in range(2):
                trial = _DummyTrial(number)
                value = objective(trial)
                self.trials.append((trial, value))
            if self.trials:
                self.best_trial = self.trials[-1][0]

    dummy_study = _DummyStudy()
    monkeypatch.setattr(checker, "_create_study", lambda name: dummy_study)

    study = checker._run_optuna(
        checker.data,
        {"fast_period": (5, 12)},
        study_name="unit-test",
    )

    assert study is dummy_study
    assert study.trials[0][1] == checker.penalty_value
    assert "low_trade_count" in study.trials[0][0].attrs
    assert study.trials[1][1] > 0
    assert study.trials[1][0].attrs["total_trades"] >= checker.min_trades


def test_build_equity_curve_handles_empty_returns(checker: OverfittingChecker) -> None:
    fallback = pd.Timestamp("2022-01-01")
    equity = checker._build_equity_curve(
        pd.Series(dtype=float),
        initial_capital=5000.0,
        fallback_index=fallback,
    )

    assert equity.iloc[0] == pytest.approx(5000.0)
    assert equity.index[0] == fallback


def test_build_equity_curve_compounds_results(checker: OverfittingChecker) -> None:
    returns = pd.Series(
        [0.10, -0.05, 0.03],
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )

    equity = checker._build_equity_curve(returns, initial_capital=1000.0)

    assert equity.iloc[-1] == pytest.approx(1000.0 * 1.10 * 0.95 * 1.03)


def test_prepare_oos_windows_with_custom_list(checker: OverfittingChecker) -> None:
    windows = [
        ("2020-01-01", "2020-01-31"),
        ("2020-02-01", "2020-02-28"),
    ]

    result = checker._prepare_oos_windows(windows=windows, years=1)

    assert len(result) == 2
    assert all(isinstance(start, pd.Timestamp) for start, _ in result)


def test_block_bootstrap_series_respects_block_sampling(checker: OverfittingChecker) -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.RangeIndex(4))

    def manual_bootstrap(seed: int) -> pd.Series:
        rng = np.random.default_rng(seed)
        values = series.to_numpy()
        n_obs = len(values)
        block_size = 2
        n_blocks = int(np.ceil(n_obs / block_size))
        sampled: List[float] = []
        for _ in range(n_blocks):
            start_idx = rng.integers(0, max(1, n_obs - block_size + 1))
            sampled.extend(values[start_idx : start_idx + block_size])
        sampled = sampled[:n_obs]
        return pd.Series(sampled, index=series.index)

    expected = manual_bootstrap(seed=7)
    rng = np.random.default_rng(7)
    result = checker._block_bootstrap_series(series, block_size=2, rng=rng)

    pd.testing.assert_series_equal(result, expected)


def test_block_bootstrap_series_empty_raises(checker: OverfittingChecker) -> None:
    with pytest.raises(ValueError):
        checker._block_bootstrap_series(
            pd.Series(dtype=float),
            block_size=3,
            rng=np.random.default_rng(0),
        )


def test_block_bootstrap_trades_validates_input_columns(checker: OverfittingChecker) -> None:
    trades = pd.DataFrame({"other": [1.0]})

    with pytest.raises(ValueError):
        checker._block_bootstrap_trades(
            trades,
            block_size=2,
            rng=np.random.default_rng(1),
            capital=1000.0,
        )


def test_block_bootstrap_trades_requires_nonempty_table(checker: OverfittingChecker) -> None:
    with pytest.raises(ValueError):
        checker._block_bootstrap_trades(
            pd.DataFrame(columns=["net_pnl"]),
            block_size=2,
            rng=np.random.default_rng(1),
            capital=1000.0,
        )


def test_extract_trades_returns_dataframe(checker: OverfittingChecker) -> None:
    trade_rows = [{"net_pnl": 10.0}, {"net_pnl": -5.0}]
    strat = types.SimpleNamespace(
        analyzers=types.SimpleNamespace(
            tradelist=types.SimpleNamespace(
                get_analysis=lambda: trade_rows,
            )
        )
    )

    df = checker._extract_trades(strat)

    assert list(df["net_pnl"]) == [10.0, -5.0]


def test_extract_returns_sorts_dict_input(checker: OverfittingChecker) -> None:
    returns = {"2020-01-05": 0.02, "2020-01-01": -0.01}
    strat = types.SimpleNamespace(
        analyzers=types.SimpleNamespace(
            timereturns=types.SimpleNamespace(
                get_analysis=lambda: returns,
            )
        )
    )

    series = checker._extract_returns(strat)

    expected_index = sorted(pd.to_datetime(list(returns.keys())))
    assert list(series.index) == expected_index


def test_compute_sharpe_from_returns_handles_constant_series(
    checker: OverfittingChecker,
) -> None:
    series = pd.Series([0.01, 0.01, 0.01], index=pd.RangeIndex(3))
    assert checker._compute_sharpe_from_returns(series) is None


def test_gather_metrics_combines_analyzers(checker: OverfittingChecker) -> None:
    def _analyzer(payload: Dict[str, Any]) -> types.SimpleNamespace:
        return types.SimpleNamespace(get_analysis=lambda payload=payload: payload)

    strat = types.SimpleNamespace(
        analyzers=types.SimpleNamespace(
            trades=_analyzer({"total": {"total": 12}, "won": {"total": 7}, "lost": {"total": 5}}),
            sharpe=_analyzer({"sharperatio": 1.5}),
            drawdown=_analyzer({"max": {"drawdown": 0.2, "len": 15}}),
            returns=_analyzer({"rtot": 0.30, "ravg": 0.12}),
        ),
        broker=types.SimpleNamespace(getvalue=lambda: 12000.0, startingcash=10000.0),
    )

    metrics = checker._gather_metrics(strat)

    assert metrics["total_trades"] == 12
    assert metrics["won_trades"] == 7
    assert metrics["max_drawdown"] == 0.2
    assert metrics["pnl"] == pytest.approx(2000.0)


def test_validate_params_enforces_fast_slow_gap(checker: OverfittingChecker) -> None:
    checker.enforce_fast_slow = True

    assert checker._validate_params({"fast_period": 5, "slow_period": 10}) is True
    assert checker._validate_params({"fast_period": 20, "slow_period": 10}) is False


def test_as_float_handles_invalid_values(checker: OverfittingChecker) -> None:
    assert checker._as_float("not-a-number", default=1.23) == pytest.approx(1.23)
    assert checker._as_float(np.nan, default=0.5) == pytest.approx(0.5)


def test_compute_metrics_from_returns_includes_drawdown(
    checker: OverfittingChecker,
) -> None:
    returns = pd.Series(
        [0.02, -0.01, 0.03, -0.02],
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    metrics = checker._compute_metrics_from_returns(returns)

    assert "sharpe_ratio" in metrics
    assert metrics["max_drawdown"] >= 0
    assert metrics["final_value"] == pytest.approx((1.02 * 0.99 * 1.03 * 0.98))


def test_summarize_simulations_with_dd_threshold(checker: OverfittingChecker) -> None:
    simulations = [
        {"sharpe_ratio": 0.5, "cagr": 0.1, "max_drawdown": 0.3, "prob_negative": 0.0},
        {"sharpe_ratio": -0.1, "cagr": -0.05, "max_drawdown": 0.6, "prob_negative": 1.0},
    ]

    summary = checker._summarize_simulations(simulations, max_dd_threshold=0.4)

    assert summary["p_sharpe_lt_0"] == pytest.approx(0.5)
    assert summary["p_cagr_lt_0"] == pytest.approx(0.5)
    assert summary["p_max_dd_gt_threshold"] == pytest.approx(0.5)


def test_report_meta_formats_dates(checker: OverfittingChecker) -> None:
    checker.run_id = "meta-case"
    meta = checker._report_meta()

    assert meta["run_id"] == "meta-case"
    assert meta["strategy"] == DummyStrategy.__name__
    assert meta["data_start"] <= meta["data_end"]


def test_json_default_handles_various_types(checker: OverfittingChecker, tmp_path: Path) -> None:
    series = pd.Series([1, 2])
    frame = pd.DataFrame({"a": [1], "b": [2]})
    payloads = [
        datetime(2020, 1, 1),
        date(2020, 1, 2),
        tmp_path,
        np.float64(1.23),
        pd.Timestamp("2020-01-03"),
        series,
        frame,
        42,
    ]

    results = [checker._json_default(obj) for obj in payloads]

    assert results[0].startswith("2020-01-01")
    assert results[1].startswith("2020-01-02")
    assert results[2] == str(tmp_path)
    assert isinstance(results[3], float)
    assert results[4].startswith("2020-01-03")
    assert results[5] == [1, 2]
    assert results[6] == frame.to_dict(orient="records")
    assert results[7] == "42"


def test_get_overfitting_value_traverses_nested_dict(checker: OverfittingChecker) -> None:
    checker.overfitting_settings = {
        "wfa": {"alpha": 0.7, "nested": {"value": 3}},
    }

    assert checker._get_overfitting_value(("wfa", "alpha")) == 0.7
    assert checker._get_overfitting_value(("wfa", "nested", "value")) == 3
    assert checker._get_overfitting_value(("missing",), default="fallback") == "fallback"


def test_resolve_badge_prioritizes_flags(checker: OverfittingChecker) -> None:
    assert checker._resolve_badge(robust=True, overfit=False) == "robust"
    assert checker._resolve_badge(robust=False, overfit=True) == "overfitted"
    assert checker._resolve_badge(robust=False, overfit=False) == "borderline"


def test_compute_wfa_robustness_uses_rules(checker: OverfittingChecker) -> None:
    checker.overfitting_settings = {
        "wfa": {
            "alpha": 0.6,
            "degradation_ratio": {"robust_min": 0.7, "overfit_max": 0.5},
            "frac_test_sharpe_lt_alpha_train": {"robust_max": 0.5, "overfit_min": 0.75},
        }
    }
    summary = {"train_sharpe_mean": 1.0, "test_sharpe_mean": 0.9}
    folds = [
        {
            "train_metrics": {"sharpe_ratio": 1.0},
            "test_metrics": {"sharpe_ratio": 0.8},
        },
        {
            "train_metrics": {"sharpe_ratio": 0.8},
            "test_metrics": {"sharpe_ratio": 0.7},
        },
    ]

    result = checker._compute_wfa_robustness(summary, folds)

    assert result["badge"] == "robust"
    assert result["metrics"]["alpha"] == 0.6


def test_compute_oos_robustness_applies_thresholds(checker: OverfittingChecker) -> None:
    checker.overfitting_settings = {
        "oos": {
            "mean_sharpe": {"robust_min": 0.8, "overfit_max": 0.5},
            "frac_sharpe_lt_0": {"robust_max": 0.2, "overfit_min": 0.5},
        }
    }
    summary = {
        "oos_sharpe_mean": 0.3,
        "oos_degradation_ratio": 0.4,
        "train_sharpe_reference": 0.9,
    }
    windows = [
        {"metrics": {"sharpe_ratio": -0.1}},
        {"metrics": {"sharpe_ratio": 0.4}},
    ]

    result = checker._compute_oos_robustness(summary, windows)

    assert result["badge"] == "overfitted"
    assert result["metrics"]["frac_oos_sharpe_lt_0"] == pytest.approx(0.5)


def test_compute_monte_carlo_robustness_includes_dd_metrics(checker: OverfittingChecker) -> None:
    checker.overfitting_settings = {
        "monte_carlo": {
            "p_sharpe_lt_0": {"robust_max": 0.2, "overfit_min": 0.5},
            "p_cagr_lt_0": {"robust_max": 0.2, "overfit_min": 0.5},
            "max_drawdown": {"threshold": 0.4, "robust_max": 0.2, "overfit_min": 0.6},
        }
    }
    summary = {
        "p_sharpe_lt_0": 0.6,
        "p_cagr_lt_0": 0.1,
        "p_max_dd_gt_threshold": 0.7,
        "prob_negative": 0.3,
    }

    result = checker._compute_monte_carlo_robustness(summary)

    assert result["badge"] == "overfitted"
    assert result["metrics"]["max_dd_threshold"] == pytest.approx(0.4)


def test_compute_stability_robustness_applies_rules(checker: OverfittingChecker) -> None:
    checker.overfitting_settings = {
        "stability": {"robust_min": 0.7, "overfit_max": 0.4}
    }
    summary = {"robust_fraction": 0.8}

    result = checker._compute_stability_robustness(summary)

    assert result["badge"] == "robust"
    assert result["metrics"]["robust_fraction"] == pytest.approx(0.8)


def test_build_html_report_wraps_tables(checker: OverfittingChecker) -> None:
    summary_df = pd.DataFrame({"metric": [1.0]})
    details_df = pd.DataFrame({"row": [1], "value": [2.0]})

    html = checker._build_html_report(
        title="Demo",
        summary_table=summary_df,
        details_table=details_df,
    )

    assert "<html" in html.lower()
    assert "Demo" in html
