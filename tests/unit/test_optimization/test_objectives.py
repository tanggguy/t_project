"""Tests unitaires pour optimization.objectives."""

from __future__ import annotations

from typing import Dict, Any

import pytest

from optimization import objectives


# --- Tests for _to_float ---
def test_to_float_valid():
    assert objectives._to_float(1.5) == 1.5
    assert objectives._to_float("3.14") == 3.14
    assert objectives._to_float(10) == 10.0


def test_to_float_invalid():
    assert objectives._to_float(None) is None
    assert objectives._to_float("invalid") is None
    assert objectives._to_float(float("nan")) is None


# --- Tests for _get_metric ---
def test_get_metric_direct_match():
    metrics = {"sharpe_ratio": 1.5}
    assert objectives._get_metric(metrics, "sharpe_ratio") == 1.5


def test_get_metric_alias():
    metrics = {"sharpe_ratio": 1.5}
    # "sharpe" is an alias for "sharpe_ratio"
    assert objectives._get_metric(metrics, "sharpe") == 1.5


def test_get_metric_case_insensitive():
    metrics = {"sharpe_ratio": 1.5}
    assert objectives._get_metric(metrics, "SHARPE") == 1.5


def test_get_metric_missing():
    metrics = {"sharpe_ratio": 1.5}
    assert objectives._get_metric(metrics, "sortino") is None


def test_get_metric_none_value():
    metrics = {"sharpe_ratio": None}
    assert objectives._get_metric(metrics, "sharpe") is None


# --- Tests for _evaluate_weighted_sum ---
def test_evaluate_weighted_sum_valid():
    metrics = {"sharpe_ratio": 1.0, "max_drawdown": 0.2}
    weights = {"sharpe": 1.0, "max_drawdown": -1.5}
    # 1.0 * 1.0 + 0.2 * -1.5 = 1.0 - 0.3 = 0.7
    assert objectives._evaluate_weighted_sum(metrics, weights) == pytest.approx(0.7)


def test_evaluate_weighted_sum_missing_metric():
    metrics = {"sharpe_ratio": 1.0}
    weights = {"sharpe": 1.0, "max_drawdown": -1.5}
    assert objectives._evaluate_weighted_sum(metrics, weights) is None


def test_evaluate_weighted_sum_no_weights():
    metrics = {"sharpe_ratio": 1.0}
    assert objectives._evaluate_weighted_sum(metrics, {}) is None


# --- Tests for _evaluate_single ---
def test_evaluate_single_default():
    metrics = {"sharpe_ratio": 1.2}
    config = {}  # Defaults to metric="sharpe"
    assert objectives._evaluate_single(metrics, config) == 1.2


def test_evaluate_single_specific_metric():
    metrics = {"sortino_ratio": 2.0}
    config = {"metric": "sortino"}
    assert objectives._evaluate_single(metrics, config) == 2.0


def test_evaluate_single_weighted_sum():
    metrics = {"sharpe_ratio": 1.0, "max_drawdown": 0.2}
    config = {
        "aggregation": "weighted_sum",
        "weights": {"sharpe": 1.0, "max_drawdown": -1.5},
    }
    assert objectives._evaluate_single(metrics, config) == pytest.approx(0.7)


# --- Tests for _evaluate_multi ---
def test_evaluate_multi_valid():
    metrics = {"sharpe_ratio": 1.0, "max_drawdown": 0.2}
    config = {
        "targets": [
            {"name": "sharpe"},
            {"name": "max_drawdown"},
        ]
    }
    assert objectives._evaluate_multi(metrics, config) == (1.0, 0.2)


def test_evaluate_multi_missing_metric():
    metrics = {"sharpe_ratio": 1.0}
    config = {
        "targets": [
            {"name": "sharpe"},
            {"name": "max_drawdown"},
        ]
    }
    assert objectives._evaluate_multi(metrics, config) is None


def test_evaluate_multi_no_targets():
    metrics = {"sharpe_ratio": 1.0}
    config = {"targets": []}
    assert objectives._evaluate_multi(metrics, config) is None


# --- Tests for evaluate_objective ---
def test_evaluate_objective_single_metric():
    metrics = {"sharpe_ratio": 1.25}
    cfg = {"metric": "sharpe"}
    assert objectives.evaluate_objective(metrics, cfg) == pytest.approx(1.25)


def test_evaluate_objective_multi_targets():
    metrics = {"sharpe_ratio": 0.9, "max_drawdown": 0.25, "cagr": 0.12}
    cfg = {
        "mode": "multi",
        "targets": [
            {"name": "cagr", "direction": "maximize"},
            {"name": "max_drawdown", "direction": "minimize"},
        ],
    }
    value = objectives.evaluate_objective(metrics, cfg)
    assert isinstance(value, tuple)
    assert value == pytest.approx((0.12, 0.25))


def test_evaluate_objective_none_config():
    metrics = {"sharpe_ratio": 1.0}
    assert objectives.evaluate_objective(metrics, {}) is None


# --- Tests for study_directions ---
def test_study_directions_single_default():
    assert objectives.study_directions({}) == "maximize"


def test_study_directions_single_minimize():
    assert objectives.study_directions({"direction": "minimize"}) == "minimize"


def test_study_directions_multi():
    cfg = {
        "mode": "multi",
        "targets": [
            {"name": "sharpe", "direction": "maximize"},
            {"name": "max_drawdown", "direction": "minimize"},
        ],
    }
    assert objectives.study_directions(cfg) == ["maximize", "minimize"]


def test_study_directions_multi_missing_targets():
    with pytest.raises(ValueError, match="objective.targets must be set"):
        objectives.study_directions({"mode": "multi"})


def test_study_directions_multi_invalid_direction():
    cfg = {
        "mode": "multi",
        "targets": [
            {"name": "sharpe", "direction": "invalid_direction"},
        ],
    }
    with pytest.raises(ValueError, match="Invalid objective direction"):
        objectives.study_directions(cfg)


# --- Tests for build_constraints_func ---
def test_build_constraints_func_none():
    assert objectives.build_constraints_func(None) is None
    assert objectives.build_constraints_func({}) is None


def test_build_constraints_func_all():
    cfg = {
        "min_trades": 5,
        "max_drawdown": 0.3,
        "fast_slow_gap": 1,
    }

    constraints_func = objectives.build_constraints_func(cfg)
    assert constraints_func is not None

    class _Trial:
        def __init__(self) -> None:
            self.user_attrs: Dict[str, Any] = {
                "total_trades": 3,
                "portfolio_metrics": {"max_drawdown": 0.4},
                "strategy_params": {"ema_fast": 10, "ema_slow": 12},
            }

    penalties = constraints_func(_Trial())
    assert len(penalties) == 3
    assert penalties[0] == pytest.approx(2.0)  # 5 - 3 trades
    assert penalties[1] == pytest.approx(0.1)  # 0.4 - 0.3
    assert penalties[2] == pytest.approx(-1.0)  # (10 + 1) - 12


def test_build_constraints_func_missing_attrs():
    cfg = {
        "min_trades": 5,
        "max_drawdown": 0.3,
    }
    constraints_func = objectives.build_constraints_func(cfg)

    class _Trial:
        def __init__(self) -> None:
            self.user_attrs: Dict[str, Any] = {}

    penalties = constraints_func(_Trial())
    # min_trades: 5 - 0 = 5.0
    # max_drawdown: inf - 0.3 = inf (since max_drawdown missing -> inf)
    assert penalties[0] == 5.0
    assert penalties[1] == float("inf")
