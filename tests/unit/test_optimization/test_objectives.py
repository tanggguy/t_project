"""Tests unitaires pour optimization.objectives."""

from __future__ import annotations

from typing import Dict, Any

import pytest

from optimization import objectives


def test_evaluate_objective_single_metric() -> None:
    metrics = {"sharpe_ratio": 1.25}
    cfg = {"metric": "sharpe"}

    assert objectives.evaluate_objective(metrics, cfg) == pytest.approx(1.25)


def test_evaluate_objective_weighted_sum() -> None:
    metrics = {"sharpe_ratio": 1.0, "max_drawdown": 0.2}
    cfg = {
        "aggregation": "weighted_sum",
        "weights": {"sharpe": 1.0, "max_drawdown": -1.5},
    }

    value = objectives.evaluate_objective(metrics, cfg)
    assert value == pytest.approx(1.0 - 0.3)


def test_evaluate_objective_multi_targets() -> None:
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


def test_study_directions_multi() -> None:
    cfg = {
        "mode": "multi",
        "targets": [
            {"name": "sharpe", "direction": "maximize"},
            {"name": "max_drawdown", "direction": "minimize"},
        ],
    }

    assert objectives.study_directions(cfg) == ["maximize", "minimize"]

    with pytest.raises(ValueError):
        objectives.study_directions({"mode": "multi", "targets": []})


def test_build_constraints_func() -> None:
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
