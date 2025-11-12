"""Utility helpers for computing optimization objectives and constraints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

ObjectiveValue = Union[float, Tuple[float, ...]]

# Mapping between generic metric names and keys present in Backtest metrics
METRIC_ALIASES = {
    "sharpe": ("sharpe_ratio",),
    "sortino": ("sortino_ratio",),
    "cagr": ("cagr", "annualized_return"),
    "calmar": ("calmar_ratio",),
    "max_drawdown": ("max_drawdown",),
    "ulcer": ("ulcer_index",),
    "pnl": ("pnl",),
    "pnl_pct": ("pnl_pct",),
    "final_value": ("final_value",),
    "total_return": ("total_return",),
    "annualized_return": ("cagr",),
}


def _to_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
        if result != result:  # NaN check without math import
            return None
        return result
    except Exception:
        return None


def _get_metric(metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
    name = str(metric_name).lower()
    keys = METRIC_ALIASES.get(name, (metric_name,))
    for key in keys:
        if key in metrics:
            value = _to_float(metrics[key])
            if value is not None:
                return value
    return None


def _evaluate_weighted_sum(metrics: Dict[str, Any], weights: Dict[str, Any]) -> Optional[float]:
    score = 0.0
    if not weights:
        return None
    for metric_name, weight in weights.items():
        value = _get_metric(metrics, metric_name)
        if value is None:
            return None
        score += float(weight) * value
    return score


def _evaluate_single(metrics: Dict[str, Any], config: Dict[str, Any]) -> Optional[float]:
    aggregation = str(config.get("aggregation", "metric")).lower()
    if aggregation == "weighted_sum":
        weights = config.get("weights") or {}
        return _evaluate_weighted_sum(metrics, weights)

    metric_name = config.get("metric", "sharpe")
    return _get_metric(metrics, metric_name)


def _evaluate_multi(metrics: Dict[str, Any], config: Dict[str, Any]) -> Optional[Tuple[float, ...]]:
    targets = config.get("targets") or []
    values: List[float] = []
    for target in targets:
        metric_name = target.get("name")
        value = _get_metric(metrics, metric_name)
        if value is None:
            return None
        values.append(value)
    if not values:
        return None
    return tuple(values)


def evaluate_objective(metrics: Dict[str, Any], config: Dict[str, Any]) -> Optional[ObjectiveValue]:
    """
    Compute the objective value (single float or tuple of floats) based on user config.
    """
    if not config:
        return None

    mode = str(config.get("mode", "single")).lower()
    if mode == "multi":
        return _evaluate_multi(metrics, config)
    return _evaluate_single(metrics, config)


def study_directions(config: Dict[str, Any]) -> Union[str, List[str]]:
    """
    Return Optuna study direction(s) based on the objective config.
    """
    mode = str(config.get("mode", "single")).lower()
    if mode != "multi":
        return str(config.get("direction", "maximize")).lower()

    targets = config.get("targets") or []
    if not targets:
        raise ValueError("objective.targets must be set when mode='multi'.")

    directions: List[str] = []
    for target in targets:
        direction = str(target.get("direction", "maximize")).lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"Invalid objective direction '{direction}'. Use 'maximize' or 'minimize'."
            )
        directions.append(direction)
    return directions


def build_constraints_func(constraints_cfg: Optional[Dict[str, Any]]):
    """
    Build an Optuna constraints function based on configuration.
    Returns None if no constraints are specified.
    """

    if not constraints_cfg:
        return None

    def constraints(trial) -> Sequence[float]:
        penalties: List[float] = []
        attrs = trial.user_attrs or {}

        min_trades = constraints_cfg.get("min_trades")
        if min_trades is not None:
            total_trades = attrs.get("total_trades", 0)
            penalties.append(float(min_trades) - float(total_trades or 0))

        max_drawdown = constraints_cfg.get("max_drawdown")
        if max_drawdown is not None:
            dd_source = attrs.get("portfolio_metrics", {})
            drawdown_value = dd_source.get("max_drawdown", attrs.get("max_drawdown"))
            dd = _to_float(drawdown_value)
            if dd is None:
                dd = float("inf")
            penalties.append(dd - float(max_drawdown))

        fast_slow_gap = constraints_cfg.get("fast_slow_gap")
        if fast_slow_gap is not None:
            params = attrs.get("strategy_params", {})
            fast = _to_float(
                params.get("fast_period")
                or params.get("ema_fast")
                or params.get("fast")
            )
            slow = _to_float(
                params.get("slow_period")
                or params.get("ema_slow")
                or params.get("slow")
            )
            gap = _to_float(fast_slow_gap)
            if fast is not None and slow is not None and gap is not None:
                penalties.append((fast + gap) - slow)

        return penalties

    return constraints


__all__ = [
    "ObjectiveValue",
    "evaluate_objective",
    "study_directions",
    "build_constraints_func",
]
