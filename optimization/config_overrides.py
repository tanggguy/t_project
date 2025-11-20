"""Helpers to apply dashboard overrides on top of a YAML config."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional


def apply_overrides(
    config: Dict[str, Any],
    *,
    data_overrides: Optional[Dict[str, Any]] = None,
    study_overrides: Optional[Dict[str, Any]] = None,
    param_space_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a new config with overrides merged into optimization.* sections.

    This function creates a deep copy of the base configuration to avoid side effects.
    Overrides are applied specifically to:
    - `optimization.data`: Updates dataset parameters (e.g., tickers, dates).
    - `optimization.study`: Updates Optuna study settings (e.g., n_trials, timeout).
    - `optimization.strategy.param_space`: Updates hyperparameter search spaces.

    Args:
        config: Base configuration dictionary (e.g., loaded from YAML).
        data_overrides: Fields to merge into optimization.data.
        study_overrides: Fields to merge into optimization.study.
        param_space_overrides: Overrides per param name inside optimization.strategy.param_space.

    Returns:
        A deep-copied configuration with overrides applied.
    """
    cfg = copy.deepcopy(config)
    opt_cfg = cfg.setdefault("optimization", cfg.get("optimization", {}))

    if data_overrides:
        data_cfg = opt_cfg.setdefault("data", {})
        data_cfg.update(data_overrides)

    if study_overrides:
        study_cfg = opt_cfg.setdefault("study", {})
        study_cfg.update(study_overrides)

    if param_space_overrides:
        strategy_cfg = opt_cfg.setdefault("strategy", {})
        param_space = strategy_cfg.setdefault("param_space", {})
        for name, spec in param_space_overrides.items():
            param_space[name] = spec

    return cfg
