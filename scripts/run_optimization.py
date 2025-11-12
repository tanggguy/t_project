#!/usr/bin/env python3
"""CLI pour lancer une optimisation Optuna basée sur un fichier YAML."""

# --- 1. Bibliothèques natives ---
import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Type
import logging

# --- 2. Bibliothèques tierces ---
import yaml

# --- Configuration du chemin de projet ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
except NameError:
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

# --- 3. Imports locaux ---
from optimization.optuna_optimizer import OptunaOptimizer
from strategies.base_strategy import BaseStrategy
from utils.logger import setup_logger


logger = setup_logger(
    __name__, log_file="logs/optimization/run_optimization.log", level=logging.ERROR
)


def load_config(path: str) -> Dict[str, Any]:
    """Charge le fichier YAML de configuration."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config: Dict[str, Any] = yaml.safe_load(stream) or {}

    return config


def discover_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Découvre les stratégies disponibles dans strategies/implementations."""

    strategies: Dict[str, Type[BaseStrategy]] = {}
    strategies_dir = PROJECT_ROOT / "strategies" / "implementations"

    if not strategies_dir.exists():
        logger.error("Le dossier des stratégies (%s) est introuvable", strategies_dir)
        return strategies

    for py_file in strategies_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = f"strategies.implementations.{py_file.stem}"

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - robustesse découverte
            logger.warning("Impossible d'importer %s: %s", module_name, exc)
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseStrategy)
                and obj is not BaseStrategy
                and name.endswith("Strategy")
            ):
                strategies[name.replace("Strategy", "")] = obj

    return strategies


def resolve_strategy(
    strategy_cfg: Dict[str, Any],
    available_strategies: Dict[str, Type[BaseStrategy]],
) -> Type[BaseStrategy]:
    """Résout la classe de stratégie à partir de la configuration."""

    module_path = strategy_cfg.get("module")
    class_name = strategy_cfg.get("class_name")

    if module_path and class_name:
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"{module_path}:{class_name} n'hérite pas de BaseStrategy")
        return strategy_class

    name = strategy_cfg.get("name")
    if not name:
        raise ValueError(
            "La configuration stratégie doit définir 'module' + 'class_name' ou 'name'."
        )

    if name not in available_strategies:
        available = ", ".join(sorted(available_strategies.keys()))
        raise ValueError(f"Stratégie '{name}' introuvable. Disponibles: {available}")

    return available_strategies[name]


def build_optimizer(config: Dict[str, Any]) -> OptunaOptimizer:
    """Instancie un OptunaOptimizer depuis la configuration YAML."""

    optimization_cfg = config.get("optimization", config)

    strategy_cfg = optimization_cfg.get("strategy") or {}
    data_cfg = optimization_cfg.get("data") or {}
    broker_cfg = optimization_cfg.get("broker") or {}
    sizing_cfg = optimization_cfg.get("position_sizing") or {}
    objective_cfg = optimization_cfg.get("objective") or {}
    study_cfg = optimization_cfg.get("study") or {}
    portfolio_cfg = optimization_cfg.get("portfolio") or {}
    output_cfg = optimization_cfg.get("output") or {}

    available_strategies = discover_strategies()
    strategy_class = resolve_strategy(strategy_cfg, available_strategies)

    strategy_name = strategy_cfg.get("name") or strategy_class.__name__
    param_space = strategy_cfg.get("param_space") or {}
    fixed_params = strategy_cfg.get("fixed_params") or {}

    if not param_space:
        raise ValueError("'param_space' doit être défini dans la section stratégie")

    optimizer = OptunaOptimizer(
        strategy_class=strategy_class,
        strategy_name=strategy_name,
        param_space=param_space,
        data_config=data_cfg,
        fixed_params=fixed_params,
        broker_config=broker_cfg,
        position_sizing_config=sizing_cfg,
        objective_config=objective_cfg,
        study_config=study_cfg,
        portfolio_config=portfolio_cfg,
        output_config=output_cfg,
    )

    return optimizer


def main() -> None:
    """Entrée principale de la CLI d'optimisation."""

    parser = argparse.ArgumentParser(
        description="Lance une optimisation de stratégie avec Optuna"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/optimization.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Nombre de trials à lancer (override config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout en secondes (override config)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Nombre de jobs parallèles Optuna",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Désactive la barre de progression Optuna",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    optimizer = build_optimizer(config)

    logger.info("Configuration chargée depuis %s", args.config)

    study = optimizer.optimize(
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=None if not args.no_progress_bar else False,
    )

    logger.info("Optimisation terminée")

    if getattr(optimizer, "is_multi_objective", False):
        pareto_trials = study.best_trials
        logger.info(
            "Étude multi-objectifs: %s points de Pareto", len(pareto_trials)
        )
        print("\n=== OPTIMISATION TERMINÉE (MULTI-OBJECTIF) ===")
        print(f"Nombre de solutions Pareto : {len(pareto_trials)}")
        for trial in pareto_trials:
            values_str = ", ".join(f"{val:.6f}" for val in trial.values)
            print(
                f"Trial #{trial.number} | valeurs = [{values_str}]"
            )
            for key, value in trial.params.items():
                print(f"  - {key}: {value}")
            print("---")
    else:
        logger.info("Meilleure valeur: %.6f", study.best_value)
        logger.info("Meilleurs paramètres: %s", study.best_params)

        print("\n=== OPTIMISATION TERMINÉE ===")
        print(f"Best value : {study.best_value:.6f}")
        print("Best params:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
