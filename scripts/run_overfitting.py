#!/usr/bin/env python3
"""CLI pour lancer les contrôles d'overfitting (WFA, OOS, Monte Carlo, Stabilité).

Ce script exploite OverfittingChecker et une configuration YAML. Il supporte:
- WFA (re-optimisation par fold via Optuna)
- Out-of-sample tests (fenêtres configurables)
- Monte Carlo (returns ou trades)
- Tests de stabilité des hyperparamètres

Usage de base:
  python scripts/run_overfitting.py --config config/overfitting_SimpleMaManaged.yaml

Pour utiliser les meilleurs paramètres issus d'une optimisation:
  python scripts/run_overfitting.py --config config/optimization_SimpleMaManaged.yaml --use-best-params
"""

from __future__ import annotations

# --- 1. Bibliothèques natives ---
import argparse
import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

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
from optimization.overfitting_check import OverfittingChecker
from strategies.base_strategy import BaseStrategy
from utils.logger import setup_logger


logger = setup_logger(
    __name__, log_file="logs/optimization/run_overfitting.log", level=logging.ERROR
)


# ---------------------------------------------------------------------------
# Chargement config, découverte des stratégies
# ---------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {path}")
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def discover_strategies() -> Dict[str, Type[BaseStrategy]]:
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
            if issubclass(obj, BaseStrategy) and obj is not BaseStrategy and name.endswith("Strategy"):
                strategies[name.replace("Strategy", "")] = obj
    return strategies


def resolve_strategy(strategy_cfg: Dict[str, Any], available: Dict[str, Type[BaseStrategy]]) -> Type[BaseStrategy]:
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
        raise ValueError("La configuration stratégie doit définir 'module'+'class_name' ou 'name'.")
    if name not in available:
        available_names = ", ".join(sorted(available.keys()))
        raise ValueError(f"Stratégie '{name}' introuvable. Disponibles: {available_names}")
    return available[name]


# ---------------------------------------------------------------------------
# Construction du Checker
# ---------------------------------------------------------------------------
def build_checker(config: Dict[str, Any]) -> tuple[OverfittingChecker, Dict[str, Any], Dict[str, Any]]:
    # Compat: certaines configs placent tout sous 'optimization'
    optimization_cfg = config.get("optimization", config)
    overfit_cfg = config.get("overfitting") or optimization_cfg.get("overfitting") or {}

    strategy_cfg = optimization_cfg.get("strategy") or {}
    data_cfg = optimization_cfg.get("data") or {}
    broker_cfg = optimization_cfg.get("broker") or {}
    sizing_cfg = optimization_cfg.get("position_sizing") or {}
    wfa_opt_cfg = overfit_cfg.get("optimization") or {}

    # Stratégie et espace de recherche
    available = discover_strategies()
    strategy_class = resolve_strategy(strategy_cfg, available)
    param_space = strategy_cfg.get("param_space") or {}
    if not param_space:
        raise ValueError("'param_space' doit être défini dans la section stratégie pour WFA.")

    checker = OverfittingChecker(
        strategy_class=strategy_class,
        param_space=param_space,
        data_config=data_cfg,
        broker_config=broker_cfg,
        position_sizing_config=sizing_cfg,
        optimization_config=wfa_opt_cfg,
        run_id=(overfit_cfg.get("run_id") or strategy_cfg.get("name") or strategy_class.__name__),
        output_dir=(overfit_cfg.get("output_dir") or "results/overfitting"),
    )

    return checker, optimization_cfg, overfit_cfg


def load_best_params(optimization_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    out_cfg = optimization_cfg.get("output") or {}
    best_path = out_cfg.get("best_params_path")
    if not best_path:
        return None
    p = Path(best_path)
    if not p.exists():
        logger.warning("Fichier best_params introuvable: %s", p)
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
            return payload.get("params")
    except Exception as exc:
        logger.error("Impossible de charger best params (%s): %s", p, exc)
        return None


# ---------------------------------------------------------------------------
# Exécution des checks
# ---------------------------------------------------------------------------
def run_checks(
    checker: OverfittingChecker,
    optimization_cfg: Dict[str, Any],
    overfit_cfg: Dict[str, Any],
    *,
    use_best_params: bool,
    checks: Optional[list[str]] = None,
) -> Optional[Path]:
    checks = [c.strip().lower() for c in (checks or []) if c and c.strip()] or []
    want_all = not checks
    summaries: Dict[str, Any] = {}

    # WFA (re-optim par fold)
    wfa_cfg = overfit_cfg.get("wfa") or {}
    if (want_all and wfa_cfg.get("enabled", True)) or ("wfa" in checks):
        logger.info("[WFA] Démarrage walk-forward analysis…")
        summaries["wfa"] = checker.walk_forward_analysis(
            fold_config=wfa_cfg.get("fold_config") or {},
            label="wfa",
        )

    # Paramètres à utiliser pour OOS/MonteCarlo/Stabilité
    params_for_tests: Optional[Dict[str, Any]] = None
    if use_best_params:
        params_for_tests = load_best_params(optimization_cfg)
        if not params_for_tests:
            logger.warning("Aucun best_params disponible. Les tests basés sur des params seront ignorés.")

    # OOS
    oos_cfg = overfit_cfg.get("oos") or {}
    if (want_all and oos_cfg.get("enabled", True)) or ("oos" in checks):
        if not params_for_tests:
            logger.warning("[OOS] params manquants. Ignoré (utiliser --use-best-params).")
        else:
            logger.info("[OOS] Démarrage out-of-sample tests…")
            summaries["oos"] = checker.out_of_sample_test(
                params_for_tests,
                windows=oos_cfg.get("windows"),
                years=int(oos_cfg.get("years", 1)),
                label="oos",
            )

    # Monte Carlo
    mc_cfg = overfit_cfg.get("monte_carlo") or {}
    if (want_all and mc_cfg.get("enabled", True)) or ("monte" in checks) or ("monte_carlo" in checks):
        if not params_for_tests:
            logger.warning("[Monte Carlo] params manquants. Ignoré (utiliser --use-best-params).")
        else:
            logger.info("[Monte Carlo] Démarrage simulations…")
            mc_result = checker.monte_carlo_simulation(
                params_for_tests,
                source=str(mc_cfg.get("source", "returns")),
                n_simulations=int(mc_cfg.get("n_simulations", 1000)),
                block_size=int(mc_cfg.get("block_size", 5)),
                seed=mc_cfg.get("seed"),
                label="monte_carlo",
            )
            summaries["monte_carlo"] = mc_result.get("summary")

    # Stabilité
    stab_cfg = overfit_cfg.get("stability") or {}
    if (want_all and stab_cfg.get("enabled", True)) or ("stability" in checks):
        if not params_for_tests:
            logger.warning("[Stability] params manquants. Ignoré (utiliser --use-best-params).")
        else:
            logger.info("[Stability] Démarrage tests de stabilité…")
            summaries["stability"] = checker.stability_tests(
                params_for_tests,
                perturbation=float(stab_cfg.get("perturbation", 0.1)),
                steps=int(stab_cfg.get("steps", 3)),
                threshold=float(stab_cfg.get("threshold", 0.95)),
                label="stability",
            )

    summary_path = checker.export_global_summary(
        wfa=summaries.get("wfa"),
        oos=summaries.get("oos"),
        monte_carlo=summaries.get("monte_carlo"),
        stability=summaries.get("stability"),
    )
    if summary_path:
        logger.info("Résumé global exporté: %s", summary_path)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lance les contrôles d'overfitting depuis YAML")
    parser.add_argument("--config", type=str, required=True, help="Chemin du fichier YAML")
    parser.add_argument("--use-best-params", action="store_true", help="Charge les best params YAML (section optimization.output)")
    parser.add_argument(
        "--checks",
        type=str,
        default=None,
        help="Liste de checks à exécuter (ex: 'wfa,oos,monte,stability'). Si absent, utilise les flags du YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    checker, optimization_cfg, overfit_cfg = build_checker(config)
    checks = args.checks.split(",") if args.checks else None
    summary_path = run_checks(
        checker,
        optimization_cfg,
        overfit_cfg,
        use_best_params=bool(args.use_best_params),
        checks=checks,
    )

    print("\n=== OVERFITTING CHECKS TERMINÉS ===")
    print(f"Sorties: {checker.output_root}")
    if summary_path:
        print(f"Résumé global: {summary_path}")


if __name__ == "__main__":
    main()
