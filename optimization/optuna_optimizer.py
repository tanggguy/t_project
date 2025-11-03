"""Optuna-based hyperparameter optimization utilities.

Ce module implémente une classe `OptunaOptimizer` capable de lancer des
backtests Backtrader pour différentes stratégies et de maximiser une métrique
de performance (par défaut le Sharpe ratio).
"""

# --- 1. Bibliothèques natives ---
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union

# --- 2. Bibliothèques tierces ---
import optuna
import pandas as pd
import yaml

# --- 3. Imports locaux du projet ---
from backtesting.engine import BacktestEngine
from risk_management.position_sizing import (
    FixedFractionalSizer,
    FixedSizer,
    VolatilityBasedSizer,
)
from strategies.base_strategy import BaseStrategy
from utils.config_loader import get_log_level
from utils.data_manager import DataManager
from utils.logger import setup_logger


ParameterSpec = Union[
    Tuple[Any, ...],
    Iterable[Any],
    Dict[str, Any],
]


class OptunaOptimizer:
    """Encapsule la logique d'optimisation bayésienne avec Optuna.

    Attributes:
        strategy_class: Classe de stratégie Backtrader à optimiser.
        strategy_name: Nom lisible de la stratégie (pour logs/rapports).
        param_space: Espace de recherche Optuna.
        fixed_params: Paramètres fixés pendant l'optimisation.
        data_config: Configuration des données (transmise au DataManager).
        broker_config: Paramètres broker (capital, commission...).
        position_sizing_config: Paramètres du position sizing optionnel.
        objective_config: Configuration de la fonction objectif.
        study_config: Configuration de l'étude Optuna (sampler/pruner/etc.).
        output_config: Configuration des sorties (fichiers, logs).
    """

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        strategy_name: str,
        param_space: Dict[str, ParameterSpec],
        data_config: Dict[str, Any],
        fixed_params: Optional[Dict[str, Any]] = None,
        broker_config: Optional[Dict[str, Any]] = None,
        position_sizing_config: Optional[Dict[str, Any]] = None,
        objective_config: Optional[Dict[str, Any]] = None,
        study_config: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise l'optimiseur Optuna.

        Args:
            strategy_class: Classe de stratégie Backtrader à optimiser.
            strategy_name: Nom lisible de la stratégie.
            param_space: Définition de l'espace de recherche Optuna.
            data_config: Configuration des données à charger via DataManager.
            fixed_params: Paramètres de stratégie figés.
            broker_config: Paramètres broker à appliquer sur Cerebro.
            position_sizing_config: Configuration du position sizing.
            objective_config: Configuration de la fonction objectif.
            study_config: Paramètres de l'étude Optuna (sampler, pruner...).
            output_config: Paramètres de sauvegarde (logs, fichiers résultats).
        """

        self.strategy_class = strategy_class
        self.strategy_name = strategy_name
        self.param_space = param_space or {}
        self.fixed_params = fixed_params or {}
        self.data_config = data_config
        self.broker_config = broker_config or {}
        self.position_sizing_config = position_sizing_config or {}
        self.objective_config = objective_config or {}
        self.study_config = study_config or {}
        self.output_config = output_config or {}

        if not self.param_space:
            raise ValueError("param_space ne peut pas être vide pour une optimisation")

        log_file = self.output_config.get(
            "log_file", "logs/optimization/optuna_optimizer.log"
        )
        log_level = get_log_level("optimization")
        self.logger = setup_logger(__name__, log_file=log_file, level="ERROR")

        self.logger.info(
            "Initialisation de l'OptunaOptimizer pour %s", self.strategy_name
        )

        self.data_manager = DataManager()
        self.data_frame = self._load_data()

        self.metric_name = self.objective_config.get("metric", "sharpe").lower()
        if self.metric_name not in {"sharpe"}:
            raise ValueError("Seule la métrique 'sharpe' est supportée pour le moment.")

        self.penalty_value = float(
            self.objective_config.get("penalize_no_trades", -1.0)
        )
        self.min_trades = int(self.objective_config.get("min_trades", 1))
        self.enforce_fast_slow = bool(
            self.objective_config.get("enforce_fast_slow_gap", True)
        )

        self.study: Optional[optuna.Study] = None

    # ------------------------------------------------------------------
    # Chargement des données et configuration Backtrader
    # ------------------------------------------------------------------
    def _load_data(self) -> pd.DataFrame:
        """Charge les données via le DataManager."""

        ticker = self.data_config.get("ticker")
        if not ticker:
            raise ValueError(
                "'ticker' doit être défini dans la configuration des données"
            )

        start_date = self.data_config.get("start_date")
        end_date = self.data_config.get("end_date")
        interval = self.data_config.get("interval")
        use_cache = self.data_config.get("use_cache", True)

        self.logger.info(
            "Chargement des données pour %s (%s -> %s)", ticker, start_date, end_date
        )

        df = self.data_manager.get_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cache=use_cache,
        )

        if df is None or df.empty:
            raise ValueError(
                f"Impossible de charger des données valides pour {ticker}."
            )

        return df

    def _create_engine(self) -> BacktestEngine:
        """Crée et configure un BacktestEngine prêt pour l'exécution."""

        engine = BacktestEngine()

        initial_capital = float(self.broker_config.get("initial_capital", 10000.0))
        engine.cerebro.broker.setcash(initial_capital)

        if "commission_pct" in self.broker_config:
            engine.cerebro.broker.setcommission(
                commission=float(self.broker_config["commission_pct"])
            )
        elif "commission_fixed" in self.broker_config:
            engine.cerebro.broker.setcommission(
                commission=float(self.broker_config["commission_fixed"])
            )

        if "slippage_pct" in self.broker_config:
            slippage = float(self.broker_config["slippage_pct"])
            if slippage > 0:
                engine.cerebro.broker.set_slippage_perc(perc=slippage)

        engine.add_data(self.data_frame.copy(), name="data0")
        self._configure_position_sizing(engine)

        return engine

    def _configure_position_sizing(self, engine: BacktestEngine) -> None:
        """Applique la configuration de position sizing si elle est activée."""

        cfg = self.position_sizing_config or {}
        if not cfg.get("enabled", False):
            return

        method = cfg.get("method", "fixed")
        self.logger.debug("Position sizing activé (%s)", method)

        try:
            if method == "fixed":
                params = cfg.get("fixed", {})
                engine.add_sizer(
                    FixedSizer,
                    stake=params.get("stake"),
                    pct_size=params.get("pct_size", 1.0),
                )
            elif method == "fixed_fractional":
                params = cfg.get("fixed_fractional", {})
                engine.add_sizer(
                    FixedFractionalSizer,
                    risk_pct=params.get("risk_pct", 0.02),
                    stop_distance=params.get("stop_distance", 0.03),
                )
            elif method == "volatility_based":
                params = cfg.get("volatility_based", {})
                engine.add_sizer(
                    VolatilityBasedSizer,
                    risk_pct=params.get("risk_pct", 0.02),
                    atr_period=params.get("atr_period", 14),
                    atr_multiplier=params.get("atr_multiplier", 2.0),
                )
            else:
                self.logger.warning(
                    "Méthode de position sizing inconnue: %s. Ignorée.", method
                )
        except Exception as exc:  # pragma: no cover - log de sécurité
            self.logger.error(
                "Erreur lors de la configuration du position sizing (%s)", exc
            )

    # ------------------------------------------------------------------
    # Suggestion de paramètres et validation
    # ------------------------------------------------------------------
    def _suggest_param(
        self, trial: optuna.Trial, name: str, spec: ParameterSpec
    ) -> Any:
        """Convertit une définition utilisateur en suggestion Optuna."""

        if isinstance(spec, dict):
            return self._suggest_from_dict(trial, name, spec)

        if isinstance(spec, tuple) or isinstance(spec, list):
            values = list(spec)
            if len(values) == 0:
                raise ValueError(f"param_space[{name}] ne peut pas être vide")

            if len(values) == 1:
                return values[0]

            if len(values) in {2, 3} and all(
                isinstance(v, (int, float)) for v in values[:2]
            ):
                low = values[0]
                high = values[1]
                step = values[2] if len(values) == 3 else None

                if (
                    isinstance(low, int)
                    and isinstance(high, int)
                    and (step is None or isinstance(step, int))
                ):
                    return trial.suggest_int(
                        name,
                        int(low),
                        int(high),
                        step=int(step) if isinstance(step, int) else 1,
                    )

                return trial.suggest_float(
                    name,
                    float(low),
                    float(high),
                    step=float(step) if isinstance(step, (int, float)) else None,
                )

            # Liste quelconque -> catégorielle
            return trial.suggest_categorical(name, list(values))

        # Valeur scalaire -> fixe
        return spec

    def _suggest_from_dict(
        self, trial: optuna.Trial, name: str, spec: Dict[str, Any]
    ) -> Any:
        """Suggestion basée sur un dictionnaire détaillé."""

        param_type = spec.get("type", "float").lower()

        if param_type in {"int", "integer"}:
            low = int(spec["low"])
            high = int(spec["high"])
            step = spec.get("step")
            if step is not None:
                step = int(step)
            return trial.suggest_int(name, low=low, high=high, step=step)

        if param_type in {"float", "uniform"}:
            low = float(spec["low"])
            high = float(spec["high"])
            step = spec.get("step")
            log_scale = bool(spec.get("log", False)) or (
                str(spec.get("scale", "")).lower() == "log"
            )
            return trial.suggest_float(
                name,
                low=low,
                high=high,
                step=float(step) if isinstance(step, (int, float)) else None,
                log=log_scale,
            )

        if param_type in {"categorical", "choice"}:
            choices = spec.get("choices")
            if not choices:
                raise ValueError(
                    f"param_space[{name}] doit définir 'choices' pour un type catégoriel"
                )
            return trial.suggest_categorical(name, list(choices))

        raise ValueError(f"Type de paramètre non supporté: {param_type}")

    def _build_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Construit le dictionnaire de paramètres pour un essai Optuna."""

        params: Dict[str, Any] = {}
        for name, spec in self.param_space.items():
            params[name] = self._suggest_param(trial, name, spec)

        # Les paramètres fixés écrasent ceux suggérés en cas de doublon
        params.update(self.fixed_params)

        return params

    def _validate_params(self, params: Dict[str, Any]) -> bool:
        """Applique les contraintes simples (ex: fast_period < slow_period)."""

        if not self.enforce_fast_slow:
            return True

        if "fast_period" in params and "slow_period" in params:
            if params["fast_period"] >= params["slow_period"]:
                return False

        return True

    # ------------------------------------------------------------------
    # Fonction objectif Optuna
    # ------------------------------------------------------------------
    def objective(self, trial: optuna.Trial) -> float:
        """Évalue un jeu de paramètres et renvoie la métrique à maximiser."""

        params = self._build_trial_params(trial)
        trial.set_user_attr("strategy_params", params)

        if not self._validate_params(params):
            self.logger.debug(
                "Trial %s rejeté (fast_period >= slow_period)", trial.number
            )
            trial.set_user_attr("constraint_violation", True)
            return self.penalty_value

        engine = self._create_engine()
        engine.add_strategy(self.strategy_class, **params)

        try:
            results = engine.run()
        except Exception as exc:  # pragma: no cover - résilience optuna
            self.logger.exception(
                "Erreur lors de l'exécution du backtest (trial %s)", trial.number
            )
            trial.set_user_attr("error", str(exc))
            return self.penalty_value

        if not results:
            trial.set_user_attr("error", "no_results")
            return self.penalty_value

        strat = results[0]
        metrics = self._gather_metrics(strat)

        for key, value in metrics.items():
            if key != "objective":
                trial.set_user_attr(key, value)

        total_trades = metrics.get("total_trades", 0)
        if total_trades < self.min_trades:
            self.logger.debug(
                "Trial %s pénalisé (trades=%s < %s)",
                trial.number,
                total_trades,
                self.min_trades,
            )
            return self.penalty_value

        objective_value = metrics.get("objective")
        if objective_value is None or not math.isfinite(objective_value):
            return self.penalty_value

        trial.report(objective_value, step=0)
        if trial.should_prune():
            self.logger.info("Trial %s interrompu par pruner", trial.number)
            raise optuna.TrialPruned()

        return float(objective_value)

    def _gather_metrics(self, strat: BaseStrategy) -> Dict[str, Any]:
        """Extrait les métriques utiles depuis la stratégie Backtrader."""

        metrics: Dict[str, Any] = {}

        try:
            trades = strat.analyzers.trades.get_analysis()
        except (AttributeError, KeyError):
            trades = {}

        try:
            sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        except (AttributeError, KeyError):
            sharpe_analysis = {}

        try:
            drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        except (AttributeError, KeyError):
            drawdown_analysis = {}

        try:
            returns_analysis = strat.analyzers.returns.get_analysis()
        except (AttributeError, KeyError):
            returns_analysis = {}

        metrics["total_trades"] = trades.get("total", {}).get("total", 0)
        metrics["won_trades"] = trades.get("won", {}).get("total", 0)
        metrics["lost_trades"] = trades.get("lost", {}).get("total", 0)

        sharpe_ratio = sharpe_analysis.get("sharperatio")
        metrics["sharpe_ratio"] = sharpe_ratio

        metrics["max_drawdown"] = drawdown_analysis.get("max", {}).get("drawdown")
        metrics["total_return"] = returns_analysis.get("rtot", 0.0)
        metrics["annualized_return"] = returns_analysis.get("ravg", 0.0)

        final_value = strat.broker.getvalue()
        metrics["final_value"] = final_value

        initial_capital = float(self.broker_config.get("initial_capital", 10000.0))
        metrics["initial_capital"] = initial_capital
        metrics["pnl"] = final_value - initial_capital
        metrics["pnl_pct"] = (
            (final_value / initial_capital) - 1.0 if initial_capital else 0.0
        )

        metrics["objective"] = self._compute_objective(sharpe_ratio)

        return metrics

    def _compute_objective(self, sharpe_ratio: Optional[float]) -> Optional[float]:
        """Calcule la valeur d'objectif selon la métrique choisie."""

        if self.metric_name == "sharpe":
            if sharpe_ratio is None:
                return None
            return float(sharpe_ratio)

        raise ValueError(f"Métrique non supportée: {self.metric_name}")

    # ------------------------------------------------------------------
    # Gestion de l'étude Optuna
    # ------------------------------------------------------------------
    def _build_sampler(self) -> Optional[optuna.samplers.BaseSampler]:
        """Construit le sampler Optuna en fonction de la config."""

        sampler_name = str(self.study_config.get("sampler", "tpe")).lower()
        sampler_kwargs = self.study_config.get("sampler_kwargs", {})

        if sampler_name in {"", "none", "null"}:
            return None

        if sampler_name == "tpe":
            return optuna.samplers.TPESampler(**sampler_kwargs)
        if sampler_name == "random":
            return optuna.samplers.RandomSampler(**sampler_kwargs)
        if sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler(**sampler_kwargs)

        raise ValueError(f"Sampler Optuna inconnu: {sampler_name}")

    def _build_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Construit le pruner Optuna selon la configuration."""

        pruner_name = str(self.study_config.get("pruner", "median")).lower()
        pruner_kwargs = self.study_config.get("pruner_kwargs", {})

        if pruner_name in {"", "none", "null"}:
            return None

        if pruner_name == "median":
            return optuna.pruners.MedianPruner(**pruner_kwargs)
        if pruner_name in {"sha", "successive_halving"}:
            return optuna.pruners.SuccessiveHalvingPruner(**pruner_kwargs)
        if pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner(**pruner_kwargs)

        raise ValueError(f"Pruner Optuna inconnu: {pruner_name}")

    def _create_study(self) -> optuna.Study:
        """Crée (ou recharge) l'étude Optuna."""

        study_name = self.study_config.get(
            "study_name", f"{self.strategy_name.lower()}_optimization"
        )
        direction = self.study_config.get("direction", "maximize")
        storage = self.study_config.get("storage")
        load_if_exists = bool(self.study_config.get("load_if_exists", True))

        sampler = self._build_sampler()
        pruner = self._build_pruner()

        storage_url = None
        if storage and str(storage).lower() not in {"", "none", "null"}:
            storage_url = str(storage)
            if storage_url.startswith("sqlite:///"):
                db_path = Path(storage_url.replace("sqlite:///", ""))
                db_path.parent.mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage_url,
            load_if_exists=load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

        self.logger.info("Étude Optuna prête (%s)", study_name)
        return study

    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        n_jobs: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
    ) -> optuna.Study:
        """Lance l'optimisation et retourne l'étude Optuna."""

        study = self._create_study()

        trials = n_trials or self.study_config.get("n_trials")
        timeout = timeout or self.study_config.get("timeout")
        jobs = n_jobs or self.study_config.get("n_jobs", 1)
        show_bar_cfg = self.study_config.get("show_progress_bar", True)
        show_bar = show_progress_bar
        if show_bar is None:
            show_bar = bool(show_bar_cfg and jobs == 1)

        self.logger.info(
            "Démarrage de l'optimisation (n_trials=%s, timeout=%s, n_jobs=%s)",
            trials,
            timeout,
            jobs,
        )

        study.optimize(
            self.objective,
            n_trials=trials,
            timeout=timeout,
            n_jobs=jobs,
            show_progress_bar=show_bar,
        )

        self.study = study
        self._handle_outputs(study)

        return study

    # ------------------------------------------------------------------
    # Sauvegardes et rapports
    # ------------------------------------------------------------------
    def _handle_outputs(self, study: optuna.Study) -> None:
        """Gère les sorties configurées après l'optimisation."""

        if self.output_config.get("save_study", False):
            self._save_study(study)

        if self.output_config.get("save_trials_csv", False):
            self._export_trials_csv(study)

        if self.output_config.get("dump_best_params", False):
            self._dump_best_params(study)

    def _save_study(self, study: optuna.Study) -> None:
        """Sauvegarde l'étude complète au format pickle (fallback simple)."""

        path_str = self.output_config.get("study_path")
        if not path_str:
            self.logger.warning(
                "Aucun chemin fourni pour save_study, opération ignorée"
            )
            return

        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = study.trials_dataframe(
            attrs=("number", "state", "value", "params", "user_attrs")
        )
        df.to_pickle(path)

        self.logger.info("Étude sauvegardée (DataFrame pickle) dans %s", path)

    def _export_trials_csv(self, study: optuna.Study) -> None:
        """Exporte l'historique des essais au format CSV."""

        path_str = self.output_config.get("trials_csv_path")
        if not path_str:
            self.logger.warning(
                "Aucun chemin fourni pour save_trials_csv, opération ignorée"
            )
            return

        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
        df.to_csv(path, index=False)

        self.logger.info("Historique des trials exporté vers %s", path)

    def _dump_best_params(self, study: optuna.Study) -> None:
        """Sauvegarde les meilleurs paramètres au format YAML."""

        path_str = self.output_config.get("best_params_path")
        if not path_str:
            self.logger.warning(
                "Aucun chemin fourni pour dump_best_params, opération ignorée"
            )
            return

        best_params = {
            "strategy": self.strategy_name,
            "objective": study.best_value,
            "params": study.best_params,
            "trial_number": study.best_trial.number,
        }

        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(best_params, stream, sort_keys=False, allow_unicode=False)

        self.logger.info("Meilleurs paramètres sauvegardés dans %s", path)
