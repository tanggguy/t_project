"""Outils de prévention de l'overfitting pour les stratégies Backtrader.

Ce module fournit une classe `OverfittingChecker` capable d'exécuter :
- une walk-forward analysis (WFA) ancrée avec ré-optimisation Optuna par fold,
- des tests out-of-sample sur des fenêtres définies,
- une analyse de robustesse via simulation Monte Carlo (bootstrap par blocs),
- des tests de stabilité locale des hyperparamètres.

Les sorties sont centralisées dans `results/overfitting/<run_id>/<timestamp>/`
avec export CSV et un mini-rapport HTML par type d'analyse.
"""

from __future__ import annotations

# --- 1. Bibliothèques natives ---
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

# --- 2. Bibliothèques tierces ---
import numpy as np
import optuna
import pandas as pd

# --- 3. Imports locaux ---
from backtesting.analyzers import drawdown as drawdown_analyzer
from backtesting.analyzers import performance as performance_analyzer
from backtesting.engine import BacktestEngine
from backtesting.portfolio import (
    aggregate_weighted_returns,
    compute_portfolio_metrics,
    normalize_weights,
)
from optimization.optuna_optimizer import ParameterSpec
from reports import overfitting_report
from strategies.base_strategy import BaseStrategy
from utils.data_manager import DataManager
from utils.config_loader import get_settings
from utils.logger import setup_logger


logger = setup_logger(__name__, level=logging.ERROR)


@dataclass
class BacktestResult:
    """Résultat consolidé d'un backtest (mono ou multi-ticker)."""

    metrics: Dict[str, Any]
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    per_ticker: Optional[Dict[str, "BacktestResult"]] = None


class OverfittingChecker:
    """Regroupe les analyses avancées de robustesse / overfitting.

    Args:
        strategy_class: Classe de stratégie héritant de BaseStrategy.
        param_space: Espace de recherche Optuna (mêmes conventions que OptunaOptimizer).
        data: DataFrame OHLCV optionnel. Si None, on charge via data_config.
        data_config: Configuration pour DataManager (ticker, dates, interval...).
        broker_config: Paramétrage du broker pour les backtests (capital, commissions).
        position_sizing_config: Configuration optionnelle du position sizing.
        optimization_config: Paramètres communs à l'optimisation Optuna (n_trials, seed...).
        run_id: Identifiant logique de la batterie de tests.
        output_dir: Répertoire racine pour stocker les résultats.
    """

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        param_space: Dict[str, ParameterSpec],
        *,
        data: Optional[pd.DataFrame] = None,
        data_config: Optional[Dict[str, Any]] = None,
        broker_config: Optional[Dict[str, Any]] = None,
        position_sizing_config: Optional[Dict[str, Any]] = None,
        optimization_config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if not param_space:
            raise ValueError("param_space ne peut pas être vide.")

        self.strategy_class = strategy_class
        self.param_space = param_space
        self.data_config = data_config or {}
        self.broker_config = broker_config or {}
        self.position_sizing_config = position_sizing_config or {}
        self.optimization_config = optimization_config or {}

        self.penalty_value = float(self.optimization_config.get("penalty_value", -1.0))
        self.min_trades = int(self.optimization_config.get("min_trades", 1))
        self.enforce_fast_slow = bool(
            self.optimization_config.get("enforce_fast_slow_gap", True)
        )

        self.data_frames = self._init_data_frames(data)
        self.tickers = list(self.data_frames.keys())
        if not self.tickers:
            raise ValueError("Aucune donnée n'a pu être chargée pour l'analyse.")

        self.primary_ticker = self.tickers[0]
        self.data = self.data_frames[self.primary_ticker]
        self.multi_ticker = len(self.tickers) > 1
        self.weights = normalize_weights(
            self.tickers, self.data_config.get("weights")
        )
        self.portfolio_alignment = str(
            self.data_config.get("alignment", "intersection")
        ).lower()
        self.analytics_settings = self._load_analytics_settings()
        self.overfitting_settings = self.analytics_settings.get("overfitting", {})
        self.warmup_bars = self._determine_warmup_bars()

        self.run_id = run_id or self.strategy_class.__name__.lower()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_output = Path(output_dir or "results/overfitting")
        self.output_root = base_output / self.run_id / timestamp
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._report_sections: List[Dict[str, str]] = []

        logger.info(
            "OverfittingChecker initialisé (run_id=%s, sorties=%s)",
            self.run_id,
            self.output_root,
        )

    # --------------------------------------------------------------------- #
    # API publique
    # --------------------------------------------------------------------- #
    def walk_forward_analysis(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fold_config: Optional[Dict[str, Any]] = None,
        param_space: Optional[Dict[str, ParameterSpec]] = None,
        label: str = "wfa",
    ) -> Dict[str, Any]:
        """Exécute une walk-forward analysis (WFA) en schéma ancré.

        Par défaut : 3 ans d'apprentissage, 6 mois de test, pas de 6 mois.

        Args:
            start_date: Date minimale (inclusive) pour la WFA.
            end_date: Date maximale (inclusive) pour la WFA.
            fold_config: Configuration des fenêtres (train_years, test_months, step_months).
            param_space: Param space facultatif pour cette WFA.
            label: Suffixe pour les fichiers de sortie.

        Returns:
            Dictionnaire récapitulatif (folds détaillés + agrégats).
        """

        cfg = {
            "train_years": 3,
            "test_months": 6,
            "step_months": 6,
        }
        if fold_config:
            cfg.update(fold_config)

        data_slice = self._filter_data(start_date, end_date)
        if len(data_slice) < 2:
            raise ValueError("Dataset insuffisant pour exécuter la WFA.")

        folds = self._build_anchored_folds(
            data_slice.index,
            train_years=int(cfg["train_years"]),
            test_months=int(cfg["test_months"]),
            step_months=int(cfg["step_months"]),
        )

        if not folds:
            raise ValueError("Impossible de construire des folds walk-forward.")

        space = param_space or self.param_space
        fold_results: List[Dict[str, Any]] = []
        train_sharpes: List[float] = []
        test_sharpes: List[float] = []

        out_dir = self.output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (train_slice, test_slice) in enumerate(folds, start=1):
            fold_id = f"fold_{idx:02d}"
            logger.info(
                "WFA %s - optimisation sur %s->%s, test %s->%s",
                fold_id,
                train_slice.index.min(),
                train_slice.index.max(),
                test_slice.index.min(),
                test_slice.index.max(),
            )

            train_dataset = self._slice_dataset(
                train_slice.index.min(), train_slice.index.max()
            )
            study = self._run_optuna(
                train_dataset,
                space,
                study_name=f"{self.run_id}_{fold_id}",
            )
            best_params = study.best_trial.user_attrs.get(
                "strategy_params", study.best_params
            )

            train_result = self._run_backtest(train_dataset, best_params)
            test_dataset = self._slice_dataset(
                test_slice.index.min(),
                test_slice.index.max(),
                include_warmup=True,
            )
            test_result = self._run_backtest(test_dataset, best_params)

            train_sharpes.append(
                self._as_float(train_result.metrics.get("sharpe_ratio"))
            )
            test_sharpes.append(
                self._as_float(test_result.metrics.get("sharpe_ratio"))
            )

            fold_results.append(
                {
                    "fold": fold_id,
                    "train_start": train_slice.index.min(),
                    "train_end": train_slice.index.max(),
                    "test_start": test_slice.index.min(),
                    "test_end": test_slice.index.max(),
                    "best_params": best_params,
                    "train_metrics": train_result.metrics,
                    "test_metrics": test_result.metrics,
                }
            )

        summary = {
            "train_sharpe_mean": float(np.mean(train_sharpes)),
            "train_sharpe_std": (
                float(np.std(train_sharpes, ddof=1)) if len(train_sharpes) > 1 else 0.0
            ),
            "test_sharpe_mean": float(np.mean(test_sharpes)),
            "test_sharpe_std": (
                float(np.std(test_sharpes, ddof=1)) if len(test_sharpes) > 1 else 0.0
            ),
            "folds": fold_results,
        }
        robustness = self._compute_wfa_robustness(summary, fold_results)
        summary["robustness_summary"] = robustness
        metrics = robustness.get("metrics", {}) or {}
        summary["degradation_ratio"] = metrics.get("degradation_ratio")
        summary["test_vs_train_gap"] = metrics.get("test_vs_train_gap")
        summary["frac_test_sharpe_lt_0"] = metrics.get("frac_test_sharpe_lt_0")
        summary["frac_test_sharpe_lt_alpha_train"] = metrics.get(
            "frac_test_sharpe_lt_alpha_train"
        )
        summary["alpha"] = metrics.get("alpha")
        summary["robustness_label"] = robustness.get("badge")

        self._export_wfa_results(fold_results, summary, out_dir)
        return summary

    def out_of_sample_test(
        self,
        params: Dict[str, Any],
        *,
        windows: Optional[Sequence[Tuple[str, str]]] = None,
        years: int = 1,
        label: str = "oos",
    ) -> Dict[str, Any]:
        """Évalue un set de paramètres sur des fenêtres OOS."""

        oos_windows = self._prepare_oos_windows(windows, years=years)
        if not oos_windows:
            raise ValueError("Aucune fenêtre OOS valide.")

        results: List[Dict[str, Any]] = []

        out_dir = self.output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (start, end) in enumerate(oos_windows, start=1):
            subset = self._slice_dataset(start, end, include_warmup=True)
            if self._is_dataset_empty(subset):
                logger.warning(
                    "Fenêtre OOS ignorée (vide) %s -> %s",
                    start.isoformat(),
                    end.isoformat(),
                )
                continue

            bt_result = self._run_backtest(subset, params)
            results.append(
                {
                    "window_id": f"oos_{idx:02d}",
                    "start": start,
                    "end": end,
                    "metrics": bt_result.metrics,
                }
            )

        if not results:
            raise ValueError("Aucun test OOS n'a pu être exécuté.")

        sharpe_values = [
            self._as_float(entry["metrics"].get("sharpe_ratio")) for entry in results
        ]
        oos_sharpe_mean = float(np.mean(sharpe_values))
        oos_sharpe_std = (
            float(np.std(sharpe_values, ddof=1)) if len(sharpe_values) > 1 else 0.0
        )
        frac_lt_zero = float(sum(val < 0.0 for val in sharpe_values)) / (
            len(sharpe_values) or 1
        )
        oos_sharpe_min = float(min(sharpe_values)) if sharpe_values else 0.0
        oos_sharpe_median = (
            float(np.median(sharpe_values)) if sharpe_values else 0.0
        )

        reference_dataset = self._slice_dataset(None, None)
        reference_result = self._run_backtest(reference_dataset, params)
        reference_sharpe = self._as_float(
            reference_result.metrics.get("sharpe_ratio")
        )
        degradation_ratio = (
            oos_sharpe_mean / reference_sharpe if reference_sharpe > 0 else 0.0
        )

        summary = {
            "oos_sharpe_mean": oos_sharpe_mean,
            "oos_sharpe_std": oos_sharpe_std,
            "windows": results,
            "frac_oos_sharpe_lt_0": frac_lt_zero,
            "oos_sharpe_min": oos_sharpe_min,
            "oos_sharpe_median": oos_sharpe_median,
            "train_sharpe_reference": reference_sharpe,
            "oos_degradation_ratio": degradation_ratio,
        }
        robustness = self._compute_oos_robustness(summary, results)
        summary["robustness_summary"] = robustness
        summary["robustness_label"] = robustness.get("badge")

        self._export_oos_results(results, summary, out_dir)
        return summary

    def monte_carlo_simulation(
        self,
        params: Dict[str, Any],
        *,
        data: Optional[pd.DataFrame] = None,
        source: str = "returns",
        n_simulations: int = 1000,
        block_size: int = 5,
        seed: Optional[int] = None,
        label: str = "monte_carlo",
    ) -> Dict[str, Any]:
        """Monte-Carlo bootstrap sur les retours ou les trades."""

        dataset = data if data is not None else self._slice_dataset(None, None)
        base_result = self._run_backtest(dataset, params)
        initial_capital = float(base_result.metrics.get("initial_capital", 1.0))

        if source not in {"returns", "trades"}:
            raise ValueError("source doit être 'returns' ou 'trades'.")

        if source == "returns" and base_result.returns.empty:
            raise ValueError("Aucun retour journalier disponible pour le bootstrap.")

        if source == "trades" and base_result.trades.empty:
            raise ValueError("Aucun trade disponible pour le bootstrap.")

        rng = np.random.default_rng(seed)
        simulations: List[Dict[str, float]] = []

        returns_series = base_result.returns
        trades_df = base_result.trades

        for sim_idx in range(1, n_simulations + 1):
            if source == "returns":
                sampled_returns = self._block_bootstrap_series(
                    returns_series, block_size=block_size, rng=rng
                )
            else:
                sampled_returns = self._block_bootstrap_trades(
                    trades_df,
                    block_size=block_size,
                    rng=rng,
                    capital=initial_capital,
                )

            sim_metrics = self._compute_metrics_from_returns(sampled_returns)
            simulations.append(
                {
                    "simulation": sim_idx,
                    "sharpe_ratio": sim_metrics.get("sharpe_ratio", 0.0),
                    "sortino_ratio": sim_metrics.get("sortino_ratio", 0.0),
                    "cagr": sim_metrics.get("cagr", 0.0),
                    "max_drawdown": sim_metrics.get("max_drawdown", 0.0),
                    "prob_negative": float(sim_metrics.get("final_value", 1.0) < 1.0),
                }
            )

        dd_rules = self._get_overfitting_value(("monte_carlo", "max_drawdown"), {}) or {}
        dd_threshold_raw = dd_rules.get("threshold")
        dd_threshold = (
            self._as_float(dd_threshold_raw) if dd_threshold_raw is not None else None
        )
        summary = self._summarize_simulations(
            simulations, max_dd_threshold=dd_threshold
        )
        out_dir = self.output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        robustness = self._compute_monte_carlo_robustness(summary)
        summary["robustness_summary"] = robustness
        summary["robustness_label"] = robustness.get("badge")
        self._export_monte_carlo(simulations, summary, out_dir, source)

        return {
            "base_metrics": base_result.metrics,
            "simulations": simulations,
            "summary": summary,
        }

    def stability_tests(
        self,
        base_params: Dict[str, Any],
        *,
        perturbation: float = 0.1,
        steps: int = 3,
        threshold: float = 0.95,
        data: Optional[pd.DataFrame] = None,
        label: str = "stability",
    ) -> Dict[str, Any]:
        """Évalue la sensibilité locale des paramètres."""

        dataset = data if data is not None else self.data
        base_result = self._run_backtest(dataset, base_params)
        base_metric = self._as_float(base_result.metrics.get("sharpe_ratio"))

        variations = self._generate_param_variations(
            base_params, perturbation=perturbation, steps=steps
        )

        results: List[Dict[str, Any]] = []
        for variation in variations:
            params = dict(base_params)
            params.update(
                {
                    key: value
                    for key, value in variation.items()
                    if not key.startswith("__")
                }
            )

            bt_result = self._run_backtest(dataset, params)
            sharpe = bt_result.metrics.get("sharpe_ratio", 0.0)
            ratio = sharpe / base_metric if base_metric else 0.0
            results.append(
                {
                    "param_name": variation["__param_name__"],
                    "param_value": variation["__param_value__"],
                    "sharpe_ratio": sharpe,
                    "relative_sharpe": ratio,
                    "metrics": bt_result.metrics,
                }
            )

        robust_count = sum(
            1 for entry in results if entry["relative_sharpe"] >= threshold
        )
        total_neighbors = len(results)

        summary = {
            "base_metric": base_metric,
            "robust_fraction": (
                float(robust_count / total_neighbors) if total_neighbors else 0.0
            ),
            "neighbors": results,
        }
        robustness = self._compute_stability_robustness(summary)
        summary["robustness_summary"] = robustness
        summary["robustness_label"] = robustness.get("badge")

        out_dir = self.output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        self._export_stability(results, summary, out_dir)

        return summary

    # ------------------------------------------------------------------ #
    # Implémentation interne
    # ------------------------------------------------------------------ #
    def _init_data_frames(
        self,
        data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
    ) -> Dict[str, pd.DataFrame]:
        if data is not None:
            if isinstance(data, pd.DataFrame):
                return {"data0": self._validate_dataframe(data, label="data0")}
            if isinstance(data, dict):
                frames: Dict[str, pd.DataFrame] = {}
                for key, df in data.items():
                    if not isinstance(df, pd.DataFrame):
                        raise TypeError(
                            "Chaque entrée de data doit être un DataFrame pandas."
                        )
                    frames[str(key)] = self._validate_dataframe(df, label=str(key))
                return frames
            raise TypeError(
                "data doit être un DataFrame ou un dictionnaire {ticker: DataFrame}."
            )

        if not self.data_config:
            raise ValueError(
                "data est None et data_config est vide : impossible de charger les données."
            )

        return self._load_data_frames_from_config()

    def _load_data_frames_from_config(self) -> Dict[str, pd.DataFrame]:
        tickers_cfg = self.data_config.get("tickers")
        if tickers_cfg:
            if isinstance(tickers_cfg, str):
                tickers = [tickers_cfg]
            else:
                tickers = [str(t).strip() for t in tickers_cfg if str(t).strip()]
        else:
            ticker = self.data_config.get("ticker")
            if not ticker:
                raise ValueError(
                    "'ticker' ou 'tickers' doit être défini dans data_config."
                )
            tickers = [str(ticker).strip()]

        tickers = [t for t in tickers if t]
        if not tickers:
            raise ValueError("Aucun ticker valide fourni pour le chargement des données.")

        if len(tickers) > 1 and not (
            self.data_config.get("start_date") and self.data_config.get("end_date")
        ):
            raise ValueError(
                "Le mode multi-ticker requiert 'start_date' et 'end_date' dans data_config."
            )

        manager = DataManager()
        frames: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = manager.get_data(
                ticker=ticker,
                start_date=self.data_config.get("start_date"),
                end_date=self.data_config.get("end_date"),
                interval=self.data_config.get("interval"),
                use_cache=self.data_config.get("use_cache", True),
            )
            frames[ticker] = self._validate_dataframe(df, label=ticker)

        return frames

    def _validate_dataframe(self, df: Optional[pd.DataFrame], *, label: str) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError(f"Aucune donnée disponible pour '{label}'.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Le DataFrame '{label}' doit avoir un index de type DatetimeIndex."
            )

        return df.sort_index()

    def _load_analytics_settings(self) -> Dict[str, Any]:
        try:
            settings = get_settings()
            return settings.get("analytics", {})
        except Exception:
            return {}

    def _determine_warmup_bars(self) -> int:
        """Détermine le nombre de barres de warmup à préfixer aux datasets."""

        override = self.optimization_config.get("warmup_bars")
        if override is not None:
            try:
                return max(int(override), 0)
            except (TypeError, ValueError):
                return 0

        inferred = self._infer_warmup_from_space()
        if inferred > 0:
            return inferred

        settings_value = self.overfitting_settings.get("warmup_bars")
        try:
            return max(int(settings_value), 0)
        except (TypeError, ValueError):
            return 0

    def _infer_warmup_from_space(self) -> int:
        """Estime le warmup à partir des bornes supérieures du param_space."""

        max_candidate = 0
        for spec in self.param_space.values():
            candidate = self._extract_int_upper_bound(spec)
            if candidate is not None:
                max_candidate = max(max_candidate, candidate)
        return max_candidate

    def _extract_int_upper_bound(self, spec: ParameterSpec) -> Optional[int]:
        if isinstance(spec, dict):
            high = spec.get("high")
            if high is None:
                return None
            param_type = str(spec.get("type", "")).lower()
            if param_type in {"int", "integer"} or isinstance(high, Integral):
                return int(high)
            if isinstance(high, float) and float(high).is_integer():
                return int(high)
            return None

        if isinstance(spec, (list, tuple)):
            values = list(spec)
            if (
                len(values) >= 2
                and all(isinstance(v, (int, float)) for v in values[:2])
            ):
                high = values[1]
                if isinstance(high, Integral):
                    return int(high)
                if isinstance(high, float) and float(high).is_integer():
                    return int(high)
            return None

        if isinstance(spec, Integral):
            return int(spec)
        if isinstance(spec, float) and spec.is_integer():
            return int(spec)

        return None

    def _filter_data(
        self, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.DataFrame:
        df = self.data
        start_ts = self._normalize_timestamp(start_date)
        end_ts = self._normalize_timestamp(end_date)
        if start_ts is not None:
            df = df.loc[start_ts:]
        if end_ts is not None:
            df = df.loc[:end_ts]
        return df

    def _apply_warmup_offset(
        self, timestamp: Optional[pd.Timestamp]
    ) -> Optional[pd.Timestamp]:
        if timestamp is None or self.warmup_bars <= 0:
            return timestamp
        return timestamp - pd.DateOffset(days=int(self.warmup_bars))

    def _slice_dataset(
        self,
        start: Optional[Union[str, pd.Timestamp]],
        end: Optional[Union[str, pd.Timestamp]],
        *,
        include_warmup: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        start_ts = self._normalize_timestamp(start)
        end_ts = self._normalize_timestamp(end)

        if include_warmup:
            start_ts = self._apply_warmup_offset(start_ts)

        if not self.multi_ticker:
            return self._slice_frame(self.data, start_ts, end_ts)

        subsets: Dict[str, pd.DataFrame] = {}
        for ticker, df in self.data_frames.items():
            subset = self._slice_frame(df, start_ts, end_ts)
            if not subset.empty:
                subsets[ticker] = subset
        return subsets

    def _slice_frame(
        self,
        df: pd.DataFrame,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        result = df
        if start is not None:
            result = result.loc[start:]
        if end is not None:
            result = result.loc[:end]
        return result

    def _normalize_timestamp(
        self, value: Optional[Union[str, pd.Timestamp]]
    ) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        ts = pd.Timestamp(value)
        data_tz = getattr(self.data.index, "tz", None)
        if data_tz is None:
            if ts.tzinfo is not None:
                return ts.tz_convert(None)
            return ts
        if ts.tzinfo is None:
            return ts.tz_localize(data_tz)
        return ts.tz_convert(data_tz)

    def _is_dataset_empty(
        self, dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> bool:
        if isinstance(dataset, dict):
            return not any(len(df) for df in dataset.values())
        return dataset.empty

    def _build_anchored_folds(
        self,
        index: pd.DatetimeIndex,
        *,
        train_years: int,
        test_months: int,
        step_months: int,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        data = self.data.loc[index.min() : index.max()]
        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

        train_start = index.min()
        last_date = index.max()

        train_end = train_start + pd.DateOffset(years=train_years)
        test_length = pd.DateOffset(months=test_months)
        step_offset = pd.DateOffset(months=step_months)

        while train_end < last_date:
            test_start = train_end
            test_end = test_start + test_length

            train_subset = data.loc[
                train_start : test_start - pd.Timedelta(nanoseconds=1)
            ]
            test_subset = data.loc[test_start:test_end]

            if len(train_subset) >= 30 and len(test_subset) >= 5:
                folds.append((train_subset, test_subset))
            else:
                logger.debug(
                    "Fold ignoré (taille train/test insuffisante). train=%s, test=%s",
                    len(train_subset),
                    len(test_subset),
                )

            train_end += step_offset
            if train_end >= last_date:
                break

        return folds

    def _run_optuna(
        self,
        data_slice: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        param_space: Dict[str, ParameterSpec],
        *,
        study_name: str,
    ) -> optuna.Study:
        def objective(trial: optuna.Trial) -> float:
            params = self._build_params(trial, param_space)
            trial.set_user_attr("strategy_params", params)

            if not self._validate_params(params):
                trial.set_user_attr("constraint_violation", True)
                return self.penalty_value

            try:
                result = self._run_backtest(data_slice, params)
            except Exception as exc:  # pragma: no cover - robustesse optuna
                logger.exception("Backtest échoué (trial %s): %s", trial.number, exc)
                trial.set_user_attr("error", str(exc))
                return self.penalty_value

            metrics = result.metrics
            sharpe = metrics.get("sharpe_ratio")
            trades = metrics.get("total_trades", 0)

            if trades < self.min_trades:
                trial.set_user_attr("low_trade_count", trades)
                return self.penalty_value

            if sharpe is None or not np.isfinite(sharpe):
                return self.penalty_value

            trial.set_user_attr("total_trades", trades)
            trial.set_user_attr("train_metrics", metrics)
            return float(sharpe)

        study = self._create_study(study_name)
        study.optimize(
            objective,
            n_trials=int(self.optimization_config.get("n_trials", 50)),
            timeout=self.optimization_config.get("timeout"),
            n_jobs=1,
            show_progress_bar=False,
        )
        return study

    def _create_study(self, name: str) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(
            seed=self.optimization_config.get("sampler_seed")
        )
        pruner = optuna.pruners.MedianPruner()

        storage = None
        if self.optimization_config.get("storage"):
            storage = str(self.optimization_config["storage"])
            if storage.startswith("sqlite:///"):
                Path(storage.replace("sqlite:///", "")).parent.mkdir(
                    parents=True, exist_ok=True
                )

        return optuna.create_study(
            study_name=name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=False,
        )

    def _run_backtest(
        self,
        data_slice: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        params: Dict[str, Any],
    ) -> BacktestResult:
        if isinstance(data_slice, dict):
            return self._run_portfolio_backtest(data_slice, params)
        return self._run_single_backtest(data_slice, params)

    def _run_single_backtest(
        self, data_slice: pd.DataFrame, params: Dict[str, Any]
    ) -> BacktestResult:
        engine = BacktestEngine()
        self._configure_broker(engine)
        self._configure_position_sizing(engine)
        engine.add_data(data_slice.copy(), name="data0")
        engine.add_strategy(self.strategy_class, **params)

        results = engine.run()
        if not results:
            raise RuntimeError("BacktestEngine n'a retourné aucun résultat.")

        strat = results[0]

        metrics = self._gather_metrics(strat)
        returns_series = self._extract_returns(strat)
        original_sharpe = metrics.get("sharpe_ratio")
        try:
            sharpe_value = float(original_sharpe)
        except (TypeError, ValueError):
            sharpe_value = None
        if sharpe_value is None or not np.isfinite(sharpe_value):
            fallback_sharpe = self._compute_sharpe_from_returns(returns_series)
            if fallback_sharpe is not None:
                metrics["sharpe_ratio"] = fallback_sharpe
        equity_series = self._build_equity_curve(
            returns_series,
            metrics.get("initial_capital", 1.0),
            fallback_index=data_slice.index.min() if len(data_slice) else None,
        )
        trades_df = self._extract_trades(strat)

        return BacktestResult(
            metrics=metrics,
            returns=returns_series,
            equity=equity_series,
            trades=trades_df,
        )

    def _run_portfolio_backtest(
        self, dataset_map: Dict[str, pd.DataFrame], params: Dict[str, Any]
    ) -> BacktestResult:
        valid_results: Dict[str, BacktestResult] = {}
        for ticker, df in dataset_map.items():
            if df is None or df.empty:
                continue
            valid_results[ticker] = self._run_single_backtest(df, params)

        if not valid_results:
            raise ValueError("Aucun dataset valide disponible pour le backtest.")

        if len(valid_results) == 1:
            return next(iter(valid_results.values()))

        returns_map = {
            ticker: result.returns
            for ticker, result in valid_results.items()
            if not result.returns.empty
        }
        weight_keys = list(valid_results.keys())
        weights = self.weights.reindex(weight_keys).fillna(0.0)
        if not np.isfinite(weights.sum()) or float(weights.sum()) <= 0:
            weights = normalize_weights(weight_keys)
        aligned_returns = aggregate_weighted_returns(
            returns_map,
            weights=weights,
            alignment=self.portfolio_alignment,
        )
        if aligned_returns.empty:
            raise ValueError("Impossible de calculer les rendements agrégés du portefeuille.")

        initial_capital = float(self.broker_config.get("initial_capital", 10000.0))
        metrics, equity_curve, _, _ = compute_portfolio_metrics(
            aligned_returns,
            initial_capital,
            self.analytics_settings,
        )

        metrics["initial_capital"] = initial_capital
        total_trades = sum(
            res.metrics.get("total_trades", 0) for res in valid_results.values()
        )
        won_trades = sum(
            res.metrics.get("won_trades", 0) for res in valid_results.values()
        )
        lost_trades = sum(
            res.metrics.get("lost_trades", 0) for res in valid_results.values()
        )
        metrics.update(
            total_trades=total_trades,
            won_trades=won_trades,
            lost_trades=lost_trades,
            per_ticker_metrics={ticker: res.metrics for ticker, res in valid_results.items()},
        )

        merged_trades = self._merge_trades(valid_results)
        return BacktestResult(
            metrics=metrics,
            returns=aligned_returns,
            equity=equity_curve,
            trades=merged_trades,
            per_ticker=valid_results,
        )

    def _merge_trades(self, per_ticker_results: Dict[str, BacktestResult]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for ticker, result in per_ticker_results.items():
            trades = result.trades
            if trades is None or trades.empty:
                continue
            enriched = trades.copy()
            if "ticker" not in enriched.columns:
                enriched["ticker"] = ticker
            else:
                enriched["ticker"] = enriched["ticker"].fillna(ticker)
            frames.append(enriched)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _configure_broker(self, engine: BacktestEngine) -> None:
        cfg = self.broker_config
        if not cfg:
            return

        initial_capital = float(cfg.get("initial_capital", 10000.0))
        engine.cerebro.broker.setcash(initial_capital)

        if "commission_pct" in cfg:
            engine.cerebro.broker.setcommission(commission=float(cfg["commission_pct"]))
        elif "commission_fixed" in cfg:
            engine.cerebro.broker.setcommission(
                commission=float(cfg["commission_fixed"])
            )

        if "slippage_pct" in cfg and float(cfg["slippage_pct"]) > 0:
            engine.cerebro.broker.set_slippage_perc(perc=float(cfg["slippage_pct"]))

    def _configure_position_sizing(self, engine: BacktestEngine) -> None:
        cfg = self.position_sizing_config
        if not cfg or not cfg.get("enabled", False):
            return

        method = cfg.get("method", "fixed")
        try:
            if method == "fixed":
                from risk_management.position_sizing import FixedSizer

                params = cfg.get("fixed", {})
                engine.add_sizer(
                    FixedSizer,
                    stake=params.get("stake"),
                    pct_size=params.get("pct_size", 1.0),
                )
            elif method == "fixed_fractional":
                from risk_management.position_sizing import FixedFractionalSizer

                params = cfg.get("fixed_fractional", {})
                engine.add_sizer(
                    FixedFractionalSizer,
                    risk_pct=params.get("risk_pct", 0.02),
                    stop_distance=params.get("stop_distance", 0.03),
                )
            elif method == "volatility_based":
                from risk_management.position_sizing import VolatilityBasedSizer

                params = cfg.get("volatility_based", {})
                engine.add_sizer(
                    VolatilityBasedSizer,
                    risk_pct=params.get("risk_pct", 0.02),
                    atr_period=params.get("atr_period", 14),
                    atr_multiplier=params.get("atr_multiplier", 2.0),
                )
        except Exception as exc:  # pragma: no cover - sizer facultatif
            logger.error("Erreur position sizing (%s): %s", method, exc)

    def _build_params(
        self, trial: optuna.Trial, param_space: Dict[str, ParameterSpec]
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in param_space.items():
            params[name] = self._suggest_param(trial, name, spec)
        return params

    def _suggest_param(
        self, trial: optuna.Trial, name: str, spec: ParameterSpec
    ) -> Any:
        if isinstance(spec, dict):
            return self._suggest_from_dict(trial, name, spec)

        if isinstance(spec, (list, tuple)):
            values = list(spec)
            if len(values) == 0:
                raise ValueError(f"param_space[{name}] est vide.")

            if len(values) == 1:
                return values[0]

            if len(values) in {2, 3} and all(
                isinstance(v, (int, float)) for v in values[:2]
            ):
                low = values[0]
                high = values[1]
                step = values[2] if len(values) == 3 else None

                if all(isinstance(v, int) for v in values[:2]):
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

            return trial.suggest_categorical(name, list(values))

        return spec

    def _suggest_from_dict(
        self, trial: optuna.Trial, name: str, spec: Dict[str, Any]
    ) -> Any:
        param_type = spec.get("type", "float").lower()

        if param_type in {"int", "integer"}:
            return trial.suggest_int(
                name,
                low=int(spec["low"]),
                high=int(spec["high"]),
                step=int(spec.get("step", 1)),
            )

        if param_type in {"float", "uniform"}:
            return trial.suggest_float(
                name,
                low=float(spec["low"]),
                high=float(spec["high"]),
                step=spec.get("step"),
                log=bool(spec.get("log", False))
                or str(spec.get("scale", "")).lower() == "log",
            )

        if param_type in {"categorical", "choice"}:
            choices = spec.get("choices")
            if not choices:
                raise ValueError(
                    f"param_space[{name}] doit définir 'choices' pour type catégoriel."
                )
            return trial.suggest_categorical(name, list(choices))

        raise ValueError(f"Type de paramètre non supporté: {param_type}")

    def _validate_params(self, params: Dict[str, Any]) -> bool:
        if not self.enforce_fast_slow:
            return True
        if "fast_period" in params and "slow_period" in params:
            return params["fast_period"] < params["slow_period"]
        return True

    def _as_float(self, value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float(default)
        return float(val) if np.isfinite(val) else float(default)

    def _gather_metrics(self, strat: BaseStrategy) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        try:
            trades = strat.analyzers.trades.get_analysis()
        except Exception:
            trades = {}

        try:
            sharpe = strat.analyzers.sharpe.get_analysis()
        except Exception:
            sharpe = {}

        try:
            drawdown = strat.analyzers.drawdown.get_analysis()
        except Exception:
            drawdown = {}

        try:
            returns = strat.analyzers.returns.get_analysis()
        except Exception:
            returns = {}

        metrics["total_trades"] = trades.get("total", {}).get("total", 0)
        metrics["won_trades"] = trades.get("won", {}).get("total", 0)
        metrics["lost_trades"] = trades.get("lost", {}).get("total", 0)
        metrics["sharpe_ratio"] = sharpe.get("sharperatio")
        metrics["max_drawdown"] = drawdown.get("max", {}).get("drawdown")
        metrics["max_drawdown_len"] = drawdown.get("max", {}).get("len")
        metrics["total_return"] = returns.get("rtot")
        metrics["annualized_return"] = returns.get("ravg")

        final_value = strat.broker.getvalue()
        initial_capital = getattr(strat.broker, "startingcash", None)
        if initial_capital is None:
            initial_capital = float(self.broker_config.get("initial_capital", 10000.0))

        metrics["final_value"] = float(final_value)
        metrics["initial_capital"] = float(initial_capital)
        metrics["pnl"] = float(final_value - initial_capital)
        metrics["pnl_pct"] = (
            (float(final_value) / float(initial_capital)) - 1.0
            if initial_capital
            else 0.0
        )

        return metrics

    def _extract_returns(self, strat: BaseStrategy) -> pd.Series:
        try:
            time_returns = strat.analyzers.timereturns.get_analysis()
            if isinstance(time_returns, dict):
                series = pd.Series(time_returns)
                series.index = pd.to_datetime(series.index)
                return series.sort_index()
        except Exception:
            pass
        return pd.Series(dtype=float)

    def _compute_sharpe_from_returns(
        self, returns: pd.Series, periods_per_year: int = 252
    ) -> Optional[float]:
        if returns is None or returns.empty:
            return None
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        if not np.isfinite(mean) or not np.isfinite(std) or std == 0:
            return None
        sharpe = (mean / std) * np.sqrt(periods_per_year)
        return float(sharpe) if np.isfinite(sharpe) else None

    def _extract_trades(self, strat: BaseStrategy) -> pd.DataFrame:
        try:
            trade_list = strat.analyzers.tradelist.get_analysis()
            if isinstance(trade_list, list) and trade_list:
                df = pd.DataFrame(trade_list)
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def _build_equity_curve(
        self,
        returns: pd.Series,
        initial_capital: float,
        *,
        fallback_index: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        if returns.empty:
            index_value = fallback_index
            if index_value is None:
                index_value = self.data.index.min() if len(self.data) else pd.Timestamp.utcnow()
            return pd.Series(data=[initial_capital], index=[index_value])
        cumulative = (1.0 + returns.fillna(0.0)).cumprod()
        equity = cumulative * float(initial_capital)
        return equity

    def _prepare_oos_windows(
        self,
        windows: Optional[Sequence[Tuple[str, str]]],
        *,
        years: int,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        if windows:
            return [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in windows]

        last_date = self.data.index.max()
        start = last_date - pd.DateOffset(years=years)
        return [(start, last_date)]

    def _block_bootstrap_series(
        self,
        series: pd.Series,
        *,
        block_size: int,
        rng: np.random.Generator,
    ) -> pd.Series:
        values = series.to_numpy()
        n_obs = len(values)
        if n_obs == 0:
            raise ValueError("Série vide pour le bootstrap.")

        block_size = max(1, int(block_size))
        n_blocks = int(np.ceil(n_obs / block_size))
        sampled = []

        for _ in range(n_blocks):
            start_idx = rng.integers(0, max(1, n_obs - block_size + 1))
            sampled.extend(values[start_idx : start_idx + block_size])

        sampled = sampled[:n_obs]
        return pd.Series(sampled, index=series.index)

    def _block_bootstrap_trades(
        self,
        trades: pd.DataFrame,
        *,
        block_size: int,
        rng: np.random.Generator,
        capital: float,
    ) -> pd.Series:
        if trades.empty:
            raise ValueError("Table de trades vide pour le bootstrap.")

        pnl_col = "net_pnl" if "net_pnl" in trades.columns else "pnl"
        if pnl_col not in trades:
            raise ValueError("Impossible de trouver une colonne pnl pour le bootstrap.")

        capital = float(capital) if float(capital or 0.0) > 0 else 1.0
        returns = trades[pnl_col].astype(float) / capital
        return self._block_bootstrap_series(returns, block_size=block_size, rng=rng)

    def _compute_metrics_from_returns(self, returns: pd.Series) -> Dict[str, float]:
        equity = (1.0 + returns).cumprod()
        dd_metrics, _ = drawdown_analyzer.analyze(equity)
        perf = performance_analyzer.compute(
            equity=equity,
            returns=returns,
            trades=None,
            periods_per_year=252,
            risk_free_rate_annual=0.0,
            mar_annual=0.0,
        )
        return {
            **perf,
            "max_drawdown": dd_metrics.get("max_drawdown", 0.0),
            "final_value": float(equity.iloc[-1]) if len(equity) else 1.0,
        }

    def _summarize_simulations(
        self,
        simulations: Sequence[Dict[str, float]],
        *,
        max_dd_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        df = pd.DataFrame(simulations)
        summary: Dict[str, Any] = {}
        for column in ["sharpe_ratio", "cagr", "max_drawdown"]:
            if column in df:
                summary[f"{column}_mean"] = float(df[column].mean())
                summary[f"{column}_std"] = float(df[column].std(ddof=1))
                summary[f"{column}_p05"] = float(df[column].quantile(0.05))
                summary[f"{column}_p50"] = float(df[column].quantile(0.50))
                summary[f"{column}_p95"] = float(df[column].quantile(0.95))
        summary["prob_negative"] = (
            float(df["prob_negative"].mean()) if "prob_negative" in df else 0.0
        )
        summary["p_sharpe_lt_0"] = (
            float((df["sharpe_ratio"] < 0.0).mean()) if "sharpe_ratio" in df else 0.0
        )
        summary["p_cagr_lt_0"] = (
            float((df["cagr"] < 0.0).mean()) if "cagr" in df else 0.0
        )
        if max_dd_threshold is not None and "max_drawdown" in df:
            summary["p_max_dd_gt_threshold"] = float(
                (df["max_drawdown"].astype(float) > float(max_dd_threshold)).mean()
            )
        else:
            summary["p_max_dd_gt_threshold"] = 0.0
        return summary

    def _generate_param_variations(
        self,
        base_params: Dict[str, Any],
        *,
        perturbation: float,
        steps: int,
    ) -> List[Dict[str, Any]]:
        variations: List[Dict[str, Any]] = []
        for name, value in base_params.items():
            if not isinstance(value, (int, float)):
                continue

            factors = np.linspace(
                1.0 - perturbation,
                1.0 + perturbation,
                num=max(steps, 3),
            )

            candidate_values: List[Any] = []
            for factor in factors:
                perturbed = value * factor
                if isinstance(value, int):
                    perturbed = int(round(perturbed))
                candidate_values.append(perturbed)

            candidate_values = sorted(set(candidate_values))
            for cand in candidate_values:
                if cand == value:
                    continue
                variations.append(
                    {
                        "__param_name__": name,
                        "__param_value__": cand,
                        name: cand,
                    }
                )

        return variations

    def _report_meta(self) -> Dict[str, Any]:
        start = self.data.index.min() if len(self.data) else None
        end = self.data.index.max() if len(self.data) else None
        return {
            "run_id": self.run_id,
            "strategy": self.strategy_class.__name__,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M %Z"),
            "data_start": start.strftime("%Y-%m-%d") if isinstance(start, pd.Timestamp) else None,
            "data_end": end.strftime("%Y-%m-%d") if isinstance(end, pd.Timestamp) else None,
        }

    def _register_report_section(
        self,
        *,
        name: str,
        description: str,
        relative_path: str,
        badge: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        status_value = status or badge
        entry = {
            "name": name,
            "description": description,
            "path": relative_path,
        }
        if badge:
            entry["badge"] = badge
        if status_value:
            entry["status"] = status_value
        # Remplacer si même path déjà présent
        self._report_sections = [
            sec for sec in self._report_sections if sec.get("path") != relative_path
        ]
        self._report_sections.append(entry)
        try:
            overfitting_report.render_overfitting_index(
                meta=self._report_meta(),
                sections=self._report_sections,
                output_path=self.output_root / "index.html",
            )
        except Exception:  # pragma: no cover - fallback silencieux
            logger.debug("Impossible de générer l'index overfitting.", exc_info=True)

    def export_global_summary(
        self,
        *,
        wfa: Optional[Dict[str, Any]] = None,
        oos: Optional[Dict[str, Any]] = None,
        monte_carlo: Optional[Dict[str, Any]] = None,
        stability: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        summaries = {
            "wfa": wfa,
            "oos": oos,
            "monte_carlo": monte_carlo,
            "stability": stability,
        }
        filtered = {key: value for key, value in summaries.items() if value is not None}
        if not filtered:
            return None

        payload: Dict[str, Any] = {
            "meta": {
                "run_id": self.run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        payload.update(filtered)

        summary_path = self.output_root / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=self._json_default,
            ),
            encoding="utf-8",
        )
        return summary_path

    def _json_default(self, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Series):
            return value.astype(object).tolist()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        return str(value)

    # ------------------------------------------------------------------ #
    # Robustness scoring helpers
    # ------------------------------------------------------------------ #
    def _get_overfitting_value(self, path: Sequence[str], default: Any = None) -> Any:
        data: Any = self.overfitting_settings
        for key in path:
            if not isinstance(data, dict):
                return default
            data = data.get(key)
            if data is None:
                return default
        return data

    def _resolve_badge(self, *, robust: bool, overfit: bool) -> str:
        if robust:
            return "robust"
        if overfit:
            return "overfitted"
        return "borderline"

    def _compute_wfa_robustness(
        self, summary: Dict[str, Any], folds: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        train_mean = self._as_float(summary.get("train_sharpe_mean"))
        test_mean = self._as_float(summary.get("test_sharpe_mean"))
        degradation_ratio = test_mean / train_mean if train_mean > 0 else 0.0
        test_vs_train_gap = test_mean - train_mean

        train_sharpes = [
            self._as_float(fold["train_metrics"].get("sharpe_ratio")) for fold in folds
        ]
        test_sharpes = [
            self._as_float(fold["test_metrics"].get("sharpe_ratio")) for fold in folds
        ]
        total_folds = len(test_sharpes) or 1
        frac_test_lt_zero = float(sum(val < 0.0 for val in test_sharpes)) / total_folds

        alpha = float(self._get_overfitting_value(("wfa", "alpha"), 0.5))
        frac_test_lt_alpha = (
            float(
                sum(
                    1
                    for train_val, test_val in zip(train_sharpes, test_sharpes)
                    if test_val < alpha * train_val
                )
            )
            / total_folds
            if test_sharpes
            else 0.0
        )

        ratio_rules = self._get_overfitting_value(("wfa", "degradation_ratio"), {}) or {}
        frac_rules = self._get_overfitting_value(
            ("wfa", "frac_test_sharpe_lt_alpha_train"), {}
        ) or {}
        robust = (
            degradation_ratio >= float(ratio_rules.get("robust_min", 0.8))
            and frac_test_lt_alpha <= float(frac_rules.get("robust_max", 0.2))
        )
        overfit = (
            degradation_ratio <= float(ratio_rules.get("overfit_max", 0.5))
            or frac_test_lt_alpha >= float(frac_rules.get("overfit_min", 0.5))
        )

        metrics = {
            "degradation_ratio": degradation_ratio,
            "test_vs_train_gap": test_vs_train_gap,
            "frac_test_sharpe_lt_0": frac_test_lt_zero,
            "frac_test_sharpe_lt_alpha_train": frac_test_lt_alpha,
            "alpha": alpha,
        }
        return {
            "badge": self._resolve_badge(robust=robust, overfit=overfit),
            "metrics": metrics,
        }

    def _compute_oos_robustness(
        self, summary: Dict[str, Any], windows: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        sharpe_values = [
            self._as_float(entry["metrics"].get("sharpe_ratio")) for entry in windows
        ]
        count = len(sharpe_values) or 1
        frac_lt_zero = float(sum(val < 0.0 for val in sharpe_values)) / count
        oos_sharpe_min = float(min(sharpe_values)) if sharpe_values else 0.0
        oos_sharpe_median = (
            float(np.median(sharpe_values)) if sharpe_values else 0.0
        )
        oos_mean = self._as_float(summary.get("oos_sharpe_mean"))

        mean_rules = self._get_overfitting_value(("oos", "mean_sharpe"), {}) or {}
        frac_rules = self._get_overfitting_value(("oos", "frac_sharpe_lt_0"), {}) or {}
        robust = (
            oos_mean >= float(mean_rules.get("robust_min", 1.0))
            and frac_lt_zero <= float(frac_rules.get("robust_max", 0.2))
        )
        overfit = (
            oos_mean <= float(mean_rules.get("overfit_max", 0.5))
            or frac_lt_zero >= float(frac_rules.get("overfit_min", 0.5))
        )

        metrics = {
            "frac_oos_sharpe_lt_0": frac_lt_zero,
            "oos_sharpe_min": oos_sharpe_min,
            "oos_sharpe_median": oos_sharpe_median,
            "oos_degradation_ratio": summary.get("oos_degradation_ratio"),
            "train_sharpe_reference": summary.get("train_sharpe_reference"),
        }
        return {
            "badge": self._resolve_badge(robust=robust, overfit=overfit),
            "metrics": metrics,
        }

    def _compute_monte_carlo_robustness(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        p_sharpe = self._as_float(summary.get("p_sharpe_lt_0"))
        p_cagr = self._as_float(summary.get("p_cagr_lt_0"))
        sharpe_rules = self._get_overfitting_value(
            ("monte_carlo", "p_sharpe_lt_0"), {}
        ) or {}
        cagr_rules = self._get_overfitting_value(
            ("monte_carlo", "p_cagr_lt_0"), {}
        ) or {}
        dd_rules = self._get_overfitting_value(
            ("monte_carlo", "max_drawdown"), {}
        ) or {}
        dd_threshold_raw = dd_rules.get("threshold")
        dd_threshold = (
            self._as_float(dd_threshold_raw) if dd_threshold_raw is not None else None
        )
        p_dd = self._as_float(summary.get("p_max_dd_gt_threshold"))

        robust = (
            p_sharpe <= float(sharpe_rules.get("robust_max", 0.2))
            and p_cagr <= float(cagr_rules.get("robust_max", 0.2))
        )
        overfit = (
            p_sharpe >= float(sharpe_rules.get("overfit_min", 0.5))
            or p_cagr >= float(cagr_rules.get("overfit_min", 0.5))
        )
        if dd_threshold is not None:
            robust = robust and (
                p_dd <= float(dd_rules.get("robust_max", 0.2))
            )
            overfit = overfit or (
                p_dd >= float(dd_rules.get("overfit_min", 0.5))
            )
        metrics = {
            "p_sharpe_lt_0": p_sharpe,
            "p_cagr_lt_0": p_cagr,
            "prob_negative": self._as_float(summary.get("prob_negative")),
            "p_max_dd_gt_threshold": p_dd,
        }
        if dd_threshold is not None:
            metrics["max_dd_threshold"] = dd_threshold
        return {
            "badge": self._resolve_badge(robust=robust, overfit=overfit),
            "metrics": metrics,
        }

    def _compute_stability_robustness(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        robust_fraction = self._as_float(summary.get("robust_fraction"))
        rules = self._get_overfitting_value(("stability", "robust_fraction"), {}) or {}
        robust = robust_fraction >= float(rules.get("robust_min", 0.8))
        overfit = robust_fraction <= float(rules.get("overfit_max", 0.5))
        metrics = {"robust_fraction": robust_fraction}
        return {
            "badge": self._resolve_badge(robust=robust, overfit=overfit),
            "metrics": metrics,
        }

    # ------------------------------------------------------------------ #
    # Export helpers
    # ------------------------------------------------------------------ #
    def _export_wfa_results(
        self,
        folds: Sequence[Dict[str, Any]],
        summary: Dict[str, Any],
        out_dir: Path,
    ) -> None:
        rows = []
        alpha = self._as_float(summary.get("alpha"))
        if alpha <= 0:
            alpha = self._as_float(
                self._get_overfitting_value(("wfa", "alpha"), 0.5), 0.5
            )
        for fold in folds:
            train_metrics = fold["train_metrics"]
            test_metrics = fold["test_metrics"]
            train_sharpe = self._as_float(train_metrics.get("sharpe_ratio"))
            test_sharpe = self._as_float(test_metrics.get("sharpe_ratio"))
            is_bad_fold = (test_sharpe < 0.0) or (
                train_sharpe > 0.0 and test_sharpe < alpha * train_sharpe
            )
            rows.append(
                {
                    "fold": fold["fold"],
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "train_total_trades": train_metrics.get("total_trades"),
                    "test_total_trades": test_metrics.get("total_trades"),
                    "train_max_dd": train_metrics.get("max_drawdown"),
                    "test_max_dd": test_metrics.get("max_drawdown"),
                    "best_params": json.dumps(fold["best_params"]),
                    "is_bad_fold": is_bad_fold,
                }
            )

        df = pd.DataFrame(rows)
        csv_path = out_dir / "wfa_folds.csv"
        df.to_csv(csv_path, index=False)

        robustness = summary.get("robustness_summary", {}) or {}
        metrics = robustness.get("metrics", {}) or {}
        summary_df = pd.DataFrame(
            [
                {
                    "train_sharpe_mean": summary["train_sharpe_mean"],
                    "train_sharpe_std": summary["train_sharpe_std"],
                    "test_sharpe_mean": summary["test_sharpe_mean"],
                    "test_sharpe_std": summary["test_sharpe_std"],
                    "degradation_ratio": metrics.get("degradation_ratio"),
                    "test_vs_train_gap": metrics.get("test_vs_train_gap"),
                    "frac_test_sharpe_lt_0": metrics.get("frac_test_sharpe_lt_0"),
                    "frac_test_sharpe_lt_alpha_train": metrics.get(
                        "frac_test_sharpe_lt_alpha_train"
                    ),
                    "robustness_label": summary.get("robustness_label"),
                    "wfa_robustness_label": summary.get("robustness_label"),
                }
            ]
        )
        summary_df.to_csv(out_dir / "wfa_summary.csv", index=False)

        report_path = out_dir / "wfa_report.html"
        try:
            overfitting_report.render_wfa_report(
                summary_df=summary_df,
                folds_df=df,
                output_path=report_path,
            )
        except Exception:  # pragma: no cover - fallback HTML simple
            logger.debug("Plotly/Jinja indisponible, fallback HTML basique.", exc_info=True)
            html_content = self._build_html_report(
                title="Walk-Forward Analysis",
                summary_table=summary_df,
                details_table=df,
            )
            report_path.write_text(html_content, encoding="utf-8")

        try:
            relative_path = report_path.relative_to(self.output_root)
        except ValueError:
            relative_path = Path(report_path.name)
        self._register_report_section(
            name="Walk-Forward Analysis",
            description="Résultats train/test par fold et métriques agrégées.",
            relative_path=str(relative_path).replace("\\", "/"),
            badge=robustness.get("badge"),
        )

    def _export_oos_results(
        self,
        windows: Sequence[Dict[str, Any]],
        summary: Dict[str, Any],
        out_dir: Path,
    ) -> None:
        rows = []
        for entry in windows:
            metrics = entry["metrics"]
            rows.append(
                {
                    "window_id": entry["window_id"],
                    "start": entry["start"],
                    "end": entry["end"],
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "total_trades": metrics.get("total_trades"),
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "oos_results.csv", index=False)

        robustness = summary.get("robustness_summary", {}) or {}
        metrics = robustness.get("metrics", {}) or {}
        summary_df = pd.DataFrame(
            [
                {
                    "oos_sharpe_mean": summary["oos_sharpe_mean"],
                    "oos_sharpe_std": summary["oos_sharpe_std"],
                    "frac_oos_sharpe_lt_0": metrics.get("frac_oos_sharpe_lt_0"),
                    "oos_sharpe_min": metrics.get("oos_sharpe_min"),
                    "oos_sharpe_median": metrics.get("oos_sharpe_median"),
                    "oos_degradation_ratio": summary.get("oos_degradation_ratio"),
                    "train_sharpe_reference": summary.get("train_sharpe_reference"),
                    "robustness_label": summary.get("robustness_label"),
                    "oos_robustness_label": summary.get("robustness_label"),
                }
            ]
        )
        summary_df.to_csv(out_dir / "oos_summary.csv", index=False)

        html_path = out_dir / "oos_report.html"
        try:
            overfitting_report.render_oos_report(
                summary_df=summary_df,
                windows_df=df,
                output_path=html_path,
            )
        except Exception:  # pragma: no cover - fallback HTML simple
            logger.debug("Plotly/Jinja indisponible pour OOS, fallback HTML.", exc_info=True)
            html_content = self._build_html_report(
                title="Out-of-Sample Testing",
                summary_table=summary_df,
                details_table=df,
            )
            html_path.write_text(html_content, encoding="utf-8")

        try:
            relative_path = html_path.relative_to(self.output_root)
        except ValueError:
            relative_path = Path(html_path.name)
        self._register_report_section(
            name="Out-of-Sample",
            description="Fenêtres OOS individuelles et métriques synthétiques.",
            relative_path=str(relative_path).replace("\\", "/"),
            badge=robustness.get("badge"),
        )

    def _export_monte_carlo(
        self,
        simulations: Sequence[Dict[str, float]],
        summary: Dict[str, Any],
        out_dir: Path,
        source: str,
    ) -> None:
        df = pd.DataFrame(simulations)
        df.to_csv(out_dir / "monte_carlo_simulations.csv", index=False)
        robustness = summary.get("robustness_summary", {}) or {}
        summary_row = {
            key: value for key, value in summary.items() if key != "robustness_summary"
        }
        summary_row["robustness_label"] = summary.get("robustness_label")
        summary_row["monte_carlo_robustness_label"] = summary.get("robustness_label")
        for key, value in (robustness.get("metrics") or {}).items():
            summary_row.setdefault(key, value)
        summary_df = pd.DataFrame([summary_row])
        summary_df.to_csv(out_dir / "monte_carlo_summary.csv", index=False)

        report_path = out_dir / "monte_carlo_report.html"
        try:
            overfitting_report.render_monte_carlo_report(
                summary_df=summary_df,
                simulations_df=df,
                output_path=report_path,
                title=f"Monte Carlo ({source})",
            )
        except Exception:  # pragma: no cover - fallback HTML simple
            logger.debug(
                "Plotly/Jinja indisponible pour Monte Carlo, fallback HTML.", exc_info=True
            )
            html_content = self._build_html_report(
                title=f"Monte Carlo ({source})",
                summary_table=summary_df,
                details_table=df,
            )
            report_path.write_text(html_content, encoding="utf-8")
        try:
            relative_path = report_path.relative_to(self.output_root)
        except ValueError:
            relative_path = Path(report_path.name)
        self._register_report_section(
            name=f"Monte Carlo ({source})",
            description="Distribution simulée des métriques de performance.",
            relative_path=str(relative_path).replace("\\", "/"),
            badge=robustness.get("badge"),
        )

    def _export_stability(
        self,
        neighbors: Sequence[Dict[str, Any]],
        summary: Dict[str, Any],
        out_dir: Path,
    ) -> None:
        df = pd.DataFrame(neighbors)
        df.drop(columns=["metrics"], inplace=True, errors="ignore")
        df.to_csv(out_dir / "stability_neighbors.csv", index=False)

        robustness = summary.get("robustness_summary", {}) or {}
        summary_df = pd.DataFrame(
            [
                {
                    "base_sharpe": summary["base_metric"],
                    "robust_fraction": summary["robust_fraction"],
                    "robustness_label": summary.get("robustness_label"),
                    "stability_robustness_label": summary.get("robustness_label"),
                }
            ]
        )
        summary_df.to_csv(out_dir / "stability_summary.csv", index=False)

        report_path = out_dir / "stability_report.html"
        try:
            overfitting_report.render_stability_report(
                summary_df=summary_df,
                neighbors_df=df,
                output_path=report_path,
            )
        except Exception:  # pragma: no cover - fallback HTML simple
            logger.debug(
                "Plotly/Jinja indisponible pour la stabilité, fallback HTML.", exc_info=True
            )
            html_content = self._build_html_report(
                title="Stability Tests",
                summary_table=summary_df,
                details_table=df,
            )
            report_path.write_text(html_content, encoding="utf-8")

        try:
            relative_path = report_path.relative_to(self.output_root)
        except ValueError:
            relative_path = Path(report_path.name)
        self._register_report_section(
            name="Stability",
            description="Sensibilité locale des hyperparamètres.",
            relative_path=str(relative_path).replace("\\", "/"),
            badge=robustness.get("badge"),
        )

    def _build_html_report(
        self,
        *,
        title: str,
        summary_table: pd.DataFrame,
        details_table: pd.DataFrame,
    ) -> str:
        summary_html = summary_table.to_html(index=False, classes="table table-summary")
        details_html = details_table.to_html(index=False, classes="table table-details")

        return f"""
<!doctype html>
<html lang="fr">
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
      body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
      h1 {{ margin-bottom: 12px; }}
      .table-summary {{ margin-bottom: 24px; border-collapse: collapse; width: 100%; }}
      .table-details {{ border-collapse: collapse; width: 100%; }}
      table, th, td {{ border: 1px solid #ccc; padding: 6px 8px; }}
      th {{ background-color: #f4f5f7; text-align: left; }}
    </style>
  </head>
  <body>
    <h1>{title}</h1>
    <section>
      <h2>Résumé</h2>
      {summary_html}
    </section>
    <section>
      <h2>Détails</h2>
      {details_html}
    </section>
  </body>
</html>
""".strip()


__all__ = ["OverfittingChecker", "BacktestResult"]
