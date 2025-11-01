# --- 1. Bibliothèques natives ---
import logging
from typing import Dict, List, Type, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

# --- 2. Bibliothèques tierces ---
import pandas as pd
import backtrader as bt

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger
from utils.data_manager import DataManager
from backtesting.engine import BacktestEngine
from strategies.base_strategy import BaseStrategy

logger = setup_logger(__name__)


class StrategyLab:
    """
    Laboratoire de développement de stratégies de trading.

    Cette classe fournit une interface simplifiée pour :
    1. Tester des indicateurs techniques
    2. Backtester plusieurs stratégies
    3. Comparer les performances
    4. Exporter les résultats automatiquement

    Example:
        >>> lab = StrategyLab(ticker='AAPL', period='2y')
        >>> lab.add_strategy(MaCrossoverStrategy, fast=10, slow=30)
        >>> lab.add_strategy(RsiStrategy, period=14)
        >>> results = lab.run_all()
        >>> lab.compare()
    """

    def __init__(
        self,
        ticker: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        period: str = "2y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        auto_export: bool = True,
        export_dir: str = "results/strategy_lab",
    ):
        """
        Initialise le laboratoire de stratégies.

        Args:
            ticker: Symbole du ticker (ex: 'AAPL')
            data: DataFrame OHLCV (si déjà chargé)
            period: Période de données (ex: '1y', '2y', '5y')
            start_date: Date de début (format 'YYYY-MM-DD')
            end_date: Date de fin (format 'YYYY-MM-DD')
            initial_capital: Capital initial pour les backtests
            commission: Commission par trade (0.001 = 0.1%)
            auto_export: Export automatique des résultats en Markdown
            export_dir: Répertoire d'export des résultats
        """
        self.ticker = ticker
        self.period = period
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = commission
        self.auto_export = auto_export
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Charger les données
        if data is not None:
            self.data = data
            logger.info(f"Données fournies : {len(data)} bougies")
        elif ticker:
            self._load_data()
        else:
            raise ValueError("Fournir soit 'ticker' soit 'data'")

        # Structures internes
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Any] = {}

        # Benchmark (Buy & Hold) - calculé automatiquement
        self._benchmark_return = self._calculate_buy_hold_return()

        logger.info(
            f"🔬 StrategyLab initialisé - Ticker: {self.ticker}, "
            f"Bougies: {len(self.data)}, "
            f"Période: {self.data.index.min().date()} à {self.data.index.max().date()}"
        )

    def _load_data(self) -> None:
        """
        Charge les données depuis yfinance via DataManager.
        """
        logger.info(f"Chargement des données pour {self.ticker}...")

        data_manager = DataManager()

        if self.period:
            self.data = data_manager.get_data(ticker=self.ticker, period=self.period)
        elif self.start_date and self.end_date:
            self.data = data_manager.get_data(
                ticker=self.ticker, start_date=self.start_date, end_date=self.end_date
            )
        else:
            raise ValueError("Spécifier 'period' OU 'start_date' + 'end_date'")

        if self.data is None or self.data.empty:
            raise ValueError(f"Impossible de charger les données pour {self.ticker}")

        logger.info(
            f"Données chargées : {len(self.data)} bougies "
            f"({self.data.index.min().date()} à {self.data.index.max().date()})"
        )

    def _calculate_buy_hold_return(self) -> float:
        """
        Calcule le rendement du Buy & Hold.

        Returns:
            float: Rendement en pourcentage
        """
        if self.data is None or len(self.data) == 0:
            return 0.0

        first_price = self.data["close"].iloc[0]
        last_price = self.data["close"].iloc[-1]

        buy_hold_return = ((last_price - first_price) / first_price) * 100

        logger.debug(f"Buy & Hold return: {buy_hold_return:.2f}%")
        return buy_hold_return

    def add_strategy(
        self, strategy_class: Type[BaseStrategy], name: Optional[str] = None, **params
    ) -> None:
        """
        Ajoute une stratégie à tester.

        Args:
            strategy_class: Classe de la stratégie
            name: Nom personnalisé (auto-généré si None)
            **params: Paramètres de la stratégie
        """
        # Générer un nom automatiquement si non fourni
        if name is None:
            # Retirer "Strategy" du nom de classe
            base_name = strategy_class.__name__.replace("Strategy", "")

            # Ajouter suffixe si nom existe déjà
            counter = 1
            name = base_name
            while name in self.strategies:
                name = f"{base_name}_{counter}"
                counter += 1

        # Stocker la configuration
        self.strategies[name] = {
            "class": strategy_class,
            "params": params,
            "status": "pending",
        }

        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        logger.info(
            f"✅ Stratégie ajoutée : {name} ({strategy_class.__name__}) "
            f"avec params: {params_str or 'defaults'}"
        )

    def run_strategy(
        self, name: str, verbose: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Exécute le backtest d'une stratégie spécifique.

        Args:
            name: Nom de la stratégie
            verbose: Afficher les logs détaillés

        Returns:
            Dictionnaire avec les résultats ou None si erreur
        """
        if name not in self.strategies:
            logger.error(f"Stratégie '{name}' non trouvée")
            return None

        strategy_config = self.strategies[name]
        strategy_class = strategy_config["class"]
        params = strategy_config["params"]

        logger.info(f"🚀 Lancement du backtest : {name}...")

        try:
            # Initialiser le moteur de backtest
            engine = BacktestEngine()

            # Configuration du broker
            engine.cerebro.broker.setcash(self.initial_capital)
            engine.cerebro.broker.setcommission(commission=self.commission)

            # Ajouter les données
            engine.add_data(self.data.copy())

            # Ajouter la stratégie
            engine.add_strategy(strategy_class, **params)

            # Exécuter
            backtest_results = engine.run()

            if not backtest_results:
                logger.error(f"Aucun résultat retourné pour {name}")
                return None

            strat = backtest_results[0]

            # Extraction des métriques
            metrics = self._extract_metrics(strat, name)

            # Marquer comme complété
            self.strategies[name]["status"] = "completed"
            self.results[name] = metrics

            logger.info(f"✅ Backtest terminé : {name}")

            if verbose:
                self._print_strategy_summary(name, metrics)

            return metrics

        except Exception as e:
            logger.error(f"❌ Erreur lors du backtest de {name}: {e}", exc_info=True)
            self.strategies[name]["status"] = "failed"
            return None

    def _extract_metrics(self, strat: bt.Strategy, name: str) -> Dict[str, Any]:
        """
        Extrait les métriques d'une stratégie exécutée.

        Args:
            strat: Instance de la stratégie Backtrader
            name: Nom de la stratégie

        Returns:
            Dictionnaire avec toutes les métriques
        """
        try:
            # Analyseurs
            trades_analyzer = strat.analyzers.trades.get_analysis()
            sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
            drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
            returns_analyzer = strat.analyzers.returns.get_analysis()

            # Valeurs finales
            final_value = strat.broker.getvalue()
            pnl = final_value - self.initial_capital
            pnl_pct = (pnl / self.initial_capital) * 100

            # Trades
            total_trades = trades_analyzer.get("total", {}).get("total", 0)
            won_trades = trades_analyzer.get("won", {}).get("total", 0)
            lost_trades = trades_analyzer.get("lost", {}).get("total", 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

            # Gains/Pertes moyens
            avg_win = trades_analyzer.get("won", {}).get("pnl", {}).get("average", 0)
            avg_loss = trades_analyzer.get("lost", {}).get("pnl", {}).get("average", 0)
            best_trade = trades_analyzer.get("won", {}).get("pnl", {}).get("max", 0)
            worst_trade = trades_analyzer.get("lost", {}).get("pnl", {}).get("max", 0)

            # Profit factor
            total_won = trades_analyzer.get("won", {}).get("pnl", {}).get("total", 0)
            total_lost = abs(
                trades_analyzer.get("lost", {}).get("pnl", {}).get("total", 0)
            )
            profit_factor = (total_won / total_lost) if total_lost > 0 else float("inf")

            # Ratios
            sharpe_ratio = sharpe_analyzer.get("sharperatio")
            if sharpe_ratio is None:
                sharpe_ratio = 0.0

            # Drawdown
            max_dd = drawdown_analyzer.get("max", {}).get("drawdown", 0)
            max_dd_duration = drawdown_analyzer.get("max", {}).get("len", 0)

            # Returns
            total_return = returns_analyzer.get("rtot", 0) * 100
            avg_annual_return = returns_analyzer.get("ravg", 0) * 100

            metrics = {
                "name": name,
                "initial_capital": self.initial_capital,
                "final_value": final_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "total_return": total_return,
                "annual_return": avg_annual_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_dd,
                "max_dd_duration": max_dd_duration,
                "total_trades": total_trades,
                "won_trades": won_trades,
                "lost_trades": lost_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
            }

            return metrics

        except Exception as e:
            logger.error(f"Erreur extraction métriques pour {name}: {e}")
            return {}

    def run_all(self, show_progress: bool = True) -> Dict[str, Any]:
        """
        Exécute tous les backtests et retourne les résultats.

        Args:
            show_progress: Afficher la progression

        Returns:
            Dictionnaire avec tous les résultats
        """
        total_strategies = len(self.strategies)
        logger.info(f"🎯 Lancement de {total_strategies} backtests...")

        for idx, name in enumerate(self.strategies.keys(), 1):
            if show_progress:
                logger.info(f"[{idx}/{total_strategies}] Backtest : {name}")

            self.run_strategy(name, verbose=False)

        # Compter les succès
        completed = sum(
            1 for s in self.strategies.values() if s["status"] == "completed"
        )
        failed = sum(1 for s in self.strategies.values() if s["status"] == "failed")

        logger.info(f"🏁 Backtests terminés : {completed} réussis, {failed} échoués")

        return self.results

    def compare(
        self,
        metrics: Optional[List[str]] = None,
        sort_by: str = "sharpe_ratio",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Compare toutes les stratégies testées.

        Args:
            metrics: Liste des métriques à comparer (None = toutes)
            sort_by: Métrique pour le tri
            ascending: Ordre croissant

        Returns:
            DataFrame de comparaison
        """
        if not self.results:
            logger.warning("Aucun résultat à comparer. Lancer run_all() d'abord.")
            return pd.DataFrame()

        # Métriques par défaut
        if metrics is None:
            metrics = [
                "sharpe_ratio",
                "total_return",
                "annual_return",
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "total_trades",
            ]

        # Construire le DataFrame
        comparison_data = []
        for name, result in self.results.items():
            row = {"strategy": name}
            for metric in metrics:
                row[metric] = result.get(metric, None)
            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)

        # Ajouter le benchmark
        benchmark_row = {
            "strategy": "Buy & Hold",
            "total_return": self._benchmark_return,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "win_rate": None,
            "profit_factor": None,
            "total_trades": 1,
            "annual_return": None,
        }
        df_comparison = pd.concat(
            [df_comparison, pd.DataFrame([benchmark_row])], ignore_index=True
        )

        # Trier
        if sort_by in df_comparison.columns:
            df_comparison = df_comparison.sort_values(
                by=sort_by, ascending=ascending
            ).reset_index(drop=True)

        # Logging
        logger.info(f"📊 Comparaison de {len(self.results)} stratégies")
        logger.info(
            f"🏆 Meilleure stratégie ({sort_by}): "
            f"{df_comparison.iloc[0]['strategy']} = "
            f"{df_comparison.iloc[0][sort_by]}"
        )

        return df_comparison

    def _print_strategy_summary(self, name: str, metrics: Dict[str, Any]) -> None:
        """
        Affiche un résumé d'une stratégie.

        Args:
            name: Nom de la stratégie
            metrics: Dictionnaire des métriques
        """
        print(f"\n{'=' * 70}")
        print(f"RÉSUMÉ : {name}")
        print(f"{'=' * 70}")
        print(f"Return Total:     {metrics['total_return']:>12.2f}%")
        print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>12.2f}")
        print(f"Max Drawdown:     {metrics['max_drawdown']:>12.2f}%")
        print(f"Win Rate:         {metrics['win_rate']:>12.2f}%")
        print(f"Total Trades:     {metrics['total_trades']:>12}")
        print(f"{'=' * 70}\n")

    def get_best_strategy(
        self, metric: str = "sharpe_ratio"
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retourne la meilleure stratégie selon une métrique.

        Args:
            metric: Métrique à maximiser

        Returns:
            Tuple (nom, métriques) ou None si aucun résultat
        """
        if not self.results:
            logger.warning("Aucun résultat disponible")
            return None

        best_name = max(self.results, key=lambda x: self.results[x].get(metric, -1e9))
        best_metrics = self.results[best_name]

        logger.info(
            f"🏆 Meilleure stratégie ({metric}): {best_name} = "
            f"{best_metrics[metric]:.2f}"
        )

        return best_name, best_metrics

    def summary(self) -> None:
        """
        Affiche un résumé global de tous les backtests.
        """
        if not self.results:
            print("Aucun backtest exécuté.")
            return

        print(f"\n{'=' * 80}")
        print(f"RÉSUMÉ GLOBAL - {self.ticker} ({len(self.data)} bougies)")
        print(f"{'=' * 80}")
        print(
            f"Période: {self.data.index.min().date()} à {self.data.index.max().date()}"
        )
        print(f"Capital Initial: {self.initial_capital:,.2f} €")
        print(f"Buy & Hold Return: {self._benchmark_return:.2f}%")
        print(f"\nStratégies testées: {len(self.results)}")

        comparison_df = self.compare()
        print(f"\n{comparison_df.to_string()}")

        print(f"\n{'=' * 80}\n")
