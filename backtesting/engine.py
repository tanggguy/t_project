# --- 1. Bibliothèques natives ---
import logging
from typing import Dict, Any, Type, List, Optional

# --- 2. Bibliothèques tierces ---
import backtrader as bt
import pandas as pd

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger
from utils.config_loader import get_settings
from strategies.base_strategy import BaseStrategy  # Pour le type hinting
from utils.data_processor import resample_data

# Initialisation du logger pour ce module
logger = setup_logger(__name__)


class BacktestEngine:
    """
    Classe de gestion du moteur de backtest (Cerebro).

    Cette classe encapsule la configuration de Backtrader (Cerebro),
    le chargement des données, l'ajout de stratégies, l'ajout de sizers
    et l'exécution du backtest.
    """

    def __init__(self) -> None:
        """
        Initialise le moteur de backtest (Cerebro) et charge la configuration.
        """
        logger.info("Initialisation du BacktestEngine...")

        # 1. Charger la configuration globale
        try:
            settings: Dict[str, Any] = get_settings()
            self.backtest_config: Dict[str, Any] = settings.get("backtest", {})
            self.broker_config: Dict[str, Any] = settings.get("broker", {})

            logger.debug("Configuration settings.yaml chargée.")
        except Exception as e:
            logger.error(f"Erreur fatale lors du chargement de la configuration: {e}")
            raise

        # 2. Initialiser Cerebro
        self.cerebro: bt.Cerebro = bt.Cerebro()

        # 3. Configurer le broker (Capital, Commissions, Slippage)
        self._setup_broker()

    def _setup_broker(self) -> None:
        """
        Configure le broker de Cerebro avec les paramètres
        du fichier settings.yaml.
        """
        # --- Capital Initial ---
        initial_capital: float = self.backtest_config.get("initial_capital", 10000.0)
        self.cerebro.broker.setcash(initial_capital)
        logger.info(f"Capital initial du broker fixé à : {initial_capital:,.2f}")

        # --- Commissions ---
        comm_type: str = self.broker_config.get("commission_type", "percentage")

        if comm_type == "percentage":
            comm_val: float = self.broker_config.get("commission_pct", 0.001)
            self.cerebro.broker.setcommission(commission=comm_val)
            logger.info(f"Commission (pourcentage) fixée à : {comm_val:.4%}")

        elif comm_type == "fixed":
            comm_val: float = self.broker_config.get("commission_fixed", 0.0)
            self.cerebro.broker.setcommission(commission=comm_val)
            logger.info(f"Commission (fixe) fixée à : {comm_val:.2f}")
        else:
            logger.warning(
                f"Type de commission '{comm_type}' non reconnu. "
                "Aucune commission appliquée."
            )

        # --- Slippage ---
        slippage: float = self.broker_config.get("slippage_pct", 0.0)
        if slippage > 0:
            self.cerebro.broker.set_slippage_perc(perc=slippage)
            logger.info(f"Slippage (pourcentage) fixé à : {slippage:.4%}")

    def _setup_analyzers(self) -> None:
        """
        Ajoute les analyseurs de performance de base à Cerebro.

        """
        logger.debug("Configuration des analyseurs...")
        # Analyseur Sharpe Ratio
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

        # Analyseur des Retours agrégés (annualisés / cumulés)
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        # Analyseur DrawDown
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

        # Analyseur des Trades (pour stats win/loss, etc.)
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        # Série de rendements par période (journalière par défaut)
        try:
            self.cerebro.addanalyzer(
                bt.analyzers.TimeReturn,
                timeframe=bt.TimeFrame.Days,
                _name="timereturns",
            )
        except Exception:
            logger.warning("Impossible d'ajouter TimeReturn analyzer (Backtrader)")

        # Liste détaillée des trades (analyzer custom si disponible)
        try:
            from backtesting.analyzers.performance import TradeListAnalyzer  # type: ignore

            self.cerebro.addanalyzer(TradeListAnalyzer, _name="tradelist")
        except Exception:
            logger.debug("TradeListAnalyzer indisponible ou non importable.")

    def add_data(self, df: pd.DataFrame, name: str = "data0") -> None:
        """
        Ajoute un DataFrame pandas comme flux de données à Cerebro.

        Args:
            df (pd.DataFrame): Le DataFrame OHLCV (avec indicateurs).
                               Doit avoir un index Datetime.
            name (str): Le nom interne du flux de données.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(
                "Le DataFrame n'a pas de DatetimeIndex. "
                "Conversion impossible en flux Backtrader."
            )
            raise ValueError("L'index du DataFrame doit être de type DatetimeIndex")

        # Note: DataManager s'assure que les colonnes sont en minuscules
        # (open, high, low, close, volume), ce qui correspond
        # aux noms attendus par PandasData.
        data_feed: bt.feeds.PandasData = bt.feeds.PandasData(dataname=df)

        self.cerebro.adddata(data_feed, name=name)
        logger.info(
            f"Flux de données '{name}' ajouté. "
            f"Période: {df.index.min().date()} à {df.index.max().date()}."
        )

    def add_resampled_data(self, df: pd.DataFrame, rule: str, name: str) -> None:
        """
        Re-échantillonne un DataFrame OHLCV via utils.data_processor.resample_data
        puis l'ajoute comme flux de données Backtrader.

        Args:
            df (pd.DataFrame): DataFrame source (index DatetimeIndex)
            rule (str): Règle pandas (ex: '4H', '1D', '1W')
            name (str): Nom du flux ajouté dans Cerebro
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(
                "Le DataFrame n'a pas de DatetimeIndex. Resampling impossible."
            )
            raise ValueError("L'index du DataFrame doit être de type DatetimeIndex")

        df_resampled = resample_data(df, rule=rule)
        if df_resampled is None or df_resampled.empty:
            logger.error(
                f"Resampling '{rule}' a retourné un DataFrame vide pour '{name}'."
            )
            return
        self.add_data(df_resampled, name=name)
        logger.info(f"Flux re-échantillonné '{name}' ajouté (rule={rule}).")

    def add_strategy(self, strategy_class: Type[BaseStrategy], **kwargs: Any) -> None:
        """
        Ajoute une classe de stratégie à Cerebro.

        Args:
            strategy_class (Type[BaseStrategy]): La classe de la stratégie
                                                 (ex: MaCrossoverStrategy).
            **kwargs: Les paramètres à passer à la stratégie
                      (ex: fast_period=10).
        """
        self.cerebro.addstrategy(strategy_class, **kwargs)
        params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(
            f"Stratégie '{strategy_class.__name__}' ajoutée. "
            f"Paramètres: ({params_str})"
        )

    def add_sizer(self, sizer_class: Type[bt.Sizer], **kwargs: Any) -> None:
        """
        Ajoute un sizer (gestionnaire de taille de position) à Cerebro.

        Le sizer contrôle la taille des positions pour chaque ordre.
        Un seul sizer peut être actif à la fois.

        Args:
            sizer_class (Type[bt.Sizer]): La classe du sizer à utiliser
                                          (ex: FixedSizer, VolatilityBasedSizer).
            **kwargs: Les paramètres à passer au sizer
                      (ex: stake=10, pct_size=0.95).

        Example:
            >>> from position_sizing.sizers import FixedSizer
            >>> engine = BacktestEngine()
            >>> engine.add_sizer(FixedSizer, pct_size=0.50)  # Investir 50% du capital
        """
        self.cerebro.addsizer(sizer_class, **kwargs)
        params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(
            f"Position Sizer '{sizer_class.__name__}' ajouté. "
            f"Paramètres: ({params_str})"
        )

    def run(self) -> List[bt.Strategy]:
        """
        Exécute le backtest et retourne les résultats.

        Returns:
            List[bt.Strategy]: Une liste contenant l'instance de la
                               stratégie exécutée (avec les analyseurs
                               remplis).
        """
        # Configurer les analyseurs juste avant l'exécution
        self._setup_analyzers()

        logger.info("--- DÉMARRAGE DU BACKTEST ---")

        # Exécuter Cerebro
        try:
            results: List[bt.Strategy] = self.cerebro.run()
            logger.info("--- BACKTEST TERMINÉ ---")
            return results

        except Exception as e:
            logger.critical(f"Erreur critique durant l'exécution de Cerebro: {e}")
            raise

    def plot(self, style: str = "candlestick") -> None:
        """
        Affiche le graphique de résultats de Backtrader.
        (Nécessite matplotlib installé et un backend graphique)

        Args:
            style (str): Style du graphique (par défaut 'candlestick').
        """
        logger.info("Génération du graphique de backtest...")
        try:
            # S'assurer que 'matplotlib' est bien importé si on l'utilise
            import matplotlib

            # Vous pouvez ajuster les paramètres de plot ici si nécessaire
            self.cerebro.plot(
                style=style,
                iplot=False,  # Utiliser matplotlib (True pour bokeh si installé)
                volume=True,
            )
        except ImportError:
            logger.error(
                "Matplotlib non trouvé. " "Impossible de générer le graphique."
            )
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique: {e}")
