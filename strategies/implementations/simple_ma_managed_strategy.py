# --- 1. Bibliothèques natives ---
import logging

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from strategies.managed_strategy import ManagedStrategy
from utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/strategies/managed_strategy")


class SimpleMaManagedStrategy(ManagedStrategy):
    """
    Stratégie simple de croisement de moyennes mobiles avec risk management.

    Cette stratégie hérite de ManagedStrategy et bénéficie donc automatiquement
    de la gestion des stop loss et take profit.

    Signaux:
    - ACHAT : Golden Cross (SMA rapide croise au-dessus de SMA lente)
    - VENTE : Gérée automatiquement par les SL/TP de ManagedStrategy

    Le risk management (type de SL, type de TP, etc.) est configurable
    via les paramètres au moment du backtest.

    Example:
        >>> # Dans run_backtest.py
        >>> engine.add_strategy(
        ...     SimpleMaManagedStrategy,
        ...     fast_period=10,
        ...     slow_period=30,
        ...     stop_loss_type='atr',
        ...     take_profit_type='fixed'
        ... )
    """

    params = (
        # --- Paramètres de la stratégie ---
        ("fast_period", 10),
        ("slow_period", 30),
        # --- Paramètres de risk management (hérités de ManagedStrategy) ---
        # Peuvent être surchargés ici ou dans le backtest
    )

    def __init__(self) -> None:
        """Initialise les indicateurs spécifiques à la stratégie."""
        # Appeler l'init de ManagedStrategy (qui gère le risk management)
        super().__init__()

        # --- Indicateurs de la stratégie ---
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow_period)

        # Indicateur de croisement
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

        logger.info(
            f"SimpleMaManagedStrategy initialisée - "
            f"Fast MA: {self.p.fast_period}, Slow MA: {self.p.slow_period}"
        )

    def next_custom(self) -> None:
        """
        Logique d'entrée de la stratégie.

        Le risk management (SL/TP) est géré automatiquement par ManagedStrategy.
        Cette méthode se concentre uniquement sur les signaux d'entrée.
        """
        # Warmup pour les SMA
        min_period = max(self.p.fast_period, self.p.slow_period)

        # Si l'ATR existe, l'attendre aussi
        if self.atr is not None:
            min_period = max(min_period, self.p.atr_period)

        if len(self) < min_period:
            return

        # Signal d'achat : Golden Cross
        if self.crossover[0] > 0:

            logger.info(
                f"Signal ACHAT - Golden Cross prix {self.data_close[0]:.2f} "
                f"(Fast: {self.sma_fast[0]:.2f}, Slow: {self.sma_slow[0]:.2f})"
            )
            self.buy()
