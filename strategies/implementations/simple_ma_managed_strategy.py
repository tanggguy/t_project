# --- 1. Bibliothèques natives ---
import logging

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from strategies.managed_strategy import ManagedStrategy


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
        self.sma_fast = bt.indicators.SMA(
            self.data.close, period=self.p.fast_period
        )
        self.sma_slow = bt.indicators.SMA(
            self.data.close, period=self.p.slow_period
        )

        # Indicateur de croisement
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

        self.log(
            f"SimpleMaManagedStrategy initialisée - "
            f"Fast MA: {self.p.fast_period}, Slow MA: {self.p.slow_period}",
            logging.INFO,
        )

    def next_custom(self) -> None:
        """
        Logique d'entrée de la stratégie.

        Le risk management (SL/TP) est géré automatiquement par ManagedStrategy.
        Cette méthode se concentre uniquement sur les signaux d'entrée.
        """
        # Éviter d'entrer si on attend que l'ATR soit prêt
        if self.atr and len(self.atr) < self.p.atr_period:
            return

        # Signal d'achat : Golden Cross
        if self.crossover[0] > 0:
            self.log(
                f"🚀 Signal ACHAT - Golden Cross "
                f"(Fast: {self.sma_fast[0]:.2f}, Slow: {self.sma_slow[0]:.2f})",
                level=logging.INFO,
            )
            self.buy()
