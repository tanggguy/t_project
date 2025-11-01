# --- 1. Bibliothèques natives ---
import logging
from typing import Optional

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from strategies.base_strategy import BaseStrategy
from risk_management import FixedStopLoss


class MaCrossoverStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur un croisement de moyennes mobiles (MA Crossover)
    avec stop loss fixe.

    - **Signal d'Achat (Golden Cross)**:
        Lorsque la moyenne mobile rapide (fast_ma) croise au-dessus
        de la moyenne mobile lente (slow_ma).
    - **Signal de Vente (Death Cross)**:
        Lorsque la moyenne mobile rapide (fast_ma) croise en dessous
        de la moyenne mobile lente (slow_ma).
    - **Stop Loss**:
        Stop loss fixe en pourcentage appliqué sur chaque position.
    """

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
        ("stop_pct", 0.02),  # Stop loss de 2% par défaut
    )

    def __init__(self) -> None:
        """Initialise la stratégie et ses indicateurs."""
        super().__init__()

        # Indicateurs de moyennes mobiles
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.slow_period
        )
        self.crossover_line = self.fast_ma - self.slow_ma

        # Gestionnaire de stop loss
        self.stop_manager = FixedStopLoss(stop_pct=self.params.stop_pct)
        self.entry_price: Optional[float] = None
        self.stop_level: Optional[float] = None

        self.log(
            f"Stratégie initialisée. Fast MA: {self.params.fast_period}, "
            f"Slow MA: {self.params.slow_period}, Stop Loss: {self.params.stop_pct*100:.1f}%"
        )

    def next(self) -> None:
        """Définit la logique de trading à chaque nouvelle bougie."""
        # Gardien : Vérifier si un ordre est déjà en cours
        if self.order:
            return

        # Si en position, vérifier d'abord le stop loss
        if self.position:
            self._check_stop_loss()

            # Si toujours en position après vérification du stop, vérifier signal de sortie
            if self.position:
                self._check_exit_signal()
        else:
            # Pas de position, chercher un signal d'entrée
            self._check_entry_signal()

    def _check_entry_signal(self) -> None:
        """Vérifie les signaux d'entrée (Golden Cross)."""
        # Signal d'Achat (Golden Cross)
        if self.crossover_line[-1] < 0 and self.crossover_line[0] > 0:
            self.entry_price = self.data_close[0]

            # Calculer le stop loss initial
            self.stop_level = self.stop_manager.calculate_stop(
                entry_price=self.entry_price, position_type="long"
            )

            self.log(
                f"SIGNAL D'ACHAT (Golden Cross) à {self.entry_price:.2f}, "
                f"Stop Loss @ {self.stop_level:.2f}",
                level=logging.INFO,
            )
            self.order = self.order_target_percent(target=0.95)

    def _check_exit_signal(self) -> None:
        """Vérifie les signaux de sortie (Death Cross)."""
        # Signal de Vente (Death Cross)
        if self.crossover_line[-1] > 0 and self.crossover_line[0] < 0:
            self.log(
                f"SIGNAL DE VENTE (Death Cross) à {self.data_close[0]:.2f}",
                level=logging.INFO,
            )
            self.order = self.order_target_percent(target=0.0)
            self._reset_stop_loss()

    def _check_stop_loss(self) -> None:
        """Vérifie si le stop loss doit être déclenché."""
        current_price = self.data_close[0]

        if self.stop_level is not None:
            if self.stop_manager.should_trigger(
                current_price=current_price,
                stop_level=self.stop_level,
                position_type="long",
            ):
                # self.log(
                #     f"STOP LOSS DÉCLENCHÉ à {current_price:.2f} "
                #     f"(Stop: {self.stop_level:.2f})",
                #     level=logging.WARNING,
                # )
                self.order = self.order_target_percent(target=0.0)
                self._reset_stop_loss()

    def _reset_stop_loss(self) -> None:
        """Réinitialise les variables du stop loss."""
        self.entry_price = None
        self.stop_level = None

    def notify_order(self, order: bt.Order) -> None:
        """Notification des ordres."""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"ACHAT EXÉCUTÉ @ {order.executed.price:.2f}, "
                    f"Coût: {order.executed.value:.2f}, "
                    f"Commission: {order.executed.comm:.2f}"
                )
            elif order.issell():
                self.log(
                    f"VENTE EXÉCUTÉE @ {order.executed.price:.2f}, "
                    f"Coût: {order.executed.value:.2f}, "
                    f"Commission: {order.executed.comm:.2f}"
                )

        # Appeler la méthode parente pour le reste de la gestion
        super().notify_order(order)
