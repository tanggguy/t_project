# --- 1. Bibliothèques natives ---
import logging

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from strategies.base_strategy import BaseStrategy


class MaCrossoverStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur un croisement de moyennes mobiles (MA Crossover).

    - **Signal d'Achat (Golden Cross)**:
        Lorsque la moyenne mobile rapide (fast_ma) croise au-dessus
        de la moyenne mobile lente (slow_ma).
    - **Signal de Vente (Death Cross)**:
        Lorsque la moyenne mobile rapide (fast_ma) croise en dessous
        de la moyenne mobile lente (slow_ma).
    """

    # Définition des paramètres de la stratégie
    params = (
        ("fast_period", 10),
        ("slow_period", 30),
    )

    def __init__(self) -> None:
        """
        Initialise la stratégie et ses indicateurs.
        """
        super().__init__()

        # --- Définition des Indicateurs ---
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.slow_period
        )

        # --- CORRECTION ---
        # Au lieu d'utiliser bt.indicators.CrossOver, nous utilisons
        # une simple soustraction. C'est plus robuste et évite le bug
        # interne de backtrader.
        # Si fast_ma > slow_ma, la ligne 'crossover_line' sera > 0.
        self.crossover_line = self.fast_ma - self.slow_ma

        self.log(
            f"Stratégie initialisée. Fast MA: {self.params.fast_period}, "
            f"Slow MA: {self.params.slow_period}"
        )

    def next(self) -> None:
        """
        Définit la logique de trading à chaque nouvelle bougie.
        """
        # 1. Gardien : Vérifier si un ordre est déjà en cours
        if self.order:
            return  # Ne rien faire si un ordre est en attente

        # 2. Vérifier si nous sommes déjà en position
        if not self.position:
            # Nous ne sommes pas en position, chercher un signal d'achat

            # --- CORRECTION DE LA LOGIQUE D'ACHAT ---
            # Signal d'Achat (Golden Cross):
            # La ligne de crossover était NÉGATIVE hier (fast < slow)
            # ET est POSITIVE aujourd'hui (fast > slow).
            if self.crossover_line[-1] < 0 and self.crossover_line[0] > 0:
                self.log(
                    f"SIGNAL D'ACHAT (Golden Cross) à {self.data_close[0]:.2f}",
                    level=logging.INFO,
                )
                self.order = self.order_target_percent(target=0.95)

        else:
            # Nous sommes en position, chercher un signal de vente

            # --- CORRECTION DE LA LOGIQUE DE VENTE ---
            # Signal de Vente (Death Cross):
            # La ligne de crossover était POSITIVE hier (fast > slow)
            # ET est NÉGATIVE aujourd'hui (fast < slow).
            if self.crossover_line[-1] > 0 and self.crossover_line[0] < 0:
                self.log(
                    f"SIGNAL DE VENTE (Death Cross) à {self.data_close[0]:.2f}",
                    level=logging.INFO,
                )
                self.order = self.order_target_percent(target=0.0)
