# --- 1. Bibliothèques natives ---
import logging

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from strategies.managed_strategy import ManagedStrategy
from utils.logger import setup_logger


logger = setup_logger(__name__, log_file="logs/strategies/ema_trend")


class EmaTrendStrategy(ManagedStrategy):
    """
    Stratégie de suivi de tendance avec filtre EMA majeur et momentum EMA court terme.

    Idée clé:
    - Filtre de tendance: n'autoriser que les achats si Prix > EMA(ema_trend),
      n'autoriser que les ventes (short) si Prix < EMA(ema_trend).
    - Momentum: valider l'entrée via croisement EMA(ema_fast) vs EMA(ema_slow).

    Paramètres principaux:
        - ema_trend (int): Période EMA de tendance (ex: 200). Filtre directionnel.
        - ema_fast (int): EMA rapide pour le momentum (ex: 12).
        - ema_slow (int): EMA lente pour le momentum (ex: 26).

    Sorties (gérées par ManagedStrategy):
        - Stop Loss (ATR): stop_loss_type='atr', stop_loss_atr_mult=2.5
        - Take Profit (ATR): take_profit_type='atr', take_profit_atr_mult=5.0

        Ces valeurs par défaut visent ~R:R ≈ 1:2 avec l'ATR.
    """

    params = (
        # --- Entrées (EMA) ---
        ("ema_trend", 200),
        ("ema_fast", 12),
        ("ema_slow", 26),
        # --- Exits (ManagedStrategy) par défaut orientés ATR ---
        ("use_stop_loss", True),
        ("stop_loss_type", "atr"),
        ("stop_loss_atr_mult", 2.5),
        ("use_take_profit", True),
        ("take_profit_type", "atr"),
        ("take_profit_atr_mult", 5.0),
        ("atr_period", 14),
    )

    def __init__(self) -> None:
        super().__init__()

        # Indicateurs EMA
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.p.ema_trend)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)

        # Croisement de momentum (EMA rapide vs EMA lente)
        self.cross_momentum = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)

        logger.info(
            "EmaTrendStrategy initialisée - Trend EMA: %s, Fast EMA: %s, Slow EMA: %s | SL: %s(%.2f ATR), TP: %s(%.2f ATR)",
            self.p.ema_trend,
            self.p.ema_fast,
            self.p.ema_slow,
            self.p.stop_loss_type,
            self.p.stop_loss_atr_mult,
            self.p.take_profit_type,
            self.p.take_profit_atr_mult,
        )

    def next_custom(self) -> None:
        """Logique d'entrée: filtre de tendance + croisement EMA court terme."""
        # Warmup: attendre la plus longue période nécessaire
        min_period = max(self.p.ema_trend, self.p.ema_fast, self.p.ema_slow)
        if self.atr is not None:
            min_period = max(min_period, self.p.atr_period)
        if len(self) < min_period:
            return

        price = self.data.close[0]
        trend_up = price > self.ema_trend[0]
        trend_down = price < self.ema_trend[0]

        cross_val = self.cross_momentum[0]

        # Entrée LONG uniquement dans régime haussier + croisement haussier
        if trend_up and cross_val > 0:
            logger.info(
                "Signal LONG - Prix %.2f > EMA(%d)=%.2f et EMA_fast croise au-dessus EMA_slow (%.2f > %.2f)",
                price,
                int(self.p.ema_trend),
                float(self.ema_trend[0]),
                float(self.ema_fast[0]),
                float(self.ema_slow[0]),
            )
            self.buy()
            return

        # Entrée SHORT uniquement dans régime baissier + croisement baissier
        if trend_down and cross_val < 0:
            logger.info(
                "Signal SHORT - Prix %.2f < EMA(%d)=%.2f et EMA_fast croise sous EMA_slow (%.2f < %.2f)",
                price,
                int(self.p.ema_trend),
                float(self.ema_trend[0]),
                float(self.ema_fast[0]),
                float(self.ema_slow[0]),
            )
            self.sell()

