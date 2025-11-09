import backtrader as bt
from strategies.managed_strategy import ManagedStrategy
from utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/strategies/rsi_mean_reversion")


class RsiMeanReversionManagedStrategy(ManagedStrategy):
    """
    Mean reversion contrôlé: surachat/survente, filtre de régime pour éviter les tendances fortes.
    """

    params = (
        ("trend_long_period", 200),
        ("avoid_strong_trend", False),  # éviter si SMA50 >> SMA200
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_exit", 50),  # target reversion
        ("bb_period", 20),
        ("bb_dev", 2.0),
        ("use_invalidation", True),
        ("reentry_cooldown_bars", 5),
        ("atr_period", 14),
    )

    def __init__(self):
        super().__init__()
        self.sma_long = bt.indicators.SMA(
            self.data.close, period=self.p.trend_long_period
        )
        self.sma_mid = bt.indicators.SMA(self.data.close, period=50)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev
        )
        self.cooldown = 0

    def next_custom(self):
        minp = max(self.p.trend_long_period, 50, self.p.rsi_period, self.p.bb_period)
        if self.atr is not None:
            minp = max(minp, self.p.atr_period)
        if len(self) < minp:
            return

        if self.cooldown > 0:
            self.cooldown -= 1

        # Éviter les tendances fortes si demandé
        if self.p.avoid_strong_trend and (self.sma_mid[0] > self.sma_long[0] * 1.01):
            if (
                self.p.use_invalidation
                and self.position.size > 0
                and self.data.close[0] < self.sma_mid[0]
            ):
                self.close()
            return

        # Signal: survente + touche bande basse, re-cross vers la moyenne
        oversold = (
            self.rsi[0] <= self.p.rsi_oversold and self.data.close[0] <= self.bb.bot[0]
        )
        revert = self.data.close[0] > self.bb.mid[0]  # sortie partielle possible
        if self.position.size == 0 and self.cooldown == 0 and oversold:
            self.buy()
            self.cooldown = self.p.reentry_cooldown_bars
        elif self.position.size > 0 and revert and self.p.use_invalidation:
            # sortie discrétionnaire complémentaire à SL/TP
            self.close()
