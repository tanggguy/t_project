import backtrader as bt
from strategies.managed_strategy import ManagedStrategy
from utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/strategies/sma_pullback")


class SmaPullbackManagedStrategy(ManagedStrategy):
    """
    Trend-following robust multi-régimes: filtre SMA200/50,
    entrée sur pullback EMA20 + confirmation RSI/Volume.
    Sorties gérées par ManagedStrategy (SL/TP/trailing), invalidation optionnelle.
    """

    params = (
        # Filtres de tendance
        ("trend_long_period", 200),
        ("trend_mid_period", 50),
        # Pullback & momentum
        ("pullback_ma_period", 20),  # EMA20
        ("rsi_period", 14),
        ("rsi_min", 50),
        ("vol_ma_period", 20),  # Volume MA
        ("require_volume_confirm", True),
        # Invalidation optionnelle (en plus de SL/TP)
        ("use_invalidation", True),
        # Cooldown entre entrées
        ("reentry_cooldown_bars", 5),
        # ATR référence (si SL ATR côté ManagedStrategy)
        ("atr_period", 14),
    )

    def __init__(self):
        super().__init__()

        # Indicateurs (sur data principale)
        self.sma_long = bt.indicators.SMA(
            self.data.close, period=self.p.trend_long_period
        )
        self.sma_mid = bt.indicators.SMA(
            self.data.close, period=self.p.trend_mid_period
        )
        self.ema_pb = bt.indicators.EMA(
            self.data.close, period=self.p.pullback_ma_period
        )
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.p.vol_ma_period)

        # Croisement/prix au-dessus de pullback MA
        self.price_above_pb = bt.indicators.CrossOver(self.data.close, self.ema_pb)

        # Compteur cooldown
        self.cooldown = 0

        logger.info(
            f"SmaPullback init - long:{self.p.trend_long_period} mid:{self.p.trend_mid_period} "
            f"pb:{self.p.pullback_ma_period} rsi:{self.p.rsi_period}"
        )

    def next_custom(self):
        # Warmup
        minp = max(
            self.p.trend_long_period,
            self.p.trend_mid_period,
            self.p.pullback_ma_period,
            self.p.rsi_period,
            self.p.vol_ma_period,
        )
        if self.atr is not None:
            minp = max(minp, self.p.atr_period)
        if len(self) < minp:
            return

        if self.cooldown > 0:
            self.cooldown -= 1

        # Filtres de régime: tendance haussière robuste
        in_uptrend = (self.data.close[0] > self.sma_long[0]) and (
            self.sma_mid[0] > self.sma_long[0]
        )
        if not in_uptrend:
            # Invalidation optionnelle si déjà en position (ex: sortie si sous SMA50)
            if (
                self.p.use_invalidation
                and self.position.size > 0
                and self.data.close[0] < self.sma_mid[0]
            ):
                logger.info(
                    f"Invalidation: close < SMA{self.p.trend_mid_period} -> close()"
                )
                self.close()
            return

        # Confirmation momentum/volume
        rsi_ok = self.rsi[0] >= self.p.rsi_min
        vol_ok = True
        if (
            self.p.require_volume_confirm
            and len(self.vol_ma) > 0
            and self.vol_ma[0] > 0
        ):
            vol_ok = self.data.volume[0] > self.vol_ma[0]

        # Entrée: pullback EMA20 puis reprise (CrossUp close vs EMA20), RSI/volume confirmés
        # et pas de position actuelle et cooldown terminé
        if self.cooldown == 0 and self.position.size == 0:
            # "Pullback sain": hier close <= EMA20 puis reprise aujourd'hui (cross up)
            had_pullback = self.data.close[-1] <= self.ema_pb[-1]
            resumed_up = self.price_above_pb[0] > 0
            if in_uptrend and had_pullback and resumed_up and rsi_ok and vol_ok:
                logger.info(
                    f"BUY pullback @ {self.data_close[0]:.2f} "
                    f"(SMA{self.p.trend_long_period}={self.sma_long[0]:.2f}, "
                    f"EMA{self.p.pullback_ma_period}={self.ema_pb[0]:.2f}, RSI={self.rsi[0]:.1f})"
                )
                self.buy()
                self.cooldown = self.p.reentry_cooldown_bars
