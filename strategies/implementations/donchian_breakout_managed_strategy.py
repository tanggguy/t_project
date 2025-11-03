import backtrader as bt
from strategies.managed_strategy import ManagedStrategy
from utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/strategies/donchian_breakout")


class DonchianBreakoutManagedStrategy(ManagedStrategy):
    """
    Breakout de range (Donchian) avec filtres de régime et volume.
    """

    params = (
        ("trend_long_period", 200),
        ("trend_mid_period", 50),
        ("donchian_period", 20),
        ("rsi_period", 14),
        ("rsi_min", 50),
        ("vol_ma_period", 20),
        ("require_volume_confirm", True),
        ("use_invalidation", True),
        ("reentry_cooldown_bars", 5),
        ("atr_period", 14),
    )

    def __init__(self):
        super().__init__()
        self.sma_long = bt.indicators.SMA(
            self.data.close, period=self.p.trend_long_period
        )
        self.sma_mid = bt.indicators.SMA(
            self.data.close, period=self.p.trend_mid_period
        )
        self.dc_high = bt.indicators.Highest(
            self.data.high, period=self.p.donchian_period
        )
        self.dc_low = bt.indicators.Lowest(self.data.low, period=self.p.donchian_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.p.vol_ma_period)
        self.cooldown = 0

    def next_custom(self):
        minp = max(
            self.p.trend_long_period,
            self.p.trend_mid_period,
            self.p.donchian_period,
            self.p.rsi_period,
            self.p.vol_ma_period,
        )
        if self.atr is not None:
            minp = max(minp, self.p.atr_period)
        if len(self) < minp:
            return

        if self.cooldown > 0:
            self.cooldown -= 1

        in_uptrend = (self.data.close[0] > self.sma_long[0]) and (
            self.sma_mid[0] > self.sma_long[0]
        )
        if not in_uptrend:
            if (
                self.p.use_invalidation
                and self.position.size > 0
                and self.data.close[0] < self.sma_mid[0]
            ):
                self.close()
            return

        # Breakout: close > plus haut Donchian (avec marge)
        breakout = self.data.close[0] > self.dc_high[-1]
        rsi_ok = self.rsi[0] >= self.p.rsi_min
        vol_ok = True
        if (
            self.p.require_volume_confirm
            and len(self.vol_ma) > 0
            and self.vol_ma[0] > 0
        ):
            vol_ok = self.data.volume[0] > self.vol_ma[0]

        if (
            self.position.size == 0
            and self.cooldown == 0
            and breakout
            and rsi_ok
            and vol_ok
        ):
            self.buy()
            self.cooldown = self.p.reentry_cooldown_bars
