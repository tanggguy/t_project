```python
#!/usr/bin/env python3
# --- 1. Bibliothèques natives ---
import sys
from pathlib import Path
from datetime import datetime

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- Configuration du Chemin ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- 3. Imports locaux du projet ---
from risk_management.position_sizing import (
    FixedSizer,
    FixedFractionalSizer,
    VolatilityBasedSizer
)
from strategies.base_strategy import BaseStrategy


class SimpleStrategy(BaseStrategy):
    """Stratégie simple pour tester les position sizers."""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )

    def __init__(self):
        super().__init__()
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.close()


def run_backtest_with_sizer(sizer_class, sizer_params):
    """
    Lance un backtest avec un sizer spécifique.
    
    Args:
        sizer_class: Classe du sizer à utiliser.
        sizer_params: Dictionnaire des paramètres du sizer.
    """
    cerebro = bt.Cerebro()
    
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addstrategy(SimpleStrategy)
    
    cerebro.addsizer(sizer_class, **sizer_params)
    
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        fromdate=datetime(2023, 1, 1),
        todate=datetime(2024, 1, 1)
    )
    cerebro.adddata(data)
    
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    print(f"\n{'='*60}")
    print(f"Test avec {sizer_class.__name__}")
    print(f"Paramètres: {sizer_params}")
    print(f"{'='*60}")
    print(f"Capital initial: {cerebro.broker.getvalue():.2f}")
    
    results = cerebro.run()
    
    print(f"Capital final: {cerebro.broker.getvalue():.2f}")
    print(f"Profit: {cerebro.broker.getvalue() - 10000.0:.2f}")
    
    strat = results[0]
    
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")


if __name__ == '__main__':
    
    run_backtest_with_sizer(
        FixedSizer,
        {'pct_size': 1.0}
    )
    
    run_backtest_with_sizer(
        FixedFractionalSizer,
        {'risk_pct': 0.02, 'stop_distance': 0.03}
    )
    
    run_backtest_with_sizer(
        VolatilityBasedSizer,
        {'risk_pct': 0.02, 'atr_period': 14, 'atr_multiplier': 2.0}
    )
```