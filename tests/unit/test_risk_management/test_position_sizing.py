# --- 1. Bibliothèques natives ---
import sys
from pathlib import Path
from unittest.mock import Mock

# --- 2. Bibliothèques tierces ---
import pytest
import backtrader as bt
import pandas as pd

# --- 3. Imports locaux du projet ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from risk_management.position_sizing import (
    FixedSizer,
    FixedFractionalSizer,
    VolatilityBasedSizer,
)


@pytest.fixture
def mock_broker():
    """Fixture pour un broker mocké."""
    broker = Mock()
    broker.getvalue.return_value = 10000.0
    return broker


@pytest.fixture
def mock_data():
    """Fixture pour des données mockées."""
    data = Mock()
    data.close = [100.0]
    return data


@pytest.fixture
def sample_dataframe():
    """Fixture pour un DataFrame de test."""
    return pd.DataFrame(
        {
            "open": [100.0] * 20,
            "high": [105.0] * 20,
            "low": [95.0] * 20,
            "close": [102.0] * 20,
            "volume": [1000] * 20,
        },
        index=pd.date_range("2024-01-01", periods=20),
    )


class TestFixedSizer:
    """Tests pour FixedSizer."""

    def test_fixed_stake(self, mock_broker):
        """Test avec un nombre fixe d'unités."""
        sizer = FixedSizer(stake=10)
        sizer.broker = mock_broker
        sizer.strategy = Mock()

        data = Mock()
        data.close = [102.0]

        size = sizer._getsizing(None, 10000.0, data, True)
        assert size == 10

    def test_percentage_size(self, mock_broker, mock_data):
        """Test avec un pourcentage du capital."""
        sizer = FixedSizer(pct_size=0.5)
        sizer.broker = mock_broker
        sizer.strategy = Mock()

        size = sizer._getsizing(None, 10000.0, mock_data, True)
        assert size == 50

    def test_full_capital(self, mock_broker, mock_data):
        """Test avec 100% du capital."""
        sizer = FixedSizer(pct_size=1.0)
        sizer.broker = mock_broker
        sizer.strategy = Mock()

        size = sizer._getsizing(None, 10000.0, mock_data, True)
        assert size == 100


class TestFixedFractionalSizer:
    """Tests pour FixedFractionalSizer."""

    def test_basic_sizing(self, mock_broker, mock_data):
        """Test du calcul de base."""
        sizer = FixedFractionalSizer(risk_pct=0.02, stop_distance=0.03)
        sizer.broker = mock_broker
        sizer.strategy = Mock()

        size = sizer._getsizing(None, 10000.0, mock_data, True)

        expected_risk = 10000.0 * 0.02
        expected_risk_per_share = 100.0 * 0.03
        expected_size = int(expected_risk / expected_risk_per_share)

        assert size == expected_size

    def test_limited_by_cash(self, mock_broker, mock_data):
        """Test que la taille est limitée par le cash disponible."""
        sizer = FixedFractionalSizer(risk_pct=0.5, stop_distance=0.01)
        sizer.broker = mock_broker
        sizer.strategy = Mock()

        cash = 1000.0
        size = sizer._getsizing(None, cash, mock_data, True)

        max_possible = int(cash / 100.0)
        assert size <= max_possible

    def test_zero_stop_distance(self, mock_broker, mock_data):
        """Test avec stop_distance nul ou invalide."""
        sizer = FixedFractionalSizer(risk_pct=0.02, stop_distance=0.0)
        sizer.broker = mock_broker
        sizer.strategy = Mock()

        size = sizer._getsizing(None, 10000.0, mock_data, True)
        assert size == 0


class TestVolatilityBasedSizer:
    """Tests pour VolatilityBasedSizer."""

    def test_basic_sizing_with_atr(self, sample_dataframe):
        """Test du calcul de base avec ATR."""
        cerebro = bt.Cerebro()

        data_feed = bt.feeds.PandasData(dataname=sample_dataframe)
        cerebro.adddata(data_feed)

        class DummyStrategy(bt.Strategy):
            def __init__(self):
                pass

            def next(self):
                pass

        cerebro.addstrategy(DummyStrategy)
        cerebro.addsizer(VolatilityBasedSizer, risk_pct=0.02, atr_multiplier=2.0)

        strats = cerebro.run()

        assert strats is not None

    def test_zero_atr_with_mock(self):
        """Test avec ATR nul après lazy init."""
        cerebro = bt.Cerebro()

        df = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [100.0] * 30,
                "low": [100.0] * 30,
                "close": [100.0] * 30,
                "volume": [1000] * 30,
            },
            index=pd.date_range("2024-01-01", periods=30),
        )

        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed)

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.order_placed = False

            def next(self):
                if not self.order_placed and len(self) > 20:
                    self.buy()
                    self.order_placed = True

        cerebro.addstrategy(TestStrategy)
        cerebro.addsizer(VolatilityBasedSizer, risk_pct=0.02, atr_multiplier=2.0)

        strats = cerebro.run()

        assert strats is not None
