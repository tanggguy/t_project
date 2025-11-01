# tests/unit/test_backtesting/test_engine.py

import pytest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
import backtrader as bt

# Mock local imports before they are imported by the module under test
@pytest.fixture(autouse=True)
def mock_imports(monkeypatch):
    """Mock modules that are not relevant for unit testing the engine."""
    monkeypatch.setattr("utils.logger.setup_logger", lambda name: MagicMock())

# Now, import the class to be tested
from backtesting.engine import BacktestEngine
from strategies.base_strategy import BaseStrategy

# --- Fixtures ---

@pytest.fixture
def mock_settings(monkeypatch):
    """Fixture to provide a controlled settings dictionary."""
    settings = {
        "backtest": {
            "initial_capital": 50000.0,
        },
        "broker": {
            "commission_type": "percentage",
            "commission_pct": 0.005,
            "slippage_pct": 0.001,
        },
    }
    # Use monkeypatch to replace get_settings where it's referenced
    monkeypatch.setattr("backtesting.engine.get_settings", lambda: settings)
    return settings

@pytest.fixture
def sample_dataframe():
    """Creates a valid DataFrame for data feed testing."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10, freq="D"))
    data = {
        "open": [100] * 10,
        "high": [105] * 10,
        "low": [99] * 10,
        "close": [102] * 10,
        "volume": [1000] * 10,
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def mock_strategy():
    """Returns a mock strategy class."""
    class MockStrategy(BaseStrategy):
        pass
    return MockStrategy

@pytest.fixture
@patch("backtesting.engine.bt.Cerebro")
def engine_instance(mock_cerebro_class, mock_settings):
    """Provides a BacktestEngine instance with a mocked Cerebro."""
    # The class is mocked, so we need to mock the instance it returns
    mock_cerebro_instance = mock_cerebro_class.return_value
    
    # Instantiate the engine, which will use the mocked Cerebro
    engine = BacktestEngine()
    
    # Attach the instance mock for assertions in tests
    engine.cerebro = mock_cerebro_instance
    return engine

# --- Tests ---

class TestBacktestEngine:

    def test_initialization(self, engine_instance, mock_settings):
        """Test that Cerebro is initialized and broker is configured on creation."""
        assert engine_instance.cerebro is not None
        
        # Check if _setup_broker was implicitly called by __init__
        engine_instance.cerebro.broker.setcash.assert_called_once_with(
            mock_settings["backtest"]["initial_capital"]
        )
        engine_instance.cerebro.broker.setcommission.assert_called_once_with(
            commission=mock_settings["broker"]["commission_pct"]
        )

    def test_setup_broker_percentage_commission(self, engine_instance, mock_settings):
        """Test broker setup with percentage commission."""
        engine_instance._setup_broker() # Called again to be explicit for this test
        
        engine_instance.cerebro.broker.setcash.assert_called_with(50000.0)
        engine_instance.cerebro.broker.setcommission.assert_called_with(commission=0.005)
        engine_instance.cerebro.broker.set_slippage_perc.assert_called_with(perc=0.001)

    @patch("backtesting.engine.get_settings")
    def test_setup_broker_fixed_commission(self, mock_get_settings, monkeypatch):
        """Test broker setup with fixed commission."""
        mock_get_settings.return_value = {
            "backtest": {"initial_capital": 10000.0},
            "broker": {"commission_type": "fixed", "commission_fixed": 1.5},
        }
        
        with patch("backtesting.engine.bt.Cerebro") as mock_cerebro_class:
            mock_cerebro_instance = mock_cerebro_class.return_value
            engine = BacktestEngine()
            engine.cerebro = mock_cerebro_instance
            
            engine.cerebro.broker.setcash.assert_called_with(10000.0)
            engine.cerebro.broker.setcommission.assert_called_with(commission=1.5)

    def test_setup_analyzers(self, engine_instance):
        """Test that all default analyzers are added to Cerebro."""
        engine_instance._setup_analyzers()
        
        calls = engine_instance.cerebro.addanalyzer.call_args_list
        assert len(calls) == 4
        
        # Check that each analyzer was added
        added_analyzers = [call[0][0] for call in calls]
        assert bt.analyzers.SharpeRatio in added_analyzers
        assert bt.analyzers.Returns in added_analyzers
        assert bt.analyzers.DrawDown in added_analyzers
        assert bt.analyzers.TradeAnalyzer in added_analyzers

    def test_add_data_success(self, engine_instance, sample_dataframe):
        """Test adding a valid DataFrame as a data feed."""
        with patch("backtesting.engine.bt.feeds.PandasData") as mock_pandas_data:
            mock_feed = mock_pandas_data.return_value
            engine_instance.add_data(sample_dataframe, name="test_data")
            
            mock_pandas_data.assert_called_once_with(dataname=sample_dataframe)
            engine_instance.cerebro.adddata.assert_called_once_with(mock_feed, name="test_data")

    def test_add_data_invalid_index(self, engine_instance):
        """Test that adding a DataFrame with a non-datetime index raises ValueError."""
        df_invalid = pd.DataFrame({"close": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="L'index du DataFrame doit être de type DatetimeIndex"):
            engine_instance.add_data(df_invalid)

    def test_add_strategy(self, engine_instance, mock_strategy):
        """Test adding a strategy to Cerebro."""
        params = {"fast_period": 10, "slow_period": 20}
        engine_instance.add_strategy(mock_strategy, **params)
        
        engine_instance.cerebro.addstrategy.assert_called_once_with(mock_strategy, **params)

    def test_run_backtest(self, engine_instance):
        """Test the execution of the backtest."""
        # Mock the return value of cerebro.run()
        mock_results = [MagicMock(spec=bt.Strategy)]
        engine_instance.cerebro.run.return_value = mock_results
        
        # Spy on _setup_analyzers to ensure it's called
        with patch.object(engine_instance, '_setup_analyzers') as mock_setup_analyzers:
            results = engine_instance.run()
            
            mock_setup_analyzers.assert_called_once()
            engine_instance.cerebro.run.assert_called_once()
            assert results == mock_results

    def test_run_backtest_critical_error(self, engine_instance):
        """Test that a critical error during cerebro.run() is raised."""
        engine_instance.cerebro.run.side_effect = RuntimeError("Cerebro failed")
        
        with pytest.raises(RuntimeError, match="Cerebro failed"):
            engine_instance.run()

    def test_plot_function(self, engine_instance):
        """Test that the plot function calls cerebro.plot."""
        # This test assumes matplotlib is installed.
        # If not, it would test the error log, but here we test the success path.
        engine_instance.plot()
        engine_instance.cerebro.plot.assert_called_once_with(
            style="candlestick", iplot=False, volume=True
        )

    def test_plot_function_matplotlib_not_found(self, engine_instance):
        """Test that an error is logged if matplotlib is not found."""
        # To simulate ImportError, we can patch the import call itself
        with patch("builtins.__import__", side_effect=ImportError):
            # Since the logger is also mocked, we can't check its output directly
            # without more complex setup. We just check that cerebro.plot is NOT called.
            engine_instance.plot()
            engine_instance.cerebro.plot.assert_not_called()
