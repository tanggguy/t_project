# tests/unit/test_strategies/test_base_strategy.py

import pytest
from unittest.mock import MagicMock, patch
import logging
import backtrader as bt

# Mock logger before importing
@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    """Mocks the logger used in the base_strategy module."""
    mock_log = MagicMock()
    monkeypatch.setattr("strategies.base_strategy.logger", mock_log)
    return mock_log

# Now import the class to be tested
from strategies.base_strategy import BaseStrategy

# --- Fixtures ---

@pytest.fixture
def strategy_instance():
    """Provides an instance of BaseStrategy in a mocked environment."""
    # Bypassing the real bt.Strategy.__init__ which requires a full Cerebro environment.
    with patch.object(bt.Strategy, '__init__', lambda self: None):
        # Create an instance without calling __init__
        strategy = BaseStrategy.__new__(BaseStrategy)

        # Manually mock the environment that BaseStrategy.__init__ expects
        strategy.data0 = MagicMock()
        strategy.data0.datetime.date.return_value.isoformat.return_value = "2023-01-01"

        # Now explicitly call the __init__ of the class we are testing
        strategy.__init__()

        # Mock other parts of the environment our strategy interacts with
        strategy.broker = MagicMock()
        strategy.broker.getvalue.return_value = 100000.0

        return strategy

@pytest.fixture
def mock_order():
    """Creates a mock backtrader Order object for testing notifications."""
    order = MagicMock(spec=bt.Order)
    order.executed = MagicMock()
    order.executed.price = 150.0
    order.executed.value = 1500.0
    order.executed.comm = 1.5
    order.isbuy.return_value = False
    order.issell.return_value = False
    return order

# --- Tests ---

class TestBaseStrategy:

    def test_initialization(self, strategy_instance):
        """Test that the strategy initializes correctly."""
        assert strategy_instance.strategy_name == "BaseStrategy"
        assert strategy_instance.order is None
        assert strategy_instance.data_close is strategy_instance.data0.close
        assert strategy_instance.data_open is strategy_instance.data0.open

    def test_log_method(self, strategy_instance, mock_logger):
        """Test the custom log method formats messages correctly."""
        message = "Test log message"
        strategy_instance.log(message, level=logging.DEBUG)
        
        expected_log = "[2023-01-01 @ BaseStrategy] --- Test log message"
        mock_logger.log.assert_called_once_with(logging.DEBUG, expected_log)

    def test_next_method_raises_not_implemented(self, strategy_instance):
        """Test that the base next() method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="La méthode 'next' doit être implémentée."):
            strategy_instance.next()

    def test_notify_order_submitted_accepted(self, strategy_instance, mock_order, mock_logger):
        """Test that Submitted and Accepted statuses are ignored."""
        mock_order.status = bt.Order.Submitted
        strategy_instance.notify_order(mock_order)
        
        mock_order.status = bt.Order.Accepted
        strategy_instance.notify_order(mock_order)
        
        mock_logger.log.assert_not_called()

    def test_notify_order_completed_buy(self, strategy_instance, mock_order, mock_logger):
        """Test notification for a completed buy order."""
        mock_order.status = bt.Order.Completed
        mock_order.isbuy.return_value = True
        strategy_instance.order = mock_order

        # Configure mock statuses to behave like a real order object
        mock_order.Completed = bt.Order.Completed
        mock_order.Submitted = bt.Order.Submitted
        mock_order.Accepted = bt.Order.Accepted
        
        strategy_instance.notify_order(mock_order)
        
        expected_msg = ("ACHAT EXÉCUTÉ, Prix: 150.00, Coût: 1500.00, Comm: 1.50")
        mock_logger.log.assert_called_once_with(logging.DEBUG, f"[2023-01-01 @ BaseStrategy] --- {expected_msg}")
        assert strategy_instance.order is None

    def test_notify_order_completed_sell(self, strategy_instance, mock_order, mock_logger):
        """Test notification for a completed sell order."""
        mock_order.status = bt.Order.Completed
        mock_order.issell.return_value = True
        strategy_instance.order = mock_order

        # Configure mock statuses
        mock_order.Completed = bt.Order.Completed
        mock_order.Submitted = bt.Order.Submitted
        mock_order.Accepted = bt.Order.Accepted
        
        strategy_instance.notify_order(mock_order)
        
        expected_msg = ("VENTE EXÉCUTÉE, Prix: 150.00, Coût: 1500.00, Comm: 1.50")
        mock_logger.log.assert_called_once_with(logging.DEBUG, f"[2023-01-01 @ BaseStrategy] --- {expected_msg}")
        assert strategy_instance.order is None

    def test_notify_order_failed_statuses(self, strategy_instance, mock_order, mock_logger):
        """Test notification for failed/rejected/canceled orders."""
        statuses_to_test = [bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected]

        # Configure mock statuses
        mock_order.Status = bt.Order.Status  # For mapping status int to string
        mock_order.Canceled = bt.Order.Canceled
        mock_order.Margin = bt.Order.Margin
        mock_order.Rejected = bt.Order.Rejected
        mock_order.Completed = bt.Order.Completed
        mock_order.Submitted = bt.Order.Submitted
        mock_order.Accepted = bt.Order.Accepted
        
        for status in statuses_to_test:
            mock_logger.reset_mock()
            strategy_instance.order = mock_order
            mock_order.status = status
            
            strategy_instance.notify_order(mock_order)
            
            status_str = bt.Order.Status[status]
            expected_msg = f"Ordre Échoué/Annulé/Rejeté: {status_str}"
            mock_logger.log.assert_called_once_with(logging.INFO, f"[2023-01-01 @ BaseStrategy] --- {expected_msg}")
            assert strategy_instance.order is None

    def test_stop_method(self, strategy_instance, mock_logger):
        """Test the stop method logs the final portfolio value."""
        strategy_instance.stop()
        
        expected_msg = "--- FIN DE LA STRATÉGIE --- Portefeuille final: 100000.00"
        mock_logger.log.assert_called_once_with(logging.INFO, f"[2023-01-01 @ BaseStrategy] --- {expected_msg}")