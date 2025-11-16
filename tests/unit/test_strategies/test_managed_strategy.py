import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from risk_management.stop_loss import FixedStopLoss, TrailingStopLoss
from strategies.managed_strategy import ManagedStrategy


DEFAULT_PARAMS = {
    "use_stop_loss": True,
    "stop_loss_type": "fixed",
    "stop_loss_pct": 0.02,
    "stop_loss_atr_mult": 2.0,
    "stop_loss_lookback": 20,
    "use_take_profit": True,
    "take_profit_type": "fixed",
    "take_profit_pct": 0.04,
    "take_profit_atr_mult": 3.0,
    "take_profit_lookback": 20,
    "atr_period": 14,
}


def build_strategy(**overrides):
    """
    Create a minimal ManagedStrategy instance with controllable params/log.
    """
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    strategy = ManagedStrategy.__new__(ManagedStrategy)
    strategy.p = SimpleNamespace(**params)
    strategy.log = MagicMock()
    return strategy


def test_needs_atr_detects_atr_stop_or_take_profit():
    default_strategy = build_strategy()
    assert not default_strategy._needs_atr()

    stop_atr = build_strategy(stop_loss_type="atr")
    assert stop_atr._needs_atr()

    take_atr = build_strategy(take_profit_type="atr")
    assert take_atr._needs_atr()


def test_create_stop_loss_manager_respects_flags():
    strategy = build_strategy()
    manager = strategy._create_stop_loss_manager()
    assert isinstance(manager, FixedStopLoss)

    no_sl = build_strategy(use_stop_loss=False)
    assert no_sl._create_stop_loss_manager() is None

    invalid = build_strategy(stop_loss_type="unknown")
    with pytest.raises(ValueError, match="Type de stop loss inconnu"):
        invalid._create_stop_loss_manager()


def test_create_take_profit_manager_handles_disabled_and_unknown():
    strategy = build_strategy()
    assert strategy._create_take_profit_manager() is not None

    disabled = build_strategy(use_take_profit=False)
    assert disabled._create_take_profit_manager() is None

    invalid = build_strategy(take_profit_type="unknown")
    with pytest.raises(ValueError, match="Type de take profit inconnu"):
        invalid._create_take_profit_manager()


def test_calculate_risk_levels_applies_managers():
    strategy = build_strategy()
    strategy.entry_price = 100.0
    strategy.position_type = "long"
    close_line = MagicMock()
    close_line.__getitem__.return_value = 105.0
    strategy.data_close = close_line

    stop_manager = MagicMock()
    stop_manager.calculate_stop.return_value = 95.0
    take_manager = MagicMock()
    take_manager.calculate_target.return_value = 110.0

    strategy.sl_manager = stop_manager
    strategy.tp_manager = take_manager

    strategy._calculate_risk_levels("long")

    stop_manager.calculate_stop.assert_called_once_with(
        entry_price=100.0,
        current_price=105.0,
        position_type="long",
    )
    take_manager.calculate_target.assert_called_once_with(
        entry_price=100.0,
        position_type="long",
    )
    assert strategy.active_stop_level == pytest.approx(95.0)
    assert strategy.active_target_level == pytest.approx(110.0)


def test_reset_position_state_clears_state_and_resets_trailing():
    strategy = build_strategy()
    manager = TrailingStopLoss(trail_pct=0.01)
    manager.reset = MagicMock()
    strategy.sl_manager = manager
    strategy.entry_price = 101.0
    strategy.active_stop_level = 99.0
    strategy.active_target_level = 107.0
    strategy.position_type = "long"

    strategy._reset_position_state()

    assert strategy.entry_price is None
    assert strategy.active_stop_level is None
    assert strategy.active_target_level is None
    assert strategy.position_type is None
    manager.reset.assert_called_once()
    strategy.log.assert_called_once_with("Trailing stop réinitialisé", logging.DEBUG)
