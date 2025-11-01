"""
Module de gestion des risques (Risk Management) pour le trading quantitatif.

Ce module fournit des outils pour gérer les risques de trading :
- Stop Loss : différentes stratégies de stop loss
- Take Profit : (à venir) différentes stratégies de prise de profit
- Position Sizing : (à venir) calcul de la taille des positions

Usage:
    from risk_management import FixedStopLoss, TrailingStopLoss

    # Utiliser un stop loss fixe
    stop = FixedStopLoss(stop_pct=0.02)
    stop_level = stop.calculate_stop(entry_price=100, position_type='long')
"""

from .stop_loss import (
    FixedStopLoss,
    TrailingStopLoss,
    ATRStopLoss,
    SupportResistanceStop,
)

__all__ = ["FixedStopLoss", "TrailingStopLoss", "ATRStopLoss", "SupportResistanceStop"]

__version__ = "0.1.0"
