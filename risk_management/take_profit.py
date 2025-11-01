"""
Module de gestion des take profit (prise de profits) pour les stratégies.

Ce module fournit différentes implémentations de take profit :
- FixedTakeProfit : Take profit en pourcentage fixe
- ATRTakeProfit : Take profit basé sur l'Average True Range
- SupportResistanceTakeProfit : Take profit basé sur les niveaux de support/résistance

Toutes les classes sont des utilitaires (non des bt.Indicator) à utiliser
dans les stratégies Backtrader.
"""

# --- 1. Bibliothèques natives ---
import logging
from typing import Optional, List

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FixedTakeProfit:
    """
    Take Profit (TP) en pourcentage fixe par rapport au prix d'entrée.

    Calcule un niveau de prise de profit fixe basé sur un pourcentage.

    Attributes:
        tp_pct (float): Pourcentage de gain visé (ex: 0.04 pour 4%)

    Example:
        >>> tp = FixedTakeProfit(tp_pct=0.04)  # 4% de gain
        >>> target_level = tp.calculate_target(entry_price=100, position_type='long')
        >>> print(target_level)  # 104.0
    """

    def __init__(self, tp_pct: float = 0.04):
        """
        Initialise le take profit fixe.

        Args:
            tp_pct (float): Pourcentage de gain (ex: 0.04 pour 4%).
                            Doit être positif.

        Raises:
            ValueError: Si tp_pct est négatif ou nul.
        """
        if tp_pct <= 0:
            raise ValueError(f"tp_pct doit être positif, reçu: {tp_pct}")

        self.tp_pct = tp_pct
        logger.debug(f"FixedTakeProfit initialisé avec tp_pct={tp_pct}")

    def calculate_target(
        self,
        entry_price: float,
        current_price: Optional[float] = None,
        position_type: str = "long",
    ) -> float:
        """
        Calcule le niveau de take profit.

        Args:
            entry_price (float): Prix d'entrée de la position.
            current_price (Optional[float]): Prix actuel (non utilisé ici).
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            float: Niveau de prix du take profit.

        Raises:
            ValueError: Si position_type n'est ni 'long' ni 'short'.
        """
        if position_type not in ["long", "short"]:
            raise ValueError(
                f"position_type doit être 'long' ou 'short', reçu: {position_type}"
            )

        if position_type == "long":
            target_level = entry_price * (1 + self.tp_pct)
        else:  # short
            target_level = entry_price * (1 - self.tp_pct)

        logger.debug(
            f"TP Fixe calculé: entry={entry_price}, type={position_type}, "
            f"target={target_level:.2f}"
        )
        return target_level

    def should_trigger(
        self, current_price: float, target_level: float, position_type: str = "long"
    ) -> bool:
        """
        Vérifie si le take profit doit être déclenché.

        Args:
            current_price (float): Prix actuel du marché.
            target_level (float): Niveau du take profit.
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            bool: True si le TP doit être déclenché, False sinon.
        """
        if position_type == "long":
            triggered = current_price >= target_level
        else:  # short
            triggered = current_price <= target_level

        if triggered:
            logger.info(
                f"Take Profit déclenché ! Prix={current_price:.2f}, "
                f"Target={target_level:.2f}, Type={position_type}"
            )

        return triggered


class ATRTakeProfit:
    """
    Take Profit basé sur l'Average True Range (ATR).

    Utilise la volatilité (ATR) pour définir un objectif de gain adaptatif.

    Attributes:
        atr_multiplier (float): Multiplicateur de l'ATR (ex: 3.0)
        atr_period (int): Période de calcul de l'ATR

    Example:
        >>> # Dans une stratégie Backtrader
        >>> tp = ATRTakeProfit(atr_multiplier=3.0, atr_period=14)
        >>> atr_value = self.atr[0]  # Valeur ATR de l'indicateur Backtrader
        >>> target_level = tp.calculate_target(
        ...     entry_price=100,
        ...     atr_value=2.5,
        ...     position_type='long'
        ... )
        >>> print(target_level)  # 107.5 (100 + 3*2.5)
    """

    def __init__(self, atr_multiplier: float = 3.0, atr_period: int = 14):
        """
        Initialise l'ATR take profit.

        Args:
            atr_multiplier (float): Multiplicateur de l'ATR (typiquement 2.0-4.0).
            atr_period (int): Période pour le calcul de l'ATR.

        Raises:
            ValueError: Si les paramètres sont invalides.
        """
        if atr_multiplier <= 0:
            raise ValueError(
                f"atr_multiplier doit être positif, reçu: {atr_multiplier}"
            )
        if atr_period <= 0:
            raise ValueError(f"atr_period doit être positif, reçu: {atr_period}")

        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        logger.debug(
            f"ATRTakeProfit initialisé avec multiplier={atr_multiplier}, "
            f"period={atr_period}"
        )

    def calculate_target(
        self,
        entry_price: float,
        atr_value: float,
        current_price: Optional[float] = None,
        position_type: str = "long",
    ) -> float:
        """
        Calcule le take profit basé sur l'ATR.

        Args:
            entry_price (float): Prix d'entrée de la position.
            atr_value (float): Valeur actuelle de l'ATR (depuis bt.indicators.ATR).
            current_price (Optional[float]): Prix actuel (non utilisé ici).
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            float: Niveau de prix du take profit.

        Raises:
            ValueError: Si position_type n'est ni 'long' ni 'short'.
        """
        if position_type not in ["long", "short"]:
            raise ValueError(
                f"position_type doit être 'long' ou 'short', reçu: {position_type}"
            )

        target_distance = atr_value * self.atr_multiplier

        if position_type == "long":
            target_level = entry_price + target_distance
        else:  # short
            target_level = entry_price - target_distance

        logger.debug(
            f"ATR TP calculé: entry={entry_price:.2f}, ATR={atr_value:.2f}, "
            f"distance={target_distance:.2f}, target={target_level:.2f}"
        )
        return target_level

    def should_trigger(
        self, current_price: float, target_level: float, position_type: str = "long"
    ) -> bool:
        """
        Vérifie si l'ATR take profit doit être déclenché.

        Args:
            current_price (float): Prix actuel du marché.
            target_level (float): Niveau du take profit.
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            bool: True si le TP doit être déclenché, False sinon.
        """
        if position_type == "long":
            triggered = current_price >= target_level
        else:  # short
            triggered = current_price <= target_level

        if triggered:
            logger.info(
                f"ATR TP déclenché ! Prix={current_price:.2f}, "
                f"Target={target_level:.2f}, Type={position_type}"
            )

        return triggered


class SupportResistanceTakeProfit:
    """
    Take Profit basé sur les niveaux de support et résistance.

    Vise le prochain niveau de résistance (pour long) ou de support (pour short).

    Attributes:
        lookback_period (int): Nombre de périodes pour détecter les pivots
        buffer_pct (float): Buffer en % sous/sur le niveau (ex: 0.001 pour 0.1%)
    """

    def __init__(self, lookback_period: int = 20, buffer_pct: float = 0.005):
        """
        Initialise le support/resistance take profit.

        Args:
            lookback_period (int): Nombre de périodes pour détecter les niveaux.
            buffer_pct (float): Buffer en pourcentage (ex: 0.005 pour 0.5%).

        Raises:
            ValueError: Si les paramètres sont invalides.
        """
        if lookback_period <= 0:
            raise ValueError(
                f"lookback_period doit être positif, reçu: {lookback_period}"
            )
        if buffer_pct < 0:
            raise ValueError(f"buffer_pct doit être positif ou nul, reçu: {buffer_pct}")

        self.lookback_period = lookback_period
        self.buffer_pct = buffer_pct
        logger.debug(
            f"SupportResistanceTakeProfit initialisé avec lookback={lookback_period}, "
            f"buffer={buffer_pct}"
        )

    def find_support(self, price_data: bt.Strategy, num_levels: int = 1) -> List[float]:
        """
        Trouve les niveaux de support récents (bas locaux).

        Args:
            price_data (bt.Strategy): Stratégie Backtrader avec accès aux données.
            num_levels (int): Nombre de niveaux de support à retourner.

        Returns:
            List[float]: Liste des niveaux de support, triés du plus proche au plus éloigné.
        """
        supports = []
        for i in range(2, min(self.lookback_period, len(price_data.data))):
            low_prev = price_data.data.low[-i - 1]
            low_curr = price_data.data.low[-i]
            low_next = price_data.data.low[-i + 1]

            if low_curr < low_prev and low_curr < low_next:
                supports.append(low_curr)

        supports.sort(reverse=True)  # Du plus haut au plus bas
        result = supports[:num_levels]
        logger.debug(f"Supports (pour TP short) trouvés: {result}")
        return result

    def find_resistance(
        self, price_data: bt.Strategy, num_levels: int = 1
    ) -> List[float]:
        """
        Trouve les niveaux de résistance récents (hauts locaux).

        Args:
            price_data (bt.Strategy): Stratégie Backtrader avec accès aux données.
            num_levels (int): Nombre de niveaux de résistance à retourner.

        Returns:
            List[float]: Liste des niveaux de résistance, triés du plus proche au plus éloigné.
        """
        resistances = []
        for i in range(2, min(self.lookback_period, len(price_data.data))):
            high_prev = price_data.data.high[-i - 1]
            high_curr = price_data.data.high[-i]
            high_next = price_data.data.high[-i + 1]

            if high_curr > high_prev and high_curr > high_next:
                resistances.append(high_curr)

        resistances.sort()  # Du plus bas au plus haut
        result = resistances[:num_levels]
        logger.debug(f"Résistances (pour TP long) trouvées: {result}")
        return result

    def calculate_target(
        self,
        entry_price: float,
        support_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        current_price: Optional[float] = None,
        position_type: str = "long",
    ) -> float:
        """
        Calcule le take profit basé sur support/résistance.

        - Pour 'long', vise la `resistance_level` (légèrement en dessous).
        - Pour 'short', vise le `support_level` (légèrement au-dessus).

        Args:
            entry_price (float): Prix d'entrée de la position.
            support_level (Optional[float]): Niveau de support (pour short).
            resistance_level (Optional[float]): Niveau de résistance (pour long).
            current_price (Optional[float]): Prix actuel (non utilisé ici).
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            float: Niveau de prix du take profit.

        Raises:
            ValueError: Si les niveaux requis ne sont pas fournis.
        """
        if position_type == "long":
            if resistance_level is None:
                raise ValueError("resistance_level requis pour position long")
            # TP légèrement sous la résistance
            target_level = resistance_level * (1 - self.buffer_pct)

        elif position_type == "short":
            if support_level is None:
                raise ValueError("support_level requis pour position short")
            # TP légèrement au-dessus du support
            target_level = support_level * (1 + self.buffer_pct)

        else:
            raise ValueError(
                f"position_type doit être 'long' ou 'short', reçu: {position_type}"
            )

        logger.debug(
            f"S/R TP calculé: entry={entry_price:.2f}, "
            f"niveau={'resistance' if position_type == 'long' else 'support'}="
            f"{resistance_level or support_level:.2f}, target={target_level:.2f}"
        )
        return target_level

    def should_trigger(
        self, current_price: float, target_level: float, position_type: str = "long"
    ) -> bool:
        """
        Vérifie si le S/R take profit doit être déclenché.

        Args:
            current_price (float): Prix actuel du marché.
            target_level (float): Niveau du take profit.
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            bool: True si le TP doit être déclenché, False sinon.
        """
        if position_type == "long":
            triggered = current_price >= target_level
        else:  # short
            triggered = current_price <= target_level

        if triggered:
            logger.info(
                f"S/R TP déclenché ! Prix={current_price:.2f}, "
                f"Target={target_level:.2f}, Type={position_type}"
            )

        return triggered
