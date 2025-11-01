"""
Module de gestion des stop loss pour les stratégies de trading.

Ce module fournit différentes implémentations de stop loss :
- FixedStopLoss : Stop loss en pourcentage fixe
- TrailingStopLoss : Stop loss suiveur (trailing)
- ATRStopLoss : Stop loss basé sur l'Average True Range
- SupportResistanceStop : Stop loss basé sur les niveaux de support/résistance

Toutes les classes sont des utilitaires (non des bt.Indicator) à utiliser
dans les stratégies Backtrader.
"""

# --- 1. Bibliothèques natives ---
import logging
from typing import Optional, Tuple, List

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FixedStopLoss:
    """
    Stop Loss en pourcentage fixe par rapport au prix d'entrée.

    Cette classe calcule un niveau de stop loss fixe basé sur un pourcentage
    défini par l'utilisateur. Simple et prévisible.

    Attributes:
        stop_pct (float): Pourcentage de stop loss (ex: 0.02 pour 2%)

    Example:
        >>> stop = FixedStopLoss(stop_pct=0.02)  # 2% de stop
        >>> stop_level = stop.calculate_stop(entry_price=100, position_type='long')
        >>> print(stop_level)  # 98.0
    """

    def __init__(self, stop_pct: float = 0.02):
        """
        Initialise le stop loss fixe.

        Args:
            stop_pct (float): Pourcentage de stop loss (ex: 0.02 pour 2%).
                             Doit être positif.

        Raises:
            ValueError: Si stop_pct est négatif ou nul.
        """
        if stop_pct <= 0:
            raise ValueError(f"stop_pct doit être positif, reçu: {stop_pct}")

        self.stop_pct = stop_pct
        logger.debug(f"FixedStopLoss initialisé avec stop_pct={stop_pct}")

    def calculate_stop(
        self,
        entry_price: float,
        current_price: Optional[float] = None,
        position_type: str = "long",
    ) -> float:
        """
        Calcule le niveau de stop loss.

        Args:
            entry_price (float): Prix d'entrée de la position.
            current_price (Optional[float]): Prix actuel (non utilisé pour fixed stop).
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            float: Niveau de prix du stop loss.

        Raises:
            ValueError: Si position_type n'est ni 'long' ni 'short'.
        """
        if position_type not in ["long", "short"]:
            raise ValueError(
                f"position_type doit être 'long' ou 'short', reçu: {position_type}"
            )

        if position_type == "long":
            stop_level = entry_price * (1 - self.stop_pct)
        else:  # short
            stop_level = entry_price * (1 + self.stop_pct)

        logger.debug(
            f"Stop calculé: entry={entry_price}, type={position_type}, "
            f"stop={stop_level:.2f}"
        )
        return stop_level

    def should_trigger(
        self, current_price: float, stop_level: float, position_type: str = "long"
    ) -> bool:
        """
        Vérifie si le stop loss doit être déclenché.

        Args:
            current_price (float): Prix actuel du marché.
            stop_level (float): Niveau du stop loss.
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            bool: True si le stop doit être déclenché, False sinon.
        """
        if position_type == "long":
            triggered = current_price <= stop_level
        else:  # short
            triggered = current_price >= stop_level

        if triggered:
            logger.info(
                f"Stop loss déclenché ! Prix={current_price:.2f}, "
                f"Stop={stop_level:.2f}, Type={position_type}"
            )

        return triggered


class TrailingStopLoss:
    """
    Stop Loss suiveur (trailing) qui s'ajuste avec le prix favorable.

    Le stop loss suit le prix lorsqu'il évolue favorablement, mais ne recule
    jamais. Permet de sécuriser les gains tout en laissant courir les profits.

    Attributes:
        trail_pct (float): Pourcentage de trailing (ex: 0.03 pour 3%)
        highest_price (Optional[float]): Plus haut prix atteint (pour long)
        lowest_price (Optional[float]): Plus bas prix atteint (pour short)

    Example:
        >>> stop = TrailingStopLoss(trail_pct=0.03)
        >>> stop_level = stop.calculate_stop(
        ...     entry_price=100,
        ...     current_price=110,
        ...     position_type='long'
        ... )
        >>> print(stop_level)  # 106.7 (110 - 3%)
    """

    def __init__(self, trail_pct: float = 0.03):
        """
        Initialise le trailing stop loss.

        Args:
            trail_pct (float): Pourcentage de trailing (ex: 0.03 pour 3%).
                              Doit être positif.

        Raises:
            ValueError: Si trail_pct est négatif ou nul.
        """
        if trail_pct <= 0:
            raise ValueError(f"trail_pct doit être positif, reçu: {trail_pct}")

        self.trail_pct = trail_pct
        self.highest_price: Optional[float] = None
        self.lowest_price: Optional[float] = None
        logger.debug(f"TrailingStopLoss initialisé avec trail_pct={trail_pct}")

    def reset(self) -> None:
        """Réinitialise les niveaux de prix extrêmes."""
        self.highest_price = None
        self.lowest_price = None
        logger.debug("TrailingStopLoss réinitialisé")

    def calculate_stop(
        self, entry_price: float, current_price: float, position_type: str = "long"
    ) -> float:
        """
        Calcule le niveau de trailing stop loss.

        Le stop suit le prix favorable mais ne recule jamais.

        Args:
            entry_price (float): Prix d'entrée de la position.
            current_price (float): Prix actuel du marché.
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            float: Niveau de prix du stop loss.

        Raises:
            ValueError: Si position_type n'est ni 'long' ni 'short'.
        """
        if position_type not in ["long", "short"]:
            raise ValueError(
                f"position_type doit être 'long' ou 'short', reçu: {position_type}"
            )

        if position_type == "long":
            # Mettre à jour le plus haut
            if self.highest_price is None:
                self.highest_price = max(entry_price, current_price)
            else:
                self.highest_price = max(self.highest_price, current_price)

            stop_level = self.highest_price * (1 - self.trail_pct)

        else:  # short
            # Mettre à jour le plus bas
            if self.lowest_price is None:
                self.lowest_price = min(entry_price, current_price)
            else:
                self.lowest_price = min(self.lowest_price, current_price)

            stop_level = self.lowest_price * (1 + self.trail_pct)

        logger.debug(
            f"Trailing stop calculé: current={current_price:.2f}, "
            f"type={position_type}, stop={stop_level:.2f}"
        )
        return stop_level

    def should_trigger(
        self, current_price: float, stop_level: float, position_type: str = "long"
    ) -> bool:
        """
        Vérifie si le trailing stop doit être déclenché.

        Args:
            current_price (float): Prix actuel du marché.
            stop_level (float): Niveau du stop loss.
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            bool: True si le stop doit être déclenché, False sinon.
        """
        if position_type == "long":
            triggered = current_price <= stop_level
        else:  # short
            triggered = current_price >= stop_level

        if triggered:
            logger.info(
                f"Trailing stop déclenché ! Prix={current_price:.2f}, "
                f"Stop={stop_level:.2f}, Type={position_type}"
            )

        return triggered


class ATRStopLoss:
    """
    Stop Loss basé sur l'Average True Range (ATR).

    Utilise la volatilité du marché (ATR) pour définir un stop loss adaptatif.
    Plus la volatilité est élevée, plus le stop est large.

    Attributes:
        atr_multiplier (float): Multiplicateur de l'ATR (ex: 2.0)
        atr_period (int): Période de calcul de l'ATR

    Example:
        >>> # Dans une stratégie Backtrader
        >>> stop = ATRStopLoss(atr_multiplier=2.0, atr_period=14)
        >>> atr_value = self.atr[0]  # Valeur ATR de l'indicateur Backtrader
        >>> stop_level = stop.calculate_stop(
        ...     entry_price=100,
        ...     atr_value=2.5,
        ...     position_type='long'
        ... )
        >>> print(stop_level)  # 95.0 (100 - 2*2.5)
    """

    def __init__(self, atr_multiplier: float = 2.0, atr_period: int = 14):
        """
        Initialise l'ATR stop loss.

        Args:
            atr_multiplier (float): Multiplicateur de l'ATR (typiquement 1.5-3.0).
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
            f"ATRStopLoss initialisé avec multiplier={atr_multiplier}, "
            f"period={atr_period}"
        )

    def calculate_stop(
        self,
        entry_price: float,
        atr_value: float,
        current_price: Optional[float] = None,
        position_type: str = "long",
    ) -> float:
        """
        Calcule le stop loss basé sur l'ATR.

        Args:
            entry_price (float): Prix d'entrée de la position.
            atr_value (float): Valeur actuelle de l'ATR (depuis bt.indicators.ATR).
            current_price (Optional[float]): Prix actuel (non utilisé ici).
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            float: Niveau de prix du stop loss.

        Raises:
            ValueError: Si position_type n'est ni 'long' ni 'short'.
        """
        if position_type not in ["long", "short"]:
            raise ValueError(
                f"position_type doit être 'long' ou 'short', reçu: {position_type}"
            )

        stop_distance = atr_value * self.atr_multiplier

        if position_type == "long":
            stop_level = entry_price - stop_distance
        else:  # short
            stop_level = entry_price + stop_distance

        logger.debug(
            f"ATR stop calculé: entry={entry_price:.2f}, ATR={atr_value:.2f}, "
            f"distance={stop_distance:.2f}, stop={stop_level:.2f}"
        )
        return stop_level

    def should_trigger(
        self, current_price: float, stop_level: float, position_type: str = "long"
    ) -> bool:
        """
        Vérifie si l'ATR stop doit être déclenché.

        Args:
            current_price (float): Prix actuel du marché.
            stop_level (float): Niveau du stop loss.
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            bool: True si le stop doit être déclenché, False sinon.
        """
        if position_type == "long":
            triggered = current_price <= stop_level
        else:  # short
            triggered = current_price >= stop_level

        if triggered:
            logger.info(
                f"ATR stop déclenché ! Prix={current_price:.2f}, "
                f"Stop={stop_level:.2f}, Type={position_type}"
            )

        return triggered


class SupportResistanceStop:
    """
    Stop Loss basé sur les niveaux de support et résistance.

    Détecte les niveaux clés de support/résistance et place le stop
    légèrement en dessous du support (pour long) ou au-dessus de la
    résistance (pour short).

    Attributes:
        lookback_period (int): Nombre de périodes pour détecter les pivots
        buffer_pct (float): Buffer en % sous/sur le niveau (ex: 0.001 pour 0.1%)

    Example:
        >>> stop = SupportResistanceStop(lookback_period=20, buffer_pct=0.005)
        >>> # Nécessite un historique de prix (high, low)
        >>> support = stop.find_support(price_history)
        >>> stop_level = stop.calculate_stop(
        ...     entry_price=100,
        ...     support_level=95,
        ...     position_type='long'
        ... )
    """

    def __init__(self, lookback_period: int = 20, buffer_pct: float = 0.005):
        """
        Initialise le support/resistance stop.

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
            f"SupportResistanceStop initialisé avec lookback={lookback_period}, "
            f"buffer={buffer_pct}"
        )

    def find_support(self, price_data: bt.Strategy, num_levels: int = 1) -> List[float]:
        """
        Trouve les niveaux de support récents.

        Détecte les bas locaux (pivots) dans l'historique de prix.

        Args:
            price_data (bt.Strategy): Stratégie Backtrader avec accès aux données.
            num_levels (int): Nombre de niveaux de support à retourner.

        Returns:
            List[float]: Liste des niveaux de support, triés du plus proche au plus éloigné.
        """
        supports = []

        # Parcourir l'historique pour trouver les bas locaux
        for i in range(2, min(self.lookback_period, len(price_data.data) - 2)):
            low_prev = price_data.data.low[-i - 1]
            low_curr = price_data.data.low[-i]
            low_next = price_data.data.low[-i + 1]

            # Bas local : plus bas que les voisins
            if low_curr < low_prev and low_curr < low_next:
                supports.append(low_curr)

        # Trier et retourner les niveaux les plus proches du prix actuel
        supports.sort(reverse=True)  # Du plus haut au plus bas
        result = supports[:num_levels]

        logger.debug(f"Supports trouvés: {result}")
        return result

    def find_resistance(
        self, price_data: bt.Strategy, num_levels: int = 1
    ) -> List[float]:
        """
        Trouve les niveaux de résistance récents.

        Détecte les hauts locaux (pivots) dans l'historique de prix.

        Args:
            price_data (bt.Strategy): Stratégie Backtrader avec accès aux données.
            num_levels (int): Nombre de niveaux de résistance à retourner.

        Returns:
            List[float]: Liste des niveaux de résistance, triés du plus proche au plus éloigné.
        """
        resistances = []

        # Parcourir l'historique pour trouver les hauts locaux
        for i in range(2, min(self.lookback_period, len(price_data.data) - 2)):
            high_prev = price_data.data.high[-i - 1]
            high_curr = price_data.data.high[-i]
            high_next = price_data.data.high[-i + 1]

            # Haut local : plus haut que les voisins
            if high_curr > high_prev and high_curr > high_next:
                resistances.append(high_curr)

        # Trier et retourner les niveaux les plus proches du prix actuel
        resistances.sort()  # Du plus bas au plus haut
        result = resistances[:num_levels]

        logger.debug(f"Résistances trouvées: {result}")
        return result

    def calculate_stop(
        self,
        entry_price: float,
        support_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        current_price: Optional[float] = None,
        position_type: str = "long",
    ) -> float:
        """
        Calcule le stop loss basé sur support/résistance.

        Args:
            entry_price (float): Prix d'entrée de la position.
            support_level (Optional[float]): Niveau de support (pour long).
            resistance_level (Optional[float]): Niveau de résistance (pour short).
            current_price (Optional[float]): Prix actuel (non utilisé ici).
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            float: Niveau de prix du stop loss.

        Raises:
            ValueError: Si les niveaux requis ne sont pas fournis.
        """
        if position_type == "long":
            if support_level is None:
                raise ValueError("support_level requis pour position long")
            # Stop légèrement sous le support
            stop_level = support_level * (1 - self.buffer_pct)

        elif position_type == "short":
            if resistance_level is None:
                raise ValueError("resistance_level requis pour position short")
            # Stop légèrement au-dessus de la résistance
            stop_level = resistance_level * (1 + self.buffer_pct)

        else:
            raise ValueError(
                f"position_type doit être 'long' ou 'short', reçu: {position_type}"
            )

        logger.debug(
            f"S/R stop calculé: entry={entry_price:.2f}, "
            f"niveau={'support' if position_type == 'long' else 'resistance'}="
            f"{support_level or resistance_level:.2f}, stop={stop_level:.2f}"
        )
        return stop_level

    def should_trigger(
        self, current_price: float, stop_level: float, position_type: str = "long"
    ) -> bool:
        """
        Vérifie si le S/R stop doit être déclenché.

        Args:
            current_price (float): Prix actuel du marché.
            stop_level (float): Niveau du stop loss.
            position_type (str): Type de position ('long' ou 'short').

        Returns:
            bool: True si le stop doit être déclenché, False sinon.
        """
        if position_type == "long":
            triggered = current_price <= stop_level
        else:  # short
            triggered = current_price >= stop_level

        if triggered:
            logger.info(
                f"S/R stop déclenché ! Prix={current_price:.2f}, "
                f"Stop={stop_level:.2f}, Type={position_type}"
            )

        return triggered
