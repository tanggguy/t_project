# --- 1. Bibliothèques natives ---
import logging
from typing import Optional

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from strategies.base_strategy import BaseStrategy


class RsiOversoldStrategy(BaseStrategy):
    """
    Stratégie basée sur le RSI (Relative Strength Index).

    Logique de trading:
    - ACHAT: Lorsque le RSI passe sous le seuil 'oversold_level' (généralement 30)
    - VENTE: Lorsque le RSI dépasse le seuil 'overbought_level' (généralement 70)

    Cette stratégie cherche à acheter les zones de survente (oversold)
    et vendre les zones de surachat (overbought).

    Attributes:
        params (tuple): Paramètres de la stratégie
            - rsi_period (int): Période de calcul du RSI (défaut: 14)
            - oversold_level (float): Seuil de survente pour achat (défaut: 30)
            - overbought_level (float): Seuil de surachat pour vente (défaut: 70)
    """

    params = (
        ("rsi_period", 14),  # Période standard du RSI
        ("oversold_level", 30.0),  # Seuil de survente (signal d'achat)
        ("overbought_level", 70.0),  # Seuil de surachat (signal de vente)
    )

    def __init__(self) -> None:
        """
        Initialise la stratégie et calcule l'indicateur RSI.
        """
        super().__init__()

        # --- Calcul de l'indicateur RSI ---
        # Utilise bt.indicators.RSI natif de Backtrader
        self.rsi = bt.indicators.RSI(self.data_close, period=self.p.rsi_period)

        # Log de l'initialisation
        self.log(
            f"Initialisation RsiOversoldStrategy - "
            f"Période RSI: {self.p.rsi_period}, "
            f"Seuil Oversold: {self.p.oversold_level}, "
            f"Seuil Overbought: {self.p.overbought_level}",
            level=logging.INFO,
        )

    def next(self) -> None:
        """
        Logique principale de la stratégie, exécutée à chaque bougie.

        Vérifie:
        1. Si un ordre est déjà en cours, ne rien faire
        2. Si pas de position: chercher signal d'ACHAT (RSI < oversold)
        3. Si position ouverte: chercher signal de VENTE (RSI > overbought)
        """
        # Si un ordre est déjà en attente, ne pas en créer un nouveau
        if self.order:
            return

        # --- Vérifier s'il y a assez de données pour l'indicateur ---
        # Le RSI a besoin d'un minimum de barres pour être calculé
        if len(self) < self.p.rsi_period:
            return

        # --- Obtenir les valeurs actuelles ---
        current_rsi = self.rsi[0]
        current_price = self.data_close[0]

        # --- Cas 1: Pas de position ouverte -> Chercher ACHAT ---
        if not self.position:
            # Signal d'achat: RSI passe sous le seuil oversold
            if current_rsi < self.p.oversold_level:
                self.log(
                    f"SIGNAL ACHAT - RSI: {current_rsi:.2f} < {self.p.oversold_level} "
                    f"(Prix: {current_price:.2f})",
                    level=logging.INFO,
                )
                # Créer un ordre d'achat
                self.order = self.buy()

        # --- Cas 2: Position ouverte -> Chercher VENTE ---
        else:
            # Signal de vente: RSI dépasse le seuil overbought
            if current_rsi > self.p.overbought_level:
                self.log(
                    f"SIGNAL VENTE - RSI: {current_rsi:.2f} > {self.p.overbought_level} "
                    f"(Prix: {current_price:.2f})",
                    level=logging.INFO,
                )
                # Créer un ordre de vente (fermer la position)
                self.order = self.sell()
