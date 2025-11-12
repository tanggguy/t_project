# --- 1. Bibliothèques natives ---
import logging
from typing import Optional

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from strategies.base_strategy import BaseStrategy


class MacdMomentumStrategy(BaseStrategy):
    """
    Stratégie basée sur le MACD (Moving Average Convergence Divergence).

    Logique de trading:
    - ACHAT: Lorsque la ligne MACD croise au-dessus de la ligne Signal (golden cross)
    - VENTE: Lorsque la ligne MACD croise en-dessous de la ligne Signal (death cross)

    Le MACD est un indicateur de momentum qui suit la tendance,
    particulièrement efficace pour détecter les changements de direction.

    Attributes:
        params (tuple): Paramètres de la stratégie
            - macd_fast (int): Période de l'EMA rapide (défaut: 12)
            - macd_slow (int): Période de l'EMA lente (défaut: 26)
            - macd_signal (int): Période de la ligne de signal (défaut: 9)
    """

    params = (
        ("macd_fast", 12),  # Période standard de l'EMA rapide
        ("macd_slow", 26),  # Période standard de l'EMA lente
        ("macd_signal", 9),  # Période standard de la ligne de signal
    )

    def __init__(self) -> None:
        """
        Initialise la stratégie et calcule l'indicateur MACD.
        """
        super().__init__()

        # --- Calcul de l'indicateur MACD ---
        # bt.indicators.MACD retourne un objet avec plusieurs lignes:
        # - self.macd.macd: La ligne MACD principale
        # - self.macd.signal: La ligne de signal
        # - self.macd.histo: L'histogramme (MACD - Signal) - optionnel
        self.macd = bt.indicators.MACD(
            self.data_close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )

        # --- Détection des croisements ---
        # CrossOver retourne:
        #  +1 quand la ligne MACD croise au-dessus de la signal (bullish)
        #  -1 quand la ligne MACD croise en-dessous de la signal (bearish)
        #   0 quand pas de croisement
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        # Log de l'initialisation
        self.log(
            f"Initialisation MacdMomentumStrategy - "
            f"Fast: {self.p.macd_fast}, "
            f"Slow: {self.p.macd_slow}, "
            f"Signal: {self.p.macd_signal}",
            level=logging.INFO,
        )

    def next(self) -> None:
        """
        Logique principale de la stratégie, exécutée à chaque bougie.

        Vérifie:
        1. Si un ordre est déjà en cours, ne rien faire
        2. Si pas de position: chercher signal d'ACHAT (crossover > 0)
        3. Si position ouverte: chercher signal de VENTE (crossover < 0)
        """
        # Si un ordre est déjà en attente, ne pas en créer un nouveau
        if self.order:
            return

        # --- Vérifier s'il y a assez de données pour l'indicateur ---
        # Le MACD a besoin d'un minimum de barres (slow + signal)
        min_period = self.p.macd_slow + self.p.macd_signal
        if len(self) < min_period:
            return

        # --- Obtenir les valeurs actuelles ---
        current_macd = self.macd.macd[0]
        current_signal = self.macd.signal[0]
        current_price = self.data_close[0]
        cross_value = self.crossover[0]

        # --- Cas 1: Pas de position ouverte -> Chercher ACHAT ---
        if not self.position:
            # Signal d'achat: MACD croise au-dessus de la Signal (cross_value > 0)
            if cross_value > 0:
                self.log(
                    f"SIGNAL ACHAT (Golden Cross) - "
                    f"MACD: {current_macd:.4f} croise au-dessus Signal: {current_signal:.4f} "
                    f"(Prix: {current_price:.2f})",
                    level=logging.INFO,
                )
                # Créer un ordre d'achat
                self.order = self.buy()

        # --- Cas 2: Position ouverte -> Chercher VENTE ---
        else:
            # Signal de vente: MACD croise en-dessous de la Signal (cross_value < 0)
            if cross_value < 0:
                self.log(
                    f"SIGNAL VENTE (Death Cross) - "
                    f"MACD: {current_macd:.4f} croise en-dessous Signal: {current_signal:.4f} "
                    f"(Prix: {current_price:.2f})",
                    level=logging.INFO,
                )
                # Créer un ordre de vente (fermer la position)
                self.order = self.close()
