# --- 1. Bibliothèques natives ---
import logging
from typing import Optional

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
# (Aucun import local nécessaire)

logger = logging.getLogger(__name__)


class FixedSizer(bt.Sizer):
    """
    Sizer pour investir un montant fixe ou un pourcentage fixe du capital.

    Si 'stake' est défini, utilise ce nombre d'unités.
    Sinon, utilise 'pct_size' du capital disponible.

    Args:
        stake: Nombre fixe d'unités à acheter (prioritaire).
        pct_size: Pourcentage du capital à investir (défaut: 1.0 = 100%).
    """

    params = (
        ("stake", None),
        ("pct_size", 1.0),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        """
        Calcule la taille de la position.

        Args:
            comminfo: Informations de commission.
            cash: Cash disponible.
            data: Données de marché.
            isbuy: True si achat, False si vente.

        Returns:
            int: Nombre d'unités à acheter/vendre.
        """
        if self.p.stake is not None:
            size = self.p.stake
        else:
            price = data.close[0]
            available_capital = cash * self.p.pct_size
            size = int(available_capital / price)

        logger.debug(
            f"FixedSizer: size={size}, cash={cash:.2f}, price={data.close[0]:.2f}"
        )
        return size


class FixedFractionalSizer(bt.Sizer):
    """
    Sizer basé sur le risque fixe par trade (Fixed Fractional).

    Formule: size = (capital * risk_pct) / (entry_price * stop_distance)

    Args:
        risk_pct: Pourcentage du capital à risquer par trade (défaut: 0.02 = 2%).
        stop_distance: Distance au stop loss en pourcentage (défaut: 0.03 = 3%).
    """

    params = (
        ("risk_pct", 0.02),
        ("stop_distance", 0.03),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        """
        Calcule la taille de la position basée sur le risque.

        Args:
            comminfo: Informations de commission.
            cash: Cash disponible.
            data: Données de marché.
            isbuy: True si achat, False si vente.

        Returns:
            int: Nombre d'unités à acheter/vendre.
        """
        portfolio_value = self.broker.getvalue()
        price = data.close[0]

        risk_amount = portfolio_value * self.p.risk_pct
        risk_per_share = price * self.p.stop_distance

        if risk_per_share <= 0:
            logger.warning(
                f"FixedFractionalSizer: risk_per_share invalide ({risk_per_share}), retour à 0"
            )
            return 0

        size = int(risk_amount / risk_per_share)

        max_size = int(cash / price)
        size = min(size, max_size)

        logger.debug(
            f"FixedFractionalSizer: size={size}, risk_amount={risk_amount:.2f}, "
            f"risk_per_share={risk_per_share:.2f}, portfolio_value={portfolio_value:.2f}"
        )

        return size


class VolatilityBasedSizer(bt.Sizer):
    """
    Sizer basé sur la volatilité (ATR).

    Ajuste la taille de position en fonction de la volatilité du marché.
    Plus la volatilité est élevée, plus la position est petite.

    Formule: size = (capital * risk_pct) / (ATR * atr_multiplier)

    Args:
        risk_pct: Pourcentage du capital à risquer par trade (défaut: 0.02 = 2%).
        atr_period: Période de l'ATR (défaut: 14).
        atr_multiplier: Multiplicateur de l'ATR pour le stop (défaut: 2.0).
    """

    params = (
        ("risk_pct", 0.02),
        ("atr_period", 14),
        ("atr_multiplier", 2.0),
    )

    def __init__(self):
        """Initialise le sizer."""
        super().__init__()
        self.atr = None

    def _getsizing(self, comminfo, cash, data, isbuy):
        """
        Calcule la taille de la position basée sur l'ATR.

        Args:
            comminfo: Informations de commission.
            cash: Cash disponible.
            data: Données de marché.
            isbuy: True si achat, False si vente.

        Returns:
            int: Nombre d'unités à acheter/vendre.
        """
        if self.atr is None:
            if self.strategy is None:
                logger.warning(
                    "VolatilityBasedSizer: strategy non disponible, retour à 0"
                )
                return 0
            self.atr = bt.indicators.ATR(self.strategy.data, period=self.p.atr_period)

        if len(self.atr) == 0:
            logger.debug(
                "VolatilityBasedSizer: ATR pas encore prêt (période de warmup)"
            )
            return 0

        portfolio_value = self.broker.getvalue()
        price = data.close[0]

        atr_value = self.atr[0]

        if atr_value <= 0:
            logger.warning(
                f"VolatilityBasedSizer: ATR invalide ({atr_value}), retour à 0"
            )
            return 0

        risk_amount = portfolio_value * self.p.risk_pct
        risk_per_share = atr_value * self.p.atr_multiplier

        size = int(risk_amount / risk_per_share)

        max_size = int(cash / price)
        size = min(size, max_size)

        logger.debug(
            f"VolatilityBasedSizer: size={size}, risk_amount={risk_amount:.2f}, "
            f"ATR={atr_value:.2f}, risk_per_share={risk_per_share:.2f}, "
            f"portfolio_value={portfolio_value:.2f}"
        )

        return size
