# --- 1. Bibliothèques natives ---
import logging
from typing import Optional

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from strategies.base_strategy import BaseStrategy
from risk_management.stop_loss import (
    FixedStopLoss,
    TrailingStopLoss,
    ATRStopLoss,
    SupportResistanceStop,
)
from risk_management.take_profit import (
    FixedTakeProfit,
    ATRTakeProfit,
    SupportResistanceTakeProfit,
)

logger = logging.getLogger(__name__)


class ManagedStrategy(BaseStrategy):
    """
    Stratégie avec risk management intégré.

    Les stratégies qui héritent de ManagedStrategy bénéficient automatiquement :
    - Calcul automatique des stop loss
    - Calcul automatique des take profit
    - Gestion de l'état des positions
    - Vérification automatique des déclenchements

    Les classes enfants doivent implémenter next_custom() au lieu de next()
    pour définir leur logique d'entrée.

    Paramètres de risque configurables via le tuple params.

    Example:
        >>> class MyStrategy(ManagedStrategy):
        ...     params = (
        ...         ('ma_period', 20),
        ...         ('stop_loss_type', 'atr'),
        ...     )
        ...
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.ma = bt.indicators.SMA(period=self.p.ma_period)
        ...
        ...     def next_custom(self):
        ...         if self.data.close[0] > self.ma[0]:
        ...             self.buy()
    """

    params = (
        # --- Stop Loss ---
        ("use_stop_loss", True),
        ("stop_loss_type", "fixed"),  # 'fixed', 'trailing', 'atr', 'support_resistance'
        ("stop_loss_pct", 0.02),  # Pour fixed/trailing
        ("stop_loss_atr_mult", 2.0),  # Pour ATR
        ("stop_loss_lookback", 20),  # Pour S/R
        # --- Take Profit ---
        ("use_take_profit", True),
        ("take_profit_type", "fixed"),  # 'fixed', 'atr', 'support_resistance'
        ("take_profit_pct", 0.04),  # Pour fixed
        ("take_profit_atr_mult", 3.0),  # Pour ATR
        ("take_profit_lookback", 20),  # Pour S/R
        # --- Indicateurs requis ---
        ("atr_period", 14),
    )

    def __init__(self) -> None:
        """Initialise la stratégie avec risk management."""
        super().__init__()

        # --- État de la position ---
        self.entry_price: Optional[float] = None
        self.active_stop_level: Optional[float] = None
        self.active_target_level: Optional[float] = None
        self.position_type: Optional[str] = None  # 'long' ou 'short'

        # --- Initialiser les indicateurs requis ---
        self.atr = None
        if self._needs_atr():
            self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
            self.log(f"ATR initialisé (période={self.p.atr_period})", logging.DEBUG)

        # --- Initialiser les gestionnaires ---
        self.sl_manager = self._create_stop_loss_manager()
        self.tp_manager = self._create_take_profit_manager()

        self.log(
            f"ManagedStrategy initialisée - "
            f"SL: {self.p.stop_loss_type if self.p.use_stop_loss else 'Désactivé'}, "
            f"TP: {self.p.take_profit_type if self.p.use_take_profit else 'Désactivé'}",
            logging.INFO,
        )

    def _needs_atr(self) -> bool:
        """
        Vérifie si l'indicateur ATR est nécessaire.

        Returns:
            bool: True si ATR nécessaire, False sinon.
        """
        return self.p.stop_loss_type == "atr" or self.p.take_profit_type == "atr"

    def _create_stop_loss_manager(self):
        """
        Crée le gestionnaire de stop loss selon le type configuré.

        Returns:
            Instance de stop loss manager ou None si désactivé.

        Raises:
            ValueError: Si le type de stop loss est inconnu.
        """
        if not self.p.use_stop_loss:
            self.log("Stop Loss désactivé", logging.INFO)
            return None

        sl_type = self.p.stop_loss_type

        if sl_type == "fixed":
            return FixedStopLoss(stop_pct=self.p.stop_loss_pct)
        elif sl_type == "trailing":
            return TrailingStopLoss(trail_pct=self.p.stop_loss_pct)
        elif sl_type == "atr":
            return ATRStopLoss(
                atr_multiplier=self.p.stop_loss_atr_mult, atr_period=self.p.atr_period
            )
        elif sl_type == "support_resistance":
            return SupportResistanceStop(
                lookback_period=self.p.stop_loss_lookback, buffer_pct=0.005
            )
        else:
            raise ValueError(f"Type de stop loss inconnu: {sl_type}")

    def _create_take_profit_manager(self):
        """
        Crée le gestionnaire de take profit selon le type configuré.

        Returns:
            Instance de take profit manager ou None si désactivé.

        Raises:
            ValueError: Si le type de take profit est inconnu.
        """
        if not self.p.use_take_profit:
            self.log("Take Profit désactivé", logging.INFO)
            return None

        tp_type = self.p.take_profit_type

        if tp_type == "fixed":
            return FixedTakeProfit(tp_pct=self.p.take_profit_pct)
        elif tp_type == "atr":
            return ATRTakeProfit(
                atr_multiplier=self.p.take_profit_atr_mult, atr_period=self.p.atr_period
            )
        elif tp_type == "support_resistance":
            return SupportResistanceTakeProfit(
                lookback_period=self.p.take_profit_lookback, buffer_pct=0.005
            )
        else:
            raise ValueError(f"Type de take profit inconnu: {tp_type}")

    def _calculate_risk_levels(self, position_type: str) -> None:
        """
        Calcule les niveaux de stop loss et take profit.

        Cette méthode est appelée automatiquement après l'entrée en position.

        Args:
            position_type (str): 'long' ou 'short'
        """
        current_price = self.data_close[0]

        # --- Calculer Stop Loss ---
        if self.sl_manager:
            sl_type = self.p.stop_loss_type

            if sl_type in ["fixed", "trailing"]:
                self.active_stop_level = self.sl_manager.calculate_stop(
                    entry_price=self.entry_price,
                    current_price=current_price,
                    position_type=position_type,
                )
            elif sl_type == "atr":
                if len(self.atr) > 0:
                    atr_value = self.atr[0]
                    self.active_stop_level = self.sl_manager.calculate_stop(
                        entry_price=self.entry_price,
                        atr_value=atr_value,
                        position_type=position_type,
                    )
                else:
                    self.log(
                        "ATR pas encore disponible pour calcul SL", logging.WARNING
                    )
            elif sl_type == "support_resistance":
                # Trouver le niveau de support/résistance
                if position_type == "long":
                    levels = self.sl_manager.find_support(self, num_levels=1)
                else:
                    levels = self.sl_manager.find_resistance(self, num_levels=1)

                if levels:
                    self.active_stop_level = self.sl_manager.calculate_stop(
                        entry_price=self.entry_price,
                        support_level=levels[0] if position_type == "long" else None,
                        resistance_level=(
                            levels[0] if position_type == "short" else None
                        ),
                        position_type=position_type,
                    )
                else:
                    self.log(
                        f"Aucun niveau S/R trouvé pour SL ({position_type})",
                        logging.WARNING,
                    )

        # --- Calculer Take Profit ---
        if self.tp_manager:
            tp_type = self.p.take_profit_type

            if tp_type == "fixed":
                self.active_target_level = self.tp_manager.calculate_target(
                    entry_price=self.entry_price, position_type=position_type
                )
            elif tp_type == "atr":
                if len(self.atr) > 0:
                    atr_value = self.atr[0]
                    self.active_target_level = self.tp_manager.calculate_target(
                        entry_price=self.entry_price,
                        atr_value=atr_value,
                        position_type=position_type,
                    )
                else:
                    self.log(
                        "ATR pas encore disponible pour calcul TP", logging.WARNING
                    )
            elif tp_type == "support_resistance":
                if position_type == "long":
                    levels = self.tp_manager.find_resistance(self, num_levels=1)
                else:
                    levels = self.tp_manager.find_support(self, num_levels=1)

                if levels:
                    self.active_target_level = self.tp_manager.calculate_target(
                        entry_price=self.entry_price,
                        resistance_level=levels[0] if position_type == "long" else None,
                        support_level=levels[0] if position_type == "short" else None,
                        position_type=position_type,
                    )
                else:
                    self.log(
                        f"Aucun niveau S/R trouvé pour TP ({position_type})",
                        logging.WARNING,
                    )

        # Log des niveaux calculés
        sl_str = f"{self.active_stop_level:.2f}" if self.active_stop_level else "N/A"
        tp_str = (
            f"{self.active_target_level:.2f}" if self.active_target_level else "N/A"
        )

        self.log(
            f"Niveaux calculés - Entry: {self.entry_price:.2f}, "
            f"SL: {sl_str}, TP: {tp_str}",
            logging.INFO,
        )

    def _check_exit_conditions(self) -> bool:
        """
        Vérifie les conditions de sortie (SL/TP).

        Returns:
            bool: True si une sortie doit être effectuée, False sinon.
        """
        current_price = self.data_close[0]

        # --- Vérifier Take Profit en priorité ---
        if self.tp_manager and self.active_target_level:
            if self.tp_manager.should_trigger(
                current_price=current_price,
                target_level=self.active_target_level,
                position_type=self.position_type,
            ):
                self.log(
                    f"✅ TAKE PROFIT déclenché @ {current_price:.2f}", logging.INFO
                )
                self.close()
                return True

        # --- Vérifier Stop Loss ---
        if self.sl_manager and self.active_stop_level:
            # Mise à jour pour trailing stop
            if self.p.stop_loss_type == "trailing":
                self.active_stop_level = self.sl_manager.calculate_stop(
                    entry_price=self.entry_price,
                    current_price=current_price,
                    position_type=self.position_type,
                )

            if self.sl_manager.should_trigger(
                current_price=current_price,
                stop_level=self.active_stop_level,
                position_type=self.position_type,
            ):
                self.log(f"🛑 STOP LOSS déclenché @ {current_price:.2f}", logging.INFO)
                self.close()
                return True

        return False

    def _reset_position_state(self) -> None:
        """Réinitialise l'état de la position."""
        self.entry_price = None
        self.active_stop_level = None
        self.active_target_level = None
        self.position_type = None

        # Réinitialiser le trailing stop si nécessaire
        if isinstance(self.sl_manager, TrailingStopLoss):
            self.sl_manager.reset()
            self.log("Trailing stop réinitialisé", logging.DEBUG)

    def next(self) -> None:
        """
        Logique principale - gère automatiquement les SL/TP.

        Cette méthode ne doit PAS être surchargée par les classes enfants.
        Les classes enfants doivent implémenter next_custom() à la place.
        """
        if self.order:
            return

        # --- Si en position : gérer les sorties ---
        if self.position:
            # Première bougie après l'entrée : calculer les niveaux
            if self.entry_price is None:
                self.entry_price = self.data_close[0]
                self.position_type = "long" if self.position.size > 0 else "short"
                self._calculate_risk_levels(self.position_type)
                return

            # Vérifier les conditions de sortie
            if self._check_exit_conditions():
                return

        # --- Si pas en position : déléguer à la stratégie enfant ---
        else:
            # Réinitialiser l'état si nécessaire
            if self.entry_price is not None:
                self._reset_position_state()

            # Appeler la logique custom de la stratégie enfant
            self.next_custom()

    def next_custom(self) -> None:
        """
        Méthode à implémenter par les classes enfants.

        Cette méthode contient la logique d'entrée en position.
        Elle est appelée uniquement quand on n'est PAS en position.

        Les classes enfants doivent surcharger cette méthode et y placer
        leur logique de génération de signaux d'achat/vente.

        Example:
            >>> def next_custom(self):
            ...     if self.crossover[0] > 0:  # Golden cross
            ...         self.buy()

        Raises:
            NotImplementedError: Si la classe enfant n'implémente pas cette méthode.
        """
        raise NotImplementedError(
            "Les stratégies héritant de ManagedStrategy doivent "
            "implémenter la méthode next_custom()"
        )

    def notify_order(self, order: bt.Order) -> None:
        """
        Gère la notification des ordres.

        Args:
            order (bt.Order): L'ordre notifié par Backtrader.
        """
        # Laisser BaseStrategy gérer le logging standard
        super().notify_order(order)

        # Si c'est une vente (clôture) complétée, réinitialiser
        if order.status == order.Completed and order.issell():
            self._reset_position_state()
