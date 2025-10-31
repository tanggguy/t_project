# --- 1. Bibliothèques natives ---
import logging

# from abc import ABC, abstractmethod  <-- SUPPRIMÉ
from typing import Optional

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
# (Aucun import local nécessaire pour cette classe de base)


# Récupérer le logger standard de Python pour ce module
logger = logging.getLogger(__name__)


# MODIFIÉ : "ABC" a été supprimé de l'héritage
class BaseStrategy(bt.Strategy):
    """
    Classe de base pour toutes les stratégies de trading.

    Cette classe fournit une structure commune pour :
    1.  Le logging standardisé (conforme au manifeste).
    2.  La gestion basique du cycle de vie des ordres (notify_order).
    3.  Les alias communs pour les lignes de données.

    Toute stratégie créée dans ce projet DOIT hériter de BaseStrategy.
    """

    def __init__(self) -> None:
        """
        Initialise la stratégie.
        Définit les alias de données et prépare la gestion des ordres.
        """
        super().__init__()

        # --- Alias pour les données (convention) ---
        # self.data0 est le flux de données principal
        self.data_close = self.data0.close
        self.data_open = self.data0.open
        self.data_high = self.data0.high
        self.data_low = self.data0.low
        self.data_volume = self.data0.volume

        # Référence pour un ordre en cours (pour éviter les ordres multiples)
        self.order: Optional[bt.Order] = None

        # Stocker le nom de la classe pour un logging plus facile
        self.strategy_name = self.__class__.__name__

    def log(self, message: str, level: int = logging.INFO) -> None:
        """
        Log un message en utilisant le logger du projet.

        Cette méthode est la seule façon de logger depuis une stratégie,
        conformément au manifeste (pas de 'print()').

        Args:
            message (str): Le message à logger.
            level (int): Le niveau de logging (ex: logging.INFO, logging.DEBUG).
        """
        # Récupère la date actuelle du backtest
        current_date = self.data0.datetime.date(0).isoformat()

        # Log avec le format : [DATE @ NOM_STRATEGIE] --- Message
        logger.log(level, f"[{current_date} @ {self.strategy_name}] --- {message}")

    # MODIFIÉ : Le décorateur "@abstractmethod" a été supprimé
    def next(self) -> None:
        """
        Méthode de logique principale (que les enfants doivent surcharger).

        Cette méthode est appelée à chaque nouvelle bougie.
        Elle DOIT être implémentée par toutes les classes enfants.
        """
        # Si une classe enfant hérite de BaseStrategy mais N'IMPLÉMENTE PAS
        # 'next', cette erreur sera levée à l'exécution.
        raise NotImplementedError("La méthode 'next' doit être implémentée.")

    def notify_order(self, order: bt.Order) -> None:
        """
        Gère les notifications de statut des ordres.

        Appelée automatiquement par Backtrader lorsqu'un ordre
        change de statut.
        """
        if order.status in [order.Submitted, order.Accepted]:
            # L'ordre est soumis/accepté par le broker - Rien à faire
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"ACHAT EXÉCUTÉ, "
                    f"Prix: {order.executed.price:.2f}, "
                    f"Coût: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}",
                    level=logging.DEBUG,
                )
            elif order.issell():
                self.log(
                    f"VENTE EXÉCUTÉE, "
                    f"Prix: {order.executed.price:.2f}, "
                    f"Coût: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}",
                    level=logging.DEBUG,
                )

            # L'ordre est complété, réinitialiser la référence
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:

            # --- CORRECTION ---
            # Suppression du 'order.Status[]' en double
            status_str = order.Status[order.status]
            self.log(f"Ordre Échoué/Annulé/Rejeté: {status_str}")
            # --- FIN CORRECTION ---

            self.order = None  # Réinitialiser

    def stop(self) -> None:
        """
        Appelée à la toute fin du backtest.
        Utilisé pour un log récapitulatif.
        """
        self.log(
            f"--- FIN DE LA STRATÉGIE --- "
            f"Portefeuille final: {self.broker.getvalue():.2f}"
        )
