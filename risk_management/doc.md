# 🚀 Guide de Démarrage Rapide : Risk Management

Ce document explique comment intégrer les modules de `stop_loss` et `take_profit` dans vos stratégies `backtrader`.

L'objectif est de garder la logique de `next()` de votre stratégie propre et de déléguer les calculs de sortie au `risk_management`.

## 1. Principes de Base de l'Intégration

L'intégration suit toujours 3 étapes :

1.  **`__init__(self)`** :
    * Initialiser les **indicateurs** requis (ex: ATR).
    * Initialiser les **classes de gestion** (ex: `ATRStopLoss`, `FixedTakeProfit`).
    * Définir des variables d'état pour suivre la position (ex: `self.stop_level = None`, `self.target_level = None`).

2.  **`next(self)`** :
    * **Si pas en position :** Vérifier les signaux d'entrée. Si un signal apparaît, passer un ordre (`self.buy()` ou `self.sell()`).
    * **Si en position (et que les niveaux ne sont pas encore définis) :** C'est la première bougie *après* l'exécution de l'ordre. On calcule et stocke les niveaux de SL et TP.
    * **Si en position (et que les niveaux sont définis) :**
        * Vérifier si le prix actuel déclenche le SL (`sl_manager.should_trigger(...)`).
        * Vérifier si le prix actuel déclenche le TP (`tp_manager.should_trigger(...)`).
        * Si l'un ou l'autre est déclenché, clôturer la position.

3.  **`notify_order(self, order)`** (Optionnel mais recommandé) :
    * Lorsque l'ordre de sortie (SL ou TP) est complété, réinitialiser vos variables d'état (ex: `self.stop_level = None`, `self.target_level = None`).
    * Si vous utilisez un `TrailingStopLoss`, appeler `self.sl_manager.reset()` ici.

---

## 2. Exemple Complet : Stratégie avec `ATRStopLoss` et `FixedTakeProfit`

Voici un exemple concret d'une stratégie (fictive) qui utilise un Stop Loss basé sur l'ATR et un Take Profit fixe de 8%.

```python
# --- 1. Bibliothèques natives ---
import logging
from typing import Optional

# --- 2. Bibliothèques tierces ---
import backtrader as bt

# --- 3. Imports locaux du projet ---
from strategies.base_strategy import BaseStrategy
from risk_management.stop_loss import ATRStopLoss
from risk_management.take_profit import FixedTakeProfit
# (Supposons que nous ayons aussi un signal d'entrée)
from strategies.signals import EntrySignalIndicator 

class AtrSlFixedTpStrategy(BaseStrategy):
    """
    Stratégie exemple :
    - ENTRÉE : Signal fictif (EntrySignalIndicator)
    - SORTIE (Perte) : Stop Loss basé sur l'ATR (x2)
    - SORTIE (Gain) : Take Profit fixe (8%)
    """
    
    params = (
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('tp_pct', 0.08),
    )

    def __init__(self) -> None:
        """Initialise les indicateurs, les gestionnaires de risque et l'état."""
        super().__init__()
        
        # 1. Initialiser les indicateurs
        self.atr = bt.indicators.ATR(
            self.data0, 
            period=self.p.atr_period
        )
        self.entry_signal = EntrySignalIndicator(self.data0) # Fictif
        
        # 2. Initialiser les gestionnaires de risque
        self.sl_manager = ATRStopLoss(
            atr_multiplier=self.p.atr_multiplier,
            atr_period=self.p.atr_period
        )
        self.tp_manager = FixedTakeProfit(
            tp_pct=self.p.tp_pct
        )
        
        # 3. Initialiser l'état de la position
        self.active_stop_level: Optional[float] = None
        self.active_target_level: Optional[float] = None

    def next(self) -> None:
        """Logique principale (bougie par bougie)."""
        
        current_price = self.data_close[0]
        
        # --- GESTION DE POSITION OUVERTE ---
        if self.position:
            
            # Cas 1: La position vient d'être ouverte, on définit les niveaux
            if self.active_stop_level is None:
                # Récupérer le prix d'entrée réel
                entry_price = self.position.price 
                atr_value = self.atr[0]
                
                # Calculer et stocker les niveaux
                self.active_stop_level = self.sl_manager.calculate_stop(
                    entry_price=entry_price,
                    atr_value=atr_value,
                    position_type="long" # (Adapter si "short")
                )
                self.active_target_level = self.tp_manager.calculate_target(
                    entry_price=entry_price,
                    position_type="long"
                )
                self.log(
                    f"Nouvelle Position @ {entry_price:.2f}. "
                    f"Cible TP: {self.active_target_level:.2f}, "
                    f"Cible SL: {self.active_stop_level:.2f}",
                    level=logging.INFO
                )
                return # Attendre la prochaine bougie pour vérifier

            # Cas 2: La position est ouverte, on vérifie les sorties
            
            # Vérification Take Profit
            if self.tp_manager.should_trigger(
                current_price=current_price,
                target_level=self.active_target_level,
                position_type="long"
            ):
                self.log(f"TAKE PROFIT DÉCLENCHÉ @ {current_price:.2f}", logging.INFO)
                self.sell() # (ou self.close())
                return

            # Vérification Stop Loss
            if self.sl_manager.should_trigger(
                current_price=current_price,
                stop_level=self.active_stop_level,
                position_type="long"
            ):
                self.log(f"STOP LOSS DÉCLENCHÉ @ {current_price:.2f}", logging.INFO)
                self.sell() # (ou self.close())
                return

        # --- GESTION D'ENTRÉE (SI PAS DE POSITION) ---
        else:
            # Réinitialiser l'état (au cas où, géré aussi dans notify_order)
            if self.active_stop_level is not None:
                self.log("Réinitialisation des niveaux SL/TP", logging.DEBUG)
                self.active_stop_level = None
                self.active_target_level = None
            
            # Vérifier le signal d'entrée
            if self.entry_signal[0] > 0: # (Logique de signal fictive)
                self.log(f"Signal d'ACHAT détecté @ {current_price:.2f}", logging.INFO)
                self.buy()

    def notify_order(self, order: bt.Order) -> None:
        """Gérer la réinitialisation de l'état après la clôture."""
        # Laisser BaseStrategy gérer le logging
        super().notify_order(order)

        # Si l'ordre est complété et que c'est une vente (clôture)
        if order.status == order.Completed and order.issell():
            # Réinitialiser nos niveaux pour la prochaine transaction
            self.active_stop_level = None
            self.active_target_level = None
            
            # Si on utilisait un TrailingStopLoss, on le réinitialiserait ici :
            # if isinstance(self.sl_manager, TrailingStopLoss):
            #     self.sl_manager.reset()
```
## 3. Utilisation des Différents Types de TP/SL
A. Fixed (Fixe)
Le plus simple. Ne nécessite aucun indicateur.

__init__:
```python
from risk_management.stop_loss import FixedStopLoss

from risk_management.take_profit import FixedTakeProfit

self.sl_manager = FixedStopLoss(stop_pct=0.03) (Stop à 3%)

self.tp_manager = FixedTakeProfit(tp_pct=0.06) (TP à 6%)

next (calcul):

sl = self.sl_manager.calculate_stop(entry_price, position_type="long")

tp = self.tp_manager.calculate_target(entry_price, position_type="long")
```
B. ATR (Volatilité)
Nécessite un indicateur ATR.

__init__:
```python
from risk_management.stop_loss import ATRStopLoss

from risk_management.take_profit import ATRTakeProfit

self.atr = bt.indicators.ATR(self.data0, period=14)

self.sl_manager = ATRStopLoss(atr_multiplier=2.0, atr_period=14)

self.tp_manager = ATRTakeProfit(atr_multiplier=4.0, atr_period=14)

next (calcul):

atr_val = self.atr[0]

sl = self.sl_manager.calculate_stop(entry_price, atr_val, "long")

tp = self.tp_manager.calculate_target(entry_price, atr_val, "long")
```
C. Support / Résistance (Pivots)
Utilise les méthodes intégrées pour trouver les pivots.

__init__:
```python
from risk_management.stop_loss import SupportResistanceStop

from risk_management.take_profit import SupportResistanceTakeProfit

self.sl_manager = SupportResistanceStop(lookback_period=30, buffer_pct=0.005)

self.tp_manager = SupportResistanceTakeProfit(lookback_period=30, buffer_pct=0.005)

next (calcul):
```
Note : Ces méthodes doivent être appelées AVANT l'entrée pour déterminer si le trade vaut le coup, ou juste après l'entrée.

Pour un Achat (Long):

SL : supports = self.sl_manager.find_support(self, num_levels=1)

TP : resistances = self.tp_manager.find_resistance(self, num_levels=1)

if supports and resistances:

sl = self.sl_manager.calculate_stop(entry_price, support_level=supports[0], position_type="long")

tp = self.tp_manager.calculate_target(entry_price, resistance_level=resistances[0], position_type="long")