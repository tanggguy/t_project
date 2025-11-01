# 🛡️ Risk Management - Module de Gestion des Risques

Module de gestion des risques pour le framework de backtesting de stratégies de swing trading.

## 📋 Vue d'ensemble

Ce module fournit des outils pour gérer les risques de trading, notamment :
- **Stop Loss** : 4 types de stop loss différents
- **Take Profit** : (à venir dans Phase 5.2)
- **Position Sizing** : (à venir dans Phase 5.3)

---

## 🎯 Stop Loss Disponibles

### 1. FixedStopLoss - Stop Loss Fixe

Stop loss en pourcentage fixe par rapport au prix d'entrée. Simple et prévisible.

**Avantages :**
- Simplicité d'utilisation
- Risque connu à l'avance
- Facile à backtester

**Inconvénients :**
- Ne s'adapte pas aux conditions de marché
- Peut être trop serré en période volatile

**Exemple d'utilisation :**
```python
from risk_management import FixedStopLoss

# Créer un stop loss de 2%
stop = FixedStopLoss(stop_pct=0.02)

# Calculer le niveau de stop pour une position long
entry_price = 100.0
stop_level = stop.calculate_stop(entry_price=entry_price, position_type='long')
print(f"Stop loss @ {stop_level}")  # 98.0

# Vérifier si le stop doit être déclenché
current_price = 97.5
if stop.should_trigger(current_price, stop_level, position_type='long'):
    print("Stop loss déclenché !")
```

---

### 2. TrailingStopLoss - Stop Loss Suiveur

Stop loss qui suit le prix lorsqu'il évolue favorablement, mais ne recule jamais.

**Avantages :**
- Laisse courir les profits
- Sécurise les gains progressivement
- Idéal pour les tendances fortes

**Inconvénients :**
- Peut sortir trop tôt en cas de pullback
- Plus complexe à gérer

**Exemple d'utilisation :**
```python
from risk_management import TrailingStopLoss

# Créer un trailing stop de 3%
stop = TrailingStopLoss(trail_pct=0.03)

# Prix d'entrée : 100
# Prix monte à 110
stop_level = stop.calculate_stop(
    entry_price=100,
    current_price=110,
    position_type='long'
)
print(f"Stop @ {stop_level}")  # 106.7 (110 - 3%)

# Prix recule à 105 : le stop ne recule PAS
stop_level = stop.calculate_stop(
    entry_price=100,
    current_price=105,
    position_type='long'
)
print(f"Stop @ {stop_level}")  # Toujours 106.7

# Réinitialiser pour une nouvelle position
stop.reset()
```

---

### 3. ATRStopLoss - Stop Loss basé sur l'ATR

Stop loss adaptatif basé sur la volatilité du marché (Average True Range).

**Avantages :**
- S'adapte automatiquement à la volatilité
- Stop large en période volatile, serré en période calme
- Réduit les faux signaux

**Inconvénients :**
- Nécessite le calcul de l'ATR
- Plus complexe à optimiser

**Exemple d'utilisation :**
```python
import backtrader as bt
from risk_management import ATRStopLoss

class MyStrategy(bt.Strategy):
    def __init__(self):
        # Créer l'indicateur ATR
        self.atr = bt.indicators.ATR(self.data, period=14)
        
        # Créer le stop loss ATR
        self.stop = ATRStopLoss(atr_multiplier=2.0, atr_period=14)
    
    def next(self):
        if self.position:
            # Calculer le stop en utilisant l'ATR actuel
            stop_level = self.stop.calculate_stop(
                entry_price=self.entry_price,
                atr_value=self.atr[0],
                position_type='long'
            )
            
            # Vérifier le déclenchement
            if self.stop.should_trigger(
                current_price=self.data.close[0],
                stop_level=stop_level,
                position_type='long'
            ):
                self.close()
```

**Paramètres recommandés :**
- `atr_period` : 14 (standard)
- `atr_multiplier` : 
  - 1.5 pour stop serré (day trading)
  - 2.0-2.5 pour stop moyen (swing trading)
  - 3.0+ pour stop large (position trading)

---

### 4. SupportResistanceStop - Stop basé sur les Niveaux Clés

Stop loss placé sous le support (long) ou au-dessus de la résistance (short).

**Avantages :**
- Logique technique solide
- Respecte la structure du marché
- Réduit les stop loss arbitraires

**Inconvénients :**
- Nécessite l'identification des niveaux S/R
- Stop peut être éloigné du prix d'entrée

**Exemple d'utilisation :**
```python
import backtrader as bt
from risk_management import SupportResistanceStop

class MyStrategy(bt.Strategy):
    def __init__(self):
        self.stop = SupportResistanceStop(
            lookback_period=20,  # Chercher sur 20 périodes
            buffer_pct=0.005     # Buffer de 0.5%
        )
    
    def next(self):
        if not self.position:
            # Signal d'achat...
            if buy_signal:
                self.buy()
                
                # Trouver le support le plus proche
                supports = self.stop.find_support(self, num_levels=1)
                
                if supports:
                    support_level = supports[0]
                    self.stop_level = self.stop.calculate_stop(
                        entry_price=self.data.close[0],
                        support_level=support_level,
                        position_type='long'
                    )
        else:
            # Vérifier le stop
            if self.stop.should_trigger(
                current_price=self.data.close[0],
                stop_level=self.stop_level,
                position_type='long'
            ):
                self.close()
```

---

## 🧪 Tests Unitaires

Lancer les tests :
```bash
# Installer pytest si nécessaire
pip install pytest

# Lancer les tests
python -m pytest test_stop_loss.py -v

# Avec couverture de code
pip install pytest-cov
python -m pytest test_stop_loss.py --cov=risk_management --cov-report=html
```

---

## 📊 Comparaison des Stop Loss

| Type | Complexité | Adaptabilité | Usage Recommandé |
|------|-----------|-------------|------------------|
| **Fixed** | ⭐ | ❌ | Débutants, stratégies simples |
| **Trailing** | ⭐⭐ | ⭐⭐ | Tendances fortes, swing trading |
| **ATR** | ⭐⭐⭐ | ⭐⭐⭐ | Tous marchés, stratégies avancées |
| **S/R** | ⭐⭐⭐⭐ | ⭐⭐ | Trading technique, support clair |

---

## 🎨 Intégration dans une Stratégie

Exemple complet d'intégration :

```python
import backtrader as bt
from risk_management import FixedStopLoss, TrailingStopLoss

class MyTradingStrategy(bt.Strategy):
    params = (
        ('stop_type', 'trailing'),
        ('stop_pct', 0.03),
    )
    
    def __init__(self):
        # Indicateurs
        self.sma = bt.indicators.SMA(self.data.close, period=20)
        
        # Stop loss manager
        if self.p.stop_type == 'fixed':
            self.stop_manager = FixedStopLoss(stop_pct=self.p.stop_pct)
        else:
            self.stop_manager = TrailingStopLoss(trail_pct=self.p.stop_pct)
        
        self.entry_price = None
        self.stop_level = None
    
    def next(self):
        if not self.position:
            # Logique d'entrée
            if self.data.close[0] > self.sma[0]:
                self.entry_price = self.data.close[0]
                self.buy()
                
                # Calculer stop initial
                self.stop_level = self.stop_manager.calculate_stop(
                    entry_price=self.entry_price,
                    current_price=self.data.close[0],
                    position_type='long'
                )
        else:
            # Mise à jour du stop (pour trailing)
            if self.p.stop_type == 'trailing':
                self.stop_level = self.stop_manager.calculate_stop(
                    entry_price=self.entry_price,
                    current_price=self.data.close[0],
                    position_type='long'
                )
            
            # Vérifier le déclenchement
            if self.stop_manager.should_trigger(
                current_price=self.data.close[0],
                stop_level=self.stop_level,
                position_type='long'
            ):
                self.close()
```

---

## 📝 Bonnes Pratiques

### 1. **Toujours définir un stop loss**
Ne jamais trader sans protection. Un stop loss limite les pertes potentielles.

### 2. **Adapter le stop au timeframe**
- Day trading : Stop serré (1-2%)
- Swing trading : Stop moyen (2-5%)
- Position trading : Stop large (5-10%)

### 3. **Tester différents types**
Chaque marché et stratégie a son stop optimal. Utiliser l'optimisation avec Optuna.

### 4. **Ratio Risk/Reward**
Viser au minimum un ratio 1:2 (risque 1% pour gagner 2%).

### 5. **Ne pas déplacer le stop contre soi**
Une fois le stop défini, ne jamais le reculer (sauf pour le sécuriser).

---

## 🔧 Configuration Logging

Pour activer les logs :

```python
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Les stop loss loggeront automatiquement
stop = FixedStopLoss(stop_pct=0.02)
```

---

## 📚 Ressources Complémentaires

- [📖 Manifeste du Projet](../gemini.md)
- [🎯 Phase 5 : Risk Management](../README.md#phase-5)
- [📊 Backtrader Documentation](https://www.backtrader.com/docu/)

---

## 🚀 Prochaines Étapes

Phase 5.2 : **Take Profit**
- Fixed Take Profit
- Trailing Take Profit
- Risk/Reward based Take Profit

Phase 5.3 : **Position Sizing**
- Fixed Position Size
- Kelly Criterion
- Volatility-based Sizing

---

## 📄 Licence

Ce code fait partie du projet de trading quantitatif et suit les mêmes règles que le manifeste principal.

---

**Développé avec ❤️ selon le manifeste KISS (Keep It Simple, Stupid)**