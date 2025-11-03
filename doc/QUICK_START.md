# 🚀 DÉMARRAGE RAPIDE - Stop Loss Management

## ⚡ En 30 secondes

```python
# 1. Importer
from risk_management import FixedStopLoss

# 2. Créer
stop = FixedStopLoss(stop_pct=0.02)  # 2% stop

# 3. Calculer
stop_level = stop.calculate_stop(entry_price=100, position_type='long')
# → 98.0

# 4. Vérifier
if stop.should_trigger(current_price=97, stop_level=98, position_type='long'):
    print("STOP LOSS !")  # ← Se déclenche ici
```

---

## 📊 Comparaison Visuelle des 4 Stops

### 1️⃣ FixedStopLoss (Le Plus Simple)

```
Prix
│
110 ┤        🟢  Prix monte, stop fixe
105 ┤      🟢
100 ┤    🔵 Entry
95  ┤
98  ├────🔴──────────────  Stop fixe @ 98 (2%)
│
└──────────────────────> Temps
```

**Quand l'utiliser ?** Stratégies simples, débutants

---

### 2️⃣ TrailingStopLoss (Suit le Prix)

```
Prix
│
110 ┤            🟢  Prix au plus haut
105 ┤        🟢
106.7├──────────🔴  Stop suit @ 106.7 (110 - 3%)
100 ┤    🔵 Entry
98  ├────🔴  Stop initial @ 97 (100 - 3%)
│
└──────────────────────> Temps
   ↑    Le stop MONTE avec le prix
   ↑    mais ne RECULE JAMAIS
```

**Quand l'utiliser ?** Tendances fortes, laisser courir les profits

---

### 3️⃣ ATRStopLoss (S'adapte à la Volatilité)

```
Marché CALME (ATR faible):
────────────────
Prix : 100
ATR  : 1.0
Stop : 98.0 (serré)  ← Stop proche

Marché VOLATILE (ATR élevé):
━━━━━━━━━━━━━━━━
Prix : 100
ATR  : 5.0
Stop : 90.0 (large) ← Stop éloigné, évite les faux signaux
```

**Quand l'utiliser ?** Tous marchés, professionnel

---

### 4️⃣ SupportResistanceStop (Niveaux Techniques)

```
Prix
│
110 ┤  Résistance ═══════════
105 ┤         🟢
100 ┤     🔵 Entry
95  ┤  Support ═══════════
94.5├────🔴  Stop sous le support (avec buffer)
│
└──────────────────────> Temps
```

**Quand l'utiliser ?** Trading technique, niveaux clairs

---

## 🎯 Intégration en 3 Étapes

### Étape 1 : Dans `__init__`

```python
def __init__(self):
    # Choisir votre stop
    self.stop = FixedStopLoss(stop_pct=0.02)
    
    # Variables de suivi
    self.entry_price = None
    self.stop_level = None
```

### Étape 2 : À l'Entrée

```python
def next(self):
    if not self.position and self.buy_signal():
        # Entrer en position
        self.entry_price = self.data.close[0]
        self.buy()
        
        # Calculer le stop
        self.stop_level = self.stop.calculate_stop(
            entry_price=self.entry_price,
            position_type='long'
        )
        print(f"Entry @ {self.entry_price}, Stop @ {self.stop_level}")
```

### Étape 3 : Surveillance

```python
    # (suite de next)
    elif self.position:
        # Vérifier le stop à chaque barre
        if self.stop.should_trigger(
            current_price=self.data.close[0],
            stop_level=self.stop_level,
            position_type='long'
        ):
            self.close()
            print("STOP LOSS DÉCLENCHÉ !")
```

---

## 🔧 Configuration Avancée

### Optimisation avec Optuna

```python
def objective(trial):
    # Suggérer le type de stop
    stop_type = trial.suggest_categorical('stop_type', 
                                         ['fixed', 'trailing', 'atr'])
    
    if stop_type == 'fixed':
        stop_pct = trial.suggest_float('stop_pct', 0.01, 0.05)
        stop = FixedStopLoss(stop_pct=stop_pct)
    
    elif stop_type == 'trailing':
        trail_pct = trial.suggest_float('trail_pct', 0.02, 0.06)
        stop = TrailingStopLoss(trail_pct=trail_pct)
    
    elif stop_type == 'atr':
        multiplier = trial.suggest_float('atr_mult', 1.5, 3.0)
        stop = ATRStopLoss(atr_multiplier=multiplier)
    
    # Lancer le backtest avec ce stop
    results = run_backtest_with_stop(stop)
    return results['sharpe_ratio']
```

---

## 📋 Checklist Avant Production

- [ ] **Tests passent** : `pytest test_stop_loss.py -v`
- [ ] **Stop défini dans `__init__`**
- [ ] **Entry price sauvegardé**
- [ ] **Stop calculé à l'entrée**
- [ ] **Stop vérifié à chaque barre**
- [ ] **Position fermée si déclenché**
- [ ] **Trailing reset entre positions**

---

## 🎓 Exemples par Niveau

### 🌱 Débutant : Stop Fixe Simple

```python
class BeginnerStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SMA(period=20)
        self.stop = FixedStopLoss(stop_pct=0.02)  # 2%
        self.entry_price = None
    
    def next(self):
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.entry_price = self.data.close[0]
                self.buy()
                self.stop_level = self.stop.calculate_stop(
                    entry_price=self.entry_price,
                    position_type='long'
                )
        else:
            if self.stop.should_trigger(
                self.data.close[0], self.stop_level, 'long'
            ):
                self.close()
```

### 🚀 Intermédiaire : Trailing Stop

```python
class IntermediateStrategy(bt.Strategy):
    params = (('trail_pct', 0.03),)
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(period=14)
        self.stop = TrailingStopLoss(trail_pct=self.p.trail_pct)
    
    def notify_order(self, order):
        if order.status == order.Completed and order.isbuy():
            self.stop.reset()  # ← IMPORTANT pour trailing
            self.entry_price = order.executed.price
    
    def next(self):
        if self.position:
            # Le stop s'ajuste automatiquement
            self.stop_level = self.stop.calculate_stop(
                entry_price=self.entry_price,
                current_price=self.data.close[0],
                position_type='long'
            )
            # Vérification...
```

### 🎯 Avancé : ATR Stop Adaptatif

```python
class AdvancedStrategy(bt.Strategy):
    def __init__(self):
        self.atr = bt.indicators.ATR(period=14)
        self.stop = ATRStopLoss(atr_multiplier=2.0)
        self.macd = bt.indicators.MACD()
    
    def next(self):
        if self.position:
            # Stop s'adapte à la volatilité
            self.stop_level = self.stop.calculate_stop(
                entry_price=self.entry_price,
                atr_value=self.atr[0],  # ← Volatilité actuelle
                position_type='long'
            )
            # Le stop sera large si volatilité élevée
```

---

## ⚠️ Erreurs Courantes à Éviter

### ❌ Oublier de reset le TrailingStop

```python
# MAUVAIS
def notify_order(self, order):
    if order.isbuy():
        self.entry_price = order.executed.price
        # ❌ Pas de reset !

# BON
def notify_order(self, order):
    if order.isbuy():
        self.stop.reset()  # ✅ Reset à chaque nouvelle position
        self.entry_price = order.executed.price
```

### ❌ Ne pas mettre à jour le Trailing Stop

```python
# MAUVAIS
def next(self):
    if self.position:
        # ❌ Stop calculé une seule fois
        if not hasattr(self, 'stop_level'):
            self.stop_level = self.stop.calculate_stop(...)

# BON
def next(self):
    if self.position:
        # ✅ Recalculé à chaque barre pour trailing
        self.stop_level = self.stop.calculate_stop(
            entry_price=self.entry_price,
            current_price=self.data.close[0],  # ← Mise à jour
            position_type='long'
        )
```

### ❌ Utiliser print() au lieu de logging

```python
# MAUVAIS
print("Stop déclenché")  # ❌ Contre le manifeste

# BON
import logging
logger = logging.getLogger(__name__)
logger.info("Stop déclenché")  # ✅ Conforme
```

---

## 📞 Aide Rapide

| Problème | Solution |
|----------|----------|
| **Import error** | Vérifier que `risk_management/` est dans le projet |
| **Stop ne se déclenche pas** | Vérifier `position_type` ('long' vs 'short') |
| **Trailing ne suit pas** | Appeler `calculate_stop()` à chaque `next()` |
| **ATR error** | Créer `self.atr = bt.indicators.ATR(...)` dans `__init__` |
| **Tests échouent** | Installer pytest : `pip install pytest` |

---

## 🎉 Vous êtes prêt !

Vous avez maintenant tout ce qu'il faut pour :
- ✅ Protéger vos positions
- ✅ Laisser courir les profits (trailing)
- ✅ S'adapter à la volatilité (ATR)
- ✅ Respecter les niveaux techniques (S/R)

**Bon trading ! 📈**

---

*Consultez `LIVRAISON.md` pour plus de détails*
