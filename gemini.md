# 💎 MANIFESTE DU PROJET (gemini.md)

Ce document sert de guide et de "règles du jeu" pour le développement de ce projet de trading quantitatif. Il est destiné à assurer la cohérence, la lisibilité et la maintenabilité du code.

Toute assistance IA (Copilot, Gemini, etc.) doit s'efforcer de suivre ces principes.

## 1. Vue d'ensemble du Projet

* **Objectif :** Créer un framework de backtesting pour des stratégies de swing trading sur actions.
* **Langage :** Python (3.13)
* **Stack Technique Principale :**
    * **Données :** `yfinance`
    * **Analyse / Indicateurs :** `pandas` 
    * **Moteur de Backtest :** `backtrader`
    * **Optimisation :** `optuna`

---

## 2. Principes Généraux

1.  **Simplicité avant tout (KISS) :** Ne pas complexifier inutilement. Une stratégie simple et robuste vaut mieux qu'une usine à gaz fragile.
2.  **Lisibilité :** Le code doit être clair. Utiliser des noms de variables explicites (ex: `fast_ma` plutôt que `f`, `rsi_level` plutôt que `r`).
3.  **Modularité :** Séparer les responsabilités.
    * La gestion des données (`utils/data_manager.py`) est séparée de la logique de stratégie (`strategies/`).
    * Le script de backtest (`scripts/run_backtest.py`) est séparé de la logique d'optimisation (`optimization/`).
4.  **"Data-Driven" :** La logique de `backtrader` doit rester simple. 
5. Pose des questions si necessaire.

---

## 3. Bonnes Pratiques de Code (Style Guide)

### 3.1. Formatage et "Linting"
* **Style :** Suivre la convention **PEP 8** (noms de variables en `snake_case`, noms de classes en `PascalCase`).
* **Imports :** Organiser les imports en haut du fichier, dans cet ordre :
    1.  Bibliothèques natives (ex: `import os`)
    2.  Bibliothèques tierces (ex: `import backtrader as bt`, ``)
    3.  Imports locaux de notre projet (ex: `from utils.data_manager import DataManager`)

### 3.2. "Type Hinting" (Annotations de type)
* **Toujours les utiliser.** C'est crucial pour l'auto-complétion de l'IA et pour éviter les bugs.
* Utiliser le module `typing` (`List`, `Dict`, `Tuple`, `Optional`).
* **Exemple :**
    ```python
    # Mauvais
    def get_data(ticker):
        # ...
        return df

    # Bon
    from typing import Optional

    def get_data(ticker: str, start_date: Optional[str] = None) -> pd.DataFrame:
        # ...
        return df
    ```

### 3.3. Docstrings et Commentaires
* **Docstrings :** Utiliser les "docstrings" (commentaires `"""..."""`) style Google ou Numpy pour toutes les classes et fonctions.
* **Commentaires :** Utiliser des commentaires `#` pour expliquer le "Pourquoi ?" d'une ligne de code complexe, pas le "Quoi ?".

### 3.4. Logging vs. `print()`
* **NE PAS UTILISER `print()`** dans les stratégies ou les modules de la bibliothèque.
* Utiliser le module `logging` de Python.
* Dans une stratégie Backtrader, utiliser `self.log('Mon message...')` (une méthode que nous définirons dans la `BaseStrategy`).

---

## 4. Conventions Spécifiques aux Librairies

### 4.1. `pandas` et Données
* Le DataManager ne gère QUE les données OHLCV brutes.
* Les indicateurs sont calculés directement dans Backtrader (approche native).
* Le DataFrame passé à Backtrader doit contenir uniquement: open, high, low, close, volume.

### 4.2. `backtrader`
* **Alias standard :** Toujours importer `import backtrader as bt`.
* **Noms de Stratégie :** Toujours finir par `Strategy` (ex: `SmaCrossStrategy`, `RsiStrategy`).
* **Paramètres :** Toujours définir les paramètres via le tuple `params = (('fast_ma', 10), ('slow_ma', 30))`.
**indicateurs :**
* Tous les indicateurs sont définis dans `__init__` de la stratégie.
* Utiliser `bt.indicators` pour indicateurs standards (SMA, RSI, MACD, etc.).
* Pour indicateurs custom, créer des classes héritant de `bt.Indicator`.
* Exemples :
```python
    self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
    self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
    self.rsi = bt.indicators.RSI(self.data.close, period=14)
```
* **Accès aux données :**
    * `self.data0` ou `self.data` est la donnée principale.
    * `self.data.close[0]` est le prix de clôture *actuel*.
    * `self.data.close[-1]` est le prix de clôture *précédent*.
* **Indicateurs (si définis dans Backtrader) :**
    * Définir dans `__init__` (ex: `self.rsi = bt.indicators.RSI(...)`).
    * Utiliser dans `next` (ex: `if self.rsi[0] < 30:`).

### 4.3. `optuna`
* **Fonction Objectif :** La fonction à optimiser doit s'appeler `objective` et prendre `trial: optuna.Trial` en argument.
* **Direction :** Toujours spécifier `direction='maximize'` (ex: pour le Sharpe) ou `direction='minimize'` (ex: pour le Drawdown).
* **Retour :** La fonction `objective` DOIT retourner un unique nombre (`float`).
* **Exemple :**
    ```python
    def objective(trial: optuna.Trial) -> float:
        # 1. Suggérer des paramètres
        fast_ma = trial.suggest_int('fast_ma', 5, 20)
        slow_ma = trial.suggest_int('slow_ma', 25, 50)

        # 2. Lancer le backtest (sans plot)
        # ...
        sharpe = results[0].analyzers.sharpe.get_analysis()['sharperatio']

        # 3. Gérer les erreurs et retourner
        if sharpe is None:
            return -1.0 # Pénaliser les essais sans trades
        
        return sharpe
    ```

---

