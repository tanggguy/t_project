# 💎 MANIFESTE DU PROJET & RÈGLES DE DÉVELOPPEMENT (gemini.md)

Ce document est la **Source de Vérité** pour le développement du projet `t_project`.
Toute intervention (humaine ou IA) doit respecter scrupuleusement ces principes pour garantir la maintenabilité et la robustesse du système.

---

## 1. 🧠 Philosophie & Concepts Architecturaux

### 1.1. Principes Fondamentaux

* **KISS (Keep It Simple, Stupid) :** La complexité est l'ennemie de la robustesse. Si une fonction fait plus de 30 lignes, elle est probablement trop complexe. Privilégier la lisibilité à l'astuce technique.
* **DRY (Don't Repeat Yourself) :** Ne jamais dupliquer de logique.
  * *Exemple :* Le calcul de la taille de position se fait **uniquement** dans `risk_management/position_sizing.py`, jamais dans la stratégie elle-même.
* **Single Responsibility Principle (SRP) :**
  * `strategies/` : Décide *quand* entrer/sortir.
  * `risk_management/` : Décide *combien* acheter et *où* placer les sécurités (SL/TP).
  * `config/` : Stocke les paramètres (pas de "magic numbers" dans le code).

### 1.2. Architecture Modulaire

Le projet est conçu comme un assemblage de blocs indépendants :

1. **Data Layer** (`utils/data_manager.py`) : Ingestion, cache et nettoyage.
2. **Strategy Layer** (`strategies/`) : Logique de trading pure, héritant de `ManagedStrategy`.
3. **Execution Layer** (`backtesting/engine.py`) : Orchestration via Cerebro.
4. **Optimization Layer** (`optimization/`) : Recherche de paramètres et validation (Overfitting).

---

## 2. 🤖 Instructions Spécifiques pour l'IA

1. **Ne réinvente pas la roue :** Avant de proposer une nouvelle fonction, vérifie si elle n'existe pas déjà dans `utils/`, `backtesting/` ou `risk_management/`.
2. **Réfléchis en "Configuration" :** Si tu dois changer une valeur (période de MA, stop loss, ticker), ne modifie pas le code Python. Propose la modification du fichier YAML correspondant dans `config/`.
3. **Protection du Capital avant tout :** Lors de la création d'une stratégie, la gestion du risque (Stop Loss) n'est pas une option, c'est une obligation. Utilise toujours les mécanismes de `ManagedStrategy`.
4. **Contexte Global :** Prends en compte que le code tourne souvent en mode multi-tickers et avec des optimisations Optuna. Évite les variables globales ou les états non réinitialisés dans `__init__`.

---

## 3. 📝 Standards de Code (Style Guide)

### 3.1. Python & PEP 8

* **Formatage :** Respect strict de la **PEP 8**.
  * Indentation : 4 espaces (pas de tabulations).
  * Lignes : Maximum 100 caractères (souplesse pour lisibilité).
* **Naming Conventions :**
  * Variables/Fonctions : `snake_case` (ex: `calculate_moving_average`).
  * Classes : `PascalCase` (ex: `ExponentialMovingAverage`).
  * Constantes : `UPPER_CASE` (ex: `DEFAULT_RISK_PCT`).

### 3.2. Type Hinting (Strictement Obligatoire)

Le typage statique aide à la compréhension et réduit les bugs.

* **Mauvais :**

    ```python
    def run(data, params):
        ...
    ```

* **Bon :**

    ```python
    from typing import Dict, Any, List
    import pandas as pd

    def run(data: pd.DataFrame, params: Dict[str, Any]) -> List[float]:
        ...
    ```

### 3.3. Documentation & Commentaires

* **Docstrings :** Format Google ou NumPy obligatoire pour chaque classe et méthode publique.

    ```python
    def get_data(self, ticker: str) -> pd.DataFrame:
        """
        Récupère les données OHLCV pour un ticker donné.

        Args:
            ticker (str): Le symbole de l'actif (ex: 'AAPL').

        Returns:
            pd.DataFrame: DataFrame contenant les données historiques.
        """
    ```

* **Commentaires :** Expliquer le *POURQUOI*, pas le *COMMENT*. Le code dit ce qu'il fait, le commentaire explique l'intention métier.

### 3.4. Logging

* **Interdit :** `print()`.
* **Obligatoire :** Utiliser `log()` dans les stratégies Backtrader ou `logging.getLogger(__name__)` ailleurs.

---

## 4. 🛠 Implémentation des Stratégies

### 4.1. Héritage

Toute stratégie doit hériter de **`strategies.managed_strategy.ManagedStrategy`**.

* Cela active automatiquement la gestion des Stop Loss, Take Profit et du Reporting.
* Ne jamais hériter directement de `bt.Strategy` sauf pour des tests techniques très bas niveau.

### 4.2. Structure type d'une stratégie

```python
from strategies.managed_strategy import ManagedStrategy
import backtrader as bt

class MaNewStrategy(ManagedStrategy):
    # 1. Paramètres par défaut (modifiables via YAML)
    params = (
        ('period_fast', 10),
        ('period_slow', 30),
    )

    def __init__(self):
        # 2. Appel obligatoire au constructeur parent
        super().__init__()
        
        # 3. Définition des indicateurs (optimisation vitesse)
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.period_fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.period_slow)

    def next_custom(self):
        # 4. Logique d'entrée UNIQUEMENT (ManagedStrategy gère les sorties SL/TP)
        # Utiliser self.buy() simplement. Le Sizer gère la quantité.
        if self.sma_fast[0] > self.sma_slow[0]:
            self.buy()

5. 🔄 Workflow et Commandes
L'IA doit privilégier l'utilisation des scripts d'entrée plutôt que des snippets isolés.

Acquisition de Données : python scripts/download_data.py --tickers AAPL MSFT --start 2020-01-01

Backtest (Recherche) : python scripts/run_backtest.py --config config/backtest_configExemple.yaml

Optimisation (Calibration) : python scripts/run_optimization.py --config config/optimization_Exemple.yaml

Validation (Robustesse) : C'est l'étape critique pour éviter l'overfitting. python scripts/run_overfitting.py --config config/overfitting_Exemple.yaml

6. Références
Documentation Technique : doc/DOCUMENTATION_TRADING_COMPLETE.md

Roadmap : TODO.md


### Points clés de cette mise à jour :

1.  **Concept KISS & DRY** : Expliqués clairement en section 1.1.
2.  **Standards PEP & Type Hinting** : Section 3 détaillée avec des exemples "Bon/Mauvais".
3.  **ManagedStrategy** : Mise en avant comme composant central obligatoire pour l'héritage des stratégies, assurant la cohérence du Risk Management.
4.  **Architecture "Config-Driven"** : Rappel que le code ne doit pas contenir de paramètres
