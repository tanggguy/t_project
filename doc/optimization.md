# 📈 Optimisation des Stratégies (Optuna)

Ce document décrit comment lancer des optimisations de paramètres pour les stratégies Backtrader au moyen d’Optuna. Vous y trouverez la structure des fichiers, la configuration YAML dédiée et les commandes à exécuter.

---

## 1. Vue d’ensemble

| Élément | Rôle |
| --- | --- |
| `optimization/optuna_optimizer.py` | Classe `OptunaOptimizer`, charge les données, configure Backtrader et pilote Optuna (mono & multi-objectifs). |
| `optimization/objectives.py` | Fonctions utilitaires pour agréger les métriques, définir les directions Optuna et déclarer des contraintes custom. |
| `scripts/run_optimization.py` | CLI pour lancer une optimisation à partir d’un fichier YAML (affiche aussi les fronts de Pareto). |
| `config/optimization.yaml` / `config/optimization_example.yaml` | Exemples de configuration (stratégie, données, objectifs simples ou multiples, étude Optuna, sorties). |

Fonctionnalités principales :
- Chargement des données via `DataManager` (cache, filtrage, validation d’index).
- Découverte dynamique de n’importe quelle stratégie héritant de `BaseStrategy`.
- Support des paramètres fixes et des espaces de recherche (entiers, flottants, catégoriels, échelle logarithmique).
- Ajout automatique des analyseurs Backtrader (Sharpe, Drawdown, Returns, Trades).
- Gestion optionnelle du position sizing (Fixed, Fixed Fractional, Volatility-based).
- Contraintes simples et avancées (gap EMA, min trades, max drawdown) + pénalités custom.
- Objectifs simples (Sharpe, pondérations) ou multi-objectifs (Sharpe vs drawdown, CAGR vs Ulcer) grâce à Optuna NSGA-II / MOTPE.
- Stockage compatible Optuna Dashboard (`sqlite:///…`).
- Exports: CSV des essais, YAML des meilleurs paramètres, DataFrame pickle des trials.

---

## 2. Configuration (`config/optimization.yaml`)

```yaml
optimization:
  strategy:
    name: "SimpleMaManaged"
    module: "strategies.implementations.simple_ma_managed_strategy"
    class_name: "SimpleMaManagedStrategy"
    fixed_params:
      use_stop_loss: true
      stop_loss_type: "fixed"
      use_take_profit: true
      take_profit_type: "atr"
    param_space:
      fast_period: [5, 20, 1]
      slow_period: [25, 50, 5]
      stop_loss_pct: [0.01, 0.05, 0.005]
      take_profit_atr_mult:
        type: "float"
        low: 2.0
        high: 5.0
        step: 0.5
      stop_loss_type:
        type: "categorical"
        choices: ["fixed", "atr", "trailing"]

  data:
    ticker: "AAPL"
    start_date: "2024-01-01"
    end_date: "2025-11-01"
    interval: "1d"
    use_cache: true

  # Mode multi-ticker (facultatif)
  # data:
  #   tickers:
  #     - "AAPL"
  #     - "MSFT"
  #   weights:
  #     AAPL: 0.4
  #     MSFT: 0.6
  #   start_date: "2018-01-01"
  #   end_date: "2025-11-01"

  # Agrégation portefeuille (alignement des dates, rapports par ticker)
  portfolio:
    alignment: "intersection"

  broker:
    initial_capital: 10000.0
    commission_pct: 0.001
    slippage_pct: 0.0

  position_sizing:
    enabled: false
    method: "fixed"
    fixed:
      pct_size: 0.5

  objective:
    mode: "single"              # "single" (défaut) ou "multi"
    aggregation: "metric"       # "metric" ou "weighted_sum"
    metric: "sharpe"
    # weights:
    #   sharpe: 1.0
    #   max_drawdown: -0.5       # Utiliser si aggregation=weighted_sum
    penalize_no_trades: -1.0
    min_trades: 1
    enforce_fast_slow_gap: true
    # Exemple multi-objectifs :
    # mode: "multi"
    # targets:
    #   - name: "sharpe"
    #     direction: "maximize"
    #   - name: "max_drawdown"
    #     direction: "minimize"
    # constraints:
    #   min_trades: 5
    #   max_drawdown: 0.30
    #   fast_slow_gap: 1

  study:
    study_name: "sma_managed_opt"
    direction: "maximize"        # Ignoré si objective.mode = multi
    storage: "sqlite:///results/optimization/optuna_studies.db"
    load_if_exists: true
    sampler: "tpe"               # Multi-objectifs : préférer "nsga2" ou "motpe"
    sampler_kwargs:
      seed: 42
    pruner: "median"             # Multi-objectifs : mettre "none"
    pruner_kwargs: {}
    n_trials: 50
    timeout: null
    n_jobs: 1
    show_progress_bar: true

  output:
    save_study: true
    study_path: "results/optimization/sma_managed_opt.pkl"
    save_trials_csv: true
    trials_csv_path: "results/optimization/sma_managed_opt_trials.csv"
    log_file: "logs/optimization/optuna_optimizer.log"
    dump_best_params: true
    best_params_path: "results/optimization/sma_managed_opt_best_params.yaml"
```

> ℹ️ **Mode multi-ticker** — lorsque `data.tickers` est présent, chaque ticker est
> backtesté dans un run indépendant, puis les rendements sont agrégés selon les
> `weights` (ou pondération égale par défaut). La section `portfolio` contrôle la
> façon d'aligner les dates (`intersection` / `union`) et reste facultative pour
> les configurations mono-ticker.

### Objectifs single vs multi

- **Single** (`mode: "single"`)
  - `aggregation: "metric"` : renvoie directement la métrique indiquée (`metric`).
  - `aggregation: "weighted_sum"` : combine plusieurs métriques via `weights` (positifs = récompense, négatifs = pénalité).
- **Multi** (`mode: "multi"`)
  - Définissez `targets` (nom + direction) pour chaque objectif. Les alias disponibles sont définis dans `optimization/objectives.py` (sharpe, sortino, cagr, max_drawdown, ulcer, pnl, etc.).
  - Optez pour `sampler: "nsga2"` (support des contraintes) ou `sampler: "motpe"`.
  - Les contraintes optionnelles (`objective.constraints`) deviennent des fonctions `constraints_func` pour Optuna (≤ 0 = faisable).
  - `dump_best_params` exporte alors l’ensemble des trials Pareto plutôt qu’un seul `best_value`.

Consultez `config/optimization_example.yaml` pour un template complet couvrant la plupart des options.

### Remarques
- `param_space` accepte :
  - `[min, max, step]` pour des int/float (utilise `suggest_int` / `suggest_float`).
  - une liste de valeurs pour un choix catégoriel (`suggest_categorical`).
  - un dictionnaire détaillé `type: float/int/categorical`, avec support `log`, `choices`, etc.
- Les clefs de `fixed_params` écrasent la suggestion (pratique pour verrouiller l’ATR ou des paramètres de risk management).
- La section `study` contrôle sampler, pruner, stockage (nécessaire pour Optuna Dashboard) et limites (`n_trials`, `timeout`).

---

## 3. Lancer une optimisation

```
python scripts/run_optimization.py \
    --config config/optimization.yaml \
    --n-trials 20 \
    --no-progress-bar
```

Options principales :
- `--config`: chemin vers le fichier YAML (défaut `config/optimization.yaml`).
- `--n-trials`, `--timeout`, `--n-jobs`: surchargent les valeurs du YAML.
- `--no-progress-bar`: force la désactivation de la barre de progression.

Étapes internes :
1. Découverte de toutes les stratégies (`strategies/implementations`).
2. Résolution de la stratégie via `name` ou `module`/`class_name`.
3. `OptunaOptimizer` charge les données, configure Cerebro, applique le broker et le position sizing.
4. À chaque essai, la stratégie est exécutée avec les paramètres suggérés ; les analyzers fournissent Sharpe, drawdown, retours, nombre de trades.
5. La fonction objectif renvoie soit une valeur unique (Sharpe, pondération custom) soit un tuple (multi-objectifs). Les contraintes/pénalités remplacent la valeur si besoin.
6. Optuna sauvegarde les essais et met à jour l’étude dans SQLite. La CLI affiche la meilleure valeur (single) ou la liste des points de Pareto (multi).

---

## 4. Sorties générées

| Fichier | Contenu |
| --- | --- |
| `results/optimization/optuna_studies.db` | Base SQLite contenant l’étude (compatible dashboard). |
| `results/optimization/sma_managed_opt.pkl` | DataFrame pickle des essais (colonnes trial/value/params/user_attrs). |
| `results/optimization/sma_managed_opt_trials.csv` | Historique des essais au format CSV. |
| `results/optimization/sma_managed_opt_best_params.yaml` | Meilleurs paramètres (single) ou front de Pareto complet (multi). |
| `logs/optimization/optuna_optimizer.log` | Logs détaillés d’exécution Optuna. |
| `logs/optimization/run_optimization.log` | Logs de la CLI. |

Les `user_attrs` des trials incluent : `strategy_params`, `constraint_violation`, `sharpe_ratio`, `total_trades`, `won_trades`, `lost_trades`, `max_drawdown`, `total_return`, `annualized_return`, `final_value`, `initial_capital`, `pnl`, `pnl_pct`.

---

## 5. Dashboard Optuna

Grâce au stockage SQLite, le dashboard officiel peut suivre l’étude en direct :

```
optuna-dashboard --storage sqlite:///results/optimization/optuna_studies.db --study-name sma_managed_opt --host 127.0.0.1 --port 4200
```

Cela offre l’historique des optimisations, l’importance des paramètres, les diagrammes parallèles, etc. Réexécuter l’optimisation avec `load_if_exists: true` reprend la même étude.

---

## 6. Conseils de personnalisation

- **Nouvelles stratégies** : Ajoutez la classe dans `strategies/implementations` (héritage `BaseStrategy`) et spécifiez son `param_space`.
- **Objectifs additionnels** : Ajoutez vos métriques/pondérations dans `optimization/objectives.py` (nouveaux alias, agrégations, tuples multi-objectifs).
- **Contraintes avancées** : Combinez `_validate_params` (contrôles locaux) et `objective.constraints` (min_trades, max_drawdown, fast_slow_gap) pour piloter NSGA-II.
- **Position sizing** : Activer selon les besoins de la stratégie testée.
- **Parallélisation** : Ajuster `n_jobs` (>1) et vérifier que le cache de données est prêt pour éviter les téléchargements concurrents.

---

## 7. Dépannage rapide

| Symptôme | Cause probable | Solution |
| --- | --- | --- |
| « param_space ne peut pas être vide » | Bloc `param_space` manquant | Définir au moins un paramètre optimisable. |
| Erreur « Impossible de charger des données » | Période invalide ou cache absent | Vérifier `start_date`/`end_date`, vider/rafraîchir le cache si nécessaire. |
| Objective = -1.0 | Contrainte violée ou pas assez de trades | Inspecter `user_attrs` (`constraint_violation`, `total_trades`). |
| RuntimeError "single best trial" | Étude lancée en mode multi-objectifs | Lire les résultats via `study.best_trials` (affichés automatiquement par `run_optimization.py`). |
| Dashboard vide | Mauvaise URL SQLite | Vérifier que CLI et dashboard pointent vers le même `sqlite:///path`. |
| Refresh lent/erreurs multi-jobs | Téléchargement de données concurrent | Pré-chauffer le cache en lançant un backtest simple avant l’optimisation. |

---

## 8. Prochaines étapes

1. Cloner `config/optimization.yaml` pour chaque famille de stratégie.
2. Enrichir les métriques (ex : combiner Sharpe, drawdown, win rate).
3. Injecter les meilleurs paramètres dans `config/backtest_config.yaml` pour validation finale.
4. Centraliser l’analyse des résultats (notebooks ou scripts dédiés).

Bonnes optimisations !

---

## 9. Prévention de l'overfitting

`optimization/overfitting_check.py` regroupe les analyses de robustesse alimentées par Optuna et par les métriques configurées dans `config/settings.yaml`. Le module produit maintenant, pour chaque scénario, des **ratios de dégradation**, des **probabilités de sur-ajustement** et des **p-values Monte Carlo** utilisées pour colorer les badges « Robust / Borderline / Overfitted » des rapports HTML.

### 9.1 Indicateurs de robustesse

#### Walk-forward ancré (WFA)
- `degradation_ratio = mean(Sharpe_test) / mean(Sharpe_train)` : un ratio < 1 indique une perte de performance entre optimisation et validation.
- `test_vs_train_gap = mean(Sharpe_test) - mean(Sharpe_train)` : gap absolu pour repérer la dérive.
- `frac_test_sharpe_lt_0` : fraction de folds avec Sharpe test négatif.
- `frac_test_sharpe_lt_alpha_train` : probabilité de sur-ajustement basée sur le seuil `alpha` (par défaut 0.5) défini dans `analytics.overfitting.wfa.alpha`. Un fold est dit “mauvais” si `Sharpe_test < alpha * Sharpe_train`.
- Les seuils `robust_min` / `overfit_max` par indicateur se trouvent sous `analytics.overfitting.wfa.*`. Ils déterminent les badges affichés dans les rapports.

#### Fenêtres out-of-sample (OOS)
- `oos_degradation_ratio = mean(Sharpe_oos) / Sharpe_train_reference` où `Sharpe_train_reference` est issu du meilleur backtest in-sample.
- Médiane / minimum des Sharpes OOS (`oos_sharpe_median`, `oos_sharpe_min`) et `frac_oos_sharpe_lt_0` (proportion de fenêtres négatives).
- Les règles `analytics.overfitting.oos.mean_sharpe` et `analytics.overfitting.oos.frac_sharpe_lt_0` contrôlent les badges.

#### Simulation Monte Carlo
- Bootstrap par blocs sur retours ou trades (`source: returns/trades`) afin d’obtenir :
  - `p_sharpe_lt_0` : proportion de simulations avec Sharpe négatif.
  - `p_cagr_lt_0` : probabilité d’un CAGR négatif.
  - `p_max_dd_gt_threshold` : probabilité que la perte maximale dépasse `max_drawdown.threshold` (0.30 par défaut).
  - `prob_negative` : fréquence des trajectoires dont la valeur finale repasse sous le capital initial.
- Ces probabilités font office de p-values Monte Carlo. Les règles associées se règlent dans `analytics.overfitting.monte_carlo.*`.

#### Tests de stabilité locale
- Génération de perturbations ±`perturbation`% sur chaque paramètre puis calcul du `robust_fraction` (part des variations dont le Sharpe reste ≥ `threshold`, 0.95 par défaut).
- Les seuils se règlent via `analytics.overfitting.stability.robust_fraction`.

### 9.2 Rapports HTML et badges

Chaque exécution `run_overfitting.py` crée `results/overfitting/<run_id>/<timestamp>/index.html`. L’index liste les sections (WFA, OOS, Monte Carlo, Stability) sous forme de cartes avec badges colorés :

```
WFA (Robust)      → ratio 0.93, 8% de folds < α · train
Monte Carlo (Borderline) → p_sharpe_lt_0 = 0.24, p_max_dd_gt_30% = 0.32
Stability (Robust) → 87% de variations conservent ≥ 95% du Sharpe
```

Chaque carte pointe vers un rapport détaillé (`wfa_report.html`, `monte_carlo_report.html`, etc.) qui inclut tables CSV et graphiques Plotly (scatter train/test, histogrammes de simulations, heatmap des perturbations, etc.). En cas de dépendances Plotly absentes, un fallback HTML minimal est généré mais conserve les badges et les métriques.

### 9.3 Exemple programmatique

Les résultats peuvent aussi être récupérés directement en Python :

```python
from optimization.overfitting_check import OverfittingChecker
from strategies.implementations.simple_ma_managed_strategy import SimpleMaManagedStrategy

checker = OverfittingChecker(
    strategy_class=SimpleMaManagedStrategy,
    param_space={
        "fast_period": [5, 20, 1],
        "slow_period": [30, 120, 5],
        "take_profit_atr_mult": {"type": "float", "low": 1.5, "high": 4.0, "step": 0.25},
    },
    data_config={
        "ticker": "AAPL",
        "start_date": "2018-01-01",
        "end_date": "2024-12-31",
        "interval": "1d",
        "use_cache": True,
    },
    broker_config={"initial_capital": 20000, "commission_pct": 0.001},
)

wfa_summary = checker.walk_forward_analysis()
```

Consultez ensuite `results/overfitting/<run_id>/<timestamp>/` pour les CSV (`*_summary.csv`, `*_folds.csv`, `monte_carlo_simulations.csv`, etc.) et l’index HTML enrichi de badges.

---

## 10. Dashboard d'Optimisation (Streamlit)

Le projet inclut une interface graphique pour lancer, surveiller et analyser les optimisations sans toucher à la ligne de commande.

### 10.1 Lancement

```bash
streamlit run visualization/dashboard.py
```

### 10.2 Fonctionnalités

1.  **Sélection de Configuration** : Charge automatiquement les fichiers `config/optimization_*.yaml`.
2.  **Éditeur de Paramètres (Overrides)** :
    -   Permet de modifier à la volée les tickers, dates, et intervalles.
    -   Détecte automatiquement l'espace de recherche (`param_space`) et génère des champs de saisie adaptés (sliders pour int/float, listes pour categorical).
    -   Estime la taille de la grille de recherche (nombre de combinaisons discrètes).
3.  **Monitoring Temps Réel** :
    -   Affiche l'état du job (Running, Done, Failed).
    -   **ETA Dynamique** : Calcule le temps restant estimé basé sur une moyenne mobile des 20 derniers essais.
    -   **Barre de Progression** : Visualise l'avancement global.
    -   Affiche les meilleurs paramètres et la meilleure valeur trouvée en direct.
4.  **Actions Post-Optimisation** :
    -   **Générer Rapport** : Lance un backtest complet avec les meilleurs paramètres et ouvre le rapport HTML.
    -   **Overfitting Checks** : Lance les tests de robustesse (WFA, Monte Carlo, etc.) sur la meilleure configuration.

### 10.3 Architecture Technique

Le dashboard repose sur trois composants clés :

#### A. Frontend (`visualization/dashboard.py`)
Interface Streamlit qui gère les interactions utilisateur. Elle ne lance pas l'optimisation directement dans son processus, mais délègue cette tâche pour ne pas bloquer l'interface.

#### B. Runner & State Management (`optimization/dashboard_runner.py`)
Gère le cycle de vie du processus d'optimisation :
-   **Job Config** : Construit une configuration d'exécution (`OptimizationJobConfig`).
-   **Processus Détaché** : Lance `scripts/run_optimization.py` via `subprocess.Popen`.
-   **Lock File** : Utilise un fichier `tmp-output/current_optimization.json` pour stocker le PID, le statut et les métriques de progression. Cela permet au dashboard de survivre à un redémarrage sans perdre la trace du job en cours.
-   **ETA Calculation** : Implémente la logique d'estimation du temps restant (`_compute_eta`) en lisant l'historique des essais depuis la base Optuna.

#### C. Système d'Overrides (`optimization/config_overrides.py`)
Permet de modifier la configuration YAML de base sans altérer le fichier original :
-   **`apply_overrides`** : Fonction qui prend un dictionnaire de config et applique des modifications ciblées sur `data`, `study`, et `strategy.param_space`.
-   **Fichiers Temporaires** : Les configurations modifiées sont sauvegardées dans `tmp-output/dashboard_config_<timestamp>.yaml` et passées au script d'optimisation.
