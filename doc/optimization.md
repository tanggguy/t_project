# 📈 Optimisation des Stratégies (Optuna)

Ce document décrit comment lancer des optimisations de paramètres pour les stratégies Backtrader au moyen d’Optuna. Vous y trouverez la structure des fichiers, la configuration YAML dédiée et les commandes à exécuter.

---

## 1. Vue d’ensemble

| Élément | Rôle |
| --- | --- |
| `optimization/optuna_optimizer.py` | Classe `OptunaOptimizer`, charge les données, configure Backtrader et pilote Optuna. |
| `scripts/run_optimization.py` | CLI pour lancer une optimisation à partir d’un fichier YAML. |
| `config/optimization.yaml` | Exemple de configuration (stratégie, données, espace de recherche, étude Optuna, sorties). |

Fonctionnalités principales :
- Chargement des données via `DataManager` (cache, filtrage, validation d’index).
- Découverte dynamique de n’importe quelle stratégie héritant de `BaseStrategy`.
- Support des paramètres fixes et des espaces de recherche (entiers, flottants, catégoriels, échelle logarithmique).
- Ajout automatique des analyseurs Backtrader (Sharpe, Drawdown, Returns, Trades).
- Gestion optionnelle du position sizing (Fixed, Fixed Fractional, Volatility-based).
- Contraintes simples, par exemple `fast_period < slow_period`.
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
    metric: "sharpe"
    penalize_no_trades: -1.0
    min_trades: 1
    enforce_fast_slow_gap: true

  study:
    study_name: "sma_managed_opt"
    direction: "maximize"
    storage: "sqlite:///results/optimization/optuna_studies.db"
    load_if_exists: true
    sampler: "tpe"
    sampler_kwargs:
      seed: 42
    pruner: "median"
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
5. La fonction objectif renvoie le Sharpe (ou la pénalité si contrainte violée / trop peu de trades).
6. Optuna sauvegarde les essais et met à jour l’étude dans SQLite.

---

## 4. Sorties générées

| Fichier | Contenu |
| --- | --- |
| `results/optimization/optuna_studies.db` | Base SQLite contenant l’étude (compatible dashboard). |
| `results/optimization/sma_managed_opt.pkl` | DataFrame pickle des essais (colonnes trial/value/params/user_attrs). |
| `results/optimization/sma_managed_opt_trials.csv` | Historique des essais au format CSV. |
| `results/optimization/sma_managed_opt_best_params.yaml` | Meilleurs paramètres et valeur objectif. |
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
- **Objectifs additionnels** : Étendre `_compute_objective` dans `OptunaOptimizer` pour supporter d’autres métriques (Sortino, Calmar…).
- **Contraintes avancées** : Ajouter des règles dans `_validate_params` au-delà de `fast_period < slow_period`.
- **Position sizing** : Activer selon les besoins de la stratégie testée.
- **Parallélisation** : Ajuster `n_jobs` (>1) et vérifier que le cache de données est prêt pour éviter les téléchargements concurrents.

---

## 7. Dépannage rapide

| Symptôme | Cause probable | Solution |
| --- | --- | --- |
| « param_space ne peut pas être vide » | Bloc `param_space` manquant | Définir au moins un paramètre optimisable. |
| Erreur « Impossible de charger des données » | Période invalide ou cache absent | Vérifier `start_date`/`end_date`, vider/rafraîchir le cache si nécessaire. |
| Objective = -1.0 | Contrainte violée ou pas assez de trades | Inspecter `user_attrs` (`constraint_violation`, `total_trades`). |
| Dashboard vide | Mauvaise URL SQLite | Vérifier que CLI et dashboard pointent vers le même `sqlite:///path`. |
| Refresh lent/erreurs multi-jobs | Téléchargement de données concurrent | Pré-chauffer le cache en lançant un backtest simple avant l’optimisation. |

---

## 8. Prochaines étapes

1. Cloner `config/optimization.yaml` pour chaque famille de stratégie.
2. Enrichir les métriques (ex : combiner Sharpe, drawdown, win rate).
3. Injecter les meilleurs paramètres dans `config/backtest_config.yaml` pour validation finale.
4. Centraliser l’analyse des résultats (notebooks ou scripts dédiés).

Bonnes optimisations !
