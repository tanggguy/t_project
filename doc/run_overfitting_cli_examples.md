# CLI `run_overfitting.py` — Exemples pratiques

Ce document regroupe les commandes les plus utiles pour lancer les contrôles d’overfitting via `scripts/run_overfitting.py`.

> **Pré-requis**
>
> - Avoir une configuration YAML valide (`optimization + overfitting`), comme `config/overfitting_SimpleMaManaged.yaml`.
> - Activer un environnement virtuel contenant les dépendances du projet.

---

## 1. Exécution standard (tous les checks)

```bash
python scripts/run_overfitting.py --config config/overfitting_SimpleMaManaged.yaml
```

- Lance WFA (avec re-optimisation par fold), OOS, Monte Carlo et Stabilité selon les indicateurs `enabled` du YAML.
- Les résultats sont écrits dans `results/overfitting/<run_id>/<timestamp>/`.

---

## 2. Basé sur les meilleurs paramètres d’une optimisation

Après avoir lancé `scripts/run_optimization.py`, vous pouvez réutiliser le fichier `best_params`:

```bash
python scripts/run_overfitting.py --config config/optimization_SimpleMaManaged.yaml --use-best-params
```

- WFA réutilise `strategy.param_space`.
- OOS / Monte Carlo / Stabilité utilisent `results/optimization/...best_params.yaml`.
- Si le fichier `best_params_path` n’existe pas, seuls les checks ne dépendant pas des paramètres (WFA) sont exécutés.

---

## 3. Limiter aux checks souhaités (`--checks`)

Utilisez `--checks` pour choisir les modules à lancer (séparés par des virgules). Exemples:

### WFA + OOS uniquement

```bash
python scripts/run_overfitting.py \
    --config config/overfitting_SimpleMaManaged.yaml \
    --use-best-params \
    --checks wfa,oos
```

### Monte Carlo (trades) puis tests de stabilité

```bash
python scripts/run_overfitting.py \
    --config config/overfitting_SimpleMaManaged.yaml \
    --use-best-params \
    --checks monte,stability
```

> Alias acceptés: `monte`, `monte_carlo`.

---

## 4. Exemple multi-ticker

Dans la config:

```yaml
  data:
    tickers: ["AAPL", "MSFT"]
    weights: { AAPL: 0.4, MSFT: 0.6 }
    alignment: "intersection"
```

Commande:

```bash
python scripts/run_overfitting.py \
    --config config/overfitting_SimpleMaManaged.yaml \
    --use-best-params
```

- `OverfittingChecker` exécute chaque ticker séparément, agrège les rendements pondérés, et exporte des métriques portefeuille + par ticker.

---

## 5. Exécution rapide (OOS + Monte Carlo)

Adapté pour des validations quotidiennes:

```bash
python scripts/run_overfitting.py \
    --config config/optimization_SimpleMaManaged.yaml \
    --use-best-params \
    --checks oos,monte
```

- Skips la WFA (coûteuse) et se concentre sur les validations hors-échantillon.

---

## 6. Changer la sortie ou le `run_id`

Dans le YAML:

```yaml
overfitting:
  run_id: "sma_managed_prod"
  output_dir: "results/overfitting/prod_checks"
```

Commande:

```bash
python scripts/run_overfitting.py \
    --config config/overfitting_SimpleMaManaged.yaml
```

Les artefacts seront stockés dans `results/overfitting/prod_checks/sma_managed_prod/<timestamp>/`.

---

## 7. Utiliser un autre fichier YAML

```bash
python scripts/run_overfitting.py \
    --config config/overfitting_MaPullback.yaml \
    --use-best-params \
    --checks wfa,stability
```

- Permet d’enchaîner plusieurs scénarios de tests (un YAML par stratégie).

---

## 8. Rappels utiles

- `--config` est obligatoire.
- `--use-best-params` n’est nécessaire que pour les checks basés sur un set de paramètres (OOS, Monte Carlo, Stabilité).
- `--checks` vide → on suit les flags `enabled` du YAML.
- Les logs CLI sont dans `logs/optimization/run_overfitting.log`.
