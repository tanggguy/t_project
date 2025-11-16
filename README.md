# t_project - plateforme de recherche quant Backtrader + Optuna

t_project est un bac a sable de recherche systematique qui combine Backtrader, Optuna et un ensemble d'outils maison pour construire, tester et durcir des strategies actions multi tickers. Le depot propose un pipeline complet: ingestion yfinance avec cache, moteur de backtest portfolio aware, gestion du risque integree, optimisation bayesienne, verifications d'overfitting et generation de rapports HTML/Plotly.

## Apercu rapide
- Python 3.11+, Backtrader 1.9.x, Optuna 4.x, pandas/yfinance/pandas-ta.
- Configurations pilots 100% par YAML (`config/backtest_*.yaml`, `config/optimization_*.yaml`, `config/overfitting_*.yaml`).
- Mode multi ticker: alignement `intersection` ou `union`, ponderations custom (`backtesting/portfolio.py`).
- Strategies modulaires (`strategies/implementations`) fondees sur `BaseStrategy` et `ManagedStrategy` (stop / take profit plug-and-play).
- Risk management: stop loss (fixed, trailing, ATR, supports), take profit, position sizing (fixed, fractional, volatilite).
- Reporting: CSV/JSON dans `results/`, rapports HTML via `reports/`, visualisations Plotly/Dash dans `visualization/`.

## Fonctionnalites majeures
- **Data pipeline**: `utils/data_manager.py` gere telechargement yfinance, mise en cache, validation, timezone, enrichissement indicateurs (pandas-ta). `scripts/download_data.py` supporte tickers multiples, periodes custom et listes marche (`config/markets/*.yaml`).
- **Strategie framework**: `strategies/base_strategy.py` apporte logging/gestion ordres, `strategies/managed_strategy.py` centralise stops et targets. Les implementations du dossier `implementations/` n'ecrivent que la logique d'entree via `next_custom()`.
- **Backtesting engine**: `backtesting/engine.py` encapsule Cerebro (brokers, analyzers, resampling) et se pilote via `scripts/run_backtest.py` (autodiscovery des strategies, multi ticker, generation de rapports).
- **Portfolio analytics**: `backtesting/portfolio.py` agrege les rendements ponderes, calcule equity/log-returns/drawdown et normalise les poids.
- **Position sizing**: `risk_management/position_sizing.py` expose fixed size, fixed fractional, volatility based (ATR). Les strategies peuvent aussi brancher des stops dynamiques depuis `risk_management/stop_loss.py` et `risk_management/take_profit.py`.
- **Optimisation & robustesse**: `optimization/optuna_optimizer.py` prend en charge single ou multi objectif (`optimization/objectives.py`), contraintes custom, et cree `study` SQLite pour Optuna Dashboard. `optimization/overfitting_check.py` ajoute walk-forward, out-of-sample, Monte Carlo, tests de stabilite **et expose les indicateurs cles** : ratios de degradation (`Sharpe_test / Sharpe_train`), probabilites de sur-ajustement (folds avec `Sharpe_test < alpha * Sharpe_train`), p-values Monte Carlo (`p_sharpe_lt_0`, `p_cagr_lt_0`, `p_max_dd_gt_threshold`) + rapports HTML avec badges.
- **Reporting / viz**: `reports/report_generator.py` + templates Jinja2 pour HTML, `reports/overfitting_report.py` pour les batteries de tests, `visualization/` (charts, performance_plots, optimization_plots, dashboard) pour analyses interactives.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate          # ou source .venv/bin/activate sous Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env          # renseigner vos cles (broker, alerting) si besoin
```
Ensuite ajuster `config/settings.yaml` (timezone, capital, commissions) et les fichiers de configuration de backtest/optimisation.

## Workflow type
1. **Preparer les donnees**
   ```bash
   python scripts/download_data.py --tickers AAPL MSFT NVDA --start 2018-01-01 --end 2025-11-01
   python scripts/download_data.py --market sp500 --interval 1d
   ```
2. **Configurer un backtest** dans `config/backtest_*.yaml` (tickers, portfolio, broker, position_sizing, output).  
3. **Lister ou lancer une strategie**
   ```bash
   python scripts/run_backtest.py --list-strategies
   python scripts/run_backtest.py --config config/backtest_configEmaTrendStrategy.yaml
   ```
4. **Analyser les resultats**: equity/metrics en console, CSV/JSON dans `results/backtesting`, rapports HTML dans `reports/generated`, logs dans `logs/backtest/`.
5. **Optimiser** les hyperparametres avec Optuna
   ```bash
   python scripts/run_optimization.py --config config/optimization_EmaTrend.yaml --n-trials 50
   optuna-dashboard sqlite:///results/optimization/optuna_studies.db --host 127.0.0.1 --port 4200
   ```
6. **Verifier l'overfitting / robustesse**
   ```bash
   python scripts/run_overfitting.py --config config/overfitting_SimpleMaManaged.yaml --checks wfa,oos,monte,stability
   python scripts/run_overfitting.py --config config/optimization_SimpleMaManaged.yaml --use-best-params
   ```

## Commandes CLI utiles
| Usage | Commande |
| --- | --- |
| Telecharger donnees | `python scripts/download_data.py --tickers AAPL --no-cache` |
| Backtest helper shell | `./scripts/run_backtest.sh config/backtest_config.yaml` |
| Lancer optimisation | `python scripts/run_optimization.py --config config/optimization_RsiOversold.yaml --n-trials 100` |
| Dashboard Optuna | `optuna-dashboard sqlite:///results/optimization/optuna_studies.db --host 127.0.0.1 --port 4200` |
| Overfitting checker | `python scripts/run_overfitting.py --config config/overfitting_SimpleMaManaged.yaml --checks wfa,oos,monte,stability` |
| Tests unitaires | `pytest -q` |

## Configuration
- `config/settings.yaml` : timezone, chemins cache, capital initial, commissions, analytics (periods_per_year, risk_free_rate).
- `config/backtest_*.yaml` : parametres strategy/backtest (tickers, portfolio, broker, sizing, report). Exemple:
  ```yaml
  backtest:
    strategy: "EmaTrend"
    strategy_params:
      ema_trend: 190
      ema_fast: 19
      ema_slow: 30
      use_stop_loss: true
      stop_loss_type: "atr"
      stop_loss_atr_mult: 3.25
      use_take_profit: true
      take_profit_type: "atr"
    data:
      tickers: ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
      start_date: "2018-01-01"
      end_date: "2025-11-01"
      interval: "1d"
    portfolio:
      alignment: "intersection"
      per_ticker_reports: true
    broker:
      initial_capital: 10000
      commission_pct: 0.0005
      slippage_pct: 0.0005
    position_sizing:
      enabled: true
      method: "fixed"
      fixed:
        pct_size: 0.4
    output:
      save_results: true
      report:
        enable: true
        out_dir: "reports/generated"
        include_trades: true
  ```
- `config/optimization_*.yaml` : definition de l'espace Optuna (`param_space`, `objective`, `study`, `portfolio`).
- `config/overfitting_*.yaml` : definition des folds WFA, fenetres OOS, scenarios Monte Carlo, tolerances de stabilite.
- `config/markets/*.yaml` : listes de tickers predefinies (`sp500.yaml`, `cac40.yaml`).
- `config/strategies/` : presets specifiques a chaque strategie.

## Strategies disponibles
| Fichier | Classe principale | Idee |
| --- | --- | --- |
| `ma_crossover.py` | `MaCrossoverStrategy` | Cross MA rapide/lente, exemple minimal. |
| `simple_ma_managed_strategy.py` | `SimpleMaManagedStrategy` | Trend following avec stops/TP geres par ManagedStrategy. |
| `sma_pullback_managed_strategy.py` | `SmaPullbackManagedStrategy` | Pullback vers SMA lente avec risk management dynamique. |
| `ema_trend_strategy.py` | `EmaTrendStrategy` | Filtre de tendance EMA 190 + momentum court terme. |
| `rsi_mean_reversion_managed_strategy.py` | `RsiMeanReversionManagedStrategy` | Mean reversion RSI avec risk manager. |
| `rsi_oversold.py` | `RsiOversoldStrategy` | Setup contrarien simple (utilise dans les tests unitaires). |
| `macd_momentum.py` | `MacdMomentumStrategy` | MACD + filtre de momentum multi timeframe. |
| `donchian_breakout_managed_strategy.py` | `DonchianBreakoutManagedStrategy` | Breakout canal Donchian avec stops trailes. |

Chaque strategie herite de `BaseStrategy` et, pour la plupart, de `ManagedStrategy` afin d'obtenir le pipeline de stops (`FixedStopLoss`, `TrailingStopLoss`, `ATRStopLoss`, `SupportResistanceStop`) et de take profits (`FixedTakeProfit`, `ATRTakeProfit`, `SupportResistanceTakeProfit`) definis dans `risk_management/`.

## Architecture du depot
| Chemin | Contenu |
| --- | --- |
| `data/` | `raw/`, `processed/`, `cache/` pour l'historique yfinance. |
| `config/` | Settings globaux, backtests, optimisations, overfitting, listes de marche. |
| `scripts/` | CLI (download, backtest, optimisation, overfitting, live_scanner). |
| `strategies/` | BaseStrategy, ManagedStrategy, implementations + doc dedie. |
| `backtesting/` | `engine`, analyzers, portfolio helpers, validators walk-forward. |
| `optimization/` | Optuna optimizer, objectifs, overfitting checker. |
| `risk_management/` | Position sizing, stops, take profit, portfolio manager a venir. |
| `utils/` | Config loader, data manager, data processor, logger, market calendar. |
| `visualization/` | Charts Plotly, performance/optimization plots, dashboard. |
| `reports/` | Templates Jinja2, exporteurs HTML/Markdown, rapports generes. |
| `results/` | `backtest_results/`, `optimization_studies/`, `best_parameters/`. |
| `logs/` | Execution logs classes par module (`backtest/`, `optimization/`, `errors/`). |
| `tests/` | Unitaires et integration (`tests/unit/test_strategies/test_rsi_oversold.py`, fixtures). |
| `notebooks/` | Workflow R&D (01_data_exploration -> 04_optimization_results). |
| `doc/` | Guides (`QUICK_START.md`, `ManagedStrategy.md`, `optimization.md`, `RISK_MANAGEMENT.md`). |

## Reporting et visualisation
- Les rapports de backtest s'ecrivent dans `reports/generated/` avec equity, underwater, table trades et metriques (Sharpe, Sortino, Calmar, PnL, drawdown, expectancy).
- Les resultats bruts (CSV/JSON) sont stockes dans `results/backtesting/` et `results/optimization/`.
- `reports/overfitting_report.py` synthetise WFA/OOS/Monte Carlo/stabilite dans `results/overfitting/<run_id>/<timestamp>/`. Chaque execution genere un `index.html` avec badges (robust, borderline, overfitted) qui renvoient vers des rapports detaillees et graphiques Plotly.
  Exemple pour ouvrir l'index apres un run :
  ```bash
  python -m webbrowser results/overfitting/sma_managed_checks16110019/20251116-010101/index.html
  ```
- `visualization/dashboard.py` et `visualization/charts.py` peuvent etre plugges dans des notebooks ou un serveur Dash pour comparer plusieurs strategies.

## Tests et qualite
- Tests unitaires: `pytest -q` (ex: `tests/unit/test_strategies/test_rsi_oversold.py`).
- Couverture: `pytest --cov=strategies --cov=utils --cov=backtesting`.
- Linting/formatting: Black, isort et Ruff peuvent etre ajoutes via pre-commit (non inclus par defaut, mais requirements contiennent les dependances).
- TODO fonctionnel: `TODO.md` suit la roadmap (phases 1 -> 14). Se referer a ce fichier avant d'ajouter une nouvelle fonctionnalite.
- Observabilite: `utils/logger.py` configure un logger structure, partage par tous les modules. Les scripts ecrivent dans `logs/<domaine>/*.log`.

## Documentation complementaire
- `doc/QUICK_START.md` : demarrage rapide sur la gestion des stop-loss.
- `doc/ManagedStrategy.md` : reference complete de la classe ManagedStrategy.
- `doc/optimization.md` : meilleures pratiques Optuna, multi objectif, contraintes.
- `doc/RISK_MANAGEMENT.md` et `doc/SL_TP.md` : details des stop/take profit.
- `GEMINI.md` : backlog idees et architecture generale.

## Prochaines etapes
Consulter `TODO.md` pour la roadmap complete: extension du broker (paper/live), moteur event-driven, scanner live (`scripts/live_scanner.py`), gestionnaire de portefeuille cross strategie, pipeline de donnees production et monitoring (alertes Telegram/email via `.env`). Les sections Phase 12-14 definissent les chantiers structurels restant a couvrir.

Bon build et bons backtests !
