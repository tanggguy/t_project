TODO COMPLET - Projet Trading Python : commenté = fait
<!-- 📋 Phase 1 : Setup Initial (Semaine 1)
    1.1 Environnement de développement

    Créer un nouveau repository Git
    Initialiser un environnement virtuel Python (venv ou conda)
    Créer le fichier requirements.txt avec les dépendances de base :

    yfinance
    backtrader
    pandas-ta
    optuna
    pandas
    numpy
    matplotlib
    plotly
    pyyaml
    python-dotenv
    jupyter
    pytest

    Installer toutes les dépendances
    Créer la structure de dossiers du projet
    Configurer .gitignore pour Python
    Créer README.md avec description du projet
    Setup du logging de base (utils/logger.py)

    1.2 Configuration

    Créer config/settings.yaml avec paramètres globaux :

    Périodes par défaut
    Commissions broker
    Capital initial
    Timezone

    Créer .env pour API keys (si nécessaire plus tard)
    Créer utils/config_loader.py pour charger les configurations

    📊 Phase 2 : Gestion des Données (Semaine 1-2)
    2.1 Data Manager

    Créer utils/data_manager.py avec classe DataManager :

    Méthode download_data() pour yfinance
    Méthode save_to_cache() pour sauvegarder en CSV
    Méthode load_from_cache() pour charger depuis cache
    Gestion des erreurs de téléchargement
    Validation des données (trous, valeurs aberrantes)

    2.2 Scripts de données

    Créer scripts/download_data.py :

    Arguments CLI (ticker, période, intervalle)
    Mode batch pour télécharger multiple tickers
    Barre de progression pour téléchargements

    Créer liste de tickers par marché :

    config/markets/sp500.yaml
    config/markets/cac40.yaml

    2.3 Data preprocessing

    Créer utils/data_processor.py :

    Calcul des returns
    Détection et gestion des outliers
    Resampling (aggrégation temporelle)

    2.4 Validation

    Notebook 01_data_exploration.ipynb :

    Visualisation des données téléchargées
    Statistiques descriptives
    Vérification de la qualité des données
    Test de téléchargement sur 5-10 tickers

    🎯 Phase 3 : Première Stratégie Simple (Semaine 2-3)
    3.1 Base Strategy

    Créer strategies/base_strategy.py :

    Classe abstraite héritant de bt.Strategy
    Méthodes template : __init__, next, notify_order
    Logging intégré
    Gestion basique des ordres

    3.2 Stratégie Moving Average Crossover

    Créer strategies/implementations/ma_crossover.py :

    Paramètres : fast_period, slow_period
    Logique : achat sur golden cross, vente sur death cross
    Position sizing simple (100% du capital)
    Pas de stop-loss pour commencer

    3.3 Premier Backtest

    Créer backtesting/engine.py :

    Classe BacktestEngine
    Configuration Cerebro (capital, commission)
    Ajout des analyseurs basiques (returns, sharpe)
    Méthode run() qui retourne les résultats

    3.4 Script de test

    Créer scripts/run_backtest.py :
    Scanne automatiquement strategies/implementations/
    Détecte toutes les classes héritant de BaseStrategy
    Affiche les paramètres par défaut de chaque stratégie
    3️⃣ Paramètres par Défaut Automatiques
    python scripts/run_backtest.py --config config/backtest_config.yaml

    Charger données d'un ticker (ex: AAPL)
    Lancer backtest sur 2 ans
    Afficher résultats basiques (P&L, nombre trades)

    Vérifier que tout fonctionne bout en bout

    📈 Phase 4 : Stategie (Semaine 3-4)

    Stratégies avec indicateurs

    Créer strategies/implementations/rsi_oversold.py :

    Achat sur RSI < 30, vente sur RSI > 70

    Créer strategies/implementations/macd_momentum.py :

    Trading sur croisements MACD

 Notebook 02_strategy_development.ipynb :

 Tests visuels des indicateurs
 Backtests comparatifs

    💰 Phase 5 : Risk Management (Semaine 4-5)
    5.1 Stop Loss et Take Profit

    Créer risk_management/stop_loss.py :

    Fixed stop loss (%)
    Trailing stop loss
    ATR-based stop loss
    Support/Resistance stops

    Créer risk_management/take_profit.py

    5.2 Position Sizing

    Créer risk_management/position_sizing.py :

    Fixed fractional (risquer X% par trade)
    
    Volatility-based sizing
 

    5.3 Intégration

    Modifier base_strategy.py pour intégrer risk management
    Ajouter paramètres de risque dans configs

    🔧 Phase 6 : Optimisation Basique (Semaine 5-6)
    6.1 Setup Optuna

    Créer optimization/optuna_optimizer.py :

    Classe OptunaOptimizer
    Définition de l'espace de recherche
    Fonction objectif (maximize Sharpe ratio)
    Sauvegarde des études

    6.2 Parameter Spaces

    Créer optimization/parameter_spaces.py :

    Espaces pour MA Crossover
    Espaces pour RSI strategy
    Contraintes et dépendances

    6.3 Première optimisation

    Script scripts/run_optimization.py :

    Optimiser MA Crossover sur données historiques
    100 trials minimum
    Sauvegarder meilleurs paramètres

    Visualisation des résultats Optuna

    📊 Phase 7 : Métriques et Analyse (Semaine 6-7)
    7.1 Analyzers avancés

    Créer backtesting/analyzers/performance.py :

    Sharpe, Sortino, Calmar ratios
    Win rate, Profit factor
    Average trade, Best/Worst trade

    7.2 Drawdown analysis

    Créer backtesting/analyzers/drawdown.py :

    Maximum drawdown
    Durée des drawdowns
    Recovery time
    Underwater curve

    7.3 Reporting

    Créer reports/report_generator.py :

    Template HTML pour rapports
    Graphiques performance
    Tableau des trades
    Export PDF -->

<!-- 🎨 Phase 8 : Visualisation (Semaine 7-8)
8.1 Charts de base

 Créer visualization/charts.py :

 Candlestick 
 Points d'entrée/sortie

 est ce que 8.1 vaut le coup, que apporter en plus du plot natif de backtrader ? -->

<!-- 8.2 Dashboard

 Créer visualization/dashboard.py :

 Dashboard Plotly/Dash
 Comparaison multi-stratégies
 Métriques temps réel
 Sélection période analyse

8.3 Notebook d'analyse

 03_backtest_analysis.ipynb :

 Analyse détaillée des trades
 Patterns gagnants/perdants
 Analyse par période -->

<!-- 🚀 Phase 9 : Optimisation Avancée (Semaine 8-9) -->
<!-- 9.1 Overfitting prevention

 Créer optimization/overfitting_check.py :

 Walk-forward analysis
 Out-of-sample testing
 Monte Carlo simulation
 Stability tests -->

<!-- esquisser scripts/run_overfitting.py prêt à l’emploi, -->

<!-- 9.2 Multi-objective

 Modifier optimization/objectives.py :

 Optimisation multi-objectifs
 Trade-off return/risque
 Contraintes custom -->
<!-- Amelioration module overffiting :
    1. Conception des métriques d’overfitting / robustesse

    Définir les indicateurs WFA/OOS cibles
    degradation_ratio = test_sharpe_mean / train_sharpe_mean
    test_vs_train_gap = test_sharpe_mean - train_sharpe_mean
    Fréquence de folds “mauvais” :
    - [ ] frac_test_sharpe_lt_0 = proportion de folds avec test_sharpe < 0
    - [ ] frac_test_sharpe_lt_alpha_train = proportion de folds avec test_sharpe < α * train_sharpe (choisir α, ex. 0.5)
    Définir les zones qualitatives (“badge” de robustesse)
    Choisir des seuils pour degradation_ratio et frac_test_sharpe_lt_alpha_train, ex :
    - [ ] Robuste : degradation_ratio >= 0.8 et frac_test_sharpe_lt_alpha_train <= 0.2
    - [ ] Borderline : entre les 2 zones
    - [ ] Sur‑ajusté : degradation_ratio <= 0.5 ou frac_test_sharpe_lt_alpha_train >= 0.5
    Définir les indicateurs Monte Carlo cibles
    p_sharpe_lt_0 = proportion des simulations avec sharpe_ratio < 0
    p_cagr_lt_0 = proportion des simulations avec cagr < 0
    
    <!-- Décider où stocker les scores de synthèse
    Ajouter un petit bloc robustness_summary dans les dictionnaires summary WFA/OOS/Monte Carlo/stabilité
    <!-- Prévoir d’utiliser ces champs dans overfitting_report.render_overfitting_index pour afficher les badges --> -->
<!-- 2. Implémentation des nouvelles métriques côté OverfittingChecker

    Enrichir le résumé WFA (optimization/overfitting_check.py)
    Dans walk_forward_analysis, après calcul des listes train_sharpes / test_sharpes et fold_results :
    Calculer degradation_ratio, test_vs_train_gap
    Calculer frac_test_sharpe_lt_0 et frac_test_sharpe_lt_alpha_train
    Ajouter ces valeurs dans summary (ex. clés "degradation_ratio", "frac_test_sharpe_lt_0", etc.)
    Déterminer un champ de statut global WFA : "robustness_label": "robust" | "borderline" | "overfitted"
    Enrichir les tests OOS
    Dans out_of_sample_test, à partir des sharpe_values :
    Calculer frac_oos_sharpe_lt_0
    Calculer éventuellement oos_degradation_ratio si tu peux rapprocher oos_sharpe_mean d’un train_sharpe_mean (option : utiliser la moyenne des Sharpe train sur la période globale)
    Ajouter ces statistiques dans summary (et un éventuel oos_robustness_label)
    Enrichir le résumé Monte Carlo
    Dans _summarize_simulations, à partir du DataFrame df :
    Ajouter p_sharpe_lt_0 = (df["sharpe_ratio"] < 0).mean()
    Ajouter p_cagr_lt_0 = (df["cagr"] < 0).mean()
    Si décision sur un seuil de drawdown, ajouter p_max_dd_gt_threshold
    (Optionnel) Calculer un monte_carlo_robustness_label basé sur ces probabilités
    Enrichir la stabilité
    Dans stability_tests, à partir de summary et neighbors :
    Vérifier que robust_fraction est bien l’indicateur principal --> -->
<!-- Ajouter un label stability_robustness_label basé sur robust_fraction (ex. robuste si ≥ 0.7, sur‑ajusté si ≤ 0.4) -->
<!-- 3. Propagation des nouvelles métriques dans les exports (CSV / HTML)

    Mettre à jour _export_wfa_results
    Ajouter les nouvelles colonnes dans summary_df (degradation_ratio, frac_test_sharpe_lt_0, etc.)
    Option : ajouter les indicateurs de “mauvais fold” par ligne si utile (ex. booléen is_bad_fold)
    Mettre à jour _export_oos_results
    Ajouter les colonnes OOS dans summary_df (frac_oos_sharpe_lt_0, oos_robustness_label, …)
    S’assurer que les CSV gardent un format simple (une seule ligne de résumé)
    Mettre à jour _export_monte_carlo
    Ajouter p_sharpe_lt_0, p_cagr_lt_0, etc. aux colonnes de summary_df
    Vérifier que les CSV restent lisibles et exploitables (nom de colonnes explicite)
    Mettre à jour _export_stability
    Ajouter stability_robustness_label dans summary_df
    (Option) Ajouter un summary.json global
    Créer un helper qui agrège les summary de WFA/OOS/Monte Carlo/Stability
    Sauvegarder ce JSON dans self.output_root / "summary.json" (utile pour des dashboards externes) --> -->
<!-- 4. Enrichissement de l’index Overfitting HTML

    Adapter _register_report_section (optimization/overfitting_check.py)
    Étendre l’entrée entry pour inclure un champ optionnel status (ex. "robust", "borderline", "overfitted")
    Passer ce status au moment de l’appel pour chaque type de rapport (WFA/OOS/Monte Carlo/Stability)
    Adapter render_overfitting_index (reports/overfitting_report.py)
    Modifier la signature pour accepter un status par section (conserver la rétro‑compatibilité)
    Dans la génération HTML des cartes :
    Afficher un badge coloré en fonction de status (par ex. petit <span> avec classes CSS)
    Définir dans_BASE_STYLE des styles simples pour les badges :
    - badge-robust (vert doux)
    - badge-borderline (orange)
    - badge-overfitted (rouge)
    Ajouter éventuellement un résumé global dans la section “Meta” (ex. “Global: Borderline (WFA robuste, MC fragile)” si tu veux fusionner les labels) -->
<!-- 5. Nouveaux graphiques WFA/OOS/Monte Carlo/Stability

    Histogramme des Sharpe OOS
    Créer une fonction render_oos_report dans reports/overfitting_report.py (analogue à render_wfa_report) :
    Paramètres : summary_df, windows_df, output_path
    Section “Résumé” : tableau des stats OOS (déjà existant)
    Section “Histogramme Sharpe OOS” :
    - [ ] Si go disponible : go.Histogram(x=windows_df["sharpe_ratio"])
    - [ ] Sinon : pas de plot (simple fallback texte)
    Section “Détails des fenêtres” : table HTML windows_df
    Modifier_export_oos_results pour utiliser render_oos_report à la place de _build_html_report
    Distribution des max drawdowns Monte Carlo
    Créer une fonction render_monte_carlo_report :
    Paramètres : summary_df, simulations_df, output_path
    Section “Résumé” (table)
    Section “Histogramme Sharpe” et/ou “Histogramme Max Drawdown” :
    - [ ] Utiliser Plotly si dispo, sinon fallback texte
    Section “Détails des simulations” : table HTML
    Modifier _export_monte_carlo pour appeler render_monte_carlo_report
    Heatmap relative_sharpe vs paramètre (stabilité)
    Créer une fonction render_stability_report :
    Paramètres : summary_df, neighbors_df, output_path
    Section “Résumé” (table)
    Pour la heatmap :
    - [ ] Utiliser neighbors_df avec colonnes param_name, param_value, relative_sharpe
    - [ ] Construire une matrice (par exemple, une heatmap par paramètre : x = param_value, y = param_name)
    - [ ] Avec Plotly : go.Heatmap ou une série de go.Scatter si c’est plus simple
    - [ ] Prevoir fallback sans plot si go absent
    Section “Détails des voisins” : table HTML
    Modifier _export_stability pour appeler render_stability_report au lieu de _build_html_report
    Conserver_build_html_report comme fallback générique
    Garder _build_html_report pour des usages simples (ou comme secours si Plotly échoue) -->
<!-- 6. tests -->
<!-- 7. Documentation & ergonomie

    Mettre à jour doc/optimization.md (section “Prévention de l’overfitting”)
    Décrire les nouveaux indicateurs de robustesse (formules, interprétation)
    Ajouter un exemple de lecture de l’index HTML avec les badges
    Mettre à jour README.md
    Mentionner explicitement que le module d’overfitting fournit :
    - [ ] Ratios de dégradation, probabilités de sur‑ajustement, p‑values Monte Carlo
    - [ ] Rapports HTML enrichis avec graphiques
    (Option) Ajouter un petit paragraphe explicatif dans config/overfitting_*.yaml
    Rappeler la signification des nouveaux indicateurs / seuils si certains sont paramétrables (ex. seuil drawdown, α) -->

<!-- Dashboard web  de lancement des optimisation :
-lancer des optimisations avec choix des stratégies, tickers, périodes et grille hyperparamètres
-estimation du temps restant
-visualisation ou lien vers des rapports html du backtest du meilleur essai
-visualisation ou lien des rapports d'overfitting

1. Architecture générale & emplacement

 Décider de l’emplacement principal du dashboard Streamlit:visualization/dashboard.py (UI uniquement, sans logique métier Optuna/Backtest).
 Introduire un petit module de “service” réutilisable pour lancer/monitorer les optimisations, par ex. optimization/dashboard_runner.py, appelé à la fois par Streamlit et éventuellement par d’autres outils.
 Vérifier que tout nouveau code respecte le style PEP8, les annotations de type (typing), et utilise logging plutôt que print().
2. API Python propre pour lancer une optimisation (sans casser la CLI)

 Extraire dans scripts/run_optimization.py une fonction de haut niveau, par ex. run_optimization_from_yaml(config_path: str, *, n_trials: int | None = None, timeout: int | None = None, n_jobs: int | None = None, show_progress_bar: bool | None = None) -> optuna.Study, qui :
 Utilise load_config() + build_optimizer() (déjà existants),
 Appelle optimizer.optimize(...) avec les bons paramètres,
 Ne fait aucun print() (la fonction retourne l’optuna.Study).
 Adapter main() dans scripts/run_optimization.py pour :
 Continuer à parser les arguments CLI exactement comme aujourd’hui,
 Appeler run_optimization_from_yaml(...),
 Gérer l’affichage CLI (prints) uniquement dans main() pour ne pas polluer l’API Python.
 Vérifier que l’exécution via CLI (python scripts/run_optimization.py --config ...) donne exactement les mêmes sorties qu’avant (non-régression fonctionnelle).
3. Service de gestion d’un “job d’optimisation”

Dans un nouveau module (ex. optimization/dashboard_runner.py) :

 Définir une dataclass, ex. OptimizationJobConfig, avec type hints, pour encapsuler :
 config_path: Path,
 n_trials, timeout, n_jobs,
 study_name, storage_url (dérivés de la config YAML via OptunaOptimizer / study_config).
 Définir une dataclass OptimizationJobStatus (ou similaire) avec :
 status: Literal["idle", "running", "done", "failed"],
 n_trials_planned: int | None,
 n_trials_completed: int,
 avg_trial_duration: float | None,
 eta_seconds: float | None,
 best_value: float | None,
 best_params: dict[str, Any] | None,
 last_update: datetime | None,
 éventuellement error_message: str | None.
 Implémenter une fonction start_optimization_job(job_cfg: OptimizationJobConfig) -> None qui :
 Démarre l’optimisation dans un process séparé (ex. multiprocessing.Process ou subprocess.Popen qui appelle la CLI), pour ne pas bloquer le thread Streamlit,
 Crée un fichier de “lock” ou un état persistant simple (ex. tmp-output/current_optimization.json) indiquant qu’un job est en cours.
 Implémenter une fonction de lecture de statut :
get_optimization_status(job_cfg: OptimizationJobConfig) -> OptimizationJobStatus qui :
 Charge l’étude avec optuna.load_study(study_name=..., storage=...),
 Calcule n_trials_completed = len([t for t in study.trials if t.state.is_finished()]),
 Détermine n_trials_planned à partir de la config YAML (study_config["n_trials"] ou param override),
 Calcule la durée moyenne par trial à partir de datetime_start / datetime_complete,
 En déduit une ETA simple (n_planned - n_completed) * avg_duration,
 Récupère best_value / best_params si disponibles,
 Gère les cas edge (aucun trial terminé, étude absente, job en erreur) proprement, avec logging.
 (Optionnel) Ajouter un petit cache en mémoire ou fichier JSON pour éviter de recharger l’étude trop fréquemment si cela s’avère coûteux.
4. Sélection des stratégies, tickers, périodes, hyperparamètres dans le dashboard

Dans visualization/dashboard.py (code Streamlit) :

 Créer une fonction load_available_optimization_configs() -> dict[str, Path] qui :
 Liste les fichiers config/optimization_*.yaml,
 Retourne un mapping “nom lisible” → chemin du YAML.
 Ajouter un sélecteur Streamlit (ex. st.selectbox) pour choisir un fichier d’optimisation YAML.
 Charger la config sélectionnée et afficher :
 Nom de la stratégie, module, class,
 Tickers (mono/multi),
 Période (start_date / end_date / interval),
 Paramètres d’Optuna (n_trials, timeout, n_jobs).
 Permettre d’overrider certains champs simples dans l’UI, dans l’esprit KISS :
 n_trials, timeout, n_jobs,
 éventuellement tickers, start_date, end_date (en restant prudents pour ne pas sur-complexifier).
 Construire un OptimizationJobConfig à partir de la config YAML + overrides UI, et l’utiliser pour start_optimization_job(...).
5. Estimation du temps restant & affichage en temps réel

Toujours dans visualization/dashboard.py :

 Mettre en place une section “Suivi de l’optimisation en cours” :
 Utiliser st_autorefresh() ou un timer pour rafraîchir le statut toutes les X secondes (ex. 5–10s),
 Appeler get_optimization_status(job_cfg) à chaque rafraîchissement.
 Afficher :
 Une barre de progression basée sur n_trials_completed / n_trials_planned,
 L’ETA human-readable (minutes / heures restantes) à partir de eta_seconds,
 La meilleure valeur trouvée (best_value) et quelques params clés (best_params).
 Gérer les états :
 idle → message “Aucune optimisation en cours”,
 running → progression + ETA,
 done → message de succès + liens vers rapports,
 failed → message d’erreur lisible (error_message), avec logs.
6. Backtest HTML du meilleur essai

 Réutiliser le pipeline existant de backtest + rapports (scripts/run_backtest.py, reports/report_generator.py) sans dupliquer la logique.
 Implémenter une fonction utilitaire (nouveau module ou extension de run_backtest.py) du style
generate_best_trial_report(config_path: Path, best_params: dict[str, Any]) -> Path qui :
 Charge la config de backtest de base (soit un YAML dédié, soit la partie “backtest” dans le YAML d’optimisation si prévu),
 Fusionne les best_params Optuna avec les paramètres de la stratégie (merge_params existe déjà dans run_backtest.py),
 Lance le backtest via les fonctions internes (pas forcément via la CLI) pour obtenir metrics/equity/trades,
 Appelle reports.report_generator.generate_report(...),
 Retourne le chemin du HTML généré (reports/generated/...).
 Dans le dashboard Streamlit :
 Ajouter un bouton “Générer rapport backtest (meilleur essai)” disponible quand le job est done et que best_params sont connus,
 Appeler generate_best_trial_report(...) dans un contexte non bloquant si nécessaire,
 Afficher soit :
 un lien vers le fichier (st.markdown("[Voir rapport](file:///...)" ou équivalent adapté),
 ou un st.components.v1.html(open(path).read(), height=...) pour intégration directe.
7. Rapports d’overfitting

 S’appuyer sur scripts/run_overfitting.py et optimization/overfitting_check.py, qui savent déjà :
 Charger la stratégie + param_space,
 Utiliser --use-best-params pour récupérer les paramètres optimaux à partir de best_params_path,
 Générer des rapports HTML (WFA, OOS, Monte Carlo, Stability) dans results/overfitting/....
 Ajouter dans visualization/dashboard.py :
 Une section “Overfitting” affichée une fois l’optimisation terminée,
 Un bouton “Lancer checks d’overfitting” qui :
 Vérifie la présence du fichier best_params_path (config output.best_params_path dans le YAML d’optimisation),
 Démarre un process séparé pour
python scripts/run_overfitting.py --config <même YAML> --use-best-params,
 Sauvegarde le répertoire racine des sorties (checker.output_root) ou l’infère à partir du log / convention (timestamp).
 Implémenter une fonction locate_overfitting_index(config_path: Path) -> Path | None qui :
 Inspecte results/overfitting/<run_id>/ pour trouver le dernier run (par timestamp),
 Retourne index.html si présent.
 Dans Streamlit :
 Afficher un lien ou intégration HTML pour index.html (page globale overfitting),
8. Gestion des erreurs & robustesse

 Ajouter du logging cohérent (via utils.logger.setup_logger) pour :
 Les démarrages de jobs (optimisation, overfitting),
 Les erreurs de chargement de YAML,
 Les problèmes de connexion à la base SQLite Optuna,
 Les erreurs dans la génération de rapports HTML.
 Dans le dashboard, afficher des messages utilisateurs clairs en cas d’erreur (sans stacktrace brute).
 Prévoir un mécanisme simple pour “réinitialiser” l’état :
 Bouton “Réinitialiser dashboard” qui efface l’état courant (fichiers de lock / job courant) et permet de relancer une optimisation proprement.
9. Respect du manifeste GEMINI

 Vérifier que tous les nouveaux modules/fonctions :
 Ont des docstrings claires (Google/Numpy style) expliquant le “Pourquoi/Comment”.
 Utilisent des noms explicites (optimization_job_status, generate_best_trial_report, etc.).
 Restent simples (KISS) : éviter d’introduire un scheduler complexe ou une queue externe tant que ce n’est pas nécessaire.
 Séparent la logique de données/optimisation (optimization/, scripts/) de la présentation (visualization/dashboard.py). -->

🏗️ Phase 10 : Stratégies Avancées (Semaine 9-10)

10.2 Stratégies complexes

 Mean reversion strategy
 Momentum breakout
 Pairs trading
 Régime detection

10.3 Machine Learning prep
IA comme Filtre de Régime
 Feature engineering
 Labeling des données
 Setup pour ML (optionnel)

<!-- 🧪 Phase 11 : Testing et Validation (Semaine 10-11)
11.1 Unit tests

 Tests pour data_manager
 Tests pour strategies
 Tests pour risk management
 Tests pour indicators -->

11.2 Integration tests

 Test pipeline complet
 Test avec données corrompues
 Test edge cases

11.3 Performance tests

 Benchmark vitesse backtest
 Optimisation du code
 Profiling mémoire

Phase 12 : Architecture de Trading Live (Paper & Live)
    - [ ] 12.1 Couche d'Abstraction Broker
        - [ ] Créer une interface `BaseBroker` (avec méthodes `submit_order`, `get_position`, `get_account_balance`).
        - [ ] Créer une implémentation `BacktestBroker` (qui wrappe le broker de Backtrader).
        - [ ] Créer une implémentation `PaperBroker` (Alpaca).
    - [ ] 12.2 Moteur d'Événements (Event-Driven Engine)
        - [ ] Migrer de la boucle `next()` de Backtrader à une boucle d'événements (Event Loop : `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`).
        - [ ] *Objectif :* Utiliser la *même* logique de stratégie pour le backtest et le live.
    - [ ] 12.3 Service de Monitoring & Alertes
        - [ ] Logger les exécutions d'ordres vers un canal dédié (Telegram).
        - [ ] Mettre en place un "Heartbeat" (service qui vérifie que le trader tourne toujours).

🎯 Phase 13 : Scanner & Gestion de Portefeuille
    - [ ] 13.1 Scanner de Marché
        - [ ] `scripts/live_scanner.py` doit être un service indépendant (ex: cron job).
        - [ ] Le scanner ne trade pas ; il *génère* des signaux (ex: "AAPL - Tendance Haussière H1") et les stocke (ex: dans un fichier, une DB Redis, ou une DB SQL).
    - [ ] 13.2 Gestionnaire de Portefeuille (Le "Cerveau")
        - [ ] Créer une classe `PortfolioManager` qui s'exécute après le Scanner.
        - [ ] *Logique :* Lire les signaux du Scanner, vérifier les positions actuelles, et allouer le capital (en utilisant `risk_management/position_sizing.py`).
        - [ ] Gérer les conflits (ex: 5 signaux d'achat mais capital pour 2 trades).
        - [ ] Gérer l'allocation inter-stratégies (Que faire si 2 stratégies différentes veulent acheter le même actif ?).

Phase 14 : Pipeline de Données de Production
    - [ ] 14.1 Fournisseur de Données
        - [ ] Sélectionner un fournisseur de données H1/Daily (payant ou API fiable, ex: Alpaca, IEX, EODHistoricalData).
    - [ ] 14.2 Base de Données Temporelle (TSDB)
        - [ ] Mettre en place une base de données optimisée pour les séries temporelles (ex: InfluxDB, TimescaleDB, ou même un stockage Parquet/S3).
        - [ ] Créer un "ETL" qui peuple cette base de données (en dehors du script de trading).
    - [ ] 14.3 Mise à jour du DataManager
        - [ ] `utils/data_manager.py` doit être modifié pour lire depuis cette nouvelle base de données (en live) au lieu des fichiers CSV (en backtest).

📈 Métriques de Succès du Projet
Court terme (1 mois)

 Backtest fonctionnel sur 3+ stratégies
 Sharpe ratio > 1 sur données historiques
 Système d'optimisation automatique

Moyen terme (3 mois)

 10+ stratégies testées
 Walk-forward validation positive
 Paper trading actif

Long terme (6 mois)

 Système de production stable
 ROI positif en paper trading
 Documentation complète

🔄 Maintenance Continue
Hebdomadaire

 Revue des performances
 Mise à jour des données
 Check des logs d'erreurs

Mensuelle

 Ré-optimisation des paramètres
 Analyse des drawdowns
 Mise à jour documentation

Trimestrielle

 Revue stratégie globale
 Benchmarking vs marché
 Planification nouvelles features
