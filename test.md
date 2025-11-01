tests/
├── __init__.py
├── conftest.py                           # Fixtures globales
│
├── unit/                                 # Tests unitaires (Phase 11.1)
│   ├── __init__.py
│   ├── test_data_manager.py
│   ├── test_config_loader.py
│   ├── test_data_processor.py
│   ├── test_logger.py
│   ├── test_market_calendar.py
│   ├── test_indicators/
│   │   ├── __init__.py
│   │   ├── test_custom_indicators.py
│   │   └── test_combinations.py
│   ├── test_strategies/
│   │   ├── __init__.py
│   │   ├── test_base_strategy.py
│   │   ├── test_ma_crossover.py
│   │   ├── test_rsi_oversold.py
│   │   └── test_multi_timeframe.py      # Phase 10
│   ├── test_risk_management/
│   │   ├── __init__.py
│   │   ├── test_position_sizing.py
│   │   ├── test_stop_loss.py
│   │   └── test_portfolio.py
│   ├── test_optimization/
│   │   ├── __init__.py
│   │   ├── test_optuna_optimizer.py
│   │   ├── test_parameter_spaces.py
│   │   └── test_overfitting_check.py
│   └── test_backtesting/
│       ├── __init__.py
│       ├── test_engine.py
│       └── test_analyzers/
│           ├── __init__.py
│           ├── test_performance.py
│           ├── test_risk.py
│           └── test_drawdown.py
│
├── integration/                          # Tests d'intégration (Phase 11.2)
│   ├── __init__.py
│   ├── test_backtest_pipeline.py        # Data -> Backtest -> Results
│   ├── test_optimization_flow.py        # Optimization end-to-end
│   ├── test_strategy_execution.py       # Exécution complète stratégie
│   ├── test_multi_strategy_portfolio.py # Portfolio multi-stratégies
│   └── test_walk_forward_validation.py  # Walk-forward (Phase 9)
│
├── end_to_end/                           # Tests E2E (Nouveau)
│   ├── __init__.py
│   ├── test_full_workflow.py            # Workflow complet
│   └── test_data_corruption_recovery.py # Gestion erreurs
│
├── live/                                 # 🆕 Tests Live Trading (Phase 12-13)
│   ├── __init__.py
│   ├── test_paper_trader.py             # Paper trading simulation
│   ├── test_broker_connection.py        # Connexion broker (mock)
│   ├── test_order_execution.py          # Exécution ordres réels
│   ├── test_position_monitor.py         # Monitoring positions
│   ├── test_live_scanner.py             # Scanner temps réel
│   └── test_alert_system.py             # Alertes (email/telegram)
│
├── performance/                          # Tests de performance (Phase 11.3)
│   ├── __init__.py
│   ├── test_backtest_speed.py           # Benchmarks vitesse
│   ├── test_memory_usage.py             # Profiling mémoire
│   ├── test_optimization_speed.py       # Vitesse optimisation
│   └── test_concurrent_backtests.py     # Multi-threading
│
├── ml/                                   # 🆕 Tests ML (Phase 10.3 optionnel)
│   ├── __init__.py
│   ├── test_feature_engineering.py
│   ├── test_labeling.py
│   └── test_model_training.py
│
├── fixtures/                             # Données de test
│   ├── __init__.py
│   ├── sample_data.csv                  # OHLCV propres
│   ├── corrupted_data.csv               # Données corrompues
│   ├── config_test.yaml                 # Config minimale
│   ├── broker_responses/                # 🆕 Réponses broker mockées
│   │   ├── order_success.json
│   │   ├── order_rejected.json
│   │   └── position_update.json
│   └── market_data/                     # 🆕 Données temps réel mockées
│       ├── live_tick_data.json
│       └── historical_snapshot.csv
│
├── mocks/                                # 🆕 Mocks pour Live (Phase 12)
│   ├── __init__.py
│   ├── mock_broker.py                   # Broker simulé
│   ├── mock_websocket.py                # WebSocket flux temps réel
│   └── mock_alert_system.py             # Système d'alertes
│
├── regression/                           # 🆕 Tests de régression
│   ├── __init__.py
│   └── test_strategy_consistency.py     # Vérifier que les résultats ne changent pas
│
└── smoke/                                # 🆕 Tests Smoke (CI/CD)
    ├── __init__.py
    ├── test_imports.py                  # Tous les imports fonctionnent
    ├── test_basic_backtest.py           # Backtest minimal fonctionne
    └── test_config_loading.py           # Configs se chargent