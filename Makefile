Exemple 1 : Télécharger un seul ticker en forçant la mise à jour du cache

Bash

python scripts/download_data.py --tickers AAPL --no-cache
Exemple 2 : Télécharger plusieurs tickers pour une période spécifique

Bash

python scripts/download_data.py -t MSFT GOOGL NVDA --start 2020-01-01 --end 2023-12-31
Exemple 3 : Télécharger tout le marché "sp500" (mode batch) (Note : Cela échouera jusqu'à ce que config/markets/sp500.yaml soit créé)

Bash

python scripts/download_data.py --market sp500



# Lister les stratégies disponibles
python scripts/run_backtest.py --list-strategies

# Lancer un backtest
python scripts/run_backtest.py --config config/backtest_config.yaml

python scripts/run_backtest.py --config config/backtest_config1111.yaml

# Avec le script helper
./run_backtest.sh config/backtest_config.yaml
./run_backtest.sh --list


pytest --cov=nom_du_module tests/
pytest --cov=utils tests/
pytest --cov=optimization --cov-report=term-missing tests/


mut.py --target utils --unit-test tests --runner pytest



Lancer une optimisation : 
python scripts/run_optimization.py --config config/optimization_SimpleMaManaged.yaml 
--n-trials 20 --no-progress-bar

python scripts/run_optimization.py --config config/optimization_RsiMeanReversionManaged.yaml 



optuna-dashboard sqlite:///results/optimization/optuna_studies.db --host 127.0.0.1 --port 4200 
      # stop_loss_type:
      #   type: "categorical"
      #   choices: ["fixed", "atr", "trailing"]

	  File "C:\Users\saill\Desktop\t_project\scripts\run_optimization.py", line 213, in <module>
    main()
  File "C:\Users\saill\Desktop\t_project\scripts\run_optimization.py", line 190, in main
    optimizer = build_optimizer(config)
  File "C:\Users\saill\Desktop\t_project\scripts\run_optimization.py", line 135, in build_optimizer
    optimizer = OptunaOptimizer(
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 108, in __init__
    self.data_frame = self._load_data()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 145, in _load_data
    df = self.data_manager.get_data(
  File "C:\Users\saill\Desktop\t_project\utils\data_manager.py", line 366, in get_data
    logger.info(
Message: '[OK] Données prêtes pour AAPL (2725 lignes de 2015-01-01 à 2025-11-01).'
Arguments: ()
[I 2025-11-04 23:54:37,365] Using an existing study with name 'rsi_mean_rev_managed_opt' instead of creating a new one.
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 32, in __init__
    logger.info("Initialisation du BacktestEngine...")
Message: 'Initialisation du BacktestEngine...'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 40, in __init__
    logger.debug("Configuration settings.yaml chargée.")
Message: 'Configuration settings.yaml chargée.'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 49, in __init__
    self._setup_broker()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 59, in _setup_broker
    logger.info(f"Capital initial du broker fixé à : {initial_capital:,.2f}")
Message: 'Capital initial du broker fixé à : 10,000.00'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 49, in __init__
    self._setup_broker()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 67, in _setup_broker
    logger.info(f"Commission (pourcentage) fixée à : {comm_val:.4%}")
Message: 'Commission (pourcentage) fixée à : 0.1000%'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 32, in __init__
    logger.info("Initialisation du BacktestEngine...")
Message: 'Initialisation du BacktestEngine...'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 40, in __init__
    logger.debug("Configuration settings.yaml chargée.")
Message: 'Configuration settings.yaml chargée.'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 49, in __init__
    self._setup_broker()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 83, in _setup_broker
    logger.info(f"Slippage (pourcentage) fixé à : {slippage:.4%}")
Message: 'Slippage (pourcentage) fixé à : 0.0500%'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 49, in __init__
    self._setup_broker()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 59, in _setup_broker
    logger.info(f"Capital initial du broker fixé à : {initial_capital:,.2f}")
Message: 'Capital initial du broker fixé à : 10,000.00'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 182, in _create_engine
    engine.add_data(self.data_frame.copy(), name="data0")
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 143, in add_data
    logger.info(
Message: "Flux de données 'data0' ajouté. Période: 2015-01-02 à 2025-10-31."
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 49, in __init__
    self._setup_broker()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 67, in _setup_broker
    logger.info(f"Commission (pourcentage) fixée à : {comm_val:.4%}")
Message: 'Commission (pourcentage) fixée à : 0.1000%'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 361, in objective
    engine.add_strategy(self.strategy_class, **params)
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 185, in add_strategy
    logger.info(
Message: "Stratégie 'RsiMeanReversionManagedStrategy' ajoutée. Paramètres: (trend_long_period=280, avoid_strong_trend=True, rsi_period=18, rsi_oversold=25, rsi_exit=50, bb_period=25, bb_dev=2.5, use_invalidation=False, reentry_cooldown_bars=9, atr_period=12, stop_loss_atr_mult=4.0, take_profit_atr_mult=2.5, use_stop_loss=True, stop_loss_type=atr, use_take_profit=True, take_profit_type=atr)"
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 163, in _create_engine
    engine = BacktestEngine()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 49, in __init__
    self._setup_broker()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 83, in _setup_broker
    logger.info(f"Slippage (pourcentage) fixé à : {slippage:.4%}")
Message: 'Slippage (pourcentage) fixé à : 0.0500%'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 364, in objective
    results = engine.run()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 225, in run
    self._setup_analyzers()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 90, in _setup_analyzers
    logger.debug("Configuration des analyseurs...")
Message: 'Configuration des analyseurs...'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 360, in objective
    engine = self._create_engine()
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 182, in _create_engine
    engine.add_data(self.data_frame.copy(), name="data0")
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 143, in add_data
    logger.info(
Message: "Flux de données 'data0' ajouté. Période: 2015-01-02 à 2025-10-31."
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 361, in objective
    engine.add_strategy(self.strategy_class, **params)
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 185, in add_strategy
    logger.info(
Message: "Stratégie 'RsiMeanReversionManagedStrategy' ajoutée. Paramètres: (trend_long_period=280, avoid_strong_trend=True, rsi_period=18, rsi_oversold=25, rsi_exit=50, bb_period=25, bb_dev=2.5, use_invalidation=False, reentry_cooldown_bars=9, atr_period=12, stop_loss_atr_mult=4.0, take_profit_atr_mult=2.5, use_stop_loss=True, stop_loss_type=atr, use_take_profit=True, take_profit_type=atr)"
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 364, in objective
    results = engine.run()
  File "C:\Users\saill\Desktop\t_project\backtesting\engine.py", line 227, in run
    logger.info("--- DÉMARRAGE DU BACKTEST ---")
Message: '--- DÉMARRAGE DU BACKTEST ---'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 80, in emit
    self.doRollover()
    ~~~~~~~~~~~~~~~^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 185, in doRollover
    self.rotate(self.baseFilename, dfn)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\logging\handlers.py", line 121, in rotate
    os.rename(source, dest)
    ~~~~~~~~~^^^^^^^^^^^^^^
PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log' -> 'C:\\Users\\saill\\Desktop\\t_project\\logs\\trading_project.log.1'
Call stack:
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1014, in _bootstrap
    self._bootstrap_inner()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 93, in _worker
    work_item.run()
  File "C:\Users\saill\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 160, in _optimize_sequential
    frozen_trial_id = _run_trial(study, func, catch)
  File "C:\Users\saill\Desktop\t_project\.venv\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "C:\Users\saill\Desktop\t_project\optimization\optuna_optimizer.py", line 364, in ob