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
