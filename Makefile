##data
Télécharger un seul ticker en forçant la mise à jour du cache :

python scripts/download_data.py --tickers AAPL --no-cache

Télécharger plusieurs tickers pour une période spécifique :

python scripts/download_data.py -t MSFT GOOGL NVDA --start 2020-01-01 --end 2023-12-31

Télécharger tout le marché "sp500"  :

python scripts/download_data.py --market sp500


##backtest
Lister les stratégies disponibles :

python scripts/run_backtest.py --list-strategies

Lancer un backtest :

python scripts/run_backtest.py --config config/backtest_config.yaml
python scripts/run_backtest.py --config config/backtest_configRsiOversold.yaml

python scripts/run_backtest.py --config config/backtest_config1111.yaml
python scripts/run_backtest.py --config config/backtest_configSmaPullbackManaged.yaml
python scripts/run_backtest.py --config config/backtest_configEmaTrendStrategy.yaml

Avec le script helper :
./run_backtest.sh config/backtest_config.yaml
./run_backtest.sh --list


##optimization
Lancer une optimisation : 
python scripts/run_optimization.py --config config/optimization_SimpleMaManaged.yaml --n-trials 1 --no-progress-bar

python scripts/run_optimization.py --config config/optimization_RsiMeanReversionManaged.yaml 
python scripts/run_optimization.py --config config/optimization_SmaPullbackManaged.yaml 
python scripts/run_optimization.py --config config/optimization_RsiOversold.yaml 
python scripts/run_optimization.py --config config/optimization_EmaTrend.yaml 



optuna-dashboard sqlite:///results/optimization/optuna_studies.db --host 127.0.0.1 --port 4200 
     

##overfitting
Lance WFA, OOS, Monte Carlo :
python scripts/run_overfitting.py --config config/overfitting_SimpleMaManaged.yaml

--use-best-params
--checks wfa,oos,monte,stability