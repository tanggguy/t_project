Exemple 1 : Télécharger un seul ticker en forçant la mise à jour du cache

Bash

python scripts/download_data.py --tickers AAPL --no-cache
Exemple 2 : Télécharger plusieurs tickers pour une période spécifique

Bash

python scripts/download_data.py -t MSFT GOOGL NVDA --start 2020-01-01 --end 2023-12-31
Exemple 3 : Télécharger tout le marché "sp500" (mode batch) (Note : Cela échouera jusqu'à ce que config/markets/sp500.yaml soit créé)

Bash

python scripts/download_data.py --market sp500