# --- 1. Bibliothèques natives ---
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- 2. Bibliothèques tierces ---
import yaml
from tqdm import tqdm

# --- Configuration du Chemin (Important pour les scripts) ---
# Ajoute la racine du projet au PYTHONPATH pour que les imports (utils, etc.) fonctionnent
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    sys.path.append(str(PROJECT_ROOT))
except NameError:
    # Si __file__ n'est pas défini (ex: dans un notebook interactif)
    PROJECT_ROOT = Path.cwd()
    sys.path.append(str(PROJECT_ROOT))

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger
from utils.data_manager import DataManager

# Initialisation du logger pour ce script
logger = setup_logger(__name__, log_file="logs/download_data.log")


def _load_market_tickers(market_name: str) -> List[str]:
    """
    Charge la liste des tickers depuis un fichier YAML de configuration.
    (Ex: 'sp500' -> 'config/markets/sp500.yaml')

    Args:
        market_name (str): Le nom du marché (ex: 'sp500', 'cac40').

    Returns:
        List[str]: Une liste de symboles de tickers.

    Raises:
        FileNotFoundError: Si le fichier de marché n'est pas trouvé.
        KeyError: Si la clé 'tickers' n'est pas dans le YAML.
    """
    market_file = PROJECT_ROOT / "config" / "markets" / f"{market_name}.yaml"
    logger.info(f"Chargement du fichier de marché : {market_file}")

    if not market_file.exists():
        logger.error(f"Fichier de marché non trouvé : {market_file}")
        raise FileNotFoundError(f"Fichier non trouvé: {market_file}")

    try:
        with open(market_file, "r", encoding="utf-8") as f:
            market_data = yaml.safe_load(f)

        tickers = market_data["tickers"]
        if not isinstance(tickers, list):
            raise TypeError(f"La clé 'tickers' dans {market_file} n'est pas une liste.")

        logger.info(f"{len(tickers)} tickers chargés pour le marché {market_name}.")
        return tickers

    except KeyError:
        logger.error(f"Clé 'tickers' non trouvée dans {market_file}.")
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de {market_file}: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """
    Configure et analyse les arguments de la ligne de commande (CLI).
    """
    parser = argparse.ArgumentParser(
        description="Script de téléchargement et de mise en cache des données."
    )

    # --- Groupe d'arguments mutuellement exclusifs pour la source des tickers ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-t",
        "--tickers",
        nargs="+",
        help="Liste de tickers à télécharger (ex: AAPL MSFT GOOG).",
    )
    group.add_argument(
        "-m",
        "--market",
        type=str,
        help="Nom du marché à télécharger (ex: 'sp500'). "
        "Charge les tickers depuis 'config/markets/[nom].yaml'.",
    )

    # --- Arguments de Période et Intervalle (Optionnels) ---
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Date de début (YYYY-MM-DD). "
        "Défaut: 'default_start_date' du settings.yaml.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Date de fin (YYYY-MM-DD). "
        "Défaut: 'default_end_date' du settings.yaml.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=str,
        default=None,
        help="Intervalle des données (ex: '1d', '1h'). "
        "Défaut: 'default_interval' du settings.yaml.",
    )

    # --- Drapeaux (Flags) booléens ---
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Force le re-téléchargement et écrase le cache existant.",
    )

    return parser.parse_args()


def main():
    """
    Fonction principale du script.
    """
    try:
        args = parse_arguments()

        ticker_list: List[str] = []

        # 1. Déterminer la liste des tickers
        if args.market:
            logger.info(f"Mode Batch activé pour le marché : {args.market}")
            try:
                ticker_list = _load_market_tickers(args.market)
            except Exception as e:
                logger.critical(
                    f"Impossible de charger la liste du marché {args.market}. Arrêt. Erreur: {e}"
                )
                return
        elif args.tickers:
            logger.info(f"Mode Ticker(s) unique(s) activé.")
            ticker_list = args.tickers

        if not ticker_list:
            logger.warning("Aucun ticker à traiter. Arrêt.")
            return

        # 2. Initialiser le DataManager
        try:
            dm = DataManager()
        except Exception as e:
            logger.critical(
                f"Impossible d'initialiser le DataManager. Arrêt. Erreur: {e}"
            )
            return

        # 3. Traiter la liste des tickers avec barre de progression
        logger.info(f"Début du traitement pour {len(ticker_list)} ticker(s).")
        logger.info(
            f"Paramètres: Start={args.start or 'défaut'}, End={args.end or 'défaut'}, "
            f"Interval={args.interval or 'défaut'}, Cache={args.use_cache}, "
        )

        successful_downloads = 0
        failed_downloads = []

        # Utilisation de tqdm pour la barre de progression
        for ticker in tqdm(
            ticker_list, desc="Téléchargement des données", unit="ticker"
        ):
            try:
                df = dm.get_data(
                    ticker=ticker,
                    start_date=args.start,
                    end_date=args.end,
                    interval=args.interval,
                    use_cache=args.use_cache,
                )

                if df.empty:
                    logger.warning(
                        f"Aucune donnée retournée pour {ticker} (ticker invalide ou plage de dates vide)."
                    )
                    failed_downloads.append(ticker)
                else:
                    successful_downloads += 1

            except Exception as e:
                logger.error(f"Échec critique lors du traitement de {ticker}: {e}")
                failed_downloads.append(ticker)

        # 4. Rapport final
        logger.info("--- Fin du script de téléchargement ---")
        logger.info(f"Succès: {successful_downloads} ticker(s)")
        logger.info(f"Échecs: {len(failed_downloads)} ticker(s)")
        if failed_downloads:
            logger.warning(f"Liste des échecs: {', '.join(failed_downloads)}")

    except Exception as e:
        logger.critical(f"Une erreur non gérée a interrompu le script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
