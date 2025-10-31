# --- 1. Bibliothèques natives ---
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# --- 2. Bibliothèques tierces ---
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import pytz

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger
from utils.config_loader import get_settings

# Initialisation du logger pour ce module
logger = setup_logger(__name__)

# Détermine le chemin racine du projet (en supposant que ce fichier est dans t_project/utils/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class DataManager:
    """
    Classe pour gérer le téléchargement, la mise en cache, la validation
    et la préparation des données financières depuis yfinance.
    """

    def __init__(self) -> None:
        """
        Initialise le DataManager en chargeant la configuration
        et en s'assurant que le répertoire de cache existe.
        """
        try:
            settings: Dict[str, Any] = get_settings()
            self.data_config: Dict[str, Any] = settings.get("data", {})
            self.project_config: Dict[str, Any] = settings.get("project", {})

            # Configuration du cache
            cache_dir_relative: str = self.data_config.get("cache_dir", "data/cache/")
            self.cache_dir: Path = PROJECT_ROOT / cache_dir_relative
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Paramètres par défaut
            self.default_start: str = self.data_config.get(
                "default_start_date", "2015-01-01"
            )
            self.default_end: str = self.data_config.get(
                "default_end_date", "2024-12-31"
            )
            self.default_interval: str = self.data_config.get("default_interval", "1d")

            # Gestion de la Timezone
            self.timezone_str: str = self.project_config.get("timezone", "UTC")
            self.timezone: Any = pytz.timezone(self.timezone_str)

            logger.info(
                f"DataManager initialisé. Cache: {self.cache_dir}. Timezone: {self.timezone_str}"
            )

        except Exception as e:
            logger.error(f"Erreur fatale lors de l'initialisation du DataManager: {e}")
            raise

    def _get_cache_filepath(self, ticker: str, interval: str) -> Path:
        """Construit le chemin de fichier standardisé pour le cache."""
        filename = f"{ticker.upper().replace('=', '_')}_{interval}.csv"
        return self.cache_dir / filename

    def load_from_cache(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Charge les données pour un ticker/intervalle depuis un fichier CSV du cache.

        Args:
            ticker (str): Le symbole du ticker.
            interval (str): L'intervalle de temps (ex: '1d', '1h').

        Returns:
            Optional[pd.DataFrame]: Un DataFrame si les données sont trouvées
                                    en cache, sinon None.
        """
        filepath: Path = self._get_cache_filepath(ticker, interval)
        if not filepath.exists():
            logger.debug(f"Cache miss pour {ticker} ({interval}) à {filepath}")
            return None

        try:
            df = pd.read_csv(filepath, index_col="Date", parse_dates=True)

            # --- CORRECTION ---
            # Si le fichier cache est vide ou invalide, df.index
            # ne sera pas un DatetimeIndex et n'aura pas '.tz'.
            if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(
                    f"Fichier cache {filepath} est vide ou invalide. "
                    "Re-téléchargement."
                )
                return None
            # --- FIN CORRECTION ---

            # Assurer la conversion correcte de la timezone depuis le cache
            if not df.index.tz:
                df.index = df.index.tz_localize("UTC")  # Sauvegardé en UTC par défaut
            df.index = df.index.tz_convert(self.timezone)

            logger.debug(f"Données chargées depuis le cache : {filepath}")
            return df
        except Exception as e:
            logger.warning(
                f"Erreur au chargement du cache {filepath}: {e}. Fichier sera re-téléchargé."
            )
            return None

    def save_to_cache(self, df: pd.DataFrame, ticker: str, interval: str) -> None:
        """
        Sauvegarde un DataFrame dans un fichier CSV du cache.

        Args:
            df (pd.DataFrame): Le DataFrame à sauvegarder.
            ticker (str): Le symbole du ticker.
            interval (str): L'intervalle de temps.
        """
        filepath: Path = self._get_cache_filepath(ticker, interval)
        try:
            df.to_csv(filepath)
            logger.debug(f"Données sauvegardées dans le cache : {filepath}")
        except Exception as e:
            logger.error(f"Échec de la sauvegarde dans le cache {filepath}: {e}")

    def _validate_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Effectue une validation simple des données téléchargées.
        Vérifie les données vides, les NaNs, les prix nuls/négatifs et les trous.
        """
        if df.empty:
            logger.warning(f"Aucune donnée retournée pour {ticker}.")
            return False

        # 1. Trous (NaNs)
        if df.isnull().values.any():
            nan_count = df.isnull().sum().sum()
            logger.warning(
                f"{nan_count} valeurs NaN trouvées pour {ticker}. Remplissage partiel..."
            )
            df.ffill(inplace=True)  # Forward-fill pour combler les trous simples

        # 2. Valeurs aberrantes (prix <= 0)
        if (df["close"] <= 0).any():
            invalid_rows = (df["close"] <= 0).sum()
            logger.warning(
                f"{invalid_rows} lignes avec prix de clôture <= 0 trouvées pour {ticker}."
            )
            # On pourrait les supprimer, mais pour l'instant on garde (KISS)

        # 3. Trous (jours de trading manquants)
        if (
            "1d" in self.default_interval
        ):  # La détection de trous n'a de sens qu'en quotidien
            date_diffs = df.index.to_series().diff().dt.days
            max_gap = date_diffs.max()
            if max_gap > 7:  # Plus de 7 jours calendaires (week-end + jours fériés)
                logger.warning(
                    f"Grand trou de données détecté pour {ticker}: {max_gap} jours."
                )

        return True

    def download_data(
        self, ticker: str, start_date: str, end_date: str, interval: str
    ) -> pd.DataFrame:
        """
        Télécharge les données depuis yfinance, les valide et les normalise.

        Args:
            ticker (str): Le symbole du ticker.
            start_date (str): Date de début (YYYY-MM-DD).
            end_date (str): Date de fin (YYYY-MM-DD).
            interval (str): L'intervalle de temps.

        Returns:
            pd.DataFrame: Un DataFrame contenant les données OHLCV,
                          ou un DataFrame vide en cas d'échec.
        """
        logger.debug(
            f"Téléchargement yfinance: {ticker} ({interval}) de {start_date} à {end_date}"
        )
        try:
            # Correction: Utiliser yf.Ticker pour une meilleure stabilité
            # avec les versions récentes de pandas/yfinance.
            tk = yf.Ticker(ticker)
            df = tk.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,  # Garde l'ajustement pour splits/dividendes
            )

            # Si le téléchargement échoue (ex: ticker invalide), tk.history
            # peut retourner un DataFrame vide avant de lever une erreur.
            if df.empty:
                logger.warning(f"Aucune donnée retournée par yf.Ticker pour {ticker}.")
                return pd.DataFrame()

            # Renommer les colonnes en minuscules (préféré par backtrader)
            df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                },
                inplace=True,
            )

            # Garder uniquement les colonnes OHLCV standard
            df = df[["open", "high", "low", "close", "volume"]]

            # Valider les données
            if not self._validate_data(df, ticker):
                return pd.DataFrame()  # Retourne vide si invalide

            # Normaliser la Timezone
            if not df.index.tz:
                df.index = df.index.tz_localize("UTC")  # yfinance retourne UTC ou None
            df.index = df.index.tz_convert(self.timezone)

            # Renommer l'index en 'Date' (parfois 'Datetime')
            df.index.name = "Date"

            logger.info(
                f"Données téléchargées avec succès pour {ticker} ({len(df)} lignes)."
            )
            return df

        except Exception as e:
            logger.error(f"Échec du téléchargement yfinance pour {ticker}: {e}")
            return pd.DataFrame()  # Retourne vide en cas d'échec

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les indicateurs techniques au DataFrame en utilisant pandas-ta.
        Conforme au principe "Data-Driven" du manifeste.

        NOTE : Cette version recrée la logique de 'ta.Strategy' manuellement
        pour contourner les problèmes de versions/dépendances, tout en
        restant modulaire.
        """
        if df.empty:
            return df

        # --- DÉFINITION MODULAIRE DE LA STRATÉGIE ---
        # (Ceci remplace 'ta.Strategy')
        # Vous pouvez modifier/ajouter/supprimer des indicateurs ici.
        # 'kind' est le nom de la fonction dans pandas-ta (ex: ta.rsi)
        # Le reste des clés/valeurs sont les paramètres.
        strategy_definition = [
            {"kind": "rsi", "length": 14},
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
            {"kind": "bbands", "length": 20, "std": 2},
            {"kind": "atr", "length": 14},
        ]
        # -----------------------------------------------

        logger.debug(
            f"Ajout de {len(strategy_definition)} indicateurs "
            f"(méthode manuelle modulaire) pour {len(df)} lignes."
        )

        try:
            for indicator_params in strategy_definition:
                # Copie pour éviter de modifier l'original
                params = indicator_params.copy()

                # Sépare le 'kind' (nom de la fonction) des 'params'
                kind = params.pop("kind")

                try:
                    # Récupère la fonction de 'df.ta' (ex: df.ta.rsi)
                    indicator_function = getattr(df.ta, kind)

                    # Appelle la fonction avec ses paramètres
                    # (ex: df.ta.rsi(length=14, append=True))
                    indicator_function(**params, append=True)

                except AttributeError:
                    logger.error(f"Indicateur non trouvé dans pandas-ta: 'ta.{kind}'")
                    raise
                except Exception as e:
                    logger.error(
                        f"Erreur lors du calcul de '{kind}' avec params {params}: {e}"
                    )
                    raise

            # Nettoyer les NaNs initiaux créés par les indicateurs
            initial_rows = len(df)
            df.dropna(inplace=True)
            final_rows = len(df)

            logger.debug(
                f"Indicateurs ajoutés. {initial_rows - final_rows} lignes supprimées (périodes de chauffe)."
            )

        except Exception as e:
            logger.error(f"Échec de l'ajout des indicateurs : {e}")
            raise e  # Propage l'erreur

        return df

    def get_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: Optional[str] = None,
        use_cache: bool = True,
        add_indicators: bool = True,
    ) -> pd.DataFrame:
        """
        Méthode principale pour récupérer les données (cache ou téléchargement).
        Gère la logique de cache :
        1. Tente de charger depuis le cache.
        2. Si le cache est manquant ou insuffisant, télécharge la plage
           par défaut complète et la sauvegarde.
        3. Filtre le DataFrame (du cache ou neuf) à la plage demandée.
        4. Ajoute les indicateurs sur le DataFrame filtré.

        Args:
            ticker (str): Le symbole du ticker (ex: "AAPL").
            start_date (Optional[str]): Date de début (YYYY-MM-DD).
            end_date (Optional[str]): Date de fin (YYYY-MM-DD).
            interval (Optional[str]): Intervalle ('1d', '1h').
            use_cache (bool): Tenter d'utiliser le cache.
            add_indicators (bool): Ajouter les indicateurs (RSI, MACD, etc.).

        Returns:
            pd.DataFrame: DataFrame final, prêt pour le backtesting.
        """
        # 1. Déterminer les dates et l'intervalle effectifs
        start_date_eff: str = start_date or self.default_start
        end_date_eff: str = end_date or self.default_end
        interval_eff: str = interval or self.default_interval

        df: Optional[pd.DataFrame] = None

        # 2. Tenter de charger depuis le cache
        if use_cache:
            df = self.load_from_cache(ticker, interval_eff)

        # 3. Vérifier si le cache est suffisant
        if df is not None:
            try:
                # Convertir les dates string en timestamps conscients de la timezone
                pd_start = pd.Timestamp(start_date_eff, tz=self.timezone)
                pd_end = pd.Timestamp(end_date_eff, tz=self.timezone)

                if df.index.min() > pd_start or df.index.max() < pd_end:
                    logger.info(
                        f"Cache insuffisant pour {ticker} (demandé: {start_date_eff} à {end_date_eff}). Re-téléchargement."
                    )
                    df = None  # Forcer le re-téléchargement
                else:
                    logger.info(f"Données pour {ticker} chargées depuis le cache.")
            except Exception as e:
                logger.warning(
                    f"Erreur de comparaison de date cache: {e}. Re-téléchargement."
                )
                df = None

        # 4. Télécharger si le cache est manquant ou insuffisant
        if df is None:
            logger.info(
                f"Pas de cache (ou cache insuffisant) pour {ticker}. Téléchargement de la plage par défaut..."
            )
            # Télécharge la *plage par défaut complète* pour construire un cache robuste
            df = self.download_data(
                ticker, self.default_start, self.default_end, interval_eff
            )
            if not df.empty and use_cache:
                self.save_to_cache(df, ticker, interval_eff)

        if df.empty:
            logger.error(f"Impossible d'obtenir des données pour {ticker}.")
            return df  # Retourne un DF vide

        # 5. Filtrer le DataFrame complet (du cache ou neuf) à la plage demandée
        df_filtered = df.loc[start_date_eff:end_date_eff].copy()

        if df_filtered.empty:
            logger.warning(
                f"Aucune donnée pour {ticker} dans la plage de dates spécifiée: {start_date_eff} à {end_date_eff}."
            )
            return df_filtered

        # 6. Ajouter les indicateurs sur le DataFrame filtré
        if add_indicators:
            df_filtered = self.add_indicators(df_filtered)

        logger.info(
            f"Données prêtes pour {ticker} ({len(df_filtered)} lignes de {start_date_eff} à {end_date_eff})."
        )
        return df_filtered


# --- Bloc de test ---
if __name__ == "__main__":
    """
    Ce bloc s'exécute uniquement lorsque vous lancez ce script directement
    (ex: `python utils/data_manager.py`) pour tester son fonctionnement.
    """
    logger.info("--- Début du test de DataManager ---")

    dm = DataManager()

    # Test 1: Téléchargement et mise en cache
    logger.info("--- Test 1: Téléchargement (ex: AAPL) ---")
    df_aapl = dm.get_data(
        "AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31",
        use_cache=False,  # Forcer le re-téléchargement
        add_indicators=True,
    )

    if not df_aapl.empty:
        logger.info(f"AAPL - Données récupérées: {df_aapl.shape}")
        logger.info(f"AAPL - Colonnes: {list(df_aapl.columns)}")
        logger.info(f"AAPL - Début: {df_aapl.index.min()}, Fin: {df_aapl.index.max()}")
        # Vérifier si les indicateurs (manifeste) sont présents
        logger.info(f"AAPL - Colonne RSI_14 présente: {'RSI_14' in df_aapl.columns}")
        logger.info(
            f"AAPL - Colonne MACDh_12_26_9 présente: {'MACDh_12_26_9' in df_aapl.columns}"
        )
    else:
        logger.error("Test 1 (AAPL) a échoué: DataFrame vide.")

    # Test 2: Chargement depuis le cache
    logger.info("--- Test 2: Chargement depuis le cache (devrait être instantané) ---")
    df_aapl_cache = dm.get_data(
        "AAPL",
        start_date="2023-06-01",
        end_date="2023-06-30",
        use_cache=True,
        add_indicators=False,
    )  # Test sans indicateurs

    if not df_aapl_cache.empty:
        logger.info(f"AAPL Cache - Données récupérées: {df_aapl_cache.shape}")
        logger.info(
            f"AAPL Cache - Début: {df_aapl_cache.index.min()}, Fin: {df_aapl_cache.index.max()}"
        )
        logger.info(f"AAPL Cache - Colonnes: {list(df_aapl_cache.columns)}")
    else:
        logger.error("Test 2 (AAPL Cache) a échoué: DataFrame vide.")

    # Test 3: Ticker invalide (gestion d'erreur)
    logger.info("--- Test 3: Ticker invalide (GESTION ERREUR) ---")
    df_invalid = dm.get_data(
        "TICKERINVALIDE123", start_date="2023-01-01", end_date="2023-01-31"
    )

    if df_invalid.empty:
        logger.info(
            "Test 3 (Ticker invalide) réussi: DataFrame vide retourné comme attendu."
        )
    else:
        logger.error(
            "Test 3 (Ticker invalide) a échoué: Des données ont été retournées."
        )

    logger.info("--- Fin du test de DataManager ---")
