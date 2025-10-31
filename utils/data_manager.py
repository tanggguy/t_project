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

# Détermine le chemin racine du projet
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

        # ✅ VÉRIFICATION 1 : Le fichier existe-t-il ?
        if not filepath.exists():
            logger.debug(f"Cache miss pour {ticker} ({interval}) - fichier inexistant")
            return None

        # ✅ VÉRIFICATION 2 : Le fichier est-il vide (0 bytes) ?
        if filepath.stat().st_size == 0:
            logger.warning(
                f"Fichier cache {filepath} est vide (0 bytes). Re-téléchargement."
            )
            return None

        try:
            df = pd.read_csv(filepath, index_col="Date", parse_dates=True)

            # ✅ VÉRIFICATION 3 : Le DataFrame est-il vide après lecture ?
            if df.empty:
                logger.warning(
                    f"DataFrame vide après lecture de {filepath}. Re-téléchargement."
                )
                return None

            # ✅ VÉRIFICATION 4 : L'index est-il valide ?
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"Index invalide dans {filepath}. Re-téléchargement.")
                return None

            # ✅ Gestion de la timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(self.timezone)

            logger.debug(
                f"[OK] Cache hit pour {ticker} ({interval}) - {len(df)} lignes chargées"
            )
            return df

        except Exception as e:
            logger.warning(
                f"Erreur au chargement du cache {filepath}: {e}. Re-téléchargement."
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
        if df.empty:
            logger.warning(
                f"Tentative de sauvegarde d'un DataFrame vide pour {ticker}. Ignoré."
            )
            return

        filepath: Path = self._get_cache_filepath(ticker, interval)
        try:
            # ✅ S'assurer que l'index est en UTC avant sauvegarde pour cohérence
            df_to_save = df.copy()
            if df_to_save.index.tz is not None:
                df_to_save.index = df_to_save.index.tz_convert("UTC")

            df_to_save.to_csv(filepath)
            file_size = filepath.stat().st_size / 1024  # Taille en KB
            logger.debug(
                f"[OK] Cache sauvegardé : {filepath} ({file_size:.1f} KB, {len(df)} lignes)"
            )

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

        # 1. Vérifier les colonnes OHLCV
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(
                f"Colonnes manquantes pour {ticker}: {missing_columns}. "
                "Données invalides."
            )
            return False

        # 2. Vérifier les valeurs NaN
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            logger.warning(
                f"Présence de NaN pour {ticker}. Colonnes affectées: "
                f"{nan_counts[nan_counts > 0].to_dict()}"
            )

        # 3. Vérifier les prix négatifs ou nuls
        for col in ["open", "high", "low", "close"]:
            if (df[col] <= 0).any():
                logger.warning(
                    f"Prix {col} négatifs ou nuls détectés pour {ticker}. "
                    "Données potentiellement corrompues."
                )
                return False

        # 4. Vérifier les trous dans les données (dates manquantes)
        date_diffs = df.index.to_series().diff()
        max_gap = date_diffs.max()
        # Pour un intervalle '1d', un trou de 5 jours est suspect (weekend + férié)
        if max_gap > pd.Timedelta(days=7):
            logger.warning(
                f"Trou de données détecté pour {ticker}: {max_gap}. "
                "Vérifier la continuité."
            )

        logger.debug(f"Validation OK pour {ticker}.")
        return True

    def download_data(
        self, ticker: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Télécharge les données d'un ticker via yfinance.

        Args:
            ticker (str): Le symbole du ticker.
            start_date (str): Date de début (YYYY-MM-DD).
            end_date (str): Date de fin (YYYY-MM-DD).
            interval (str): Intervalle ('1d', '1h', etc.).

        Returns:
            pd.DataFrame: DataFrame OHLCV avec index Datetime, ou DataFrame vide si échec.
        """
        logger.info(
            f"Téléchargement de {ticker} ({start_date} à {end_date}, {interval})..."
        )

        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=interval,
                actions=False,
                auto_adjust=True,
            )

            if df.empty:
                logger.warning(f"Aucune donnée retournée par yfinance pour {ticker}.")
                return pd.DataFrame()

            # Nettoyer les noms de colonnes (mettre en minuscules)
            df.columns = df.columns.str.lower()

            # S'assurer que l'index est un DatetimeIndex avec timezone
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error(f"L'index pour {ticker} n'est pas un DatetimeIndex.")
                return pd.DataFrame()

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

            df.index = df.index.tz_convert(self.timezone)

            # Validation
            if not self._validate_data(df, ticker):
                logger.error(f"Validation échouée pour {ticker}.")
                return pd.DataFrame()

            logger.info(
                f"Données téléchargées avec succès pour {ticker} ({len(df)} lignes)."
            )
            return df

        except Exception as e:
            logger.error(f"Erreur lors du téléchargement de {ticker}: {e}")
            return pd.DataFrame()

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des indicateurs techniques au DataFrame en utilisant pandas-ta.

        Args:
            df (pd.DataFrame): DataFrame OHLCV.

        Returns:
            pd.DataFrame: DataFrame avec les indicateurs ajoutés.
        """
        if df.empty:
            logger.warning("DataFrame vide. Impossible d'ajouter des indicateurs.")
            return df

        logger.debug("Ajout des indicateurs techniques avec pandas-ta...")

        try:
            # Calcul avec pandas-ta
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.atr(length=14, append=True)

            # Supprimer les NaN générés par les périodes de chauffe
            df.dropna(inplace=True)

            logger.debug(f"Indicateurs ajoutés. Nouvelles colonnes: {list(df.columns)}")
            return df

        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des indicateurs: {e}")
            return df

    def get_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: Optional[str] = None,
        use_cache: bool = True,
        add_indicators: bool = False,
    ) -> pd.DataFrame:
        """
        Méthode principale pour obtenir les données.

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
                # Convertir les dates string en timestamps
                pd_start = pd.Timestamp(start_date_eff, tz=self.timezone)
                pd_end = pd.Timestamp(end_date_eff, tz=self.timezone)

                if df.index.min() > pd_start or df.index.max() < pd_end:
                    logger.info(
                        f"Cache insuffisant pour {ticker} (demandé: {start_date_eff} à {end_date_eff}). Re-téléchargement."
                    )
                    df = None
                else:
                    logger.info(f"[OK] Données pour {ticker} chargées depuis le cache.")
            except Exception as e:
                logger.warning(
                    f"Erreur de comparaison de date: {e}. Re-téléchargement."
                )
                df = None

        # 4. Télécharger si le cache est manquant ou insuffisant
        if df is None:
            logger.info(
                f"Téléchargement pour {ticker} (plage par défaut : {self.default_start} à {self.default_end})..."
            )
            df = self.download_data(
                ticker, self.default_start, self.default_end, interval_eff
            )
            if not df.empty and use_cache:
                self.save_to_cache(df, ticker, interval_eff)

        if df.empty:
            logger.error(f"Impossible d'obtenir des données pour {ticker}.")
            return df

        # 5. Filtrer à la plage demandée
        df_filtered = df.loc[start_date_eff:end_date_eff].copy()

        if df_filtered.empty:
            logger.warning(
                f"Aucune donnée pour {ticker} dans la plage {start_date_eff} à {end_date_eff}."
            )
            return df_filtered

        # 6. Ajouter les indicateurs
        if add_indicators:
            df_filtered = self.add_indicators(df_filtered)

        logger.info(
            f"[OK] Données prêtes pour {ticker} ({len(df_filtered)} lignes de {start_date_eff} à {end_date_eff})."
        )
        return df_filtered


# --- Bloc de test ---
if __name__ == "__main__":
    """Test du DataManager"""
    logger.info("--- Début du test de DataManager ---")

    dm = DataManager()

    # Test 1: Téléchargement initial
    logger.info("--- Test 1: Téléchargement (AAPL) ---")
    df_aapl = dm.get_data(
        "AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31",
        use_cache=True,
        add_indicators=True,
    )

    if not df_aapl.empty:
        logger.info(f"[OK] AAPL - {df_aapl.shape[0]} lignes récupérées")
        logger.info(f"✓ Colonnes: {list(df_aapl.columns)[:5]}...")
    else:
        logger.error("[ERREUR] Test 1 échoué")

    # Test 2: Chargement depuis cache
    logger.info("--- Test 2: Rechargement (devrait utiliser le cache) ---")
    df_aapl_cache = dm.get_data(
        "AAPL",
        start_date="2023-06-01",
        end_date="2023-06-30",
        use_cache=True,
        add_indicators=False,
    )

    if not df_aapl_cache.empty:
        logger.info(f"[OK] Cache - {df_aapl_cache.shape[0]} lignes chargées")
    else:
        logger.error("[ERREUR] Test 2 échoué")

    logger.info("--- Fin du test de DataManager ---")
