# test_data_manager.py
# -*- coding: utf-8 -*-

"""
Suite de tests unitaires pour la classe DataManager.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
from pandas.testing import assert_frame_equal
import pytz
from pathlib import Path
import numpy as np

# Importe le module à tester.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from utils.data_manager import DataManager

# --- Fixtures de Test ---


@pytest.fixture
def mock_dependencies(mocker, tmp_path):
    """
    Fixture centralisée pour mocker toutes les dépendances externes
    (config, logger, I/O fichier, réseau).
    """

    # 1. Mock la configuration (get_settings)
    mock_settings = {
        "data": {
            "cache_dir": "test_cache/",
            "default_start_date": "2015-01-01",
            "default_end_date": "2024-12-31",
            "default_interval": "1d",
        },
        "project": {"timezone": "Europe/Paris"},
    }
    mocker.patch("data_manager.get_settings", return_value=mock_settings)

    # 2. Mock le logger
    mock_logger = MagicMock()
    mocker.patch("data_manager.setup_logger", return_value=mock_logger)

    # 3. Mock PROJECT_ROOT pour utiliser tmp_path
    # C'est crucial pour isoler les tests du système de fichiers réel.
    mocker.patch("data_manager.PROJECT_ROOT", tmp_path)

    # 4. Mock pytz.timezone (appelé dans __init__)
    mock_tz = pytz.timezone("Europe/Paris")
    mocker.patch("data_manager.pytz.timezone", return_value=mock_tz)

    # 5. Mock les méthodes de Path (appelées dans __init__ et autres)
    mock_mkdir = mocker.patch("data_manager.Path.mkdir")
    mock_exists = mocker.patch("data_manager.Path.exists")
    mock_stat = mocker.patch("data_manager.Path.stat")

    # 6. Mock les appels réseau (yfinance)
    mock_yf_ticker_cls = mocker.patch("data_manager.yf.Ticker")
    mock_yf_ticker_instance = MagicMock()
    mock_yf_ticker_cls.return_value = mock_yf_ticker_instance

    # 7. Mock les I/O de pandas
    mock_read_csv = mocker.patch("data_manager.pd.read_csv")
    mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")  # Patch sur la classe DF

    # Retourne les mocks pour que les tests puissent les configurer
    return {
        "mock_logger": mock_logger,
        "mock_settings": mock_settings,
        "mock_tz": mock_tz,
        "mock_mkdir": mock_mkdir,
        "mock_exists": mock_exists,
        "mock_stat": mock_stat,
        "mock_yf_ticker_cls": mock_yf_ticker_cls,
        "mock_yf_ticker_instance": mock_yf_ticker_instance,
        "mock_read_csv": mock_read_csv,
        "mock_to_csv": mock_to_csv,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def manager(mock_dependencies):
    """
    Retourne une instance de DataManager avec toutes les dépendances mockées.
    """
    # L'instanciation déclenche __init__, qui utilise les mocks
    # de mock_dependencies (get_settings, setup_logger, PROJECT_ROOT, pytz, mkdir)
    return DataManager()


@pytest.fixture
def sample_df_paris():
    """Un DataFrame échantillon avec la timezone de Paris."""
    tz = pytz.timezone("Europe/Paris")
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="1D", tz=tz)
    data = {
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [99, 100, 101, 102, 103],
        "close": [101, 102, 103, 104, 105],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_df_utc():
    """Un DataFrame échantillon avec la timezone UTC."""
    tz = pytz.timezone("UTC")
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="1D", tz=tz)
    data = {
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [99, 100, 101, 102, 103],
        "close": [101, 102, 103, 104, 105],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_df_raw_yfinance():
    """Un DataFrame échantillon brut tel que retourné par yfinance (caps, no tz)."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="1D")  # Pas de TZ
    data = {
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [99, 100, 101, 102, 103],
        "Close": [101, 102, 103, 104, 105],
        "Volume": [1000, 1100, 1200, 1300, 1400],
        "Dividends": [0, 0, 0, 0, 0],  # yfinance peut retourner ceci
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


# --- Tests de la Classe DataManager ---


class TestDataManager:

    def test_init_success(self, manager, mock_dependencies):
        """
        Vérifie que l'initialisation configure correctement les attributs
        et crée le répertoire de cache.
        """
        tmp_path = mock_dependencies["tmp_path"]

        # Vérifie la configuration
        assert manager.default_start == "2015-01-01"
        assert manager.default_end == "2024-12-31"
        assert manager.default_interval == "1d"
        assert manager.timezone_str == "Europe/Paris"
        assert manager.timezone == mock_dependencies["mock_tz"]

        # Vérifie le chemin du cache
        expected_cache_dir = tmp_path / "test_cache"
        assert manager.cache_dir == expected_cache_dir

        # Vérifie que mkdir a été appelé
        mock_dependencies["mock_mkdir"].assert_called_once_with(
            parents=True, exist_ok=True
        )

        # Vérifie le logging
        mock_dependencies["mock_logger"].info.assert_called()

    def test_init_failure_on_get_settings(self, mocker):
        """
        Vérifie qu'une exception est levée si get_settings échoue.
        """
        mocker.patch(
            "data_manager.get_settings", side_effect=Exception("Config load error")
        )
        mocker.patch("data_manager.setup_logger", return_value=MagicMock())

        with pytest.raises(Exception, match="Config load error"):
            DataManager()

    def test_get_cache_filepath(self, manager, mock_dependencies):
        """
        Vérifie la construction correcte du chemin de fichier cache.
        """
        tmp_path = mock_dependencies["tmp_path"]

        # Cas nominal
        path1 = manager._get_cache_filepath("AAPL", "1d")
        assert path1 == tmp_path / "test_cache" / "AAPL_1d.csv"

        # Cas avec ticker spécial (ex: futures)
        path2 = manager._get_cache_filepath("ES=F", "1h")
        assert path2 == tmp_path / "test_cache" / "ES_F_1h.csv"

        # Cas avec minuscules
        path3 = manager._get_cache_filepath("msft", "30m")
        assert path3 == tmp_path / "test_cache" / "MSFT_30m.csv"

    def test_load_from_cache_success_naive_index(
        self, manager, mock_dependencies, sample_df_utc, sample_df_paris
    ):
        """
        Vérifie le chargement réussi depuis le cache quand l'index est 'naive' (lu comme UTC).
        """
        # Configure les mocks
        mock_dependencies["mock_exists"].return_value = True
        mock_dependencies["mock_stat"].return_value = MagicMock(st_size=1024)
        # pd.read_csv lit les dates sans TZ, qui seront localisées en UTC
        df_naive = sample_df_utc.tz_localize(None)
        mock_dependencies["mock_read_csv"].return_value = df_naive

        # Appelle la méthode
        result_df = manager.load_from_cache("AAPL", "1d")

        # Vérifie les appels
        expected_path = manager._get_cache_filepath("AAPL", "1d")
        mock_dependencies["mock_exists"].assert_called_once_with()
        mock_dependencies["mock_stat"].assert_called_once_with()
        mock_dependencies["mock_read_csv"].assert_called_once_with(
            expected_path, index_col="Date", parse_dates=True
        )

        # Vérifie le résultat (doit être converti en 'Europe/Paris')
        assert_frame_equal(result_df, sample_df_paris)

    def test_load_from_cache_success_aware_index(
        self, manager, mock_dependencies, sample_df_utc, sample_df_paris
    ):
        """
        Vérifie le chargement réussi si le CSV a déjà un index aware (ex: UTC).
        """
        # Configure les mocks
        mock_dependencies["mock_exists"].return_value = True
        mock_dependencies["mock_stat"].return_value = MagicMock(st_size=1024)
        mock_dependencies["mock_read_csv"].return_value = sample_df_utc

        # Appelle la méthode
        result_df = manager.load_from_cache("AAPL", "1d")

        # Vérifie le résultat (doit être converti en 'Europe/Paris')
        assert_frame_equal(result_df, sample_df_paris)

    def test_load_from_cache_miss_file_not_exists(self, manager, mock_dependencies):
        """Vérifie le cas où le fichier cache n'existe pas."""
        mock_dependencies["mock_exists"].return_value = False

        result = manager.load_from_cache("AAPL", "1d")

        assert result is None
        mock_dependencies["mock_stat"].assert_not_called()
        mock_dependencies["mock_read_csv"].assert_not_called()

    def test_load_from_cache_miss_file_is_empty(self, manager, mock_dependencies):
        """Vérifie le cas où le fichier cache a une taille de 0 bytes."""
        mock_dependencies["mock_exists"].return_value = True
        mock_dependencies["mock_stat"].return_value = MagicMock(st_size=0)

        result = manager.load_from_cache("AAPL", "1d")

        assert result is None
        mock_dependencies["mock_read_csv"].assert_not_called()

    def test_load_from_cache_miss_dataframe_is_empty(self, manager, mock_dependencies):
        """Vérifie le cas où le fichier CSV est lu mais le DataFrame est vide."""
        mock_dependencies["mock_exists"].return_value = True
        mock_dependencies["mock_stat"].return_value = MagicMock(st_size=1024)
        mock_dependencies["mock_read_csv"].return_value = pd.DataFrame()

        result = manager.load_from_cache("AAPL", "1d")

        assert result is None

    def test_load_from_cache_miss_invalid_index(self, manager, mock_dependencies):
        """Vérifie le cas où le DataFrame lu n'a pas un DatetimeIndex."""
        mock_dependencies["mock_exists"].return_value = True
        mock_dependencies["mock_stat"].return_value = MagicMock(st_size=1024)
        # Crée un DF avec un index numérique
        df_bad_index = pd.DataFrame({"close": [1, 2, 3]})
        mock_dependencies["mock_read_csv"].return_value = df_bad_index

        result = manager.load_from_cache("AAPL", "1d")

        assert result is None

    def test_load_from_cache_read_exception(self, manager, mock_dependencies):
        """Vérifie le cas où pd.read_csv lève une exception."""
        mock_dependencies["mock_exists"].return_value = True
        mock_dependencies["mock_stat"].return_value = MagicMock(st_size=1024)
        mock_dependencies["mock_read_csv"].side_effect = pd.errors.ParserError(
            "Fichier corrompu"
        )

        result = manager.load_from_cache("AAPL", "1d")

        assert result is None
        # Vérifie que l'erreur a été loggée
        assert (
            "Erreur au chargement du cache"
            in mock_dependencies["mock_logger"].warning.call_args[0][0]
        )

    def test_save_to_cache_success(
        self, manager, mock_dependencies, sample_df_paris, sample_df_utc
    ):
        """Vérifie la sauvegarde réussie d'un DataFrame dans le cache."""
        mock_dependencies["mock_stat"].return_value = MagicMock(st_size=1024)

        manager.save_to_cache(sample_df_paris, "AAPL", "1d")

        expected_path = manager._get_cache_filepath("AAPL", "1d")

        # Vérifie que to_csv a été appelé
        mock_dependencies["mock_to_csv"].assert_called_once()

        # Vérifie que le DF sauvegardé a été converti en UTC
        # L'argument 'self' de to_csv est le DataFrame
        args, kwargs = mock_dependencies["mock_to_csv"].call_args
        df_saved = args[0]
        assert kwargs["path_or_buf"] == expected_path
        assert df_saved.index.tz.zone == "UTC"
        assert_frame_equal(df_saved, sample_df_utc)

    def test_save_to_cache_empty_dataframe(self, manager, mock_dependencies):
        """Vérifie que rien n'est sauvegardé si le DataFrame est vide."""
        manager.save_to_cache(pd.DataFrame(), "AAPL", "1d")

        mock_dependencies["mock_to_csv"].assert_not_called()
        assert (
            "Tentative de sauvegarde d'un DataFrame vide"
            in mock_dependencies["mock_logger"].warning.call_args[0][0]
        )

    def test_save_to_cache_exception(self, manager, mock_dependencies, sample_df_paris):
        """Vérifie la gestion d'erreur si to_csv échoue."""
        mock_dependencies["mock_to_csv"].side_effect = IOError("Disque plein")

        manager.save_to_cache(sample_df_paris, "AAPL", "1d")

        mock_dependencies["mock_to_csv"].assert_called_once()
        assert (
            "Échec de la sauvegarde"
            in mock_dependencies["mock_logger"].error.call_args[0][0]
        )

    def test_validate_data_success(self, manager, sample_df_paris):
        """Vérifie qu'un DataFrame valide passe la validation."""
        assert manager._validate_data(sample_df_paris, "AAPL") == True

    def test_validate_data_empty(self, manager):
        """Vérifie qu'un DataFrame vide échoue la validation."""
        assert manager._validate_data(pd.DataFrame(), "AAPL") == False

    def test_validate_data_missing_column(self, manager, sample_df_paris):
        """Vérifie qu'un DF avec une colonne manquante échoue."""
        df_invalid = sample_df_paris.drop(columns=["volume"])
        assert manager._validate_data(df_invalid, "AAPL") == False

    def test_validate_data_nan_values(
        self, manager, sample_df_paris, mock_dependencies
    ):
        """Vérifie que les NaNs sont signalés (mais la validation passe)."""
        df_with_nan = sample_df_paris.copy()
        df_with_nan.loc[df_with_nan.index[1], "open"] = np.nan

        assert manager._validate_data(df_with_nan, "AAPL") == True
        assert (
            "Présence de NaN"
            in mock_dependencies["mock_logger"].warning.call_args[0][0]
        )

    def test_validate_data_negative_price(self, manager, sample_df_paris):
        """Vérifie que les prix négatifs échouent la validation."""
        df_invalid = sample_df_paris.copy()
        df_invalid.loc[df_invalid.index[1], "low"] = -10

        assert manager._validate_data(df_invalid, "AAPL") == False

    def test_validate_data_zero_price(self, manager, sample_df_paris):
        """Vérifie que les prix nuls échouent la validation."""
        df_invalid = sample_df_paris.copy()
        df_invalid.loc[df_invalid.index[1], "close"] = 0

        assert manager._validate_data(df_invalid, "AAPL") == False

    def test_validate_data_gap_warning(self, manager, mock_dependencies):
        """Vérifie qu'un grand écart de dates déclenche un avertissement (mais passe)."""
        tz = pytz.timezone("Europe/Paris")
        dates = [pd.Timestamp("2023-01-01", tz=tz), pd.Timestamp("2023-01-20", tz=tz)]
        df_gap = pd.DataFrame(
            {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=dates
        )

        assert manager._validate_data(df_gap, "AAPL") == True
        assert (
            "Trou de données détecté"
            in mock_dependencies["mock_logger"].warning.call_args[0][0]
        )

    def test_download_data_success(
        self, manager, mock_dependencies, sample_df_raw_yfinance, sample_df_paris
    ):
        """Vérifie un téléchargement réussi (nettoyage, timezone, validation)."""
        mock_dependencies["mock_yf_ticker_instance"].history.return_value = (
            sample_df_raw_yfinance
        )

        result_df = manager.download_data("AAPL", "2023-01-01", "2023-01-05", "1d")

        # Vérifie l'appel à yfinance
        mock_dependencies["mock_yf_ticker_cls"].assert_called_with("AAPL")
        mock_dependencies["mock_yf_ticker_instance"].history.assert_called_with(
            start="2023-01-01",
            end="2023-01-05",
            interval="1d",
            actions=False,
            auto_adjust=True,
        )

        # Le DF retourné doit avoir des colonnes en minuscules et la bonne TZ
        expected_df = sample_df_paris.copy()
        assert_frame_equal(result_df, expected_df)

    def test_download_data_yfinance_returns_empty(self, manager, mock_dependencies):
        """Vérifie le cas où yfinance ne retourne aucune donnée."""
        mock_dependencies["mock_yf_ticker_instance"].history.return_value = (
            pd.DataFrame()
        )

        result_df = manager.download_data("AAPL", "2023-01-01", "2023-01-05", "1d")

        assert result_df.empty
        assert (
            "Aucune donnée retournée par yfinance"
            in mock_dependencies["mock_logger"].warning.call_args[0][0]
        )

    def test_download_data_yfinance_exception(self, manager, mock_dependencies):
        """Vérifie le cas où yfinance lève une exception."""
        mock_dependencies["mock_yf_ticker_instance"].history.side_effect = Exception(
            "Erreur API"
        )

        result_df = manager.download_data("AAPL", "2023-01-01", "2023-01-05", "1d")

        assert result_df.empty
        assert (
            "Erreur lors du téléchargement"
            in mock_dependencies["mock_logger"].error.call_args[0][0]
        )

    def test_download_data_invalid_index(self, manager, mock_dependencies):
        """Vérifie le cas où yfinance retourne un index non-datetime."""
        df_bad_index = pd.DataFrame({"Close": [1, 2, 3]})
        mock_dependencies["mock_yf_ticker_instance"].history.return_value = df_bad_index

        result_df = manager.download_data("AAPL", "2023-01-01", "2023-01-05", "1d")

        assert result_df.empty
        assert (
            "n'est pas un DatetimeIndex"
            in mock_dependencies["mock_logger"].error.call_args[0][0]
        )

    def test_download_data_validation_fails(
        self, manager, mock_dependencies, sample_df_raw_yfinance
    ):
        """Vérifie que si les données téléchargées échouent à la validation, un DF vide est retourné."""
        df_invalid = sample_df_raw_yfinance.copy()
        df_invalid.loc[df_invalid.index[1], "Low"] = -10  # Prix négatif

        mock_dependencies["mock_yf_ticker_instance"].history.return_value = df_invalid

        result_df = manager.download_data("AAPL", "2023-01-01", "2023-01-05", "1d")

        assert result_df.empty
        assert (
            "Validation échouée"
            in mock_dependencies["mock_logger"].error.call_args[0][0]
        )

    def test_get_data_cache_hit_full_range(
        self, manager, mock_dependencies, sample_df_paris
    ):
        """Scénario : Le cache est activé, trouvé, et couvre toute la plage demandée."""
        # Crée un cache couvrant 2023
        tz = pytz.timezone("Europe/Paris")
        full_dates = pd.date_range(
            start="2023-01-01", end="2023-12-31", freq="1D", tz=tz
        )
        full_df = pd.DataFrame(
            {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=full_dates
        )

        mock_dependencies["mock_load_from_cache"] = mocker.patch.object(
            manager, "load_from_cache", return_value=full_df
        )
        mock_dependencies["mock_download_data"] = mocker.patch.object(
            manager, "download_data"
        )

        # Demande une sous-plage
        start_req = "2023-02-01"
        end_req = "2023-02-10"
        result_df = manager.get_data("AAPL", start_date=start_req, end_date=end_req)

        # Vérifie
        mock_dependencies["mock_load_from_cache"].assert_called_once_with("AAPL", "1d")
        mock_dependencies["mock_download_data"].assert_not_called()

        expected_df = full_df.loc[start_req:end_req]
        assert_frame_equal(result_df, expected_df)
        assert (
            "chargées depuis le cache"
            in mock_dependencies["mock_logger"].info.call_args[0][0]
        )

    def test_get_data_cache_miss_download_success(
        self, manager, mock_dependencies, sample_df_paris
    ):
        """Scénario : Cache vide, téléchargement réussi, sauvegarde en cache."""
        # Crée le DF qui sera "téléchargé"
        tz = pytz.timezone("Europe/Paris")
        full_dates = pd.date_range(
            start="2015-01-01", end="2024-12-31", freq="1D", tz=tz
        )
        full_df = pd.DataFrame(
            {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=full_dates
        )

        mock_dependencies["mock_load_from_cache"] = mocker.patch.object(
            manager, "load_from_cache", return_value=None
        )
        mock_dependencies["mock_download_data"] = mocker.patch.object(
            manager, "download_data", return_value=full_df
        )
        mock_dependencies["mock_save_to_cache"] = mocker.patch.object(
            manager, "save_to_cache"
        )

        start_req = "2023-01-01"
        end_req = "2023-01-05"
        result_df = manager.get_data("AAPL", start_date=start_req, end_date=end_req)

        # Vérifie
        mock_dependencies["mock_load_from_cache"].assert_called_once_with("AAPL", "1d")

        # Doit télécharger la plage PAR DÉFAUT
        mock_dependencies["mock_download_data"].assert_called_once_with(
            "AAPL", "2015-01-01", "2024-12-31", "1d"
        )

        # Doit sauvegarder le DF téléchargé
        mock_dependencies["mock_save_to_cache"].assert_called_once_with(
            full_df, "AAPL", "1d"
        )

        # Le résultat doit être le DF filtré
        expected_df = full_df.loc[start_req:end_req]
        assert_frame_equal(result_df, expected_df)

    def test_get_data_cache_disabled(self, manager, mock_dependencies, sample_df_paris):
        """Scénario : use_cache=False. Doit télécharger et ne pas sauvegarder."""
        # Crée le DF qui sera "téléchargé"
        tz = pytz.timezone("Europe/Paris")
        full_dates = pd.date_range(
            start="2015-01-01", end="2024-12-31", freq="1D", tz=tz
        )
        full_df = pd.DataFrame(
            {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=full_dates
        )

        mock_dependencies["mock_load_from_cache"] = mocker.patch.object(
            manager, "load_from_cache"
        )
        mock_dependencies["mock_download_data"] = mocker.patch.object(
            manager, "download_data", return_value=full_df
        )
        mock_dependencies["mock_save_to_cache"] = mocker.patch.object(
            manager, "save_to_cache"
        )

        start_req = "2023-01-01"
        end_req = "2023-01-05"
        result_df = manager.get_data(
            "AAPL", start_date=start_req, end_date=end_req, use_cache=False
        )

        # Vérifie
        mock_dependencies["mock_load_from_cache"].assert_not_called()
        mock_dependencies["mock_download_data"].assert_called_once_with(
            "AAPL", "2015-01-01", "2024-12-31", "1d"
        )
        mock_dependencies["mock_save_to_cache"].assert_not_called()

        expected_df = full_df.loc[start_req:end_req]
        assert_frame_equal(result_df, expected_df)

    @pytest.mark.parametrize("scenario", ["start", "end"])
    def test_get_data_cache_insufficient(self, scenario, manager, mock_dependencies):
        """Scénario : Le cache est trouvé mais insuffisant (trop tard ou trop tôt)."""
        # Cache de 2023-02-01 à 2023-10-31
        tz = pytz.timezone("Europe/Paris")
        cache_dates = pd.date_range(
            start="2023-02-01", end="2023-10-31", freq="1D", tz=tz
        )
        cache_df = pd.DataFrame(
            {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=cache_dates
        )

        # DF téléchargé (plage complète)
        full_dates = pd.date_range(
            start="2015-01-01", end="2024-12-31", freq="1D", tz=tz
        )
        full_df = pd.DataFrame(
            {"open": 2, "high": 2, "low": 2, "close": 2, "volume": 2}, index=full_dates
        )

        mock_dependencies["mock_load_from_cache"] = mocker.patch.object(
            manager, "load_from_cache", return_value=cache_df
        )
        mock_dependencies["mock_download_data"] = mocker.patch.object(
            manager, "download_data", return_value=full_df
        )
        mock_dependencies["mock_save_to_cache"] = mocker.patch.object(
            manager, "save_to_cache"
        )

        if scenario == "start":
            # Demande commence *avant* le cache
            start_req = "2023-01-15"
            end_req = "2023-02-15"
        else:  # "end"
            # Demande finit *après* le cache
            start_req = "2023-10-15"
            end_req = "2023-11-15"

        result_df = manager.get_data("AAPL", start_date=start_req, end_date=end_req)

        # Vérifie : doit charger, voir que c'est insuffisant, télécharger, et sauvegarder
        mock_dependencies["mock_load_from_cache"].assert_called_once()
        mock_dependencies["mock_download_data"].assert_called_once()
        mock_dependencies["mock_save_to_cache"].assert_called_once()

        # Le résultat doit venir du NOUVEAU DF (valeurs = 2)
        expected_df = full_df.loc[start_req:end_req]
        assert_frame_equal(result_df, expected_df)
        assert (result_df["open"] == 2).all()
        assert (
            "Cache insuffisant" in mock_dependencies["mock_logger"].info.call_args[0][0]
        )

    def test_get_data_download_fails(self, manager, mock_dependencies):
        """Scénario : Cache vide et téléchargement échoue."""
        mock_dependencies["mock_load_from_cache"] = mocker.patch.object(
            manager, "load_from_cache", return_value=None
        )
        mock_dependencies["mock_download_data"] = mocker.patch.object(
            manager, "download_data", return_value=pd.DataFrame()  # Échec
        )
        mock_dependencies["mock_save_to_cache"] = mocker.patch.object(
            manager, "save_to_cache"
        )

        result_df = manager.get_data(
            "AAPL", start_date="2023-01-01", end_date="2023-01-05"
        )

        mock_dependencies["mock_load_from_cache"].assert_called_once()
        mock_dependencies["mock_download_data"].assert_called_once()
        mock_dependencies["mock_save_to_cache"].assert_not_called()  # Car DF vide

        assert result_df.empty
        assert (
            "Impossible d'obtenir des données"
            in mock_dependencies["mock_logger"].error.call_args[0][0]
        )

    def test_get_data_filtered_result_is_empty(self, manager, mock_dependencies):
        """Scénario : Données chargées (cache ou DL), mais le filtre final ne donne rien."""
        tz = pytz.timezone("Europe/Paris")
        full_dates = pd.date_range(
            start="2023-01-01", end="2023-12-31", freq="1D", tz=tz
        )
        full_df = pd.DataFrame(
            {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=full_dates
        )

        mock_dependencies["mock_load_from_cache"] = mocker.patch.object(
            manager, "load_from_cache", return_value=full_df
        )
        mock_dependencies["mock_download_data"] = mocker.patch.object(
            manager, "download_data"
        )

        # Demande une plage hors des données
        result_df = manager.get_data(
            "AAPL", start_date="2025-01-01", end_date="2025-01-31"
        )

        mock_dependencies["mock_load_from_cache"].assert_called_once()
        mock_dependencies["mock_download_data"].assert_not_called()

        assert result_df.empty
        assert (
            "Aucune donnée pour AAPL dans la plage"
            in mock_dependencies["mock_logger"].warning.call_args[0][0]
        )

    def test_get_data_cache_date_comparison_fails(
        self, manager, mock_dependencies, sample_df_paris
    ):
        """
        Scénario : Le chargement du cache réussit, mais la comparaison de date
        échoue (ex: format de date invalide). Doit re-télécharger.
        """
        mock_dependencies["mock_load_from_cache"] = mocker.patch.object(
            manager, "load_from_cache", return_value=sample_df_paris
        )
        mock_dependencies["mock_download_data"] = mocker.patch.object(
            manager, "download_data", return_value=sample_df_paris
        )
        mock_dependencies["mock_save_to_cache"] = mocker.patch.object(
            manager, "save_to_cache"
        )

        # Fait échouer la conversion en Timestamp
        mocker.patch(
            "data_manager.pd.Timestamp", side_effect=ValueError("Date invalide")
        )

        manager.get_data("AAPL", start_date="DATE_INVALIDE", end_date="2023-01-05")

        # Vérifie
        mock_dependencies["mock_load_from_cache"].assert_called_once()
        assert (
            "Erreur de comparaison de date"
            in mock_dependencies["mock_logger"].warning.call_args[0][0]
        )
        # Doit avoir re-téléchargé
        mock_dependencies["mock_download_data"].assert_called_once()
        mock_dependencies["mock_save_to_cache"].assert_called_once()
