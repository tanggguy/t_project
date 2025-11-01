# test_data_manager.py
# --- 1. Bibliothèques natives ---
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import timezone  # Importer timezone standard

# --- 2. Bibliothèques tierces ---
import pandas as pd
import pytest
from pytest_mock import MockerFixture
import pytz
from freezegun import freeze_time

# --- 3. Imports locaux du projet ---
from utils.data_manager import DataManager

# Configuration de base pour les tests
TEST_CACHE_DIR = "data/test_cache/"
DEFAULT_SETTINGS: Dict[str, Any] = {
    "project": {"timezone": "Europe/Paris"},
    "data": {
        "cache_dir": TEST_CACHE_DIR,
        "default_start_date": "2015-01-01",
        "default_end_date": "2024-12-31",
        "default_interval": "1d",
    },
}
PARIS_TZ = pytz.timezone("Europe/Paris")
UTC_TZ = pytz.timezone("UTC")


@pytest.fixture(autouse=True)
def mock_project_root(mocker: MockerFixture) -> Path:
    """Mock le PROJECT_ROOT pour pointer vers un 'faux' répertoire."""
    mock_path = mocker.MagicMock(spec=Path)
    mock_path_instance = mocker.MagicMock(spec=Path)

    mock_path.return_value.resolve.return_value.parent.parent = mock_path_instance
    mock_path_instance.__truediv__.side_effect = lambda other: Path(
        f"/fake/project/{other}"
    )

    mocker.patch("utils.data_manager.PROJECT_ROOT", mock_path_instance)
    mocker.patch("utils.data_manager.Path", mock_path)

    return mock_path_instance


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> Dict[str, MockerFixture]:
    """Mock toutes les dépendances externes (logger, settings, mkdir)."""

    # Patcher la variable 'logger' NIVEAU MODULE
    mock_log_instance = mocker.patch("utils.data_manager.logger", spec=logging.Logger)

    # Mock get_settings
    mock_get_settings = mocker.patch("utils.data_manager.get_settings")
    mock_get_settings.return_value = DEFAULT_SETTINGS

    # Mock pytz
    mock_pytz = mocker.patch("utils.data_manager.pytz")
    mock_pytz.timezone.side_effect = pytz.timezone

    # Mock Path.mkdir
    mock_mkdir = mocker.patch("pathlib.Path.mkdir")

    # Mock yfinance
    mock_yf_ticker_cls = mocker.patch("utils.data_manager.yf.Ticker")
    mock_yf_ticker_instance = mocker.MagicMock()
    mock_yf_ticker_cls.return_value = mock_yf_ticker_instance

    return {
        "logger": mock_log_instance,
        "get_settings": mock_get_settings,
        "pytz": mock_pytz,
        "mkdir": mock_mkdir,
        "yf_ticker": mock_yf_ticker_instance,
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fournit un DataFrame valide pour les tests."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-31", freq="D", tz=PARIS_TZ)
    data = {
        # Utiliser des floats pour les prix
        "open": [100.0 + i for i in range(len(dates))],
        "high": [105.0 + i for i in range(len(dates))],
        "low": [99.0 + i for i in range(len(dates))],
        "close": [102.0 + i for i in range(len(dates))],
        "volume": [1000 * i for i in range(len(dates))],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def data_manager(mock_dependencies: Dict[str, MockerFixture]) -> DataManager:
    """Fixture pour un DataManager initialisé avec des mocks."""
    return DataManager()


class TestDataManager:
    """Suite de tests pour la classe DataManager."""

    def test_init_success(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mock_project_root: Path,
    ):
        """Teste l'initialisation réussie du DataManager."""

        expected_cache_dir = Path(f"/fake/project/{TEST_CACHE_DIR}")

        mock_dependencies["logger"].info.assert_called_with(
            f"DataManager initialisé. Cache: {expected_cache_dir}. Timezone: {data_manager.timezone_str}"
        )

    def test_init_raises_exception(self, mock_dependencies: Dict[str, MockerFixture]):
        """Teste que l'init lève une exception si get_settings échoue."""

        mock_dependencies["get_settings"].side_effect = Exception(
            "Config file not found"
        )

        with pytest.raises(Exception, match="Config file not found"):
            DataManager()

        mock_dependencies["logger"].error.assert_called_once_with(
            "Erreur fatale lors de l'initialisation du DataManager: Config file not found"
        )

    @pytest.mark.parametrize(
        "ticker, interval, expected_filename",
        [
            ("AAPL", "1d", "AAPL_1d.csv"),
            ("BTC=F", "1h", "BTC_F_1h.csv"),
            ("msft", "5m", "MSFT_5m.csv"),
        ],
    )
    def test_get_cache_filepath(
        self,
        data_manager: DataManager,
        ticker: str,
        interval: str,
        expected_filename: str,
    ):
        filepath = data_manager._get_cache_filepath(ticker, interval)
        expected_path = data_manager.cache_dir / expected_filename
        assert filepath == expected_path

    def test_load_from_cache_success(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste le chargement réussi depuis le cache (cache hit)."""

        mock_filepath = mocker.MagicMock(spec=Path)
        mocker.patch.object(
            data_manager, "_get_cache_filepath", return_value=mock_filepath
        )
        mock_filepath.exists.return_value = True
        mock_filepath.stat.return_value.st_size = 1024

        naive_dates = pd.date_range("2020-01-01", "2020-01-05")
        df_from_csv = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=naive_dates)
        df_from_csv.index.name = "Date"

        mock_read_csv = mocker.patch("utils.data_manager.pd.read_csv")
        mock_read_csv.return_value = df_from_csv

        df = data_manager.load_from_cache("AAPL", "1d")

        assert df is not None
        mock_dependencies["logger"].debug.assert_called_with(
            f"[OK] Cache hit pour AAPL (1d) - {len(df)} lignes chargées"
        )

    def test_load_from_cache_file_not_found(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste le cache miss si le fichier n'existe pas."""

        mock_filepath = mocker.MagicMock(spec=Path)
        mocker.patch.object(
            data_manager, "_get_cache_filepath", return_value=mock_filepath
        )
        mock_filepath.exists.return_value = False

        df = data_manager.load_from_cache("AAPL", "1d")

        assert df is None
        mock_dependencies["logger"].debug.assert_called_with(
            "Cache miss pour AAPL (1d) - fichier inexistant"
        )

    def test_load_from_cache_file_is_empty(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste le cache miss si le fichier a une taille de 0 bytes."""

        mock_filepath = mocker.MagicMock(spec=Path)
        mocker.patch.object(
            data_manager, "_get_cache_filepath", return_value=mock_filepath
        )
        mock_filepath.exists.return_value = True
        mock_filepath.stat.return_value.st_size = 0

        df = data_manager.load_from_cache("AAPL", "1d")

        assert df is None
        mock_dependencies["logger"].warning.assert_called_with(
            f"Fichier cache {mock_filepath} est vide (0 bytes). Re-téléchargement."
        )

    def test_load_from_cache_dataframe_empty_after_read(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste le cache miss si le DataFrame lu est vide."""

        mock_filepath = mocker.MagicMock(spec=Path)
        mocker.patch.object(
            data_manager, "_get_cache_filepath", return_value=mock_filepath
        )
        mock_filepath.exists.return_value = True
        mock_filepath.stat.return_value.st_size = 1024

        mock_read_csv = mocker.patch("utils.data_manager.pd.read_csv")
        mock_read_csv.return_value = pd.DataFrame()

        df = data_manager.load_from_cache("AAPL", "1d")

        assert df is None
        mock_dependencies["logger"].warning.assert_called_with(
            f"DataFrame vide après lecture de {mock_filepath}. Re-téléchargement."
        )

    def test_load_from_cache_invalid_index(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste le cache miss si l'index n'est pas un DatetimeIndex."""

        mock_filepath = mocker.MagicMock(spec=Path)
        mocker.patch.object(
            data_manager, "_get_cache_filepath", return_value=mock_filepath
        )
        mock_filepath.exists.return_value = True
        mock_filepath.stat.return_value.st_size = 1024

        df_bad_index = pd.DataFrame({"close": [1, 2]}, index=[0, 1])
        mock_read_csv = mocker.patch("utils.data_manager.pd.read_csv")
        mock_read_csv.return_value = df_bad_index

        df = data_manager.load_from_cache("AAPL", "1d")

        assert df is None
        mock_dependencies["logger"].warning.assert_called_with(
            f"Index invalide dans {mock_filepath}. Re-téléchargement."
        )

    def test_save_to_cache_empty_dataframe(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste que la sauvegarde est ignorée si le DataFrame est vide."""

        mock_to_csv = mocker.patch.object(pd.DataFrame, "to_csv")
        empty_df = pd.DataFrame()

        data_manager.save_to_cache(empty_df, "AAPL", "1d")

        mock_to_csv.assert_not_called()
        mock_dependencies["logger"].warning.assert_called_with(
            "Tentative de sauvegarde d'un DataFrame vide pour AAPL. Ignoré."
        )

    def test_save_to_cache_exception(
        self,
        data_manager: DataManager,
        sample_dataframe: pd.DataFrame,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste la gestion d'erreur si to_csv échoue."""

        mock_filepath = mocker.MagicMock(spec=Path)
        mocker.patch.object(
            data_manager, "_get_cache_filepath", return_value=mock_filepath
        )

        mock_to_csv = mocker.patch.object(pd.DataFrame, "to_csv")
        mock_to_csv.side_effect = Exception("Disk full")

        data_manager.save_to_cache(sample_dataframe, "AAPL", "1d")

        mock_to_csv.assert_called_once()
        mock_dependencies["logger"].error.assert_called_with(
            f"Échec de la sauvegarde dans le cache {mock_filepath}: Disk full"
        )

    def test_validate_data_success(
        self,
        data_manager: DataManager,
        sample_dataframe: pd.DataFrame,
        mock_dependencies: Dict[str, MockerFixture],
    ):
        """Teste la validation réussie de données valides."""

        is_valid = data_manager._validate_data(sample_dataframe, "AAPL")

        assert is_valid is True
        mock_dependencies["logger"].debug.assert_called_with("Validation OK pour AAPL.")

    def test_validate_data_empty(
        self, data_manager: DataManager, mock_dependencies: Dict[str, MockerFixture]
    ):
        """Teste la validation échouée (False) si le DataFrame est vide."""

        is_valid = data_manager._validate_data(pd.DataFrame(), "AAPL")

        assert is_valid is False
        mock_dependencies["logger"].warning.assert_called_with(
            "Aucune donnée retournée pour AAPL."
        )

    def test_validate_data_missing_columns(
        self,
        data_manager: DataManager,
        sample_dataframe: pd.DataFrame,
        mock_dependencies: Dict[str, MockerFixture],
    ):
        """Teste la validation échouée (False) si des colonnes manquent."""

        df_invalid = sample_dataframe.drop(columns=["volume", "open"])

        is_valid = data_manager._validate_data(df_invalid, "AAPL")

        assert is_valid is False
        mock_dependencies["logger"].error.assert_called_with(
            "Colonnes manquantes pour AAPL: ['open', 'volume']. Données invalides."
        )

    def test_validate_data_with_nans(
        self,
        data_manager: DataManager,
        sample_dataframe: pd.DataFrame,
        mock_dependencies: Dict[str, MockerFixture],
    ):
        """Teste la validation réussie (True) mais avec un avertissement pour les NaNs."""

        sample_dataframe.loc[sample_dataframe.index[1], "close"] = pd.NA
        sample_dataframe.loc[sample_dataframe.index[3], "volume"] = pd.NA

        is_valid = data_manager._validate_data(sample_dataframe, "AAPL")

        assert is_valid is True
        mock_dependencies["logger"].warning.assert_called_with(
            "Présence de NaN pour AAPL. Colonnes affectées: {'close': 1, 'volume': 1}"
        )

    @pytest.mark.parametrize("bad_price", [0, -10.5])
    def test_validate_data_negative_or_zero_price(
        self,
        data_manager: DataManager,
        sample_dataframe: pd.DataFrame,
        bad_price: float,
        mock_dependencies: Dict[str, MockerFixture],
    ):
        """Teste la validation échouée (False) pour des prix nuls ou négatifs."""

        sample_dataframe.loc[sample_dataframe.index[2], "low"] = bad_price

        is_valid = data_manager._validate_data(sample_dataframe, "AAPL")

        assert is_valid is False
        mock_dependencies["logger"].warning.assert_called_with(
            f"Prix low négatifs ou nuls détectés pour AAPL. Données potentiellement corrompues."
        )

    def test_validate_data_large_gap(
        self, data_manager: DataManager, mock_dependencies: Dict[str, MockerFixture]
    ):
        """Teste la validation réussie (True) mais avec un avertissement pour les grands trous."""

        dates = pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-20"])
        df_gaps = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [1, 2, 3],
                "low": [1, 2, 3],
                "close": [1, 2, 3],
                "volume": [1, 2, 3],
            },
            index=dates,
        )

        is_valid = data_manager._validate_data(df_gaps, "AAPL")

        assert is_valid is True
        mock_dependencies["logger"].warning.assert_called_with(
            f"Trou de données détecté pour AAPL: 15 days 00:00:00. Vérifier la continuité."
        )

    def test_download_data_success(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste un téléchargement réussi via yfinance."""

        raw_dates = pd.date_range("2020-01-01", "2020-01-05", tz="America/New_York")
        raw_df = pd.DataFrame(
            {
                "Open": [1, 2, 3, 4, 5],
                "High": [1, 2, 3, 4, 5],
                "Low": [1, 2, 3, 4, 5],
                "Close": [1, 2, 3, 4, 5],
                "Volume": [1, 2, 3, 4, 5],
            },
            index=raw_dates,
        )

        mock_dependencies["yf_ticker"].history.return_value = raw_df
        mock_validate = mocker.patch.object(
            data_manager, "_validate_data", return_value=True
        )

        df = data_manager.download_data("MSFT", "2020-01-01", "2020-01-05", "1d")

        assert not df.empty
        mock_validate.assert_called_once()

        mock_dependencies["logger"].info.assert_any_call(
            "Téléchargement de MSFT (2020-01-01 à 2020-01-05, 1d)..."
        )
        mock_dependencies["logger"].info.assert_any_call(
            f"Données téléchargées avec succès pour MSFT ({len(df)} lignes)."
        )

    def test_download_data_yfinance_returns_empty(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste le cas où yfinance ne retourne aucune donnée."""

        mock_dependencies["yf_ticker"].history.return_value = pd.DataFrame()
        mocker.patch.object(data_manager, "_validate_data")

        df = data_manager.download_data("EMPTY", "2020-01-01", "2020-01-05", "1d")

        assert df.empty
        mock_dependencies["logger"].warning.assert_called_with(
            "Aucune donnée retournée par yfinance pour EMPTY."
        )

    def test_download_data_invalid_index_type(
        self, data_manager: DataManager, mock_dependencies: Dict[str, MockerFixture]
    ):
        """Teste le cas où yfinance retourne un index non-Datetime."""

        bad_df = pd.DataFrame({"Close": [1, 2]}, index=[0, 1])
        mock_dependencies["yf_ticker"].history.return_value = bad_df

        df = data_manager.download_data("BADIDX", "2020-01-01", "2020-01-05", "1d")

        assert df.empty
        mock_dependencies["logger"].error.assert_called_with(
            "L'index pour BADIDX n'est pas un DatetimeIndex."
        )

    def test_download_data_validation_fails(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste le cas où les données téléchargées échouent à la validation."""

        raw_dates = pd.date_range("2020-01-01", "2020-01-05", tz=UTC_TZ)
        raw_df = pd.DataFrame(
            {"Close": [1, 2, 3, 4, 5], "Volume": [1, 2, 3, 4, 5]}, index=raw_dates
        )
        mock_dependencies["yf_ticker"].history.return_value = raw_df

        mocker.patch.object(data_manager, "_validate_data", return_value=False)

        df = data_manager.download_data("BADVAL", "2020-01-01", "2020-01-05", "1d")

        assert df.empty
        mock_dependencies["logger"].error.assert_called_with(
            "Validation échouée pour BADVAL."
        )

    def test_download_data_yfinance_exception(
        self, data_manager: DataManager, mock_dependencies: Dict[str, MockerFixture]
    ):
        """Teste le cas où yf.Ticker.history lève une exception."""

        mock_dependencies["yf_ticker"].history.side_effect = Exception(
            "API limit reached"
        )

        df = data_manager.download_data("ERR", "2020-01-01", "2020-01-05", "1d")

        assert df.empty
        mock_dependencies["logger"].error.assert_called_with(
            "Erreur lors du téléchargement de ERR: API limit reached"
        )

    @freeze_time("2024-01-10")
    def test_get_data_from_cache_success_full_match(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste get_data: succès complet depuis le cache."""

        full_dates = pd.date_range(
            DEFAULT_SETTINGS["data"]["default_start_date"], "2024-01-09", tz=PARIS_TZ
        )
        cached_df = pd.DataFrame({"close": range(len(full_dates))}, index=full_dates)

        mocker.patch.object(data_manager, "load_from_cache", return_value=cached_df)
        mocker.patch.object(data_manager, "download_data")

        df = data_manager.get_data(
            "AAPL", start_date="2020-01-01", end_date="2021-12-31"
        )

        assert not df.empty
        mock_dependencies["logger"].info.assert_any_call(
            "[OK] Données pour AAPL chargées depuis le cache."
        )
        mock_dependencies["logger"].info.assert_any_call(
            "[OK] Données prêtes pour AAPL (731 lignes de 2020-01-01 à 2021-12-31)."
        )

    def test_get_data_cache_miss_then_download(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste get_data: cache miss, suivi d'un téléchargement et d'une sauvegarde."""

        mocker.patch.object(data_manager, "load_from_cache", return_value=None)

        full_dates = pd.date_range(
            DEFAULT_SETTINGS["data"]["default_start_date"],
            DEFAULT_SETTINGS["data"]["default_end_date"],
            tz=PARIS_TZ,
        )
        downloaded_df = pd.DataFrame(
            {"close": range(len(full_dates))}, index=full_dates
        )
        mock_download = mocker.patch.object(
            data_manager, "download_data", return_value=downloaded_df
        )
        mock_save = mocker.patch.object(data_manager, "save_to_cache")

        df = data_manager.get_data(
            "MSFT", start_date="2023-01-01", end_date="2023-12-31", interval="1d"
        )

        assert not df.empty
        assert len(df) == 365
        mock_dependencies["logger"].info.assert_any_call(
            "[OK] Données prêtes pour MSFT (365 lignes de 2023-01-01 à 2023-12-31)."
        )

    @pytest.mark.parametrize("case", ["insufficient_start", "insufficient_end"])
    def test_get_data_cache_insufficient(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
        case: str,
    ):
        """Teste get_data: cache partiel, forçant le re-téléchargement."""

        partial_dates = pd.date_range("2020-01-01", "2022-12-31", tz=PARIS_TZ)
        cached_df = pd.DataFrame(
            {"close": range(len(partial_dates))}, index=partial_dates
        )
        mocker.patch.object(data_manager, "load_from_cache", return_value=cached_df)

        full_dates = pd.date_range(
            DEFAULT_SETTINGS["data"]["default_start_date"],
            DEFAULT_SETTINGS["data"]["default_end_date"],
            tz=PARIS_TZ,
        )
        downloaded_df = pd.DataFrame(
            {"close": range(len(full_dates))}, index=full_dates
        )
        mock_download = mocker.patch.object(
            data_manager, "download_data", return_value=downloaded_df
        )
        mocker.patch.object(data_manager, "save_to_cache")

        start_req, end_req = "2019-01-01", "2021-01-01"
        if case == "insufficient_end":
            start_req, end_req = "2021-01-01", "2024-01-01"

        df = data_manager.get_data("TSLA", start_date=start_req, end_date=end_req)

        mock_dependencies["logger"].info.assert_any_call(
            f"Cache insuffisant pour TSLA (demandé: {start_req} à {end_req}). Re-téléchargement."
        )
        assert not df.empty

    def test_get_data_no_cache_flag(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste get_data: use_cache=False force le téléchargement et empêche la sauvegarde."""

        mock_load = mocker.patch.object(data_manager, "load_from_cache")

        full_dates = pd.date_range(
            DEFAULT_SETTINGS["data"]["default_start_date"],
            DEFAULT_SETTINGS["data"]["default_end_date"],
            tz=PARIS_TZ,
        )
        downloaded_df = pd.DataFrame(
            {"close": range(len(full_dates))}, index=full_dates
        )
        mock_download = mocker.patch.object(
            data_manager, "download_data", return_value=downloaded_df
        )

        mock_save = mocker.patch.object(data_manager, "save_to_cache")

        df = data_manager.get_data(
            "GOOG", start_date="2023-01-01", end_date="2023-01-31", use_cache=False
        )

        mock_load.assert_not_called()
        mock_download.assert_called_once()
        mock_save.assert_not_called()
        assert len(df) == 31

    def test_get_data_download_fails(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste get_data: échec du chargement ET échec du téléchargement."""

        mocker.patch.object(data_manager, "load_from_cache", return_value=None)
        mocker.patch.object(data_manager, "download_data", return_value=pd.DataFrame())
        mocker.patch.object(data_manager, "save_to_cache")

        df = data_manager.get_data("FAIL", "2020-01-01", "2020-01-31")

        assert df.empty
        mock_dependencies["logger"].error.assert_called_with(
            "Impossible d'obtenir des données pour FAIL."
        )

    def test_get_data_filtered_is_empty(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste get_data: succès du cache/téléchargement, mais le filtrage ne donne rien."""

        # Arrange
        # 1. Le cache (2015-2024) est chargé
        full_dates = pd.date_range("2015-01-01", "2024-12-31", tz=PARIS_TZ)
        cached_df = pd.DataFrame({"close": range(len(full_dates))}, index=full_dates)
        mocker.patch.object(data_manager, "load_from_cache", return_value=cached_df)

        # --- CORRECTION 2: Simuler le re-téléchargement (qui ramène les mêmes données) ---
        # Le code va voir que 2026 manque et appeler download_data
        # Nous simulons que le téléchargement retourne les mêmes données (2015-2024)
        mocker.patch.object(data_manager, "download_data", return_value=cached_df)

        # Act
        # Demande une plage future
        df = data_manager.get_data(
            "AAPL", start_date="2026-01-01", end_date="2026-12-31"
        )

        # Assert
        # 1. Vérifier que le DataFrame final est bien vide
        assert df.empty

        # 2. Vérifier que le log "Cache insuffisant" a été appelé (car 2026 > 2024)
        mock_dependencies["logger"].info.assert_any_call(
            "Cache insuffisant pour AAPL (demandé: 2026-01-01 à 2026-12-31). Re-téléchargement."
        )

        # 3. Vérifier que le warning final a été appelé
        mock_dependencies["logger"].warning.assert_called_with(
            "Aucune donnée pour AAPL dans la plage 2026-01-01 à 2026-12-31."
        )

    def test_get_data_cache_date_comparison_error(
        self,
        data_manager: DataManager,
        mock_dependencies: Dict[str, MockerFixture],
        mocker: MockerFixture,
    ):
        """Teste get_data: une erreur de conversion de date dans le cache force le re-téléchargement."""

        full_dates = pd.date_range("2015-01-01", "2024-12-31", tz=PARIS_TZ)
        cached_df = pd.DataFrame({"close": range(len(full_dates))}, index=full_dates)
        mocker.patch.object(data_manager, "load_from_cache", return_value=cached_df)

        mocker.patch(
            "utils.data_manager.pd.Timestamp", side_effect=Exception("Invalid date")
        )

        mock_download = mocker.patch.object(
            data_manager, "download_data", return_value=cached_df
        )

        data_manager.get_data("AAPL", start_date="2020-01-01", end_date="2021-01-01")

        mock_dependencies["logger"].warning.assert_called_with(
            "Erreur de comparaison de date: Invalid date. Re-téléchargement."
        )
        mock_download.assert_called_once()
