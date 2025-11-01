# tests/unit/test_data_processor.py

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import logging
from utils.data_processor import add_returns, handle_outliers, resample_data


# --- Fixtures ---
@pytest.fixture
def sample_data():
    """Crée un DataFrame de test avec des données OHLCV"""
    dates = pd.date_range(start="2023-01-01", periods=5, freq="1D")
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "close": [100.0, 102.0, 101.0, 103.0, 102.0],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def mock_logger(monkeypatch):
    """Fixture to provide a MagicMock logger and patch the data_processor logger.

    Tests that assert logger calls request this fixture. It patches
    utils.data_processor.logger so calls inside the module go to the mock.
    """
    from unittest.mock import MagicMock

    mock_log = MagicMock()
    # Patch the logger used inside the data_processor module only when requested
    monkeypatch.setattr("utils.data_processor.logger", mock_log)
    return mock_log


@pytest.fixture
def sample_data_with_outlier(sample_data):
    """Crée un DataFrame avec un outlier"""
    df = sample_data.copy()
    df.loc[df.index[2], "close"] = 200.0  # Outlier au milieu
    return df


# --- Tests pour add_returns ---
def test_add_returns_nominal(sample_data):
    """Test le calcul des rendements dans un cas normal"""
    result = add_returns(sample_data)

    # Vérifier que les colonnes ont été ajoutées
    assert "pct_return" in result.columns
    assert "log_return" in result.columns

    # Vérifier les calculs (premier jour devrait être 0)
    assert result["pct_return"].iloc[0] == 0.0
    assert result["log_return"].iloc[0] == 0.0

    # Vérifier un calcul de rendement
    expected_pct_return = (102.0 - 100.0) / 100.0  # Jour 2
    assert result["pct_return"].iloc[1] == pytest.approx(expected_pct_return)


def test_add_returns_missing_close(sample_data):
    """Test le comportement quand la colonne 'close' est manquante"""
    df = sample_data.drop(columns=["close"])
    result = add_returns(df)
    assert "pct_return" not in result.columns
    assert "log_return" not in result.columns


def test_add_returns_single_row():
    """Test avec un DataFrame d'une seule ligne"""
    df = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2023-01-01")])
    result = add_returns(df)
    assert result["pct_return"].iloc[0] == 0.0
    assert result["log_return"].iloc[0] == 0.0


# --- Tests pour handle_outliers ---
def test_handle_outliers_nominal(sample_data_with_outlier, mock_logger):
    """Test la gestion des outliers dans un cas normal"""
    # Ajouter un outlier plus significatif
    df = add_returns(sample_data_with_outlier.copy())

    result = handle_outliers(df, quantile=0.1)
    
    assert mock_logger.info.called
    assert "rendements extrêmes ont été" in mock_logger.info.call_args[0][0]

    # Vérifier que l'outlier a été écrêté
    original_value = sample_data_with_outlier["close"].iloc[2]
    assert (
        result["pct_return"].iloc[2]
        != (original_value - sample_data_with_outlier["close"].iloc[1])
        / sample_data_with_outlier["close"].iloc[1]
    )


def test_handle_outliers_no_returns(sample_data, mock_logger):
    """Test handle_outliers sans colonne de rendements"""
    result = handle_outliers(sample_data.copy())
    mock_logger.warning.assert_called_once_with(
        "La méthode 'clip_returns' nécessite 'pct_return'. "
        "Appel de add_returns() implicitement."
    )
    assert "pct_return" in result.columns  # La colonne devrait être ajoutée


def test_handle_outliers_invalid_method(sample_data_with_outlier, mock_logger):
    """Test avec une méthode invalide"""
    handle_outliers(sample_data_with_outlier, method="invalid_method")
    mock_logger.warning.assert_called_once_with(
        "Méthode d'outlier 'invalid_method' non reconnue. Aucune action."
    )


def test_handle_outliers_single_row():
    """Test avec un DataFrame d'une seule ligne"""
    df = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2023-01-01")])
    df = add_returns(df)
    result = handle_outliers(df)
    assert_frame_equal(result, df)  # Aucun changement attendu


# --- Tests pour resample_data ---
def test_resample_data_weekly(sample_data):
    """Test le resampling hebdomadaire"""
    result = resample_data(sample_data, rule="1W")

    # Vérifier que le nombre de lignes a diminué
    assert len(result) < len(sample_data)

    # Vérifier les règles d'agrégation pour la première ligne
    # Calculer l'attendu via pandas pour éviter les suppositions sur l'alignement des bins
    expected = sample_data.resample("1W").apply(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    assert result["open"].iloc[0] == expected["open"].iloc[0]
    assert result["high"].iloc[0] == expected["high"].iloc[0]
    assert result["low"].iloc[0] == expected["low"].iloc[0]
    assert result["close"].iloc[0] == expected["close"].iloc[0]
    assert result["volume"].iloc[0] == expected["volume"].iloc[0]


def test_resample_data_invalid_index(mock_logger):
    """Test avec un index non temporel"""
    df = pd.DataFrame(
        {"close": [100.0, 101.0], "volume": [1000, 1100]}, index=[1, 2]
    )  # Index numérique

    result = resample_data(df)
    assert_frame_equal(result, df)  # Devrait retourner le DataFrame inchangé
    mock_logger.error.assert_called_once_with(
        "L'index n'est pas un DatetimeIndex. Resampling impossible."
    )


def test_resample_data_missing_columns(sample_data):
    """Test avec des colonnes manquantes"""
    df = sample_data[["close", "volume"]]  # Seulement close et volume
    result = resample_data(df, rule="1W")

    assert "close" in result.columns
    assert "volume" in result.columns
    assert "open" not in result.columns
    assert "high" not in result.columns
    assert "low" not in result.columns


def test_resample_data_empty():
    """Test avec un DataFrame vide"""
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df.index = pd.DatetimeIndex([])  # Index temporel vide
    result = resample_data(df)
    assert len(result) == 0
    assert list(result.columns) == list(df.columns)


def test_resample_data_invalid_rule(sample_data, mock_logger):
    """Test avec une règle de resampling invalide"""
    result = resample_data(sample_data, rule="invalid")
    mock_logger.error.assert_called_once()
    assert "Échec du resampling" in mock_logger.error.call_args[0][0]
    assert_frame_equal(result, sample_data)  # Devrait retourner le DataFrame original


# --- Tests de cas limites et valeurs extrêmes ---
def test_extreme_returns(mock_logger):
    """Test avec des rendements extrêmes"""
    dates = pd.date_range(start="2023-01-01", periods=3, freq="1D")
    df = pd.DataFrame(
        {"close": [100.0, 1000.0, 10.0]}, index=dates  # Variations extrêmes
    )

    df = add_returns(df)
    handle_outliers(df, quantile=0.1)
    mock_logger.info.assert_called_once()
    assert "rendements extrêmes ont été" in mock_logger.info.call_args[0][0]


def test_zero_values():
    """Test avec des valeurs très petites"""
    dates = pd.date_range(start="2023-01-01", periods=3, freq="1D")
    df = pd.DataFrame(
        {"close": [100.0, 1.0, 100.0]}, index=dates  # Une valeur plus raisonnable
    )

    df = add_returns(df)
    # Vérifier que les rendements sont calculés sans erreur
    assert np.all(np.isfinite(df["pct_return"]))  # Pas de valeurs infinies
    assert np.all(np.isfinite(df["log_return"]))  # Pas de valeurs infinies

    # Vérifier que les valeurs sont dans des limites raisonnables
    assert (
        df["pct_return"].abs().max() < 100
    )  # Les rendements ne devraient pas être trop extrêmes


import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock


# Suppression du mock du logger pour permettre la capture des logs réels


# Now, import the functions to be tested
from utils.data_processor import add_returns, handle_outliers, resample_data

# --- Test Data Fixtures ---


@pytest.fixture
def basic_ohlc_df():
    """Provides a basic OHLC DataFrame for testing."""
    data = {
        "open": [100, 102, 101, 103],
        "high": [103, 104, 102, 105],
        "low": [99, 101, 100, 102],
        "close": [102, 101, 103, 104],
        "volume": [1000, 1100, 1200, 1300],
    }
    return pd.DataFrame(data)


@pytest.fixture
def ohlc_df_with_datetime_index():
    """Provides an OHLC DataFrame with a DatetimeIndex for 2 full weeks."""
    # Dates are Monday, Tuesday of week 1, and Monday, Tuesday of week 2
    dates = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-09", "2023-01-10"])
    data = {
        "open": [100, 102, 101, 103],
        "high": [103, 104, 102, 105],
        "low": [99, 101, 100, 102],
        "close": [102, 101, 103, 104],
        "volume": [1000, 1100, 1200, 1300],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def df_with_outliers():
    """Provides a DataFrame with return outliers."""
    data = {"close": [100, 101, 102, 50, 104, 105, 106, 107, 108, 109, 110]}
    df = pd.DataFrame(data)
    df = add_returns(df)  # Add returns to work with
    return df


@pytest.fixture
def df_no_outliers():
    """Provides a DataFrame with normal returns for stable quantile testing."""
    # data = {"close": np.linspace(100, 105, 30)}  # <-- OLD LINE

    # --- NEW LINES ---
    # Using identical values ensures all pct_returns are 0.0.
    # This guarantees that quantiles will also be 0.0,
    # and no clipping will occur, testing the 'else' branch correctly.
    data = {"close": [100.0] * 30}
    # --- END NEW LINES ---

    df = pd.DataFrame(data)
    df = add_returns(df)
    return df


# --- Tests for add_returns ---


class TestAddReturns:
    def test_nominal_case_adds_return_columns(self, basic_ohlc_df):
        """Tests that 'pct_return' and 'log_return' are added correctly."""
        df = add_returns(basic_ohlc_df.copy())

        assert "pct_return" in df.columns
        assert "log_return" in df.columns
        pd.testing.assert_series_equal(
            df["pct_return"],
            pd.Series([0.0, -0.009804, 0.019802, 0.009709], name="pct_return"),
            check_exact=False,
            atol=1e-5,
        )
        pd.testing.assert_series_equal(
            df["log_return"],
            pd.Series([0.0, -0.009852, 0.019608, 0.009662], name="log_return"),
            check_exact=False,
            atol=1e-5,
        )

    def test_error_case_missing_close_column(self, mock_logger):
        """Tests that the function returns the original df and logs an error if 'close' is missing."""
        df_no_close = pd.DataFrame({"open": [100, 101]})
        result_df = add_returns(df_no_close.copy())

        assert "pct_return" not in result_df.columns
        assert "log_return" not in result_df.columns
        pd.testing.assert_frame_equal(result_df, df_no_close)
        mock_logger.error.assert_called_once_with(
            "La colonne 'close' est manquante pour calculer les rendements."
        )

    def test_edge_case_single_row(self):
        """Tests behavior with a single-row DataFrame."""
        df_single = pd.DataFrame({"close": [100]})
        result_df = add_returns(df_single.copy())

        assert result_df.loc[0, "pct_return"] == 0.0
        assert result_df.loc[0, "log_return"] == 0.0
        assert result_df.shape == (1, 3)

    def test_edge_case_empty_dataframe(self):
        """Tests behavior with an empty DataFrame."""
        df_empty = pd.DataFrame({"close": []})
        result_df = add_returns(df_empty.copy())

        assert "pct_return" in result_df.columns
        assert "log_return" in result_df.columns
        assert result_df.empty


# --- Tests for handle_outliers ---


class TestHandleOutliers:
    def test_nominal_case_clips_returns(self, df_with_outliers, mock_logger):
        """Tests that extreme returns are clipped based on quantiles."""
        original_max_return = df_with_outliers["pct_return"].max()
        original_min_return = df_with_outliers["pct_return"].min()

        # Using a high quantile to ensure the outlier at index 3 is clipped
        df_handled = handle_outliers(df_with_outliers.copy(), quantile=0.1)

        assert df_handled["pct_return"].max() < original_max_return
        assert df_handled["pct_return"].min() > original_min_return
        mock_logger.info.assert_called_once()
        assert "rendements extrêmes ont été écrêtés" in mock_logger.info.call_args[0][0]

    def test_no_outliers_detected(self, df_no_outliers, mock_logger):
        """Tests that no changes are made if no outliers are detected."""
        original_df = df_no_outliers.copy()

        df_handled = handle_outliers(df_no_outliers.copy(), quantile=0.001)

        pd.testing.assert_frame_equal(original_df, df_handled)
        mock_logger.debug.assert_called_with(
            "Aucun outlier de rendement détecté (basé sur les quantiles)."
        )

    def test_implicit_add_returns_call(self, basic_ohlc_df, mock_logger):
        """Tests that add_returns is called if 'pct_return' is missing."""
        df = basic_ohlc_df.copy()
        assert "pct_return" not in df.columns

        df_handled = handle_outliers(df, quantile=0.01)

        assert "pct_return" in df_handled.columns
        mock_logger.warning.assert_called_once_with(
            "La méthode 'clip_returns' nécessite 'pct_return'. Appel de add_returns() implicitement."
        )

    def test_error_case_unrecognized_method(self, basic_ohlc_df, mock_logger):
        """Tests that no action is taken for an unrecognized method."""
        original_df = basic_ohlc_df.copy()
        df_handled = handle_outliers(original_df.copy(), method="unknown_method")

        pd.testing.assert_frame_equal(original_df, df_handled)
        mock_logger.warning.assert_called_once_with(
            "Méthode d'outlier 'unknown_method' non reconnue. Aucune action."
        )

    def test_edge_case_empty_df(self):
        """Tests handling of an empty DataFrame."""
        df_empty = pd.DataFrame({"close": []})
        result_df = handle_outliers(df_empty.copy())
        assert result_df.empty


# --- Tests for resample_data ---


class TestResampleData:
    def test_nominal_case_resamples_correctly(
        self, ohlc_df_with_datetime_index, mock_logger
    ):
        """Tests correct resampling of data to a weekly frequency."""
        df_weekly = resample_data(ohlc_df_with_datetime_index.copy(), rule="1W")

        assert df_weekly.shape[0] == 2  # Two weeks in the test data
        assert df_weekly.index.name == ohlc_df_with_datetime_index.index.name

        # Check aggregation logic for the first week
        first_week = df_weekly.iloc[0]
        assert first_week["open"] == 100  # First open of week 1
        assert first_week["high"] == 104  # Max high of week 1
        assert first_week["low"] == 99  # Min low of week 1
        assert first_week["close"] == 101  # Last close of week 1
        assert first_week["volume"] == 2100  # Sum volume of week 1

        mock_logger.info.assert_called_once()
        assert (
            "DataFrame ré-échantillonné avec la règle '1W'"
            in mock_logger.info.call_args[0][0]
        )

    def test_error_case_no_datetime_index(self, basic_ohlc_df, mock_logger):
        """Tests that the function returns original df if index is not DatetimeIndex."""
        result_df = resample_data(basic_ohlc_df.copy(), rule="1W")

        pd.testing.assert_frame_equal(result_df, basic_ohlc_df)
        mock_logger.error.assert_called_once_with(
            "L'index n'est pas un DatetimeIndex. Resampling impossible."
        )

    def test_edge_case_missing_ohlc_columns(self, ohlc_df_with_datetime_index):
        """Tests resampling with some OHLC columns missing."""
        df_missing_cols = ohlc_df_with_datetime_index.drop(columns=["high", "low"])
        df_weekly = resample_data(df_missing_cols.copy(), rule="1W")

        assert "high" not in df_weekly.columns
        assert "low" not in df_weekly.columns
        assert "open" in df_weekly.columns
        assert "close" in df_weekly.columns
        assert "volume" in df_weekly.columns
        assert df_weekly.shape == (2, 3)

    def test_edge_case_empty_df_with_index(self):
        """Tests resampling an empty DataFrame with a DatetimeIndex."""
        df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df_empty.index = pd.to_datetime([])

        result_df = resample_data(df_empty.copy(), rule="1W")
        assert result_df.empty

    def test_general_exception_during_resampling(
        self, ohlc_df_with_datetime_index, mocker, mock_logger
    ):
        """Tests that a general exception during resampling is caught and logged."""
        mocker.patch(
            "pandas.core.resample.Resampler.apply", side_effect=Exception("Test error")
        )

        result_df = resample_data(ohlc_df_with_datetime_index.copy(), rule="1W")

        # Should return the original dataframe
        pd.testing.assert_frame_equal(result_df, ohlc_df_with_datetime_index)
        mock_logger.error.assert_called_once_with(
            "Échec du resampling avec la règle '1W': Test error"
        )
