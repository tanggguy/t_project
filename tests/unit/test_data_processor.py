# c:/Users/saill/Desktop/t_project/tests/unit/test_data_processor.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock


# Mock the logger before importing the module under test
@pytest.fixture(autouse=True)
def mock_logger(mocker):
    """Fixture to mock the logger used in the data_processor module."""
    mock_log = MagicMock()
    mocker.patch("utils.data_processor.logger", mock_log)
    return mock_log


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
