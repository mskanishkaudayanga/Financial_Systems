"""
Unit test suite for the market data ingestion module.

Mocks external yfinance API calls to ensure test suite executes offline,
reliably, and deterministically.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.data.market_data import (
    DataValidationError,
    InvalidTickerError,
    MarketDataError,
    REQUIRED_COLUMNS,
    fetch_market_data,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Fixture providing a standard valid 5-row OHLCV DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    data = {
        "Open": [150.0, 151.0, 152.0, 153.0, 154.0],
        "High": [155.0, 156.0, 157.0, 158.0, 159.0],
        "Low": [149.0, 150.0, 151.0, 152.0, 153.0],
        "Close": [152.0, 153.0, 154.0, 155.0, 156.0],
        "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
    }
    return pd.DataFrame(data, index=dates)


def test_fetch_market_data_success(sample_ohlcv_df):
    """Test successful market data fetch returning structured OHLCV DataFrame."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_ohlcv_df

    with patch("src.data.market_data.yf.Ticker", return_value=mock_ticker_instance):
        df = fetch_market_data("AAPL", years=2)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 5
    assert not df.isna().any().any()


def test_fetch_market_data_empty_returns_invalid_ticker_error():
    """Test that an empty DataFrame response triggers InvalidTickerError."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame()

    with patch("src.data.market_data.yf.Ticker", return_value=mock_ticker_instance):
        with pytest.raises(InvalidTickerError, match="No market data returned"):
            fetch_market_data("INVALID_TICKER")


def test_fetch_market_data_missing_columns(sample_ohlcv_df):
    """Test that missing required columns triggers DataValidationError."""
    incomplete_df = sample_ohlcv_df.drop(columns=["Close"])

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = incomplete_df

    with patch("src.data.market_data.yf.Ticker", return_value=mock_ticker_instance):
        with pytest.raises(DataValidationError, match="missing required columns"):
            fetch_market_data("AAPL")


def test_fetch_market_data_nan_imputation(sample_ohlcv_df):
    """Test that missing values (NaNs) inside time series are properly imputed."""
    df_with_nans = sample_ohlcv_df.copy()
    df_with_nans.iloc[2, df_with_nans.columns.get_loc("Close")] = np.nan

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = df_with_nans

    with patch("src.data.market_data.yf.Ticker", return_value=mock_ticker_instance):
        df = fetch_market_data("AAPL")

    assert not df.isna().any().any()
    # Forward fill check: row 2 Close should equal row 1 Close (153.0)
    assert df.iloc[2]["Close"] == 153.0


def test_fetch_market_data_api_failure():
    """Test that underlying API/network exceptions are wrapped in MarketDataError."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.side_effect = Exception("Connection Timeout")

    with patch("src.data.market_data.yf.Ticker", return_value=mock_ticker_instance):
        with pytest.raises(MarketDataError, match="Failed to fetch market data"):
            fetch_market_data("AAPL")


def test_invalid_input_parameters():
    """Test parameter validation for ticker string and lookback years."""
    with pytest.raises(ValueError, match="non-empty string"):
        fetch_market_data("")

    with pytest.raises(ValueError, match="non-empty string"):
        fetch_market_data(None)  # type: ignore

    with pytest.raises(ValueError, match="positive integer"):
        fetch_market_data("AAPL", years=0)
