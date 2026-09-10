"""
Market data ingestion module using yfinance.

Provides robust, dynamic daily OHLCV data retrieval with custom exception handling,
schema validation, and missing value imputation.
"""

import collections
import collections.abc

# Python 3.10+ compatibility shim for legacy dependencies (e.g., outdated frozendict)
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping

from datetime import date, timedelta
from typing import List, Optional
import pandas as pd
import yfinance as yf

from src.config import config


class MarketDataError(Exception):
    """Base exception for market data operations."""
    pass


class InvalidTickerError(MarketDataError):
    """Raised when a ticker symbol is empty, invalid, or yields no data."""
    pass


class DataValidationError(MarketDataError):
    """Raised when market data fails schema or structural validation."""
    pass


REQUIRED_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Volume"]


def fetch_market_data(
    ticker: str,
    years: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch daily OHLCV market data for a given ticker over a dynamic date range.

    Args:
        ticker: Equity ticker symbol (e.g., 'AAPL', 'MSFT').
        years: Number of lookback years. Defaults to config.LOOKBACK_YEARS.

    Returns:
        pd.DataFrame: Cleaned pandas DataFrame indexed by DatetimeIndex containing
                      ['Open', 'High', 'Low', 'Close', 'Volume'].

    Raises:
        ValueError: If ticker parameter is invalid.
        InvalidTickerError: If the ticker yields empty or non-existent data.
        DataValidationError: If required columns are missing or data cannot be validated.
        MarketDataError: For general yfinance API failures or network errors.
    """
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker symbol must be a non-empty string.")

    cleaned_ticker = ticker.strip().upper()
    lookback_years = years if years is not None else config.LOOKBACK_YEARS

    if lookback_years <= 0:
        raise ValueError("Lookback years must be a positive integer.")

    # Calculate dynamic date range relative to today
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * lookback_years + 5)

    try:
        ticker_obj = yf.Ticker(cleaned_ticker)
        df = ticker_obj.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=False
        )
    except Exception as exc:
        raise MarketDataError(
            f"Failed to fetch market data for ticker '{cleaned_ticker}' due to API error: {exc}"
        ) from exc

    if df is None or df.empty:
        raise InvalidTickerError(
            f"No market data returned for ticker '{cleaned_ticker}'. Verify ticker symbol."
        )

    # Handle yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Validate required OHLCV columns exist
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Downloaded data for '{cleaned_ticker}' is missing required columns: {missing_cols}"
        )

    # Filter required OHLCV columns
    df = df[REQUIRED_COLUMNS].copy()

    # Impute missing values (forward-fill then back-fill for leading NaNs)
    df = df.ffill().bfill()

    # Drop any remaining unfillable NaN rows if whole dataset was empty
    if df.isna().any().any():
        df = df.dropna()

    if df.empty:
        raise DataValidationError(
            f"Market data for '{cleaned_ticker}' contains no valid rows after cleaning."
        )

    # Ensure index is datetime sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df
