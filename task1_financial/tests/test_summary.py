"""
Unit test suite for financial summary generation and deterministic momentum signal.
"""

from datetime import date
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.features.summary import (
    calculate_momentum_signal,
    fetch_pe_ratio,
    generate_financial_summary,
)


@pytest.fixture
def sample_indicators_df() -> pd.DataFrame:
    """Fixture providing a synthetic DataFrame with prices and calculated technical indicators."""
    current_year = date.today().year
    dates = pd.date_range(start=f"{current_year - 1}-01-01", periods=300, freq="D")

    # Generate synthetic price series rising from 100 to 200
    prices = pd.Series(range(100, 400), index=dates[:300], dtype=float)

    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.05,
            "Low": prices * 0.95,
            "Close": prices,
            "Volume": 100000,
            "SMA_50": prices * 0.95,
            "SMA_200": prices * 0.85,
            "RSI_14": 65.0,
            "MACD_Hist": 1.5,
        },
        index=dates,
    )
    return df


def test_generate_financial_summary_fields_and_math(sample_indicators_df):
    """Test financial summary dictionary contains all required fields and correct math."""
    with patch("src.features.summary.fetch_pe_ratio", return_value=25.5):
        summary = generate_financial_summary("AAPL", sample_indicators_df)

    assert summary["ticker"] == "AAPL"
    assert "current_price" in summary
    assert "52_week_high" in summary
    assert "52_week_low" in summary
    assert summary["pe_ratio"] == 25.5
    assert "ytd_return" in summary
    assert "momentum_signal" in summary

    # Verify 52-week high is maximum high of last 252 rows
    expected_high = sample_indicators_df["High"].tail(252).max()
    assert summary["52_week_high"] == round(expected_high, 2)


def test_unavailable_pe_ratio_handled_gracefully(sample_indicators_df):
    """Test that missing or unlisted P/E ratio returns None without fabricating values."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {}  # Empty info dictionary

    with patch("src.features.summary.yf.Ticker", return_value=mock_ticker_instance):
        pe = fetch_pe_ratio("UNKNOWN_TICKER")
        assert pe is None

        summary = generate_financial_summary("UNKNOWN_TICKER", sample_indicators_df)
        assert summary["pe_ratio"] is None


def test_ytd_return_calculation():
    """Test dynamic YTD return calculation from start of current year."""
    current_year = date.today().year
    dates = pd.date_range(start=f"{current_year}-01-01", periods=10, freq="D")
    prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 120.0]

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": 1000,
        },
        index=dates,
    )

    with patch("src.features.summary.fetch_pe_ratio", return_value=None):
        summary = generate_financial_summary("TEST", df)

    # YTD return: (120 - 100) / 100 * 100 = 20.0%
    assert summary["ytd_return"] == 20.0


def test_momentum_signal_rules():
    """Test deterministic momentum signal classification across strong bullish, neutral, and bearish rules."""
    # 1. Strong Bullish row
    strong_bullish_row = pd.Series(
        {
            "Close": 150.0,
            "SMA_50": 140.0,
            "SMA_200": 120.0,
            "RSI_14": 60.0,
            "MACD_Hist": 2.0,
        }
    )
    assert calculate_momentum_signal(strong_bullish_row) == "STRONG_BULLISH"

    # 2. Strong Bearish row
    strong_bearish_row = pd.Series(
        {
            "Close": 100.0,
            "SMA_50": 120.0,
            "SMA_200": 140.0,
            "RSI_14": 40.0,
            "MACD_Hist": -2.0,
        }
    )
    assert calculate_momentum_signal(strong_bearish_row) == "STRONG_BEARISH"


def test_summary_invalid_dataframe():
    """Test invalid input DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="must be non-empty"):
        generate_financial_summary("AAPL", pd.DataFrame())
