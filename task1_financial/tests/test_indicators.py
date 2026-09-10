"""
Deterministic unit test suite for technical indicator calculation module.

Validates mathematical correctness, edge-case behavior, and structural integrity
using synthetic data without relying on external network/TA-Lib dependencies.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.technical_indicators import (
    calculate_bollinger_bands,
    calculate_indicators,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)


@pytest.fixture
def sample_price_df() -> pd.DataFrame:
    """Fixture providing a 250-row synthetic price DataFrame for testing long SMA windows."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=250, freq="D")
    base_price = 100.0
    returns = np.random.normal(0.001, 0.015, size=250)
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 100000,
        },
        index=dates,
    )


def test_sma_manual_calculation():
    """Verify Simple Moving Average against hand-computed rolling average values."""
    prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma3 = calculate_sma(prices, window=3)

    # Expected: [NaN, NaN, (10+20+30)/3=20, (20+30+40)/3=30, (30+40+50)/3=40]
    expected = [np.nan, np.nan, 20.0, 30.0, 40.0]

    np.testing.assert_allclose(sma3.values[2:], expected[2:], rtol=1e-5)
    assert np.isnan(sma3.iloc[0]) and np.isnan(sma3.iloc[1])


def test_rsi_monotonic_and_constant_trends():
    """Verify RSI behavior on strictly increasing, strictly decreasing, and constant series."""
    # 1. Strictly increasing series -> RSI should approach/equal 100 after period
    inc_prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    rsi_inc = calculate_rsi(inc_prices, period=3)
    np.testing.assert_allclose(rsi_inc.iloc[3:].values, 100.0)

    # 2. Strictly decreasing series -> RSI should equal 0 after period
    dec_prices = pd.Series([20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0])
    rsi_dec = calculate_rsi(dec_prices, period=3)
    np.testing.assert_allclose(rsi_dec.iloc[3:].values, 0.0)

    # 3. Constant price series -> zero change -> RSI = 50.0
    const_prices = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    rsi_const = calculate_rsi(const_prices, period=3)
    np.testing.assert_allclose(rsi_const.iloc[1:].values, 50.0)


def test_macd_structure_and_convergence(sample_price_df):
    """Verify MACD line, signal line, and histogram arithmetic relationships."""
    close = sample_price_df["Close"]
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)

    # Histogram must equal (MACD_Line - MACD_Signal)
    expected_hist = macd_line - macd_signal
    np.testing.assert_allclose(macd_hist.values, expected_hist.values, rtol=1e-5)

    # Check series length integrity
    assert len(macd_line) == len(close)
    assert len(macd_signal) == len(close)


def test_bollinger_bands_relationships(sample_price_df):
    """Verify Bollinger Bands upper/lower standard deviation offsets and band ordering."""
    close = sample_price_df["Close"]
    middle, upper, lower = calculate_bollinger_bands(close, window=20, std_dev=2.0)

    # Middle band should equal 20-period SMA
    expected_middle = close.rolling(window=20).mean()
    np.testing.assert_allclose(middle.dropna().values, expected_middle.dropna().values)

    # Upper band must be > Middle band, and Lower band must be < Middle band
    valid_mask = ~middle.isna()
    assert (upper[valid_mask] > middle[valid_mask]).all()
    assert (lower[valid_mask] < middle[valid_mask]).all()

    # Band width check: Upper - Middle == Middle - Lower == 2 * StdDev
    np.testing.assert_allclose(
        (upper[valid_mask] - middle[valid_mask]).values,
        (middle[valid_mask] - lower[valid_mask]).values,
        rtol=1e-5,
    )


def test_calculate_indicators_orchestrator(sample_price_df):
    """Verify that calculate_indicators appends all 9 expected indicator columns."""
    df_out = calculate_indicators(sample_price_df)

    expected_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_50", "SMA_200", "RSI_14",
        "MACD_Line", "MACD_Signal", "MACD_Hist",
        "BB_Middle", "BB_Upper", "BB_Lower"
    ]

    for col in expected_columns:
        assert col in df_out.columns

    # Verify input DataFrame was NOT mutated
    assert "SMA_50" not in sample_price_df.columns


def test_insufficient_data_handling():
    """Verify that small datasets populate expected NaNs without throwing exceptions."""
    small_df = pd.DataFrame({"Close": [10.0, 11.0, 12.0, 13.0, 14.0]})
    df_out = calculate_indicators(small_df, sma_short=50, sma_long=200)

    # SMA_50 and SMA_200 should be all NaNs for a 5-row dataset
    assert df_out["SMA_50"].isna().all()
    assert df_out["SMA_200"].isna().all()
    assert len(df_out) == 5


def test_missing_close_column_raises():
    """Verify that missing 'Close' column raises ValueError."""
    bad_df = pd.DataFrame({"Open": [10.0, 11.0], "High": [12.0, 13.0]})

    with pytest.raises(ValueError, match="missing required 'Close' column"):
        calculate_indicators(bad_df)


def test_invalid_input_raises():
    """Verify invalid DataFrame input parameter raises ValueError."""
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        calculate_indicators(pd.DataFrame())

    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        calculate_indicators(None)  # type: ignore
