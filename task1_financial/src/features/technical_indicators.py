"""
Quantitative technical indicator computation module.

Implements SMA (50 & 200), RSI-14 (Wilder's method), MACD (12, 26, 9),
and Bollinger Bands (20, 2) from mathematical first principles using Pandas and NumPy.
TA-Lib is explicitly avoided to maintain zero external C-library dependencies.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def calculate_sma(series: pd.Series, window: int = 50) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA) over a specified rolling window.

    Mathematical Definition:
        SMA_N(t) = (1 / N) * sum_{i=0}^{N-1} P(t - i)

    Args:
        series: Price series (e.g., Close prices).
        window: Rolling window length (e.g., 50 or 200).

    Returns:
        pd.Series: SMA values with initial (window - 1) NaNs preserved.
    """
    if window <= 0:
        raise ValueError("SMA window length must be a positive integer.")
    return series.rolling(window=window).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) using Wilder's smoothing method.

    Mathematical Definition:
        Delta_t = P_t - P_{t-1}
        Gain_t = max(Delta_t, 0), Loss_t = max(-Delta_t, 0)
        AvgGain_t = (AvgGain_{t-1} * (N - 1) + Gain_t) / N   (Wilder's EMA: alpha = 1 / N)
        AvgLoss_t = (AvgLoss_{t-1} * (N - 1) + Loss_t) / N
        RSI_t = 100 * (AvgGain_t / (AvgGain_t + AvgLoss_t))

    Handles zero-loss cases safely without division-by-zero errors.

    Args:
        series: Price series (e.g., Close prices).
        period: Lookback period for RSI (default: 14).

    Returns:
        pd.Series: RSI values bounded between 0.0 and 100.0.
    """
    if period <= 0:
        raise ValueError("RSI period must be a positive integer.")

    delta = series.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Apply Wilder's Exponential Smoothing (alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Calculate RSI safely using sum formulation: RSI = 100 * (AG / (AG + AL))
    total_change = avg_gain + avg_loss

    # Use np.where to handle zero change (e.g., constant series -> RSI = 50)
    rsi_values = np.where(
        total_change == 0.0,
        50.0,
        100.0 * (avg_gain / total_change)
    )

    rsi_series = pd.Series(rsi_values, index=series.index, name=f"RSI_{period}")

    # Preserve initial NaN for period 0 where diff() is NaN
    rsi_series.iloc[0] = np.nan

    return rsi_series


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Mathematical Definition:
        MACD_Line = EMA_{fast}(Price) - EMA_{slow}(Price)
        MACD_Signal = EMA_{signal}(MACD_Line)
        MACD_Hist = MACD_Line - MACD_Signal

    Args:
        series: Price series (e.g., Close prices).
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal EMA period (default: 9).

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (MACD_Line, MACD_Signal, MACD_Hist)
    """
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("MACD periods must be positive integers.")
    if fast >= slow:
        raise ValueError("Fast EMA period must be strictly less than slow EMA period.")

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return macd_line, macd_signal, macd_hist


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands over a specified rolling window.

    Mathematical Definition:
        BB_Middle = SMA_N(Price)
        BB_Upper = BB_Middle + (K * StdDev_N(Price))
        BB_Lower = BB_Middle - (K * StdDev_N(Price))

    Args:
        series: Price series (e.g., Close prices).
        window: Rolling window length (default: 20).
        std_dev: Standard deviation multiplier K (default: 2.0).

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (BB_Middle, BB_Upper, BB_Lower)
    """
    if window <= 0:
        raise ValueError("Bollinger Bands window length must be a positive integer.")
    if std_dev <= 0:
        raise ValueError("Standard deviation multiplier must be positive.")

    middle_band = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std(ddof=1)

    upper_band = middle_band + (std_dev * rolling_std)
    lower_band = middle_band - (std_dev * rolling_std)

    return middle_band, upper_band, lower_band


def calculate_indicators(
    df: pd.DataFrame,
    sma_short: int = 50,
    sma_long: int = 200,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Calculate all five technical indicators and append them to a copy of the input DataFrame.

    Indicators Added:
        - SMA_50, SMA_200
        - RSI_14
        - MACD_Line, MACD_Signal, MACD_Hist
        - BB_Middle, BB_Upper, BB_Lower

    Args:
        df: Input DataFrame containing at least a 'Close' price column.
        sma_short: Short SMA window (default: 50).
        sma_long: Long SMA window (default: 200).
        rsi_period: RSI period (default: 14).
        macd_fast: Fast EMA period for MACD (default: 12).
        macd_slow: Slow EMA period for MACD (default: 26).
        macd_signal: Signal EMA period for MACD (default: 9).
        bb_window: Bollinger Bands window (default: 20).
        bb_std_dev: Bollinger Bands standard deviation multiplier (default: 2.0).

    Returns:
        pd.DataFrame: Copy of input DataFrame augmented with indicator columns.

    Raises:
        ValueError: If input is invalid or missing required 'Close' column.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")

    if "Close" not in df.columns:
        raise ValueError("Input DataFrame is missing required 'Close' column.")

    # Operate on a copy to avoid mutating original DataFrame
    df_out = df.copy()

    close_series = df_out["Close"]

    # Calculate SMAs
    df_out[f"SMA_{sma_short}"] = calculate_sma(close_series, window=sma_short)
    df_out[f"SMA_{sma_long}"] = calculate_sma(close_series, window=sma_long)

    # Calculate RSI
    df_out[f"RSI_{rsi_period}"] = calculate_rsi(close_series, period=rsi_period)

    # Calculate MACD
    macd_line, macd_signal_line, macd_hist = calculate_macd(
        close_series, fast=macd_fast, slow=macd_slow, signal=macd_signal
    )
    df_out["MACD_Line"] = macd_line
    df_out["MACD_Signal"] = macd_signal_line
    df_out["MACD_Hist"] = macd_hist

    # Calculate Bollinger Bands
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(
        close_series, window=bb_window, std_dev=bb_std_dev
    )
    df_out["BB_Middle"] = bb_mid
    df_out["BB_Upper"] = bb_upper
    df_out["BB_Lower"] = bb_lower

    return df_out
