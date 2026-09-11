# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Implement get_price_data tool using yfinance with SMA20, SMA50, EMA20, RSI14, daily return calculations and defensive error handling', Date: 2026-09-11
"""
Market Price Data Tool.

Retrieves historical OHLCV data for an equity ticker using yfinance,
calculates key technical indicators (SMA20, SMA50, EMA20, RSI14, daily return),
and returns structured output with defensive validation.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import yfinance as yf

from src.schemas.tools_schemas import PriceDataOutput, OHLCVRecord, TechnicalIndicators

VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) using Wilder's exponential smoothing.

    Args:
        series: Price pandas Series (e.g. Close price).
        period: RSI window period (default 14).

    Returns:
        pd.Series: Calculated RSI values between 0.0 and 100.0.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def get_price_data(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Retrieve historical OHLCV market data and calculate technical indicators.

    Args:
        ticker: Equity symbol (e.g., 'AAPL', 'MSFT', 'NVDA').
        period: Historical lookback period (default '1y').
                Valid options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'.

    Returns:
        Dict[str, Any]: Serialized dictionary conforming to PriceDataOutput schema.
    """
    # 1. Validate inputs
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        return PriceDataOutput(
            status="error",
            ticker=str(ticker),
            period=str(period),
            error="Ticker symbol must be a non-empty string."
        ).model_dump()

    cleaned_ticker = ticker.strip().upper()
    cleaned_period = period.strip().lower() if isinstance(period, str) else "1y"

    if cleaned_period not in VALID_PERIODS:
        return PriceDataOutput(
            status="error",
            ticker=cleaned_ticker,
            period=cleaned_period,
            error=f"Invalid lookback period '{cleaned_period}'. Must be one of {sorted(list(VALID_PERIODS))}."
        ).model_dump()

    # 2. Fetch data via yfinance
    try:
        ticker_obj = yf.Ticker(cleaned_ticker)
        df = ticker_obj.history(period=cleaned_period, auto_adjust=False)
    except Exception as exc:
        return PriceDataOutput(
            status="error",
            ticker=cleaned_ticker,
            period=cleaned_period,
            error=f"Failed to fetch market data from yfinance: {str(exc)}"
        ).model_dump()

    if df is None or df.empty:
        return PriceDataOutput(
            status="error",
            ticker=cleaned_ticker,
            period=cleaned_period,
            error=f"No price data found for ticker symbol '{cleaned_ticker}'."
        ).model_dump()

    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return PriceDataOutput(
            status="error",
            ticker=cleaned_ticker,
            period=cleaned_period,
            error=f"Downloaded market data is missing required OHLCV columns: {missing_cols}"
        ).model_dump()

    df = df[required_cols].copy().ffill().bfill()
    df = df.dropna()

    if df.empty:
        return PriceDataOutput(
            status="error",
            ticker=cleaned_ticker,
            period=cleaned_period,
            error=f"No valid price rows remaining after cleaning for ticker '{cleaned_ticker}'."
        ).model_dump()

    # 3. Calculate Technical Indicators
    df["daily_return"] = df["Close"].pct_change()
    df["sma20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["sma50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["rsi14"] = _calculate_rsi(df["Close"], period=14)

    # 4. Construct response payload
    records: List[OHLCVRecord] = []
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        records.append(
            OHLCVRecord(
                date=date_str,
                open=round(float(row["Open"]), 2),
                high=round(float(row["High"]), 2),
                low=round(float(row["Low"]), 2),
                close=round(float(row["Close"]), 2),
                volume=int(row["Volume"]),
                daily_return=round(float(row["daily_return"]), 4) if pd.notna(row["daily_return"]) else None,
                sma20=round(float(row["sma20"]), 2) if pd.notna(row["sma20"]) else None,
                sma50=round(float(row["sma50"]), 2) if pd.notna(row["sma50"]) else None,
                ema20=round(float(row["ema20"]), 2) if pd.notna(row["ema20"]) else None,
                rsi14=round(float(row["rsi14"]), 2) if pd.notna(row["rsi14"]) else None,
            )
        )

    latest_row = df.iloc[-1]
    latest_indicators = TechnicalIndicators(
        latest_close=round(float(latest_row["Close"]), 2),
        latest_sma20=round(float(latest_row["sma20"]), 2) if pd.notna(latest_row["sma20"]) else None,
        latest_sma50=round(float(latest_row["sma50"]), 2) if pd.notna(latest_row["sma50"]) else None,
        latest_ema20=round(float(latest_row["ema20"]), 2) if pd.notna(latest_row["ema20"]) else None,
        latest_rsi14=round(float(latest_row["rsi14"]), 2) if pd.notna(latest_row["rsi14"]) else None,
        latest_daily_return=round(float(latest_row["daily_return"]), 4) if pd.notna(latest_row["daily_return"]) else None,
    )

    output = PriceDataOutput(
        status="success",
        ticker=cleaned_ticker,
        period=cleaned_period,
        records_count=len(records),
        latest_indicators=latest_indicators,
        data=records,
    )

    return output.model_dump()
