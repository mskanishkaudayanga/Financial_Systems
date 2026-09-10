"""
Financial summary generation and deterministic technical momentum signal classification.

Calculates key equity research metrics (YTD Return, 52-week High/Low, P/E Ratio)
and classifies technical momentum deterministically without LLM reliance.
"""

from datetime import date
from typing import Any, Dict, Optional
import pandas as pd
import yfinance as yf


def calculate_momentum_signal(row: pd.Series) -> str:
    """
    Classify equity momentum deterministically using technical indicator signals.

    Rules & Scoring System:
        1. Moving Average Alignment:
           - Close > SMA_50: +1 | Close < SMA_50: -1
           - Close > SMA_200: +1 | Close < SMA_200: -1
           - SMA_50 > SMA_200 (Golden Cross): +1 | SMA_50 < SMA_200 (Death Cross): -1
        2. RSI (14) Momentum:
           - RSI > 50: +1 | RSI < 50: -1
           - RSI > 70 (Overbought): -1 penalty | RSI < 30 (Oversold): -1 penalty
        3. MACD Histogram:
           - MACD_Hist > 0 (Bullish Momentum): +1
           - MACD_Hist < 0 (Bearish Momentum): -1

    Classification Thresholds:
        Score >= +3 : "STRONG_BULLISH"
        Score in [+1, +2]: "BULLISH"
        Score == 0  : "NEUTRAL"
        Score in [-2, -1]: "BEARISH"
        Score <= -3 : "STRONG_BEARISH"

    Args:
        row: Latest data row containing prices and indicator values.

    Returns:
        str: Deterministic momentum signal string.
    """
    score = 0

    close = row.get("Close")
    sma_50 = row.get("SMA_50")
    sma_200 = row.get("SMA_200")
    rsi_14 = row.get("RSI_14")
    macd_hist = row.get("MACD_Hist")

    # 1. SMA Trend Rules
    if pd.notna(close) and pd.notna(sma_50):
        score += 1 if close > sma_50 else -1

    if pd.notna(close) and pd.notna(sma_200):
        score += 1 if close > sma_200 else -1

    if pd.notna(sma_50) and pd.notna(sma_200):
        score += 1 if sma_50 > sma_200 else -1

    # 2. RSI Rules
    if pd.notna(rsi_14):
        if rsi_14 > 50.0:
            score += 1
        else:
            score -= 1

        if rsi_14 > 70.0 or rsi_14 < 30.0:
            score -= 1  # Extreme boundaries indicate overbought/oversold exhaustion

    # 3. MACD Histogram Rules
    if pd.notna(macd_hist):
        score += 1 if macd_hist > 0.0 else -1

    # Final Classification Mapping
    if score >= 3:
        return "STRONG_BULLISH"
    elif 1 <= score <= 2:
        return "BULLISH"
    elif score == 0:
        return "NEUTRAL"
    elif -2 <= score <= -1:
        return "BEARISH"
    else:
        return "STRONG_BEARISH"


def fetch_pe_ratio(ticker: str) -> Optional[float]:
    """
    Fetch Price-to-Earnings (P/E) ratio safely from API.

    Returns None if P/E ratio is unavailable, unlisted, or invalid.

    Args:
        ticker: Target stock ticker.

    Returns:
        Optional[float]: Trailing P/E ratio or None.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if isinstance(info, dict):
            pe = info.get("trailingPE") or info.get("forwardPE")
            if pe is not None and isinstance(pe, (int, float)) and pe > 0:
                return round(float(pe), 2)
    except Exception:
        pass
    return None


def generate_financial_summary(
    ticker: str,
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate financial summary dictionary including 52-week bounds, YTD return,
    P/E ratio, and deterministic momentum classification.

    Args:
        ticker: Equity ticker symbol.
        df: Processed DataFrame containing OHLCV prices and technical indicators.

    Returns:
        Dict[str, Any]: Financial summary dictionary.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame must be non-empty.")

    if "Close" not in df.columns or "High" not in df.columns or "Low" not in df.columns:
        raise ValueError("DataFrame is missing required price columns ('Close', 'High', 'Low').")

    latest_row = df.iloc[-1]
    current_price = float(latest_row["Close"])

    # Compute 52-Week (252 trading days) High and Low dynamically
    window_52w = min(252, len(df))
    df_52w = df.tail(window_52w)
    high_52w = float(df_52w["High"].max())
    low_52w = float(df_52w["Low"].min())

    # Calculate YTD Return dynamically from first trading day of current year
    current_year = date.today().year
    ytd_df = df[df.index.year == current_year]

    if not ytd_df.empty:
        ytd_start_price = float(ytd_df["Close"].iloc[0])
    else:
        # Fallback to first available price in dataset if date range doesn't cross current year
        ytd_start_price = float(df["Close"].iloc[0])

    ytd_return = round(((current_price - ytd_start_price) / ytd_start_price) * 100.0, 2)

    # Fetch P/E ratio without data fabrication
    pe_ratio = fetch_pe_ratio(ticker)

    # Calculate deterministic momentum signal
    momentum = calculate_momentum_signal(latest_row)

    return {
        "ticker": ticker.upper(),
        "current_price": round(current_price, 2),
        "52_week_high": round(high_52w, 2),
        "52_week_low": round(low_52w, 2),
        "pe_ratio": pe_ratio,
        "ytd_return": ytd_return,
        "momentum_signal": momentum,
    }
