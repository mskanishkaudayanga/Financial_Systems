# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Convert calculate_volatility into a LangChain @tool with CalculateVolatilityArgs Pydantic input schema and explicit LLM tool description', Date: 2026-09-11
"""
Volatility Calculation Tool.

Computes the annualized historical volatility of an equity ticker based on daily returns
over a specified trading window using standard deviation scaled by sqrt(252).
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

from src.schemas.tools_schemas import VolatilityOutput, CalculateVolatilityArgs


@tool(args_schema=CalculateVolatilityArgs)
def calculate_volatility(ticker: str, window: int = 252) -> Dict[str, Any]:
    """Calculate the annualized historical volatility and daily return standard deviation for an equity ticker over a specified trading window. Use this tool when evaluating stock risk, return variance, or price volatility."""
    # 1. Validate inputs
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        return VolatilityOutput(
            status="error",
            ticker=str(ticker),
            window=int(window) if isinstance(window, int) else 252,
            error="Ticker symbol must be a non-empty string."
        ).model_dump()

    cleaned_ticker = ticker.strip().upper()

    if not isinstance(window, int) or window < 5:
        return VolatilityOutput(
            status="error",
            ticker=cleaned_ticker,
            window=int(window) if isinstance(window, int) else 0,
            error="Window must be an integer of at least 5 trading days."
        ).model_dump()

    # 2. Map window length to yfinance period query
    if window <= 30:
        period = "3mo"
    elif window <= 126:
        period = "6mo"
    elif window <= 252:
        period = "1y"
    elif window <= 504:
        period = "2y"
    else:
        period = "max"

    # 3. Retrieve historical price data directly (independent tool execution)
    try:
        ticker_obj = yf.Ticker(cleaned_ticker)
        df = ticker_obj.history(period=period, auto_adjust=False)
    except Exception as exc:
        return VolatilityOutput(
            status="error",
            ticker=cleaned_ticker,
            window=window,
            error=f"Failed to fetch market data for volatility calculation: {str(exc)}"
        ).model_dump()

    if df is None or df.empty or "Close" not in df.columns:
        return VolatilityOutput(
            status="error",
            ticker=cleaned_ticker,
            window=window,
            error=f"No price history returned for ticker '{cleaned_ticker}'."
        ).model_dump()

    # Clean close prices
    close_prices = df["Close"].copy().dropna()
    if len(close_prices) < 5:
        return VolatilityOutput(
            status="error",
            ticker=cleaned_ticker,
            window=window,
            trading_days_used=len(close_prices),
            error=f"Insufficient price history ({len(close_prices)} points) to calculate volatility."
        ).model_dump()

    # 4. Calculate daily returns and standard deviation
    daily_returns = close_prices.pct_change().dropna()
    # Trim to requested window length
    recent_returns = daily_returns.tail(window)

    if len(recent_returns) < 5:
        return VolatilityOutput(
            status="error",
            ticker=cleaned_ticker,
            window=window,
            trading_days_used=len(recent_returns),
            error=f"Insufficient daily return records ({len(recent_returns)}) after cleaning."
        ).model_dump()

    daily_std = float(recent_returns.std())
    annualized_vol = float(daily_std * np.sqrt(252))

    return VolatilityOutput(
        status="success",
        ticker=cleaned_ticker,
        window=window,
        annualized_volatility=round(annualized_vol, 6),
        daily_volatility=round(daily_std, 6),
        trading_days_used=len(recent_returns)
    ).model_dump()
