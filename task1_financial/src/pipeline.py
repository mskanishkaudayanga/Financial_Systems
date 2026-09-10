"""
Main execution pipeline for financial analysis and signal generation.

Orchestrates market data fetching (Phase 1) and technical indicator calculations (Phase 2).
"""

from src.config import config
from src.data.market_data import fetch_market_data
from src.features.technical_indicators import calculate_indicators


def run_pipeline(ticker: str = config.DEFAULT_TICKER, years: int = config.LOOKBACK_YEARS):
    """
    Run Phase 1 & Phase 2 pipeline: Fetch market data and compute technical indicators.

    Args:
        ticker: Target stock ticker symbol.
        years: Lookback years.

    Returns:
        pd.DataFrame: Processed market data with computed technical indicators.
    """
    print("=" * 70)
    print(f"1. Fetching Market Data for Ticker: '{ticker}' ({years} years lookback)...")
    df_market = fetch_market_data(ticker=ticker, years=years)
    print(f"   Success! Fetched {len(df_market)} daily OHLCV bars.")
    print(f"   Date Range: {df_market.index.min().strftime('%Y-%m-%d')} to {df_market.index.max().strftime('%Y-%m-%d')}")

    print("\n2. Computing Technical Indicators (SMA-50, SMA-200, RSI-14, MACD, Bollinger Bands)...")
    df_indicators = calculate_indicators(df_market)
    print(f"   Success! Final DataFrame shape: {df_indicators.shape}")
    print(f"   Columns Added: {[c for c in df_indicators.columns if c not in df_market.columns]}")

    print("\n3. Sample Output (Latest 5 Trading Days):")
    print("-" * 70)
    display_cols = ["Close", "SMA_50", "SMA_200", "RSI_14", "MACD_Hist", "BB_Middle", "BB_Upper", "BB_Lower"]
    print(df_indicators[display_cols].tail(5).to_string())
    print("=" * 70)

    return df_indicators


if __name__ == "__main__":
    run_pipeline()
