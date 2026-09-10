"""
Main execution pipeline for financial analysis and signal generation.

Orchestrates market data fetching (Phase 1), technical indicator calculations (Phase 2),
financial summary generation & news retrieval (Phase 3), LLM sentiment analysis (Phase 4),
and LLM trading recommendation reasoning (Phase 5).
"""

from typing import Any, Dict, Tuple
import pandas as pd

from src.config import config
from src.data.market_data import fetch_market_data
from src.data.news_data import fetch_news_data
from src.features.summary import generate_financial_summary
from src.features.technical_indicators import calculate_indicators
from src.llm.sentiment import analyze_batch_sentiment
from src.llm.signal import generate_trading_signal
from src.schemas.models import AggregatedSentiment, TradingRecommendation


def run_pipeline(
    ticker: str = config.DEFAULT_TICKER,
    years: int = config.LOOKBACK_YEARS,
    news_limit: int = config.NEWS_HEADLINES_LIMIT,
) -> Tuple[pd.DataFrame, Dict[str, Any], list, AggregatedSentiment, TradingRecommendation]:
    """
    Run full end-to-end Task 1 pipeline (Phase 1 through Phase 5).

    Args:
        ticker: Target stock ticker symbol.
        years: Market data lookback years.
        news_limit: Headline count limit.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], list, AggregatedSentiment, TradingRecommendation]:
            (Market DataFrame, Summary Dict, News List, Aggregated Sentiment, Trading Recommendation)
    """
    print("=" * 80)
    print(f"1. Fetching Market Data for Ticker: '{ticker}' ({years} years lookback)...")
    df_market = fetch_market_data(ticker=ticker, years=years)
    print(f"   Success! Fetched {len(df_market)} daily OHLCV bars.")

    print("\n2. Computing Technical Indicators (SMA, RSI, MACD, Bollinger Bands)...")
    df_indicators = calculate_indicators(df_market)
    print(f"   Success! Final DataFrame shape: {df_indicators.shape}")

    print("\n3. Generating Financial Summary & Deterministic Momentum Signal...")
    summary = generate_financial_summary(ticker=ticker, df=df_indicators)
    for key, val in summary.items():
        print(f"   - {key}: {val}")

    print(f"\n4. Retrieving & Normalizing News Headlines (Target: {news_limit})...")
    news_list = fetch_news_data(ticker=ticker, limit=news_limit)
    print(f"   Success! Retrieved {len(news_list)} normalized headlines.")

    print("\n5. Running LLM News Sentiment Analysis & Confidence-Weighted Aggregation...")
    sentiment_result = analyze_batch_sentiment(news_list, ticker=ticker)
    print(f"   - Overall Sentiment Label: {sentiment_result.overall_label}")
    print(f"   - Weighted Sentiment Score: {sentiment_result.weighted_sentiment_score:.4f}")

    print("\n6. Synthesizing LLM Trading Recommendation & Evidence-Based Reasoning...")
    latest_row = df_indicators.iloc[-1]
    recommendation = generate_trading_signal(
        ticker=ticker,
        summary=summary,
        latest_indicators=latest_row,
        news_sentiment=sentiment_result
    )
    print(f"   - RECOMMENDATION: {recommendation.recommendation}")
    print(f"   - REASONING: {recommendation.reasoning}")
    print("=" * 80)

    return df_indicators, summary, news_list, sentiment_result, recommendation


if __name__ == "__main__":
    run_pipeline()
