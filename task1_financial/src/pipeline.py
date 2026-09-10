"""
Main execution pipeline for the Financial AI Equity Research system.

High-level orchestrator connecting data retrieval, feature engineering, financial summary,
news sentiment extraction, and LLM trading recommendation reasoning.
Contains zero indicator math, zero raw prompt strings, and zero raw API requests.
"""

import logging
from typing import Any, Dict, Optional
import pandas as pd

from src.config import config
from src.data.market_data import fetch_market_data
from src.data.news_data import NewsDataError, fetch_news_data
from src.features.summary import generate_financial_summary
from src.features.technical_indicators import calculate_indicators
from src.llm.client import LLMClient
from src.llm.sentiment import analyze_batch_sentiment
from src.llm.signal import generate_trading_signal
from src.schemas.models import (
    AggregatedSentiment,
    FinancialPipelineResult,
    TradingRecommendation,
)

logger = logging.getLogger(__name__)


def extract_latest_indicators(df_indicators: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Extract latest row of technical indicator values into a clean float dictionary.

    Args:
        df_indicators: DataFrame with indicator columns.

    Returns:
        Dict[str, Optional[float]]: Key-value pairs of indicator names and latest values.
    """
    if df_indicators.empty:
        return {}

    latest_row = df_indicators.iloc[-1]
    indicator_cols = [
        "Close", "SMA_50", "SMA_200", "RSI_14",
        "MACD_Line", "MACD_Signal", "MACD_Hist",
        "BB_Middle", "BB_Upper", "BB_Lower"
    ]

    result: Dict[str, Optional[float]] = {}
    for col in indicator_cols:
        if col in latest_row:
            val = latest_row[col]
            result[col] = round(float(val), 4) if pd.notna(val) else None
        else:
            result[col] = None

    return result


def run_pipeline(
    ticker: str = config.DEFAULT_TICKER,
    years: int = config.LOOKBACK_YEARS,
    news_limit: int = config.NEWS_HEADLINES_LIMIT,
    client: Optional[LLMClient] = None,
) -> FinancialPipelineResult:
    """
    Execute the complete end-to-end Financial AI Equity Research pipeline.

    Orchestration Flow:
        1. Fetch daily OHLCV market data (Phase 1).
        2. Calculate technical indicators (Phase 2).
        3. Generate financial summary & fetch news headlines (Phase 3).
        4. Run LLM news sentiment analysis & aggregation (Phase 4).
        5. Synthesize LLM trading recommendation signal (Phase 5).
        6. Validate & assemble final FinancialPipelineResult.

    Args:
        ticker: Target equity ticker symbol.
        years: Market data lookback period in years.
        news_limit: Maximum headlines to retrieve.
        client: Optional LLMClient instance.

    Returns:
        FinancialPipelineResult: Validated end-to-end pipeline result object.
    """
    clean_ticker = ticker.strip().upper()
    print("=" * 80)
    print(f"🚀 Starting Financial AI Pipeline Execution for '{clean_ticker}'...")

    # 1. Market Data Retrieval (Phase 1)
    print(f"\n[1/5] Fetching Market Data ({years} years lookback)...")
    df_market = fetch_market_data(ticker=clean_ticker, years=years)
    print(f"      Fetched {len(df_market)} daily OHLCV bars ({df_market.index.min().strftime('%Y-%m-%d')} to {df_market.index.max().strftime('%Y-%m-%d')}).")

    # 2. Technical Indicator Calculation (Phase 2)
    print("\n[2/5] Calculating Technical Indicators (SMA, RSI, MACD, Bollinger Bands)...")
    df_indicators = calculate_indicators(df_market)
    print(f"      Calculated indicators. Data shape: {df_indicators.shape}.")

    # 3. Financial Summary & News Retrieval (Phase 3)
    print("\n[3/5] Generating Financial Summary & Retrieving News...")
    summary = generate_financial_summary(ticker=clean_ticker, df=df_indicators)

    news_list = []
    try:
        news_list = fetch_news_data(ticker=clean_ticker, limit=news_limit)
        print(f"      Retrieved {len(news_list)} normalized news headlines.")
    except NewsDataError as news_err:
        logger.warning(f"News retrieval failed for '{clean_ticker}': {news_err}. Proceeding with empty news list.")
        print(f"      News retrieval warning: {news_err}. Proceeding without news.")

    # 4. LLM News Sentiment Analysis (Phase 4)
    print("\n[4/5] Running LLM News Sentiment Analysis & Aggregation...")
    sentiment_result: AggregatedSentiment = analyze_batch_sentiment(
        headlines=news_list,
        ticker=clean_ticker,
        client=client
    )
    print(f"      Overall Sentiment: {sentiment_result.overall_label} (Score: {sentiment_result.weighted_sentiment_score:.4f})")

    # 5. LLM Trading Recommendation (Phase 5)
    print("\n[5/5] Synthesizing LLM Trading Recommendation & Reasoning...")
    latest_row = df_indicators.iloc[-1]
    recommendation: TradingRecommendation = generate_trading_signal(
        ticker=clean_ticker,
        summary=summary,
        latest_indicators=latest_row,
        news_sentiment=sentiment_result,
        client=client
    )
    print(f"      Recommendation: {recommendation.recommendation}")
    print(f"      Reasoning: {recommendation.reasoning}")

    # Extract latest indicators dictionary
    latest_indicators_dict = extract_latest_indicators(df_indicators)

    # Assemble and validate final Pydantic result model
    result = FinancialPipelineResult(
        ticker=clean_ticker,
        market_summary=summary,
        latest_technical_indicators=latest_indicators_dict,
        news_sentiment=sentiment_result,
        recommendation=recommendation
    )

    print("\n✅ Pipeline execution completed successfully!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    run_pipeline()
