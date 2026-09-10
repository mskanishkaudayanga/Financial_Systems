"""
Trading signal reasoning module using LLM over pre-calculated Python indicators.

Synthesizes combinations of technical metrics, financial summaries, and news sentiment
into validated Buy/Hold/Sell recommendations.
"""

import json
import logging
from typing import Any, Dict, Optional
import pandas as pd

from src.config import config
from src.llm.client import LLMClient, LLMClientError
from src.llm.sentiment import extract_json_payload
from src.prompts.recommendation import RECOMMENDATION_SYSTEM_PROMPT, RECOMMENDATION_USER_PROMPT_TEMPLATE
from src.schemas.models import AggregatedSentiment, RecommendationType, TradingRecommendation

logger = logging.getLogger(__name__)


def create_fallback_trading_recommendation(
    ticker: str,
    summary: Dict[str, Any],
    latest_indicators: pd.Series
) -> TradingRecommendation:
    """
    Construct a deterministic fallback TradingRecommendation when LLM reasoning fails.

    Args:
        ticker: Equity ticker.
        summary: Financial summary dictionary.
        latest_indicators: Series containing technical indicator values.

    Returns:
        TradingRecommendation: Validated fallback recommendation.
    """
    momentum = summary.get("momentum_signal", "NEUTRAL")

    if momentum in ("STRONG_BULLISH", "BULLISH"):
        rec: RecommendationType = "BUY"
    elif momentum in ("STRONG_BEARISH", "BEARISH"):
        rec = "SELL"
    else:
        rec = "HOLD"

    close = latest_indicators.get("Close", "N/A")
    sma_50 = latest_indicators.get("SMA_50", "N/A")
    rsi = latest_indicators.get("RSI_14", "N/A")

    close_str = f"${float(close):.2f}" if isinstance(close, (int, float)) else str(close)
    sma_str = f"${float(sma_50):.2f}" if isinstance(sma_50, (int, float)) else str(sma_50)
    rsi_str = f"{float(rsi):.2f}" if isinstance(rsi, (int, float)) else str(rsi)

    reasoning = (
        f"Automated deterministic fallback recommendation for {ticker.upper()} is set to {rec} based on technical alignment. "
        f"The current price of {close_str} is evaluated against a 50-day moving average of {sma_str} and an RSI reading of {rsi_str}. "
        f"Pre-calculated technical momentum for the asset is currently categorized as {momentum}. "
        f"This fallback ensures continuous system operation while LLM reasoning is temporarily unavailable."
    )

    return TradingRecommendation(
        recommendation=rec,
        reasoning=reasoning
    )


def generate_trading_signal(
    ticker: str,
    summary: Dict[str, Any],
    latest_indicators: pd.Series,
    news_sentiment: Optional[AggregatedSentiment] = None,
    client: Optional[LLMClient] = None
) -> TradingRecommendation:
    """
    Generate an evidence-based trading recommendation by providing pre-calculated indicators to the LLM.

    Args:
        ticker: Target equity ticker symbol.
        summary: Financial summary dictionary.
        latest_indicators: Pandas Series containing the latest row of technical indicators.
        news_sentiment: Optional AggregatedSentiment model from Phase 4.
        client: Optional LLMClient instance.

    Returns:
        TradingRecommendation: Validated recommendation model containing action and reasoning.
    """
    if not summary or not isinstance(summary, dict):
        raise ValueError("Summary must be a non-empty dictionary.")

    llm_client = client if client is not None else LLMClient(temperature=0.1)

    # Format sentiment inputs
    sentiment_label = news_sentiment.overall_label if news_sentiment else "NEUTRAL"
    sentiment_score = news_sentiment.weighted_sentiment_score if news_sentiment else 0.0

    # Format user prompt template with pre-calculated Python values
    user_prompt = RECOMMENDATION_USER_PROMPT_TEMPLATE.format(
        ticker=ticker.upper(),
        current_price=summary.get("current_price", "N/A"),
        high_52w=summary.get("52_week_high", "N/A"),
        low_52w=summary.get("52_week_low", "N/A"),
        ytd_return=summary.get("ytd_return", "N/A"),
        pe_ratio=summary.get("pe_ratio") if summary.get("pe_ratio") is not None else "N/A",
        sma_50=latest_indicators.get("SMA_50", "N/A"),
        sma_200=latest_indicators.get("SMA_200", "N/A"),
        rsi_14=latest_indicators.get("RSI_14", "N/A"),
        macd_line=latest_indicators.get("MACD_Line", "N/A"),
        macd_signal=latest_indicators.get("MACD_Signal", "N/A"),
        macd_hist=latest_indicators.get("MACD_Hist", "N/A"),
        bb_middle=latest_indicators.get("BB_Middle", "N/A"),
        bb_upper=latest_indicators.get("BB_Upper", "N/A"),
        bb_lower=latest_indicators.get("BB_Lower", "N/A"),
        news_sentiment_label=sentiment_label,
        news_sentiment_score=sentiment_score,
    )

    try:
        raw_response = llm_client.generate(
            prompt=user_prompt,
            system_prompt=RECOMMENDATION_SYSTEM_PROMPT,
            temperature=0.1
        )

        json_str = extract_json_payload(raw_response)
        parsed_dict = json.loads(json_str)

        if not isinstance(parsed_dict, dict):
            raise ValueError("Parsed LLM output payload is not a dictionary.")

        validated = TradingRecommendation.model_validate(parsed_dict)
        return validated

    except (LLMClientError, json.JSONDecodeError, ValueError, Exception) as exc:
        logger.warning(f"LLM trading recommendation failed for {ticker}: {exc}. Employing fallback.")
        return create_fallback_trading_recommendation(
            ticker=ticker,
            summary=summary,
            latest_indicators=latest_indicators
        )
