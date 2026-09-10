"""
Unit test suite for LLM trading signal generation and Pydantic validation.
"""

from unittest.mock import MagicMock
import pandas as pd
import pytest
from pydantic import ValidationError

from src.llm.client import LLMClientError
from src.llm.signal import create_fallback_trading_recommendation, generate_trading_signal
from src.prompts.recommendation import RECOMMENDATION_SYSTEM_PROMPT
from src.schemas.models import AggregatedSentiment, TradingRecommendation


def test_trading_recommendation_pydantic_validation_success():
    """Test valid TradingRecommendation model instantiation."""
    valid_data = {
        "recommendation": "BUY",
        "reasoning": (
            "The price is trading firmly above both the 50-day and 200-day moving averages, confirming strong bullish trend alignment. "
            "RSI at 62 indicates healthy buying momentum without reaching overbought territory. "
            "Additionally, the positive MACD histogram expansion signals continued upward price acceleration. "
            "Combining these technical metrics supports a BUY recommendation."
        ),
    }
    model = TradingRecommendation.model_validate(valid_data)
    assert model.recommendation == "BUY"
    assert len(model.reasoning) > 20


def test_trading_recommendation_invalid_action():
    """Test that invalid recommendation string raises ValidationError."""
    invalid_data = {
        "recommendation": "STRONG_BUY",  # Invalid enum value
        "reasoning": "Sentence 1. Sentence 2. Sentence 3. Sentence 4.",
    }
    with pytest.raises(ValidationError):
        TradingRecommendation.model_validate(invalid_data)


def test_trading_recommendation_invalid_sentence_count():
    """Test that reasoning with fewer than 3 sentences raises ValidationError."""
    too_short = {
        "recommendation": "HOLD",
        "reasoning": "Sentence 1 only.",  # Only 1 sentence
    }
    with pytest.raises(ValidationError):
        TradingRecommendation.model_validate(too_short)


def test_generate_trading_signal_success():
    """Test trading signal generation with mocked LLMClient."""
    mock_client = MagicMock()
    mock_client.generate.return_value = """
    {
      "recommendation": "BUY",
      "reasoning": "Price is above SMA 50 and 200 indicating an upward trend. RSI is 58 showing positive momentum. MACD histogram is positive supporting bullish continuation. Overall analysis warrants a BUY recommendation."
    }
    """

    summary = {
        "current_price": 240.0,
        "52_week_high": 250.0,
        "52_week_low": 180.0,
        "ytd_return": 15.0,
        "pe_ratio": 30.0,
        "momentum_signal": "BULLISH",
    }

    latest_indicators = pd.Series(
        {
            "Close": 240.0,
            "SMA_50": 230.0,
            "SMA_200": 210.0,
            "RSI_14": 58.0,
            "MACD_Line": 2.5,
            "MACD_Signal": 1.5,
            "MACD_Hist": 1.0,
            "BB_Middle": 235.0,
            "BB_Upper": 245.0,
            "BB_Lower": 225.0,
        }
    )

    rec = generate_trading_signal("AAPL", summary, latest_indicators, client=mock_client)

    assert isinstance(rec, TradingRecommendation)
    assert rec.recommendation == "BUY"
    assert "SMA 50" in rec.reasoning or "RSI" in rec.reasoning or "above" in rec.reasoning


def test_generate_trading_signal_fallback_on_api_error():
    """Test that API errors trigger deterministic fallback without crashing."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = LLMClientError("API Connection Timeout")

    summary = {"momentum_signal": "BULLISH"}
    latest_indicators = pd.Series({"Close": 200.0, "SMA_50": 190.0, "RSI_14": 55.0})

    rec = generate_trading_signal("AAPL", summary, latest_indicators, client=mock_client)

    assert isinstance(rec, TradingRecommendation)
    assert rec.recommendation == "BUY"
    assert "fallback" in rec.reasoning.lower()


def test_system_prompt_contains_no_dynamic_values():
    """Verify system prompt contains instructions only, zero hardcoded stock values."""
    assert "{current_price}" not in RECOMMENDATION_SYSTEM_PROMPT
    assert "AAPL" not in RECOMMENDATION_SYSTEM_PROMPT
