"""
Unit test suite for LLM news sentiment analysis, Pydantic schema validation, and weighted aggregation.
"""

from unittest.mock import MagicMock
from pydantic import ValidationError
import pytest

from src.llm.client import LLMClientError
from src.llm.sentiment import (
    analyze_batch_sentiment,
    analyze_headline_sentiment,
    calculate_weighted_sentiment_score,
    extract_json_payload,
)
from src.schemas.models import AggregatedSentiment, HeadlineSentiment


def test_headline_sentiment_pydantic_validation_success():
    """Test valid HeadlineSentiment model instantiation."""
    valid_data = {
        "headline": "Apple Reports Quarterly Profit Surge",
        "sentiment": "positive",
        "confidence": 0.95,
        "brief_reason": "Quarterly earnings exceeded Wall Street expectations.",
    }
    model = HeadlineSentiment.model_validate(valid_data)
    assert model.sentiment == "positive"
    assert model.confidence == 0.95


def test_headline_sentiment_pydantic_validation_out_of_bounds_confidence():
    """Test that confidence outside [0.0, 1.0] raises ValidationError."""
    invalid_high = {
        "headline": "Test Headline",
        "sentiment": "positive",
        "confidence": 1.5,  # Invalid: > 1.0
        "brief_reason": "Good news",
    }
    with pytest.raises(ValidationError):
        HeadlineSentiment.model_validate(invalid_high)

    invalid_low = {
        "headline": "Test Headline",
        "sentiment": "negative",
        "confidence": -0.2,  # Invalid: < 0.0
        "brief_reason": "Bad news",
    }
    with pytest.raises(ValidationError):
        HeadlineSentiment.model_validate(invalid_low)


def test_headline_sentiment_pydantic_validation_invalid_sentiment():
    """Test that invalid sentiment string raises ValidationError."""
    invalid_label = {
        "headline": "Test Headline",
        "sentiment": "bullish_hype",  # Invalid enum value
        "confidence": 0.8,
        "brief_reason": "Reason",
    }
    with pytest.raises(ValidationError):
        HeadlineSentiment.model_validate(invalid_label)


def test_extract_json_payload_markdown_stripping():
    """Test extracting clean JSON from markdown code fences for both objects and arrays."""
    raw_markdown = """Here is the result:
```json
[
  {
    "headline": "H1",
    "sentiment": "positive",
    "confidence": 0.9,
    "brief_reason": "Revenue grew 10%"
  }
]
```
Thank you!"""

    cleaned = extract_json_payload(raw_markdown)
    assert cleaned.startswith("[") and cleaned.endswith("]")
    assert '"sentiment": "positive"' in cleaned


def test_analyze_headline_sentiment_success():
    """Test headline analysis with mocked LLMClient returning valid payload."""
    mock_client = MagicMock()
    mock_client.generate.return_value = '{"sentiment": "positive", "confidence": 0.85, "brief_reason": "Strong sales."}'

    res = analyze_headline_sentiment("Apple Sales Up", "AAPL", client=mock_client)
    assert isinstance(res, HeadlineSentiment)
    assert res.sentiment == "positive"
    assert res.confidence == 0.85
    assert res.brief_reason == "Strong sales."


def test_analyze_headline_sentiment_fallback_on_api_error():
    """Test that API failures return a neutral fallback model with 0.0 confidence without crashing."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = LLMClientError("Network Timeout")

    res = analyze_headline_sentiment("Apple Sales Up", "AAPL", client=mock_client)
    assert res.sentiment == "neutral"
    assert res.confidence == 0.0
    assert "processing error" in res.brief_reason.lower()


def test_calculate_weighted_sentiment_score_math():
    """Test confidence-weighted aggregate sentiment score calculation."""
    results = [
        HeadlineSentiment(headline="H1", sentiment="positive", confidence=0.8, brief_reason="R1"),
        HeadlineSentiment(headline="H2", sentiment="positive", confidence=1.0, brief_reason="R2"),
        HeadlineSentiment(headline="H3", sentiment="negative", confidence=0.2, brief_reason="R3"),
    ]

    # Weighted score = (1.0*0.8 + 1.0*1.0 + (-1.0)*0.2) / (0.8 + 1.0 + 0.2)
    #                = (0.8 + 1.0 - 0.2) / 2.0 = 1.6 / 2.0 = 0.8
    score = calculate_weighted_sentiment_score(results)
    assert score == 0.8


def test_analyze_batch_sentiment_single_llm_call():
    """Test that analyze_batch_sentiment executes in a SINGLE LLM API call for all headlines."""
    mock_client = MagicMock()
    mock_client.generate.return_value = """
    [
      {"headline": "Apple Launches Product A", "sentiment": "positive", "confidence": 0.9, "brief_reason": "Good launch"},
      {"headline": "Apple Opens New Store", "sentiment": "positive", "confidence": 0.8, "brief_reason": "Expansion"}
    ]
    """

    headlines = [
        {"headline": "Apple Launches Product A"},
        {"headline": "Apple Opens New Store"},
    ]

    aggregated = analyze_batch_sentiment(headlines, "AAPL", client=mock_client)

    # Verify ONLY 1 LLM call was executed for the batch
    mock_client.generate.assert_called_once()

    assert isinstance(aggregated, AggregatedSentiment)
    assert aggregated.total_headlines == 2
    assert aggregated.positive_count == 2
    assert aggregated.weighted_sentiment_score == 0.8529 or round(aggregated.weighted_sentiment_score, 2) == 0.85
    assert aggregated.overall_label == "POSITIVE"
