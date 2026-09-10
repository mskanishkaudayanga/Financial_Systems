"""
Integration test suite for the complete end-to-end Financial AI pipeline.

Uses offline mocks for market data, news retrieval, and LLM communication to test
pipeline orchestration, error propagation, and structured output production.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.data.market_data import InvalidTickerError, MarketDataError
from src.data.news_data import NewsDataError
from src.llm.client import LLMClientError
from src.pipeline import run_pipeline
from src.schemas.models import FinancialPipelineResult


@pytest.fixture
def mock_ohlcv_df():
    """Fixture providing a 250-row synthetic market data DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=250, freq="D")
    prices = [100.0 + i * 0.5 for i in range(250)]

    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 2.0 for p in prices],
            "Low": [p - 2.0 for p in prices],
            "Close": prices,
            "Volume": 100000,
        },
        index=dates,
    )


@pytest.fixture
def mock_news_list():
    """Fixture providing mock normalized news list."""
    return [
        {
            "headline": "Apple Reports Strong Quarterly Profits",
            "source": "Reuters",
            "published_at": "2024-01-15 12:00:00 UTC",
            "url": "https://reuters.com/aapl-1",
        },
        {
            "headline": "Apple Expands Data Center Infrastructure",
            "source": "Bloomberg",
            "published_at": "2024-01-16 14:00:00 UTC",
            "url": "https://bloomberg.com/aapl-2",
        },
    ]


def test_full_pipeline_success(mock_ohlcv_df, mock_news_list):
    """Test successful end-to-end pipeline execution returning FinancialPipelineResult."""
    mock_llm_client = MagicMock()
    # Mock sentiment batch call return value
    mock_llm_client.generate.side_effect = [
        # Batch sentiment LLM response
        """
        [
          {"headline": "Apple Reports Strong Quarterly Profits", "sentiment": "positive", "confidence": 0.9, "brief_reason": "Profits up."},
          {"headline": "Apple Expands Data Center Infrastructure", "sentiment": "positive", "confidence": 0.8, "brief_reason": "Expansion."}
        ]
        """,
        # Recommendation LLM response
        """
        {
          "recommendation": "BUY",
          "reasoning": "Price is above SMA 50 and 200 showing strong upward trend. RSI at 60 indicates bullish momentum without overbought risk. MACD histogram remains positive. Combining these metrics justifies a BUY recommendation."
        }
        """,
    ]

    with patch("src.pipeline.fetch_market_data", return_value=mock_ohlcv_df), \
         patch("src.pipeline.fetch_news_data", return_value=mock_news_list), \
         patch("src.features.summary.fetch_pe_ratio", return_value=30.0):

        result = run_pipeline("AAPL", years=2, news_limit=10, client=mock_llm_client)

    assert isinstance(result, FinancialPipelineResult)
    assert result.ticker == "AAPL"
    assert result.market_summary["current_price"] == 224.5
    assert result.latest_technical_indicators["Close"] == 224.5
    assert "SMA_50" in result.latest_technical_indicators
    assert result.news_sentiment.overall_label == "POSITIVE"
    assert result.recommendation.recommendation == "BUY"


def test_pipeline_news_failure_graceful(mock_ohlcv_df):
    """Test that news retrieval failure logs warning and completes pipeline gracefully."""
    mock_llm_client = MagicMock()
    mock_llm_client.generate.return_value = """
    {
      "recommendation": "HOLD",
      "reasoning": "Price is hovering near technical moving averages. RSI indicates neutral momentum. MACD histogram is flat. Current setup favors a HOLD recommendation."
    }
    """

    with patch("src.pipeline.fetch_market_data", return_value=mock_ohlcv_df), \
         patch("src.pipeline.fetch_news_data", side_effect=NewsDataError("News API Down")), \
         patch("src.features.summary.fetch_pe_ratio", return_value=None):

        result = run_pipeline("AAPL", client=mock_llm_client)

    assert isinstance(result, FinancialPipelineResult)
    assert result.news_sentiment.total_headlines == 0
    assert result.news_sentiment.overall_label == "NEUTRAL"
    assert result.recommendation.recommendation == "HOLD"


def test_pipeline_llm_failure_fallback(mock_ohlcv_df, mock_news_list):
    """Test that LLM client failure triggers fallback mechanisms for sentiment and recommendation."""
    mock_llm_client = MagicMock()
    mock_llm_client.generate.side_effect = LLMClientError("LLM Provider Timeout")

    with patch("src.pipeline.fetch_market_data", return_value=mock_ohlcv_df), \
         patch("src.pipeline.fetch_news_data", return_value=mock_news_list), \
         patch("src.features.summary.fetch_pe_ratio", return_value=None):

        result = run_pipeline("AAPL", client=mock_llm_client)

    assert isinstance(result, FinancialPipelineResult)
    # Sentiment fallback: neutral 0.0 confidence
    assert result.news_sentiment.weighted_sentiment_score == 0.0
    # Recommendation fallback: deterministic momentum recommendation
    assert result.recommendation.recommendation in ("BUY", "HOLD", "SELL")
    assert "fallback" in result.recommendation.reasoning.lower()


def test_pipeline_market_data_failure():
    """Test that critical market data failure raises MarketDataError without attempting pipeline execution."""
    with patch("src.pipeline.fetch_market_data", side_effect=InvalidTickerError("Invalid Ticker Symbol")):
        with pytest.raises(MarketDataError):
            run_pipeline("INVALID_SYMBOL")
