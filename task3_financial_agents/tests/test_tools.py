# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create pytest unit tests for five Phase 1 tools verifying schemas, calculations, input validation, and defensive failure handling', Date: 2026-09-11
"""
Unit test suite for Phase 1 Financial Research Tools.
"""

import pytest
from src.tools import (
    get_price_data,
    get_news,
    calculate_volatility,
    llm_sentiment,
    web_search,
)


def test_get_price_data_valid_ticker():
    """Verify get_price_data returns valid OHLCV records and technical indicators."""
    res = get_price_data("AAPL", period="1mo")
    assert res["status"] == "success"
    assert res["ticker"] == "AAPL"
    assert res["records_count"] > 0
    assert len(res["data"]) > 0

    latest = res["latest_indicators"]
    assert latest is not None
    assert "latest_close" in latest
    assert "latest_rsi14" in latest
    assert "latest_sma20" in latest
    assert "latest_ema20" in latest


def test_get_price_data_invalid_inputs():
    """Verify get_price_data handles invalid ticker or period gracefully."""
    res1 = get_price_data("", period="1y")
    assert res1["status"] == "error"
    assert "error" in res1

    res2 = get_price_data("INVALID_TICKER_XYZ9999", period="1y")
    assert res2["status"] == "error"

    res3 = get_price_data("AAPL", period="invalid_period")
    assert res3["status"] == "error"


def test_get_news_valid():
    """Verify get_news returns structured news items for valid ticker."""
    res = get_news("AAPL", n=5)
    assert res["status"] == "success"
    assert res["ticker"] == "AAPL"
    assert "count" in res
    assert isinstance(res["news"], list)


def test_get_news_invalid():
    """Verify get_news handles invalid ticker or count limit."""
    res1 = get_news("", n=5)
    assert res1["status"] == "error"

    res2 = get_news("AAPL", n=-1)
    assert res2["status"] == "error"


def test_calculate_volatility_valid():
    """Verify calculate_volatility returns positive annualized volatility."""
    res = calculate_volatility("AAPL", window=60)
    assert res["status"] == "success"
    assert res["ticker"] == "AAPL"
    assert res["annualized_volatility"] is not None
    assert res["annualized_volatility"] > 0.0
    assert res["trading_days_used"] > 0


def test_calculate_volatility_invalid():
    """Verify calculate_volatility handles invalid parameters."""
    res1 = calculate_volatility("", window=252)
    assert res1["status"] == "error"

    res2 = calculate_volatility("AAPL", window=2)
    assert res2["status"] == "error"


def test_llm_sentiment_validation_and_fallback():
    """Verify llm_sentiment handles invalid inputs or fallback safely."""
    # Test empty list
    res1 = llm_sentiment([])
    assert res1["status"] == "error"

    # Test fallback behavior when API key is missing or dummy
    res2 = llm_sentiment(["Company X reports record earnings", "Stock rises 5%"])
    assert res2["status"] in ("success", "fallback")
    assert "sentiment" in res2
    assert "score" in res2["sentiment"]
    assert "label" in res2["sentiment"]


def test_web_search():
    """Verify web_search retrieves results or handles exceptions gracefully."""
    res = web_search("Apple stock financial performance", max_results=3)
    assert res["status"] in ("success", "error")
    assert "count" in res
    assert isinstance(res["results"], list)
