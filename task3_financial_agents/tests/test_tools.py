# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update pytest suite for five LangChain @tool objects testing .name, .description, .args_schema, and .invoke()', Date: 2026-09-11
"""
Unit test suite for Phase 2 LangChain Financial Research Tools.
"""

import pytest
from src.tools import (
    get_price_data,
    get_news,
    calculate_volatility,
    llm_sentiment,
    web_search,
)


def test_tool_metadata():
    """Verify all five tools expose required LangChain tool metadata attributes."""
    tools = [get_price_data, get_news, calculate_volatility, llm_sentiment, web_search]
    expected_names = ["get_price_data", "get_news", "calculate_volatility", "llm_sentiment", "web_search"]

    for tool_obj, name in zip(tools, expected_names):
        assert hasattr(tool_obj, "name")
        assert tool_obj.name == name
        assert hasattr(tool_obj, "description")
        assert len(tool_obj.description) > 20
        assert hasattr(tool_obj, "args_schema")
        assert tool_obj.args_schema is not None


def test_get_price_data_langchain_invoke():
    """Verify get_price_data tool invocation via .invoke()."""
    res = get_price_data.invoke({"ticker": "AAPL", "period": "1mo"})
    assert res["status"] == "success"
    assert res["ticker"] == "AAPL"
    assert res["records_count"] > 0
    assert "latest_indicators" in res


def test_get_price_data_invalid_inputs():
    """Verify get_price_data tool handles invalid inputs gracefully."""
    res1 = get_price_data.invoke({"ticker": "", "period": "1y"})
    assert res1["status"] == "error"

    res2 = get_price_data.invoke({"ticker": "AAPL", "period": "invalid_period"})
    assert res2["status"] == "error"


def test_get_news_langchain_invoke():
    """Verify get_news tool invocation via .invoke()."""
    res = get_news.invoke({"ticker": "AAPL", "n": 5})
    assert res["status"] == "success"
    assert res["ticker"] == "AAPL"
    assert "count" in res


def test_calculate_volatility_langchain_invoke():
    """Verify calculate_volatility tool invocation via .invoke()."""
    res = calculate_volatility.invoke({"ticker": "AAPL", "window": 60})
    assert res["status"] == "success"
    assert res["annualized_volatility"] > 0.0


def test_llm_sentiment_langchain_invoke():
    """Verify llm_sentiment tool invocation via .invoke()."""
    res = llm_sentiment.invoke({"headlines": ["Apple reports quarterly revenue growth", "Analysts upgrade AAPL target"]})
    assert res["status"] in ("success", "fallback")
    assert "sentiment" in res


def test_web_search_langchain_invoke():
    """Verify web_search tool invocation via .invoke()."""
    res = web_search.invoke({"query": "Apple financial performance", "max_results": 3})
    assert res["status"] in ("success", "error")
    assert "count" in res
