# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Re-export all five financial research tools in src/tools/__init__.py', Date: 2026-09-11
"""
Tools Package.

Provides five independent financial research tools:
1. get_price_data: Retrieve OHLCV and compute SMA20, SMA50, EMA20, RSI14, daily_return.
2. get_news: Retrieve recent financial news articles with structured metadata.
3. calculate_volatility: Compute annualized historical volatility (\sigma * \sqrt{252}).
4. llm_sentiment: Execute qualitative headline sentiment analysis via structured LLM output.
5. web_search: Search web for analyst commentary and equity intelligence via DuckDuckGo.
"""

from src.tools.market_data import get_price_data
from src.tools.news import get_news
from src.tools.volatility import calculate_volatility
from src.tools.sentiment import llm_sentiment
from src.tools.search import web_search

__all__ = [
    "get_price_data",
    "get_news",
    "calculate_volatility",
    "llm_sentiment",
    "web_search",
]
