# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create src/schemas/tools_schemas.py with Pydantic models for inputs and outputs of five research tools', Date: 2026-09-11
"""
Pydantic schemas for Phase 1 Financial Research Tools.

Defines strict type definitions, input validation models, and output structures
for price data, financial news, volatility, LLM sentiment, and web search tools.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Market Price Data Schemas
# ---------------------------------------------------------------------------

class OHLCVRecord(BaseModel):
    """Single trading session OHLCV data record with indicators."""

    date: str = Field(description="ISO format date string (YYYY-MM-DD)")
    open: float = Field(description="Opening price")
    high: float = Field(description="High price during trading session")
    low: float = Field(description="Low price during trading session")
    close: float = Field(description="Closing price")
    volume: int = Field(description="Trading volume")
    daily_return: Optional[float] = Field(default=None, description="Daily percentage price change")
    sma20: Optional[float] = Field(default=None, description="20-day Simple Moving Average")
    sma50: Optional[float] = Field(default=None, description="50-day Simple Moving Average")
    ema20: Optional[float] = Field(default=None, description="20-day Exponential Moving Average")
    rsi14: Optional[float] = Field(default=None, description="14-day Relative Strength Index")


class TechnicalIndicators(BaseModel):
    """Latest calculated technical indicators snapshot."""

    latest_close: float = Field(description="Most recent closing price")
    latest_sma20: Optional[float] = Field(default=None, description="Latest 20-day SMA")
    latest_sma50: Optional[float] = Field(default=None, description="Latest 50-day SMA")
    latest_ema20: Optional[float] = Field(default=None, description="Latest 20-day EMA")
    latest_rsi14: Optional[float] = Field(default=None, description="Latest 14-day RSI")
    latest_daily_return: Optional[float] = Field(default=None, description="Latest daily return percentage")


class PriceDataOutput(BaseModel):
    """Standardized output container for get_price_data tool."""

    status: Literal["success", "error"] = Field(description="Operation status")
    ticker: str = Field(description="Ticker symbol requested")
    period: str = Field(description="Historical lookback period")
    records_count: int = Field(default=0, description="Total OHLCV records returned")
    latest_indicators: Optional[TechnicalIndicators] = Field(default=None, description="Snapshot of latest indicators")
    data: List[OHLCVRecord] = Field(default_factory=list, description="Historical OHLCV data points")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


# ---------------------------------------------------------------------------
# 2. Financial News Schemas
# ---------------------------------------------------------------------------

class NewsItem(BaseModel):
    """Structured financial news item."""

    title: str = Field(description="News headline title")
    publisher: str = Field(description="Publishing news organization or source")
    published_date: str = Field(description="Publication timestamp or formatted date")
    url: str = Field(description="Direct URL link to full article")
    summary: str = Field(description="Summary or excerpt of news content")


class NewsOutput(BaseModel):
    """Standardized output container for get_news tool."""

    status: Literal["success", "error"] = Field(description="Operation status")
    ticker: str = Field(description="Ticker symbol requested")
    count: int = Field(default=0, description="Total news articles retrieved")
    news: List[NewsItem] = Field(default_factory=list, description="Retrieved news items")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


# ---------------------------------------------------------------------------
# 3. Volatility Schemas
# ---------------------------------------------------------------------------

class VolatilityOutput(BaseModel):
    """Standardized output container for calculate_volatility tool."""

    status: Literal["success", "error"] = Field(description="Operation status")
    ticker: str = Field(description="Ticker symbol requested")
    window: int = Field(description="Historical trading window length in days")
    annualized_volatility: Optional[float] = Field(default=None, description="Annualized historical volatility percentage")
    daily_volatility: Optional[float] = Field(default=None, description="Daily return standard deviation")
    trading_days_used: int = Field(default=0, description="Number of trading days used in calculation")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


# ---------------------------------------------------------------------------
# 4. LLM Sentiment Schemas
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    """Structured LLM output for news sentiment analysis."""

    score: float = Field(
        description="Sentiment score ranging from -1.0 (extremely bearish) to +1.0 (extremely bullish)"
    )
    label: Literal["Bullish", "Bearish", "Neutral"] = Field(
        description="Qualitative sentiment classification label"
    )
    confidence: float = Field(
        description="Model confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Concise rationale explaining the sentiment classification"
    )


class SentimentOutput(BaseModel):
    """Standardized output container for llm_sentiment tool."""

    status: Literal["success", "error", "fallback"] = Field(description="Operation status")
    headlines_count: int = Field(default=0, description="Total headlines analyzed")
    sentiment: SentimentResult = Field(description="Aggregated sentiment breakdown")
    error: Optional[str] = Field(default=None, description="Error message if operation failed or fallback used")


# ---------------------------------------------------------------------------
# 5. Web Search Schemas
# ---------------------------------------------------------------------------

class SearchResultItem(BaseModel):
    """Normalized web search result item."""

    title: str = Field(description="Search result page title")
    snippet: str = Field(description="Text snippet or abstract from page")
    url: str = Field(description="Target web page URL")


class WebSearchOutput(BaseModel):
    """Standardized output container for web_search tool."""

    status: Literal["success", "error"] = Field(description="Operation status")
    query: str = Field(description="Search query executed")
    count: int = Field(default=0, description="Number of search results retrieved")
    results: List[SearchResultItem] = Field(default_factory=list, description="List of search result items")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
