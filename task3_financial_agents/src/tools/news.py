# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Optimize get_news payload size by truncating long summaries to 250 characters to stay within LLM TPM rate limits', Date: 2026-09-11
"""
Financial News Tool.

Retrieves recent market and equity news for a ticker symbol using yfinance,
normalizes news payloads across API version schemas, and returns structured data
optimized for LLM context token limits.
"""

from typing import Dict, Any, List
from datetime import datetime
import yfinance as yf
from langchain_core.tools import tool

from src.schemas.tools_schemas import NewsOutput, NewsItem, GetNewsArgs


@tool(args_schema=GetNewsArgs)
def get_news(ticker: str, n: int = 10) -> Dict[str, Any]:
    """Retrieve recent financial market news articles and headlines for a given equity ticker symbol. Use this tool when you need recent news coverage, corporate announcements, press releases, or qualitative news context for a company."""
    # 1. Validate inputs
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        return NewsOutput(
            status="error",
            ticker=str(ticker),
            count=0,
            error="Ticker symbol must be a non-empty string."
        ).model_dump()

    cleaned_ticker = ticker.strip().upper()

    if not isinstance(n, int) or n <= 0:
        return NewsOutput(
            status="error",
            ticker=cleaned_ticker,
            count=0,
            error="Article count limit 'n' must be a positive integer."
        ).model_dump()

    # Cap n at 10 to keep payload lightweight
    limit_n = min(n, 10)

    # 2. Retrieve news via yfinance
    try:
        ticker_obj = yf.Ticker(cleaned_ticker)
        raw_news = ticker_obj.news
    except Exception as exc:
        return NewsOutput(
            status="error",
            ticker=cleaned_ticker,
            count=0,
            error=f"Failed to fetch news from yfinance: {str(exc)}"
        ).model_dump()

    if not raw_news or not isinstance(raw_news, list):
        return NewsOutput(
            status="success",
            ticker=cleaned_ticker,
            count=0,
            news=[]
        ).model_dump()

    # 3. Normalize news items
    news_items: List[NewsItem] = []
    for item in raw_news[:limit_n]:
        if not isinstance(item, dict):
            continue

        content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}

        title = (
            item.get("title")
            or content.get("title")
            or "No Title Available"
        )

        publisher = (
            item.get("publisher")
            or content.get("provider", {}).get("displayName")
            or "Financial Market News"
        )

        pub_time = item.get("providerPublishTime") or content.get("pubDate")
        if isinstance(pub_time, (int, float)):
            published_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(pub_time, str):
            published_date = pub_time
        else:
            published_date = datetime.utcnow().strftime("%Y-%m-%d")

        url = (
            item.get("link")
            or content.get("canonicalUrl", {}).get("url")
            or content.get("clickThroughUrl", {}).get("url")
            or f"https://finance.yahoo.com/quote/{cleaned_ticker}"
        )

        raw_summary = (
            item.get("summary")
            or content.get("summary")
            or content.get("description")
            or title
        )
        # Truncate summary to 200 chars for token efficiency
        clean_summary = str(raw_summary).strip()
        if len(clean_summary) > 200:
            clean_summary = clean_summary[:197] + "..."

        news_items.append(
            NewsItem(
                title=str(title).strip(),
                publisher=str(publisher).strip(),
                published_date=str(published_date).strip(),
                url=str(url).strip(),
                summary=clean_summary
            )
        )

    return NewsOutput(
        status="success",
        ticker=cleaned_ticker,
        count=len(news_items),
        news=news_items
    ).model_dump()
