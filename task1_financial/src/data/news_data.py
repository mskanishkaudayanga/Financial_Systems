"""
Financial news retrieval and normalization module.

Retrieves and normalizes recent financial news headlines using yfinance.
Handles payload variations, deduplication, missing fields, and API failures gracefully.
"""

from datetime import datetime, timezone

import logging
from typing import Any, Dict, List, Optional
import warnings
import yfinance as yf

from src.config import config

logger = logging.getLogger(__name__)


class NewsDataError(Exception):
    """Base exception for news data operations."""
    pass


class InsufficientNewsWarning(UserWarning):
    """Warning issued when retrieved headlines count is below requested threshold."""
    pass


def normalize_news_item(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Normalize a raw yfinance news dictionary across different API payload formats.

    Args:
        item: Raw news dictionary payload.

    Returns:
        Optional[Dict[str, str]]: Normalized news item dict containing
                                  ['headline', 'source', 'published_at', 'url'],
                                  or None if essential fields are missing.
    """
    if not isinstance(item, dict):
        return None

    # Handle nested 'content' dictionary format present in newer yfinance versions
    content = item.get("content")
    if not isinstance(content, dict):
        content = item

    # Extract headline title
    headline = content.get("title") or item.get("title") or item.get("headline")
    if not headline or not isinstance(headline, str) or not headline.strip():
        return None

    # Extract publisher / source
    provider = content.get("provider")
    if isinstance(provider, dict):
        source = provider.get("displayName") or provider.get("name") or "Yahoo Finance"
    else:
        source = content.get("publisher") or item.get("publisher") or item.get("source") or "Yahoo Finance"

    # Extract article URL
    url = ""
    canonical_url = content.get("canonicalUrl")
    if isinstance(canonical_url, dict):
        url = canonical_url.get("url", "")
    if not url:
        url = content.get("link") or item.get("link") or content.get("url") or item.get("url") or ""

    # Extract and format publication timestamp
    published_at = ""
    pub_date = content.get("pubDate") or item.get("pubDate")
    if pub_date:
        published_at = str(pub_date)
    else:
        publish_time = content.get("providerPublishTime") or item.get("providerPublishTime")
        if publish_time:
            try:
                dt = datetime.fromtimestamp(int(publish_time), tz=timezone.utc)
                published_at = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                published_at = str(publish_time)

    return {
        "headline": headline.strip(),
        "source": str(source).strip(),
        "published_at": published_at,
        "url": str(url).strip(),
    }


def fetch_news_data(
    ticker: str,
    limit: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Retrieve and normalize recent financial news headlines for a ticker.

    Args:
        ticker: Target equity ticker symbol.
        limit: Desired headline count. Defaults to config.NEWS_HEADLINES_LIMIT (min 10).

    Returns:
        List[Dict[str, str]]: List of normalized news items.

    Raises:
        ValueError: If ticker parameter is invalid.
        NewsDataError: If API execution encounters a fatal exception.
    """
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker symbol must be a non-empty string.")

    cleaned_ticker = ticker.strip().upper()
    target_limit = limit if limit is not None else config.NEWS_HEADLINES_LIMIT

    try:
        ticker_obj = yf.Ticker(cleaned_ticker)
        raw_news = ticker_obj.news
    except Exception as exc:
        raise NewsDataError(
            f"Failed to retrieve news headlines for ticker '{cleaned_ticker}': {exc}"
        ) from exc

    if not raw_news or not isinstance(raw_news, list):
        warnings.warn(
            f"No news headlines available for ticker '{cleaned_ticker}'.",
            InsufficientNewsWarning
        )
        return []

    normalized_list: List[Dict[str, str]] = []
    seen_headlines = set()

    for item in raw_news:
        normalized = normalize_news_item(item)
        if normalized:
            # Deduplicate based on lowercase headline text
            h_key = normalized["headline"].lower()
            if h_key not in seen_headlines:
                seen_headlines.add(h_key)
                normalized_list.append(normalized)

    # Check headline threshold count requirement
    if len(normalized_list) < target_limit:
        warnings.warn(
            f"Retrieved {len(normalized_list)} headlines for '{cleaned_ticker}', "
            f"which is fewer than the requested target of {target_limit}.",
            InsufficientNewsWarning
        )

    return normalized_list
