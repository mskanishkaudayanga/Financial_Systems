"""
Unit test suite for news retrieval and payload normalization.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.data.news_data import (
    InsufficientNewsWarning,
    NewsDataError,
    fetch_news_data,
    normalize_news_item,
)


@pytest.fixture
def raw_yfinance_news_payload():
    """Fixture providing raw yfinance news sample payloads (both legacy and modern formats)."""
    return [
        {
            "id": "111",
            "content": {
                "title": "Apple Reports Quarterly Revenue Growth",
                "pubDate": "2024-01-15T12:00:00Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://reuters.com/apple-1"},
            },
        },
        {
            "title": "Apple Launches New iPhone Model",
            "publisher": "Bloomberg",
            "link": "https://bloomberg.com/apple-2",
            "providerPublishTime": 1700000000,
        },
        # Duplicate title payload to test deduplication
        {
            "title": "apple reports quarterly revenue growth",
            "publisher": "Yahoo Finance",
            "link": "https://finance.yahoo.com/dup",
        },
        # Malformed item missing headline
        {
            "publisher": "Unknown",
            "link": "https://example.com/bad",
        },
    ]


def test_normalize_news_item_legacy_and_modern():
    """Test normalization across legacy flat format and modern nested content format."""
    modern_item = {
        "content": {
            "title": "Modern Headline",
            "provider": {"displayName": "TechCrunch"},
            "canonicalUrl": {"url": "https://techcrunch.com/item"},
            "pubDate": "2024-02-01",
        }
    }
    norm_modern = normalize_news_item(modern_item)
    assert norm_modern["headline"] == "Modern Headline"
    assert norm_modern["source"] == "TechCrunch"
    assert norm_modern["url"] == "https://techcrunch.com/item"

    legacy_item = {
        "title": "Legacy Headline",
        "publisher": "Wall Street Journal",
        "link": "https://wsj.com/item",
        "providerPublishTime": 1700000000,
    }
    norm_legacy = normalize_news_item(legacy_item)
    assert norm_legacy["headline"] == "Legacy Headline"
    assert norm_legacy["source"] == "Wall Street Journal"
    assert norm_legacy["url"] == "https://wsj.com/item"


def test_normalize_news_item_malformed():
    """Test that malformed raw news items missing titles return None."""
    bad_item = {"publisher": "No Title Source"}
    assert normalize_news_item(bad_item) is None
    assert normalize_news_item(None) is None  # type: ignore


def test_fetch_news_data_normalization_and_deduplication(raw_yfinance_news_payload):
    """Test fetch_news_data returns deduplicated, normalized headlines."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.news = raw_yfinance_news_payload

    with patch("src.data.news_data.yf.Ticker", return_value=mock_ticker_instance):
        with pytest.warns(InsufficientNewsWarning, match="fewer than the requested target"):
            headlines = fetch_news_data("AAPL", limit=10)

    # 4 raw items: 1 malformed removed, 1 duplicate removed -> 2 unique headlines remaining
    assert len(headlines) == 2
    assert headlines[0]["headline"] == "Apple Reports Quarterly Revenue Growth"
    assert headlines[1]["headline"] == "Apple Launches New iPhone Model"


def test_fetch_news_data_api_failure():
    """Test that API failures raise NewsDataError."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.news = Exception("News Service Down")

    with patch("src.data.news_data.yf.Ticker", side_effect=Exception("API Connection Failed")):
        with pytest.raises(NewsDataError, match="Failed to retrieve news headlines"):
            fetch_news_data("AAPL")


def test_fetch_news_data_invalid_ticker():
    """Test parameter validation for ticker string."""
    with pytest.raises(ValueError, match="non-empty string"):
        fetch_news_data("")
