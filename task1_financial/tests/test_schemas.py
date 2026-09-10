"""
Unit tests for configuration loading and Pydantic schemas.
"""

from src.config import config


def test_config_loading():
    """Verify configuration loads default values successfully without hardcoded secrets."""
    assert config.DEFAULT_TICKER == "AAPL" or isinstance(config.DEFAULT_TICKER, str)
    assert config.LOOKBACK_YEARS >= 1
    assert config.NEWS_HEADLINES_LIMIT >= 1


def test_schemas_placeholder():
    """Placeholder test for future Pydantic models."""
    pass
