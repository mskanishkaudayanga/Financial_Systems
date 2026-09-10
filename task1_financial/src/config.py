"""
Configuration management for the Financial AI Equity Research pipeline.

Loads environment variables using python-dotenv with type-annotated settings.
Secrets and credentials MUST be provided via environment variables or a local .env file.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Root directory of the task module (task1_financial)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load local .env file if present (never committed to repository)
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass(frozen=True)
class Config:
    """Centralized configuration container for pipeline settings."""

    # Market Data Parameters
    DEFAULT_TICKER: str = os.getenv("DEFAULT_TICKER", "AAPL")
    LOOKBACK_YEARS: int = int(os.getenv("LOOKBACK_YEARS", "2"))

    # News Parameters
    NEWS_HEADLINES_LIMIT: int = int(os.getenv("NEWS_HEADLINES_LIMIT", "10"))

    # LLM Configuration (Credentials MUST come from environment)
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "15"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# Expose configuration instance
config = Config()
