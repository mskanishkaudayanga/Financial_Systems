# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create src/config.py with python-dotenv loading and dataclass for LLM provider, model name, API key, and app settings', Date: 2026-09-11
"""
Configuration management for Task 3 Financial Research Agent System.

Loads environment variables using python-dotenv with type-annotated settings.
Secrets and credentials MUST be provided via environment variables or a local .env file.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Root directory of task3_financial_agents module
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load local .env file if present
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass(frozen=True)
class Config:
    """Centralized configuration container for financial agents system."""

    # LLM Provider & Model Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: Optional[str] = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Application Settings & Directories
    CACHE_DIR: Path = BASE_DIR / os.getenv("CACHE_DIR", "cache")
    TRACE_FILE: Path = BASE_DIR / os.getenv("TRACE_FILE", "agent_trace.jsonl")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_RECURSION_LIMIT: int = int(os.getenv("MAX_RECURSION_LIMIT", "25"))


# Single configuration instance for project
config = Config()
