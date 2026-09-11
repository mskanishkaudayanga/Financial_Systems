# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create pytest unit tests for Task 3C Memory verifying deterministic cache key, corrupt cache validation, and short-term memory checkpointer', Date: 2026-09-11
"""
Unit test suite for Task 3C Memory System (Short-Term & Persistent Memory).
"""

import json
import pytest
from pathlib import Path
from src.memory import (
    get_cache_key,
    get_cache_file_path,
    validate_cache_payload,
    load_persistent_cache,
    save_persistent_cache,
    get_short_term_checkpointer,
)


def test_deterministic_cache_key_generation():
    """Verify get_cache_key produces deterministic TICKER_YYYY-MM-DD string."""
    key = get_cache_key("aapl", "2026-09-11")
    assert key == "AAPL_2026-09-11"

    path = get_cache_file_path("AAPL", "2026-09-11")
    assert path.name == "AAPL_2026-09-11.json"


def test_validate_cache_payload_valid_and_corrupt():
    """Verify validate_cache_payload correctly identifies valid vs corrupted payloads."""
    valid_payload = {
        "ticker": "AAPL",
        "timestamp": "2026-09-11T12:00:00Z",
        "data_brief": {"current_price": 225.0, "relevant_technical_indicators": {"sma20": 220.0}},
        "final_report": "# Financial Report\nHealthy trends."
    }
    assert validate_cache_payload(valid_payload, "AAPL") is True

    # Missing required key
    corrupt_payload_1 = {
        "ticker": "AAPL",
        "data_brief": {"current_price": 225.0},
        # Missing final_report & timestamp
    }
    assert validate_cache_payload(corrupt_payload_1, "AAPL") is False

    # Ticker mismatch
    corrupt_payload_2 = dict(valid_payload)
    corrupt_payload_2["ticker"] = "MSFT"
    assert validate_cache_payload(corrupt_payload_2, "AAPL") is False


def test_save_and_load_persistent_cache(tmp_path, monkeypatch):
    """Verify saving and loading persistent JSON cache."""
    sample_payload = {
        "data_brief": {"current_price": 225.0, "relevant_technical_indicators": {"sma20": 220.0}},
        "final_report": "# Test Report"
    }

    # Save cache
    saved = save_persistent_cache("TEST", sample_payload, date_str="2026-09-11")
    assert saved is True

    # Load cache
    loaded = load_persistent_cache("TEST", date_str="2026-09-11")
    assert loaded is not None
    assert loaded["ticker"] == "TEST"
    assert loaded["data_brief"]["current_price"] == 225.0


def test_short_term_checkpointer_instance():
    """Verify get_short_term_checkpointer returns valid MemorySaver instance."""
    checkpointer = get_short_term_checkpointer()
    assert checkpointer is not None
    assert hasattr(checkpointer, "get_tuple")
