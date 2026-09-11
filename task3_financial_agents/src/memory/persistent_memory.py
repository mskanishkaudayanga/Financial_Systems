# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create src/memory/persistent_memory.py implementing deterministic cache key cache/{TICKER}_{YYYY-MM-DD}.json, load, save, and corrupt file validation', Date: 2026-09-11
"""
Persistent Memory Module.

Provides deterministic file-based disk caching for final research briefs
using key pattern: cache/{TICKER}_{YYYY-MM-DD}.json. Features defensive JSON validation
to detect corrupted or incomplete cache files gracefully.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import config
from src.observability import log_trace_event


def get_cache_key(ticker: str, date_str: Optional[str] = None) -> str:
    """
    Generate deterministic cache key string: {TICKER}_{YYYY-MM-DD}.

    Args:
        ticker: Equity ticker symbol (e.g. 'AAPL').
        date_str: Optional YYYY-MM-DD string. Defaults to current UTC date.
    """
    clean_ticker = ticker.strip().upper()
    current_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{clean_ticker}_{current_date}"


def get_cache_file_path(ticker: str, date_str: Optional[str] = None) -> Path:
    """Return full filesystem Path for deterministic cache file."""
    cache_key = get_cache_key(ticker, date_str)
    return config.CACHE_DIR / f"{cache_key}.json"


def validate_cache_payload(payload: Dict[str, Any], expected_ticker: str) -> bool:
    """
    Defensively validate cache payload schema integrity.

    Required keys: 'ticker', 'data_brief', 'final_report', 'timestamp'.
    """
    if not isinstance(payload, dict):
        return False

    required_keys = ["ticker", "data_brief", "final_report", "timestamp"]
    for key in required_keys:
        if key not in payload or payload[key] is None:
            return False

    # Check ticker match
    if str(payload.get("ticker")).upper() != expected_ticker.upper():
        return False

    # Check data_brief structure
    brief = payload.get("data_brief")
    if not isinstance(brief, dict) or "current_price" not in brief:
        return False

    return True


def load_persistent_cache(ticker: str, date_str: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Attempt to load research brief from persistent disk cache.

    Returns:
        Dict[str, Any] if valid cache hit occurs, None on cache miss or corrupted file.
    """
    cache_file = get_cache_file_path(ticker, date_str)

    if not cache_file.exists():
        log_trace_event(
            event_type="AGENT",
            content=f"⚡ [PERSISTENT CACHE MISS] No cache file found at {cache_file.name}. Executing graph workflow.",
            metadata={"cache_hit": False, "file": str(cache_file)}
        )
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if validate_cache_payload(payload, ticker):
            log_trace_event(
                event_type="AGENT",
                content=f"🎯 [PERSISTENT CACHE HIT] Loaded valid research brief from {cache_file.name}. Bypassing tool execution.",
                metadata={"cache_hit": True, "file": str(cache_file), "timestamp": payload.get("timestamp")}
            )
            return payload
        else:
            log_trace_event(
                event_type="AGENT",
                content=f"⚠️ [CORRUPT CACHE DETECTED] File {cache_file.name} is incomplete or corrupted. Bypassing cache.",
                metadata={"cache_hit": False, "file": str(cache_file), "reason": "schema_validation_failed"}
            )
            return None

    except Exception as exc:
        log_trace_event(
            event_type="AGENT",
            content=f"⚠️ [CORRUPT CACHE DETECTED] Failed to read {cache_file.name} due to JSON parse error: {str(exc)}. Bypassing cache.",
            metadata={"cache_hit": False, "file": str(cache_file), "error": str(exc)}
        )
        return None


def save_persistent_cache(ticker: str, payload: Dict[str, Any], date_str: Optional[str] = None) -> bool:
    """
    Save structured research brief payload to persistent JSON disk cache.

    Args:
        ticker: Equity ticker symbol.
        payload: Final state payload containing data_brief, final_report, etc.
        date_str: Optional YYYY-MM-DD string.
    """
    cache_file = get_cache_file_path(ticker, date_str)

    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "ticker": ticker.upper(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_brief": payload.get("data_brief"),
            "clarification_request": payload.get("clarification_request"),
            "clarification_response": payload.get("clarification_response"),
            "final_report": payload.get("final_report"),
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, default=str)

        log_trace_event(
            event_type="AGENT",
            content=f"💾 [PERSISTENT CACHE SAVED] Saved research brief to {cache_file.name}.",
            metadata={"file": str(cache_file)}
        )
        return True

    except Exception as exc:
        log_trace_event(
            event_type="AGENT",
            content=f"⚠️ Failed to save persistent cache to {cache_file.name}: {str(exc)}",
            metadata={"error": str(exc)}
        )
        return False
