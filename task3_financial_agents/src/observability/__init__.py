# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Re-export log_trace_event in src/observability/__init__.py', Date: 2026-09-11
"""
Observability Package.

Provides trace logging and agent audit event tracking.
"""

from src.observability.tracer import log_trace_event

__all__ = ["log_trace_event"]
