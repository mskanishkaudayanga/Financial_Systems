# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create src/memory/short_term_memory.py providing LangGraph MemorySaver checkpointer for short-term thread state retention', Date: 2026-09-11
"""
Short-Term Memory Module.

Provides in-memory LangGraph state checkpointers (MemorySaver) to retain
conversation history, tool observations, and agent outputs across multi-turn session threads.
"""

from langgraph.checkpoint.memory import MemorySaver

# Shared in-memory checkpointer instance
_MEMORY_CHECKPOINTER = MemorySaver()


def get_short_term_checkpointer() -> MemorySaver:
    """Return shared LangGraph MemorySaver checkpointer instance."""
    return _MEMORY_CHECKPOINTER
