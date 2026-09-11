# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Re-export agent_node and execute_tools_node in src/agents/__init__.py', Date: 2026-09-11
"""
Agents Package.

Exposes autonomous financial research agent node and tool execution node.
"""

from src.agents.research_agent import (
    agent_node,
    execute_tools_node,
    RESEARCH_AGENT_SYSTEM_PROMPT,
)

__all__ = [
    "agent_node",
    "execute_tools_node",
    "RESEARCH_AGENT_SYSTEM_PROMPT",
]
