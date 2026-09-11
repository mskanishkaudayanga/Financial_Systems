# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Re-export Agent B Research Writer nodes and RESEARCH_WRITER_TOOLS in src/agents/__init__.py', Date: 2026-09-11
"""
Agents Package.

Exposes autonomous research agent, Agent A (Data Analyst), and Agent B (Research Writer).
"""

from src.agents.research_agent import (
    agent_node,
    execute_tools_node,
    RESEARCH_AGENT_SYSTEM_PROMPT,
)
from src.agents.data_analyst_agent import (
    data_analyst_agent_node,
    data_analyst_tools_node,
    synthesize_and_validate_data_brief,
    DATA_ANALYST_TOOLS,
    DATA_ANALYST_SYSTEM_PROMPT,
)
from src.agents.research_writer_agent import (
    research_writer_agent_node,
    research_writer_tools_node,
    RESEARCH_WRITER_TOOLS,
    RESEARCH_WRITER_SYSTEM_PROMPT,
)

__all__ = [
    "agent_node",
    "execute_tools_node",
    "RESEARCH_AGENT_SYSTEM_PROMPT",
    "data_analyst_agent_node",
    "data_analyst_tools_node",
    "synthesize_and_validate_data_brief",
    "DATA_ANALYST_TOOLS",
    "DATA_ANALYST_SYSTEM_PROMPT",
    "research_writer_agent_node",
    "research_writer_tools_node",
    "RESEARCH_WRITER_TOOLS",
    "RESEARCH_WRITER_SYSTEM_PROMPT",
]
