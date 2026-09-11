# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Re-export create_single_agent_graph and run_financial_research_agent in src/workflows/__init__.py', Date: 2026-09-11
"""
Workflows Package.

Orchestrates multi-agent interactions and single-agent state graph execution.
"""

from src.workflows.single_agent_workflow import (
    create_single_agent_graph,
    run_financial_research_agent,
)

__all__ = [
    "create_single_agent_graph",
    "run_financial_research_agent",
]
