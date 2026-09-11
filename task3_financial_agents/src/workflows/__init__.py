# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Re-export create_multi_agent_graph and run_multi_agent_research in src/workflows/__init__.py', Date: 2026-09-11
"""
Workflows Package.

Orchestrates single-agent state graphs, Agent A workflow, and Multi-Agent Sequential Workflow.
"""

from src.workflows.single_agent_workflow import (
    create_single_agent_graph,
    run_financial_research_agent,
)
from src.workflows.data_analyst_workflow import (
    create_data_analyst_graph,
    run_data_analyst_agent,
)
from src.workflows.multi_agent_workflow import (
    create_multi_agent_graph,
    run_multi_agent_research,
)

__all__ = [
    "create_single_agent_graph",
    "run_financial_research_agent",
    "create_data_analyst_graph",
    "run_data_analyst_agent",
    "create_multi_agent_graph",
    "run_multi_agent_research",
]
