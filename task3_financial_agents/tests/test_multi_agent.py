# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create pytest unit tests for Task 3B Multi-Agent workflow verifying Agent B tool restrictions and StateGraph nodes', Date: 2026-09-11
"""
Unit test suite for Task 3B Multi-Agent System (Agent A + DataBrief Handoff + Agent B).
"""

import pytest
from src.agents.research_writer_agent import RESEARCH_WRITER_TOOLS
from src.workflows.multi_agent_workflow import create_multi_agent_graph


def test_agent_b_tool_restriction():
    """Verify Agent B is strictly restricted to allowed qualitative tools only."""
    tool_names = [t.name for t in RESEARCH_WRITER_TOOLS]

    # Allowed tools for Agent B
    assert "get_news" in tool_names
    assert "web_search" in tool_names

    # Prohibited tools (NOT allowed for Agent B)
    assert "get_price_data" not in tool_names
    assert "calculate_volatility" not in tool_names
    assert len(RESEARCH_WRITER_TOOLS) == 2


def test_multi_agent_graph_compilation_and_nodes():
    """Verify Multi-Agent StateGraph compiles with correct node architecture."""
    graph = create_multi_agent_graph()
    assert graph is not None
    node_names = list(graph.nodes.keys())

    assert "agent_a" in node_names
    assert "agent_a_tools" in node_names
    assert "agent_a_handoff" in node_names
    assert "agent_b" in node_names
    assert "agent_b_tools" in node_names
