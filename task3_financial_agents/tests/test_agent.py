# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create pytest unit tests for Task 3A single agent StateGraph structure, nodes, and conditional edges', Date: 2026-09-11
"""
Unit test suite for Task 3A Autonomous Financial Research Agent Graph.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.workflows import create_single_agent_graph, should_continue
from src.schemas.agent_schemas import AgentState
from langgraph.graph import END


def test_graph_compilation_and_nodes():
    """Verify single agent StateGraph compiles with correct node names."""
    graph = create_single_agent_graph()
    assert graph is not None
    node_names = list(graph.nodes.keys())
    assert "agent" in node_names
    assert "tools" in node_names


def test_should_continue_with_tool_calls():
    """Verify conditional edge routes to 'tools' when AIMessage contains tool_calls."""
    ai_msg_with_tools = AIMessage(
        content="I will check market data.",
        tool_calls=[{"name": "get_price_data", "args": {"ticker": "AAPL"}, "id": "call_1"}]
    )
    state: AgentState = {
        "messages": [HumanMessage(content="Analyze AAPL"), ai_msg_with_tools],
        "ticker": "AAPL",
        "research_question": "Analyze AAPL",
        "final_report": None,
    }
    decision = should_continue(state)
    assert decision == "tools"


def test_should_continue_synthesis_finish():
    """Verify conditional edge routes to END when AIMessage has no tool_calls."""
    ai_msg_final = AIMessage(content="# Financial Health Report\nEverything looks strong.")
    state: AgentState = {
        "messages": [HumanMessage(content="Analyze AAPL"), ai_msg_final],
        "ticker": "AAPL",
        "research_question": "Analyze AAPL",
        "final_report": None,
    }
    decision = should_continue(state)
    assert decision == END
