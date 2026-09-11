# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update test_agent.py testing Observe-Replan-Act observation record generation and graceful failure recovery', Date: 2026-09-11
"""
Unit test suite for Task 3A Autonomous Financial Research Agent Graph.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from src.workflows import create_single_agent_graph, should_continue
from src.agents.research_agent import _format_observation_summary, _generate_replan_decision
from src.schemas.agent_schemas import AgentState
from langgraph.graph import END


def test_graph_compilation_and_nodes():
    """Verify single agent StateGraph compiles with correct node names."""
    graph = create_single_agent_graph()
    assert graph is not None
    node_names = list(graph.nodes.keys())
    assert "agent" in node_names
    assert "tools" in node_names


def test_observation_summary_formatting():
    """Verify _format_observation_summary creates concise observation strings."""
    res_price = {
        "status": "success",
        "ticker": "AAPL",
        "latest_indicators": {"latest_close": 225.0, "latest_rsi14": 55.0, "latest_sma20": 220.0}
    }
    obs_str = _format_observation_summary("get_price_data", res_price)
    assert "Observed OHLCV price trend" in obs_str
    assert "AAPL" in obs_str
    assert "225.0" in obs_str


def test_graceful_recovery_replan_decision():
    """Verify agent replans with graceful recovery when an observation indicates a tool failure."""
    failed_obs = [{
        "tool_name": "web_search",
        "status": "error",
        "summary": "Web search returned 0 items",
        "has_error": True
    }]
    ai_response = AIMessage(
        content="",
        tool_calls=[{"name": "get_news", "args": {"ticker": "AAPL"}, "id": "call_99"}]
    )
    decision = _generate_replan_decision(ai_response, failed_obs, "AAPL")
    assert "Observed failure/empty output" in decision
    assert "web_search" in decision
    assert "get_news" in decision
