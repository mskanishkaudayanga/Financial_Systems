# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create pytest unit tests for Task 3B critique clarification loop verifying schemas, node routing, and single-loop guard', Date: 2026-09-11
"""
Unit test suite for Task 3B Critique / Clarification Loop.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from src.schemas.agent_schemas import ClarificationRequest, ClarificationResponse, AgentState
from src.agents.data_analyst_agent import agent_a_clarification_node
from src.workflows.multi_agent_workflow import create_multi_agent_graph, should_continue_agent_b
from langgraph.graph import END


def test_clarification_schemas():
    """Verify ClarificationRequest and ClarificationResponse schema validation."""
    req = ClarificationRequest(
        ticker="AAPL",
        request_id="req_101",
        specific_question="Calculate percentage distance between current price and SMA50.",
        metric_type="percentage_distance_sma50"
    )
    assert req.ticker == "AAPL"
    assert req.metric_type == "percentage_distance_sma50"

    resp = ClarificationResponse(
        request_id="req_101",
        ticker="AAPL",
        metric_name="percentage_distance_sma50",
        calculated_value=4.27,
        formatted_result="Current price ($224.50) is +4.27% above SMA50 ($215.30).",
        supporting_details={"current_price": 224.50, "sma50": 215.30}
    )
    assert resp.calculated_value == 4.27
    assert "+4.27%" in resp.formatted_result


def test_should_continue_agent_b_clarification_routing():
    """Verify conditional edge routes to 'agent_a_clarification' when clarification_count == 0."""
    sample_state: AgentState = {
        "messages": [HumanMessage(content="Analyze AAPL"), AIMessage(content="Need metric.")],
        "ticker": "AAPL",
        "research_question": "Analyze AAPL",
        "observations": [],
        "data_brief": {"current_price": 224.5, "relevant_technical_indicators": {"sma50": 215.3}},
        "clarification_request": {"request_id": "req_1", "specific_question": "Calc SMA50 pct", "metric_type": "pct_sma50"},
        "clarification_response": None,
        "clarification_count": 0,
        "final_report": None,
    }

    # First pass (clarification_count == 0) -> routes to agent_a_clarification
    route_1 = should_continue_agent_b(sample_state)
    assert route_1 == "agent_a_clarification"

    # Second pass (clarification_count == 1) -> routes to END
    sample_state["clarification_count"] = 1
    sample_state["clarification_request"] = None
    route_2 = should_continue_agent_b(sample_state)
    assert route_2 == END


def test_agent_a_clarification_node_execution():
    """Verify agent_a_clarification_node processes ClarificationRequest and calculates metric."""
    sample_state: AgentState = {
        "messages": [HumanMessage(content="Analyze AAPL")],
        "ticker": "AAPL",
        "research_question": "Analyze AAPL",
        "observations": [],
        "data_brief": {"current_price": 224.50, "relevant_technical_indicators": {"latest_sma50": 215.30}},
        "clarification_request": {
            "request_id": "req_1",
            "specific_question": "Calculate percentage distance between current price and SMA50.",
            "metric_type": "percentage_distance_sma50"
        },
        "clarification_response": None,
        "clarification_count": 0,
        "final_report": None,
    }

    result = agent_a_clarification_node(sample_state)
    assert "clarification_response" in result
    assert result["clarification_count"] == 1
    assert result["clarification_request"] is None

    resp = result["clarification_response"]
    assert resp["calculated_value"] == 4.27
    assert "4.27%" in resp["formatted_result"]


def test_multi_agent_graph_with_clarification_node():
    """Verify Multi-Agent StateGraph compiles with agent_a_clarification node."""
    graph = create_multi_agent_graph()
    assert graph is not None
    node_names = list(graph.nodes.keys())
    assert "agent_a_clarification" in node_names
