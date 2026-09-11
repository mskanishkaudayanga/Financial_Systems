# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update test_data_analyst.py testing state[data_brief] field population and defensive Pydantic validation', Date: 2026-09-11
"""
Unit test suite for Task 3B Agent A (Quantitative Data Analyst).
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agents.data_analyst_agent import DATA_ANALYST_TOOLS, synthesize_and_validate_data_brief
from src.workflows.data_analyst_workflow import create_data_analyst_graph
from src.schemas.agent_schemas import DataBrief, AgentState


def test_agent_a_tool_restriction():
    """Verify Agent A is strictly restricted to allowed quantitative tools only."""
    tool_names = [t.name for t in DATA_ANALYST_TOOLS]

    assert "get_price_data" in tool_names
    assert "calculate_volatility" in tool_names
    assert "llm_sentiment" in tool_names

    assert "get_news" not in tool_names
    assert "web_search" not in tool_names
    assert len(DATA_ANALYST_TOOLS) == 3


def test_data_analyst_graph_nodes():
    """Verify Agent A StateGraph contains analyst, tools, and handoff nodes."""
    graph = create_data_analyst_graph()
    assert graph is not None
    node_names = list(graph.nodes.keys())
    assert "analyst" in node_names
    assert "tools" in node_names
    assert "handoff" in node_names


def test_databrief_pydantic_schema():
    """Verify DataBrief Pydantic schema validation."""
    brief = DataBrief(
        ticker="AAPL",
        current_price=225.50,
        relevant_technical_indicators={"sma20": 220.0, "rsi14": 55.0},
        volatility=21.4,
        sentiment_score=0.35,
        sentiment_label="Bullish",
        quantitative_observations=["Trading above 20-day SMA", "Volatility at 21.4%"]
    )
    assert brief.ticker == "AAPL"
    assert brief.current_price == 225.50
    assert brief.volatility == 21.4
    assert len(brief.quantitative_observations) == 2


def test_synthesize_and_validate_data_brief_node():
    """Verify synthesize_and_validate_data_brief populates valid data_brief dictionary in state."""
    sample_state: AgentState = {
        "messages": [
            HumanMessage(content="Analyze AAPL"),
            AIMessage(content="Quantitative analysis done."),
        ],
        "ticker": "AAPL",
        "research_question": "Analyze AAPL",
        "observations": [{"tool_name": "get_price_data", "status": "success", "summary": "Close=$225", "has_error": False}],
        "data_brief": None,
        "final_report": None,
    }

    res = synthesize_and_validate_data_brief(sample_state)
    assert "data_brief" in res
    brief = res["data_brief"]
    assert brief is not None
    assert brief["ticker"] == "AAPL"
    assert brief["current_price"] > 0
    assert "relevant_technical_indicators" in brief
