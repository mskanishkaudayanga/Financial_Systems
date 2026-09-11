# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update data_analyst_workflow.py compiling LangGraph StateGraph with synthesize_and_validate_data_brief node populating state[data_brief]', Date: 2026-09-11
"""
Agent A Workflow Package.

Compiles the cyclic LangGraph StateGraph for Agent A (Quantitative Data Analyst)
and populates the dedicated data_brief state field for structured handoff.
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.config import config
from src.schemas.agent_schemas import AgentState, DataBrief
from src.agents.data_analyst_agent import (
    data_analyst_agent_node,
    data_analyst_tools_node,
    synthesize_and_validate_data_brief,
    DATA_ANALYST_SYSTEM_PROMPT,
)


def should_continue_data_analyst(state: AgentState) -> str:
    """
    Conditional Routing Edge for Agent A.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "handoff"


def create_data_analyst_graph():
    """
    Construct and compile the LangGraph StateGraph for Agent A (Data Analyst).

    Graph Architecture:
        START -> analyst -> should_continue -> tools -> analyst ... -> handoff -> END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("analyst", data_analyst_agent_node)
    workflow.add_node("tools", data_analyst_tools_node)
    workflow.add_node("handoff", synthesize_and_validate_data_brief)

    workflow.add_edge(START, "analyst")

    workflow.add_conditional_edges(
        source="analyst",
        path=should_continue_data_analyst,
        path_map={
            "tools": "tools",
            "handoff": "handoff"
        }
    )

    workflow.add_edge("tools", "analyst")
    workflow.add_edge("handoff", END)

    return workflow.compile()


def run_data_analyst_agent(ticker: str) -> Dict[str, Any]:
    """
    Run Agent A (Data Analyst) workflow to perform quantitative analysis,
    populate state["data_brief"], and return the final graph state.

    Args:
        ticker: Equity ticker symbol (e.g., 'AAPL', 'NVDA').

    Returns:
        Dict[str, Any]: Final graph state containing history messages and data_brief dictionary.
    """
    cleaned_ticker = ticker.strip().upper()
    prompt = (
        f"Perform quantitative equity data analysis for {cleaned_ticker}. "
        f"Fetch historical OHLCV price trends, calculate 252-day annualized volatility, "
        f"and analyze headline sentiment. Synthesize all numerical findings into a structured DataBrief."
    )

    initial_messages = [
        SystemMessage(content=DATA_ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    initial_state: AgentState = {
        "messages": initial_messages,
        "ticker": cleaned_ticker,
        "research_question": prompt,
        "observations": [],
        "data_brief": None,
        "final_report": None,
    }

    graph = create_data_analyst_graph()

    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": config.MAX_RECURSION_LIMIT}
    )

    return final_state
