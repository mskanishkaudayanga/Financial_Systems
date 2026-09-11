# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create src/workflows/multi_agent_workflow.py compiling Multi-Agent StateGraph for Agent A -> Handoff -> Agent B -> Final Report', Date: 2026-09-11
"""
Multi-Agent Workflow Package.

Compiles the multi-agent sequential LangGraph StateGraph connecting Agent A (Data Analyst),
structured DataBrief handoff, and Agent B (Research Writer).
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.config import config
from src.schemas.agent_schemas import AgentState
from src.agents.data_analyst_agent import (
    data_analyst_agent_node,
    data_analyst_tools_node,
    synthesize_and_validate_data_brief,
    DATA_ANALYST_SYSTEM_PROMPT,
)
from src.agents.research_writer_agent import (
    research_writer_agent_node,
    research_writer_tools_node,
)


def should_continue_agent_a(state: AgentState) -> str:
    """Conditional Routing Edge for Agent A."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "agent_a_tools"
    return "agent_a_handoff"


def should_continue_agent_b(state: AgentState) -> str:
    """Conditional Routing Edge for Agent B."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "agent_b_tools"
    return END


def create_multi_agent_graph():
    """
    Construct and compile the Multi-Agent Sequential LangGraph StateGraph.

    Graph Architecture:
        START -> agent_a -> should_continue_agent_a -> agent_a_tools -> agent_a
                         -> agent_a_handoff -> agent_b -> should_continue_agent_b -> agent_b_tools -> agent_b -> END
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("agent_a", data_analyst_agent_node)
    workflow.add_node("agent_a_tools", data_analyst_tools_node)
    workflow.add_node("agent_a_handoff", synthesize_and_validate_data_brief)

    workflow.add_node("agent_b", research_writer_agent_node)
    workflow.add_node("agent_b_tools", research_writer_tools_node)

    # 2. Agent A Flow
    workflow.add_edge(START, "agent_a")
    workflow.add_conditional_edges(
        source="agent_a",
        path=should_continue_agent_a,
        path_map={
            "agent_a_tools": "agent_a_tools",
            "agent_a_handoff": "agent_a_handoff"
        }
    )
    workflow.add_edge("agent_a_tools", "agent_a")

    # 3. Handoff to Agent B Flow
    workflow.add_edge("agent_a_handoff", "agent_b")

    # 4. Agent B Flow
    workflow.add_conditional_edges(
        source="agent_b",
        path=should_continue_agent_b,
        path_map={
            "agent_b_tools": "agent_b_tools",
            END: END
        }
    )
    workflow.add_edge("agent_b_tools", "agent_b")

    # 5. Compile Multi-Agent Graph
    return workflow.compile()


def run_multi_agent_research(ticker: str) -> Dict[str, Any]:
    """
    Run the end-to-end multi-agent research workflow (Agent A -> DataBrief -> Agent B -> Final Report).

    Args:
        ticker: Equity ticker symbol (e.g., 'AAPL', 'NVDA', 'MSFT').

    Returns:
        Dict[str, Any]: Final graph state containing history messages, data_brief, and final_report.
    """
    cleaned_ticker = ticker.strip().upper()
    prompt = (
        f"Conduct multi-agent equity research on {cleaned_ticker}. "
        f"Agent A will analyze quantitative price trends and volatility into a DataBrief. "
        f"Agent B will receive the DataBrief, perform qualitative news research, and synthesize the final report."
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

    graph = create_multi_agent_graph()

    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": config.MAX_RECURSION_LIMIT}
    )

    # Extract final report text from last AIMessage
    last_msg = final_state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        final_state["final_report"] = str(last_msg.content)

    return final_state
