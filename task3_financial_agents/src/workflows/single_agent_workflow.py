# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create src/workflows/single_agent_workflow.py compiling LangGraph StateGraph with conditional tool loop and runner function', Date: 2026-09-11
"""
Single Agent Workflow Package.

Compiles a cyclic LangGraph StateGraph orchestrating the autonomous financial research agent,
conditional tool execution loop, state transitions, and report synthesis.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.config import config
from src.schemas.agent_schemas import AgentState
from src.agents.research_agent import (
    agent_node,
    execute_tools_node,
    RESEARCH_AGENT_SYSTEM_PROMPT,
)


def should_continue(state: AgentState) -> str:
    """
    Conditional Routing Edge.

    Determines whether to route to the 'tools' node for tool execution,
    or to END if the agent has finished tool calls and generated the report.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def create_single_agent_graph():
    """
    Construct and compile the LangGraph StateGraph for Task 3A.

    Graph Architecture:
        START -> agent_node -> should_continue -> tools_node -> agent_node ... -> END
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", execute_tools_node)

    # 2. Add Edges & Conditional Routing
    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        source="agent",
        path=should_continue,
        path_map={
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge("tools", "agent")

    # 3. Compile Graph
    return workflow.compile()


def run_financial_research_agent(
    ticker: str,
    question: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the autonomous financial research agent workflow for a given equity ticker.

    Args:
        ticker: Equity ticker symbol (e.g., 'AAPL', 'NVDA', 'MSFT').
        question: Optional specific research question. If omitted, uses standard task prompt.

    Returns:
        Dict[str, Any]: Final graph state containing history messages and final_report.
    """
    cleaned_ticker = ticker.strip().upper()
    default_prompt = (
        f"Analyse the current financial health and market sentiment of {cleaned_ticker}. "
        f"Identify the top three risks to its share price over the next 90 days "
        f"and suggest one data-driven hedge strategy."
    )
    user_question = question if question and question.strip() else default_prompt

    initial_messages = [
        SystemMessage(content=RESEARCH_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=user_question),
    ]

    initial_state: AgentState = {
        "messages": initial_messages,
        "ticker": cleaned_ticker,
        "research_question": user_question,
        "final_report": None,
    }

    graph = create_single_agent_graph()

    # Execute StateGraph with recursion limit safety guard
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": config.MAX_RECURSION_LIMIT}
    )

    # Extract final synthesis text
    last_msg = final_state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        final_state["final_report"] = str(last_msg.content)

    return final_state
