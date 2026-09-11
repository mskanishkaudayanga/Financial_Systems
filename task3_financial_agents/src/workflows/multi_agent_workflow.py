# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update multi_agent_workflow.py integrating persistent JSON cache check and LangGraph MemorySaver checkpointer for thread sessions', Date: 2026-09-11
"""
Multi-Agent Workflow Package.

Compiles the multi-agent sequential LangGraph StateGraph connecting Agent A (Data Analyst),
structured DataBrief handoff, Agent B (Research Writer), Critique Loop, and Memory System.
"""

import time
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.config import config
from src.schemas.agent_schemas import AgentState
from src.agents.data_analyst_agent import (
    data_analyst_agent_node,
    data_analyst_tools_node,
    synthesize_and_validate_data_brief,
    agent_a_clarification_node,
    DATA_ANALYST_SYSTEM_PROMPT,
)
from src.agents.research_writer_agent import (
    research_writer_agent_node,
    research_writer_tools_node,
)
from src.memory import (
    load_persistent_cache,
    save_persistent_cache,
    get_short_term_checkpointer,
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

    if state.get("clarification_request") and state.get("clarification_count", 0) == 0:
        return "agent_a_clarification"

    return END


def create_multi_agent_graph(checkpointer=None):
    """
    Construct and compile the Multi-Agent Sequential LangGraph StateGraph with Memory Checkpointer.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("agent_a", data_analyst_agent_node)
    workflow.add_node("agent_a_tools", data_analyst_tools_node)
    workflow.add_node("agent_a_handoff", synthesize_and_validate_data_brief)
    workflow.add_node("agent_a_clarification", agent_a_clarification_node)

    workflow.add_node("agent_b", research_writer_agent_node)
    workflow.add_node("agent_b_tools", research_writer_tools_node)

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

    workflow.add_edge("agent_a_handoff", "agent_b")

    workflow.add_conditional_edges(
        source="agent_b",
        path=should_continue_agent_b,
        path_map={
            "agent_b_tools": "agent_b_tools",
            "agent_a_clarification": "agent_a_clarification",
            END: END
        }
    )
    workflow.add_edge("agent_b_tools", "agent_b")
    workflow.add_edge("agent_a_clarification", "agent_b")

    # Compile Graph with optional short-term checkpointer
    return workflow.compile(checkpointer=checkpointer)


def run_multi_agent_research(
    ticker: str,
    question: Optional[str] = None,
    use_cache: bool = True,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the end-to-end multi-agent research workflow with persistent disk cache & short-term memory checkpointer.

    Args:
        ticker: Equity ticker symbol (e.g. 'AAPL').
        question: Optional research prompt.
        use_cache: If True, checks persistent disk cache (cache/{TICKER}_{YYYY-MM-DD}.json) before running.
        thread_id: Optional session thread ID for short-term graph memory follow-up turns.

    Returns:
        Dict[str, Any]: Final graph state containing data_brief, final_report, and state history.
    """
    cleaned_ticker = ticker.strip().upper()

    # 1. Check Persistent Disk Cache (if use_cache is True and no follow-up thread_id)
    if use_cache and not thread_id:
        cached_payload = load_persistent_cache(cleaned_ticker)
        if cached_payload is not None:
            return cached_payload

    # 2. Execute Multi-Agent Graph with Short-Term Memory Checkpointer
    default_prompt = (
        f"Conduct multi-agent equity research on {cleaned_ticker}. "
        f"Agent A will analyze quantitative price trends into a DataBrief. "
        f"Agent B will receive the DataBrief, perform qualitative news research, issue a ClarificationRequest if needed, and synthesize the report."
    )
    user_question = question if question and question.strip() else default_prompt

    initial_messages = [
        SystemMessage(content=DATA_ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=user_question),
    ]

    initial_state: AgentState = {
        "messages": initial_messages,
        "ticker": cleaned_ticker,
        "research_question": user_question,
        "observations": [],
        "data_brief": None,
        "clarification_request": None,
        "clarification_response": None,
        "clarification_count": 0,
        "final_report": None,
    }

    checkpointer = get_short_term_checkpointer()
    graph = create_multi_agent_graph(checkpointer=checkpointer)

    session_thread = thread_id or f"session_{cleaned_ticker}_{int(time.time())}"
    config_dict = {
        "configurable": {"thread_id": session_thread},
        "recursion_limit": config.MAX_RECURSION_LIMIT
    }

    final_state = graph.invoke(
        initial_state,
        config=config_dict
    )

    last_msg = final_state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        final_state["final_report"] = str(last_msg.content)

    # 3. Save payload to Persistent Disk Cache upon successful run
    if use_cache:
        save_persistent_cache(cleaned_ticker, final_state)

    return final_state
