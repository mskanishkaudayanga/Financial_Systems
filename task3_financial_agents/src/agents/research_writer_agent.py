# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update research_writer_agent.py implementing ClarificationRequest issuance on first pass and ClarificationResponse integration on second pass', Date: 2026-09-11
"""
Agent B: Qualitative Research Writer Agent.

Specialized sub-agent responsible for qualitative financial research, headline parsing,
analyst commentary retrieval via web search, issuing ClarificationRequests to Agent A,
and synthesizing the final equity research report.
"""

import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from src.config import config
from src.tools import (
    get_news,
    web_search,
)
from src.schemas.agent_schemas import (
    AgentState,
    ClarificationRequest,
)
from src.observability import log_trace_event

# ---------------------------------------------------------------------------
# Strict Tool Access Control for Agent B (Research Writer)
# EXCLUDED: get_price_data, calculate_volatility
# ---------------------------------------------------------------------------
RESEARCH_WRITER_TOOLS = [get_news, web_search]
RESEARCH_WRITER_TOOLS_BY_NAME = {t.name: t for t in RESEARCH_WRITER_TOOLS}

RESEARCH_WRITER_SYSTEM_PROMPT = """You are Agent B: Qualitative Research Writer Agent.
Your responsibility is qualitative market research, news analysis, analyst commentary evaluation,
and synthesizing the final equity research report for an equity ticker.

STRICT ACCESS CONTROL:
- You have access ONLY to 2 qualitative tools: `get_news` and `web_search`.
- You do NOT have access to `get_price_data` or `calculate_volatility`.

CRITIQUE & CLARIFICATION LOOP RULES:
- You receive Agent A's structured DataBrief (price trends, technical indicators, volatility, sentiment).
- If you require a specific quantitative metric calculation to strengthen the 90-day risk analysis (e.g. percentage distance between current price and SMA50), you can issue a ClarificationRequest to Agent A.
- Autonomously use `get_news` and `web_search` to gather qualitative context, recent corporate announcements, and analyst risk notes.
- Compare qualitative findings against Agent A's quantitative DataBrief and returned ClarificationResponse.
- Synthesize a comprehensive final equity research report containing:
  1. Financial Health & Price Trend Summary
  2. Top Three Share Price Risks over the next 90 days (with explicit empirical supporting evidence for each)
  3. Data-Driven Hedge Strategy Recommendation.
"""


def _get_llm():
    """Initialize base LLM with configuration settings."""
    return ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        openai_api_key=config.LLM_API_KEY,
        openai_api_base=config.LLM_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        max_retries=2,
        timeout=30.0,
    )


def research_writer_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent B (Research Writer) LLM Node.

    Handles initial qualitative analysis, issues structured ClarificationRequest when clarification_count == 0,
    or synthesizes final report when clarification_response is received.
    """
    messages = list(state["messages"])
    ticker = state.get("ticker", "AAPL")
    data_brief = state.get("data_brief") or {}
    clarification_count = state.get("clarification_count", 0)
    clarification_response = state.get("clarification_response")

    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=RESEARCH_WRITER_SYSTEM_PROMPT))

    # First Pass (clarification_count == 0): Issue structured ClarificationRequest to Agent A
    if clarification_count == 0 and not state.get("clarification_request"):
        specific_q = f"Calculate the exact percentage distance between the current price and the 50-day moving average (SMA50) for {ticker}."
        req_obj = ClarificationRequest(
            ticker=ticker,
            request_id="req_1",
            specific_question=specific_q,
            metric_type="percentage_distance_sma50"
        )
        req_dict = req_obj.model_dump()

        log_trace_event(
            event_type="CLARIFICATION REQUEST",
            content=specific_q,
            metadata={"from": "Agent B (Research Writer)", "to": "Agent A (Data Analyst)", "request": req_dict}
        )

        log_trace_event(
            event_type="AGENT DECISION",
            content=f"[Agent B] Identified missing quantitative spread metric; requesting clarification from Agent A.",
            metadata={"agent": "ResearchWriter"}
        )

        return {
            "clarification_request": req_dict,
            "messages": [AIMessage(content=f"Requesting clarification from Agent A: {specific_q}")]
        }

    # Second Pass (after receiving ClarificationResponse from Agent A): Continue qualitative research or synthesize report
    if clarification_response and not any("[Clarification Received]" in str(m.content) for m in messages):
        formatted_res = clarification_response.get("formatted_result", "")
        messages.append(
            HumanMessage(content=f"[Clarification Received from Agent A]\nCalculated Metric: {formatted_res}")
        )
        log_trace_event(
            event_type="AGENT",
            content=f"[Agent B: Research Writer] Incorporated Agent A's ClarificationResponse ({formatted_res}). Proceeding to report synthesis.",
            metadata={"agent": "ResearchWriter", "clarification": clarification_response}
        )

    llm = _get_llm()
    llm_with_tools = llm.bind_tools(RESEARCH_WRITER_TOOLS)

    response: AIMessage = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        decision_str = f"[Agent B: Research Writer] Dispatching qualitative tools: {', '.join(tool_names)}."
    else:
        decision_str = f"[Agent B: Research Writer] Qualitative research and critique loop complete. Synthesizing final report."

    log_trace_event(
        event_type="AGENT DECISION",
        content=decision_str,
        metadata={"agent": "ResearchWriter", "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0}
    )

    return {"messages": [response]}


def research_writer_tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent B (Research Writer) Tool Execution Node.
    """
    last_message = state["messages"][-1]
    tool_messages: List[ToolMessage] = []
    new_observations: List[Dict[str, Any]] = []

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": [], "observations": []}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        call_id = tool_call["id"]

        if tool_name not in RESEARCH_WRITER_TOOLS_BY_NAME:
            status = "error"
            summary_str = json.dumps({"status": "error", "error": f"Access Denied: Tool '{tool_name}' is not allowed for Agent B."})
            obs_str = f"Agent A Access Denied: Attempted to call unauthorized quantitative tool '{tool_name}'."
        else:
            log_trace_event(
                event_type="TOOL CALL",
                content=f"[Agent B] Invoking {tool_name} with arguments: {json.dumps(tool_args)}",
                metadata={"agent": "ResearchWriter", "tool_name": tool_name, "args": tool_args, "call_id": call_id}
            )

            tool_obj = RESEARCH_WRITER_TOOLS_BY_NAME[tool_name]
            try:
                result = tool_obj.invoke(tool_args)
                status = result.get("status", "success") if isinstance(result, dict) else "success"
                summary_str = json.dumps(result, default=str)
                obs_str = f"[Agent B Observation] {tool_name} returned output (status: {status})."
            except Exception as exc:
                status = "error"
                result = {"status": "error", "error": f"Tool execution failed: {str(exc)}"}
                summary_str = json.dumps(result)
                obs_str = f"[Agent B Observation] {tool_name} returned error: {str(exc)}."

        log_trace_event(
            event_type="TOOL RESULT",
            content=summary_str,
            metadata={"agent": "ResearchWriter", "tool_name": tool_name, "status": status, "call_id": call_id}
        )

        log_trace_event(
            event_type="UPDATED OBSERVATION",
            content=obs_str,
            metadata={"agent": "ResearchWriter", "tool_name": tool_name, "status": status}
        )

        new_observations.append({
            "tool_name": tool_name,
            "status": status,
            "summary": obs_str,
            "has_error": (status == "error")
        })

        tool_messages.append(
            ToolMessage(
                content=summary_str,
                tool_call_id=call_id,
                name=tool_name
            )
        )

    return {
        "messages": tool_messages,
        "observations": new_observations
    }
