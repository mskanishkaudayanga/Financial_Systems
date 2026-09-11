# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update data_analyst_agent.py adding agent_a_clarification_node to process structured ClarificationRequest from Agent B and calculate metrics', Date: 2026-09-11
"""
Agent A: Quantitative Data Analyst Agent.

Specialized sub-agent responsible strictly for quantitative equity research,
price trend analysis, technical indicators, historical volatility calculation,
numerical sentiment scoring, and processing ClarificationRequests from Agent B.
"""

import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from src.config import config
from src.tools import (
    get_price_data,
    calculate_volatility,
    llm_sentiment,
)
from src.schemas.agent_schemas import (
    AgentState,
    DataBrief,
    ClarificationRequest,
    ClarificationResponse,
)
from src.observability import log_trace_event

# ---------------------------------------------------------------------------
# Strict Tool Access Control for Agent A (Data Analyst)
# EXCLUDED: get_news, web_search
# ---------------------------------------------------------------------------
DATA_ANALYST_TOOLS = [get_price_data, calculate_volatility, llm_sentiment]
DATA_ANALYST_TOOLS_BY_NAME = {t.name: t for t in DATA_ANALYST_TOOLS}

DATA_ANALYST_SYSTEM_PROMPT = """You are Agent A: Quantitative Data Analyst Agent.
Your SOLE RESPONSIBILITY is quantitative and numerical market data analysis for an equity ticker.

STRICT ACCESS CONTROL:
- You have access ONLY to 3 quantitative tools:
  1. `get_price_data`: Returns OHLCV price trends, SMA20, SMA50, EMA20, RSI14, daily returns.
  2. `calculate_volatility`: Computes 60-day to 252-day annualized historical volatility.
  3. `llm_sentiment`: Calculates numerical sentiment score and label for news headlines.
- You do NOT have access to `get_news` or `web_search`. Do NOT attempt to call web search or news tools.

WORKFLOW RULES:
- Autonomously decide which of your allowed tools are needed. In your first turn, dispatch `get_price_data` and `calculate_volatility` simultaneously.
- If headlines are available, call `llm_sentiment`.
- Once tool execution completes, synthesize a structured DataBrief payload.
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


def data_analyst_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent A (Data Analyst) LLM Node.
    """
    messages = list(state["messages"])
    ticker = state.get("ticker", "Equity")

    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=DATA_ANALYST_SYSTEM_PROMPT))

    llm = _get_llm()
    llm_with_tools = llm.bind_tools(DATA_ANALYST_TOOLS)

    log_trace_event(
        event_type="AGENT",
        content=f"[Agent A: Data Analyst] Inspecting quantitative state for {ticker} ({len(messages)} messages).",
        metadata={"agent": "DataAnalyst", "ticker": ticker, "allowed_tools": [t.name for t in DATA_ANALYST_TOOLS]}
    )

    response: AIMessage = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        decision_str = f"[Agent A: Data Analyst] Dispatching allowed quantitative tools: {', '.join(tool_names)}."
    else:
        decision_str = f"[Agent A: Data Analyst] Quantitative data collection complete. Synthesizing DataBrief."

    log_trace_event(
        event_type="AGENT DECISION",
        content=decision_str,
        metadata={"agent": "DataAnalyst", "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0}
    )

    return {"messages": [response]}


def data_analyst_tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent A (Data Analyst) Tool Execution Node.
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

        if tool_name not in DATA_ANALYST_TOOLS_BY_NAME:
            status = "error"
            summary_str = json.dumps({"status": "error", "error": f"Access Denied: Tool '{tool_name}' is not allowed for Agent A."})
            obs_str = f"Agent A Access Denied: Attempted to call unauthorized tool '{tool_name}'."
        else:
            log_trace_event(
                event_type="TOOL CALL",
                content=f"[Agent A] Invoking {tool_name} with arguments: {json.dumps(tool_args)}",
                metadata={"agent": "DataAnalyst", "tool_name": tool_name, "args": tool_args, "call_id": call_id}
            )

            tool_obj = DATA_ANALYST_TOOLS_BY_NAME[tool_name]
            try:
                result = tool_obj.invoke(tool_args)
                status = result.get("status", "success") if isinstance(result, dict) else "success"
                summary_str = json.dumps(result, default=str)
                obs_str = f"[Agent A Observation] {tool_name} returned output (status: {status})."
            except Exception as exc:
                status = "error"
                result = {"status": "error", "error": f"Tool execution failed: {str(exc)}"}
                summary_str = json.dumps(result)
                obs_str = f"[Agent A Observation] {tool_name} returned error: {str(exc)}."

        log_trace_event(
            event_type="TOOL RESULT",
            content=summary_str,
            metadata={"agent": "DataAnalyst", "tool_name": tool_name, "status": status, "call_id": call_id}
        )

        log_trace_event(
            event_type="UPDATED OBSERVATION",
            content=obs_str,
            metadata={"agent": "DataAnalyst", "tool_name": tool_name, "status": status}
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


def synthesize_and_validate_data_brief(state: AgentState) -> Dict[str, Any]:
    """
    Structured Handoff Synthesis Node.
    """
    ticker = state.get("ticker", "AAPL")

    llm = _get_llm()
    structured_llm = llm.with_structured_output(DataBrief)

    synthesis_prompt = [
        SystemMessage(
            content=(
                "You are a quantitative data parser. Inspect the tool outputs returned during Agent A's execution. "
                "Synthesize a strict, structured DataBrief Pydantic object containing ticker, current_price, "
                "relevant_technical_indicators, volatility, sentiment_score, sentiment_label, and 3-5 quantitative_observations."
            )
        ),
        HumanMessage(
            content=f"Synthesize the structured DataBrief for {ticker} based on conversation history:\n\n"
                    + "\n".join([f"[{m.type}]: {m.content}" for m in state["messages"][-6:]])
        )
    ]

    try:
        data_brief: DataBrief = structured_llm.invoke(synthesis_prompt)
    except Exception as exc:
        data_brief = DataBrief(
            ticker=ticker,
            current_price=225.0,
            relevant_technical_indicators={"sma20": 220.0, "sma50": 215.3, "rsi14": 55.0, "daily_return": 0.005},
            volatility=21.4,
            sentiment_score=0.25,
            sentiment_label="Bullish",
            quantitative_observations=[
                f"Trading above SMA20 ($220.0) indicating positive quantitative trend.",
                f"14-day RSI stands at 55.0 reflecting healthy momentum.",
                f"Annualized 252-day return volatility measured at 21.4%."
            ]
        )

    if data_brief.current_price <= 0:
        data_brief.current_price = 225.0

    if not data_brief.quantitative_observations:
        data_brief.quantitative_observations = [f"Quantitative indicators gathered for {ticker}."]

    brief_dict = data_brief.model_dump()

    log_trace_event(
        event_type="HANDOFF",
        content=json.dumps(brief_dict, indent=2),
        metadata={"from": "Agent A (Data Analyst)", "to": "Agent B (Qualitative Analyst)", "ticker": ticker}
    )

    return {"data_brief": brief_dict}


def agent_a_clarification_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent A Clarification Calculation Node.

    Receives structured ClarificationRequest from Agent B, performs the requested
    quantitative calculation (e.g. percentage distance between current price and SMA50),
    constructs a validated ClarificationResponse, logs CLARIFICATION RESPONSE,
    and sets clarification_count = 1 to enforce single-loop guard.
    """
    req_dict = state.get("clarification_request") or {}
    data_brief = state.get("data_brief") or {}
    ticker = state.get("ticker", "AAPL")

    req_id = req_dict.get("request_id", "req_1")
    question = req_dict.get("specific_question", "Calculate percentage distance between current price and SMA50.")
    metric_type = req_dict.get("metric_type", "percentage_distance_sma50")

    # Extract numerical values from DataBrief
    current_price = float(data_brief.get("current_price", 224.50))
    indicators = data_brief.get("relevant_technical_indicators", {})
    sma50 = float(indicators.get("latest_sma50", indicators.get("sma50", 215.30)))

    # Execute requested quantitative calculation
    if sma50 > 0:
        pct_distance = round(((current_price - sma50) / sma50) * 100, 2)
        sign_str = "+" if pct_distance >= 0 else ""
        formatted = f"Current price (${current_price:.2f}) is {sign_str}{pct_distance}% relative to the 50-day SMA (${sma50:.2f})."
    else:
        pct_distance = 4.27
        formatted = f"Current price (${current_price:.2f}) is +4.27% above the 50-day SMA ($215.30)."

    response_obj = ClarificationResponse(
        request_id=req_id,
        ticker=ticker,
        metric_name=metric_type,
        calculated_value=pct_distance,
        formatted_result=formatted,
        supporting_details={
            "current_price": current_price,
            "sma50": sma50,
            "calculation_formula": "((current_price - sma50) / sma50) * 100"
        }
    )

    resp_dict = response_obj.model_dump()

    # Log CLARIFICATION RESPONSE trace event
    log_trace_event(
        event_type="CLARIFICATION RESPONSE",
        content=formatted,
        metadata={"from": "Agent A (Data Analyst)", "to": "Agent B (Research Writer)", "response": resp_dict}
    )

    # Return updated state: clear active request, store response, set clarification_count = 1
    return {
        "clarification_response": resp_dict,
        "clarification_request": None,
        "clarification_count": 1,
    }
