# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update research_agent.py implementing Observe-Replan-Act state transitions, observation record generation, and graceful failure recovery', Date: 2026-09-11
"""
Autonomous Financial Research Agent Nodes.

Implements single agent node and custom tool execution node with explicit
Observe -> Replan -> Act state loop, observation record appending, and graceful failure recovery.
"""

import json
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from src.config import config
from src.tools import (
    get_price_data,
    get_news,
    calculate_volatility,
    llm_sentiment,
    web_search,
)
from src.schemas.agent_schemas import AgentState, ObservationRecord
from src.observability import log_trace_event

# Bind all 5 research tools
TOOLS = [get_price_data, get_news, calculate_volatility, llm_sentiment, web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

RESEARCH_AGENT_SYSTEM_PROMPT = """You are a Senior Financial Equity Research Analyst Agent.
Your task is to conduct structured equity research on a ticker to answer:
"Analyse the current financial health and market sentiment of [TICKER]. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy."

You have access to 5 specialized tools:
1. `get_price_data`: Returns OHLCV price trends, SMA20, SMA50, EMA20, RSI14, daily returns.
2. `get_news`: Retrieves recent financial news articles and headlines.
3. `calculate_volatility`: Computes 60 to 252-day annualized historical volatility.
4. `llm_sentiment`: Performs qualitative LLM sentiment analysis on news headlines.
5. `web_search`: Searches DuckDuckGo for analyst price targets or market commentary.

OBSERVE -> REPLAN -> ACT RULES:
- After every tool execution, you must inspect the returned observation in state history.
- **GRACEFUL RECOVERY**: If a tool returns `status="error"` or `count=0` (empty results), observe that failure and immediately REPLAN by selecting an alternative tool or proceeding with available metrics.
- Keep your decisions state-driven, professional, and concise (1 sentence). Do NOT output raw chain-of-thought.
- Synthesize a final research report containing:
  1. Financial Health & Price Trend Summary
  2. Top Three Share Price Risks (with explicit supporting evidence for each)
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


def _format_observation_summary(tool_name: str, result: Dict[str, Any]) -> str:
    """Format a clean, 1-sentence state observation summary from tool results."""
    status = result.get("status", "success")

    if status == "error":
        err_msg = result.get("error", "Unknown error")
        return f"Tool {tool_name} returned error: {err_msg}. Alternative fallback required."

    if tool_name == "get_price_data":
        indicators = result.get("latest_indicators") or {}
        close = indicators.get("latest_close", "N/A")
        rsi = indicators.get("latest_rsi14", "N/A")
        sma20 = indicators.get("latest_sma20", "N/A")
        return f"Observed OHLCV price trend for {result.get('ticker')}: Latest Close=${close}, SMA20=${sma20}, RSI14={rsi}."

    elif tool_name == "calculate_volatility":
        vol = result.get("annualized_volatility")
        days = result.get("trading_days_used", 0)
        return f"Observed historical volatility for {result.get('ticker')}: {vol}% annualized across {days} trading sessions."

    elif tool_name == "get_news":
        count = result.get("count", 0)
        if count == 0:
            return f"Observed 0 news headlines for {result.get('ticker')}. Graceful fallback to web search or technical metrics."
        return f"Observed {count} recent financial news headlines for {result.get('ticker')}."

    elif tool_name == "llm_sentiment":
        sent = result.get("sentiment") or {}
        label = sent.get("label", "Neutral")
        score = sent.get("score", 0.0)
        return f"Observed Qualitative Headline Sentiment: {label} (Score: {score:+.2f})."

    elif tool_name == "web_search":
        count = result.get("count", 0)
        if count == 0:
            return f"Web search returned 0 analyst commentary items for query. Fallback to existing news headlines."
        return f"Observed {count} web search intelligence results for analyst targets."

    return f"Observed {tool_name} execution output (status: {status})."


def _generate_replan_decision(response: AIMessage, observations: List[Dict[str, Any]], ticker: str) -> str:
    """Generate a clean 1-sentence decision summary incorporating recent observations."""
    if not response.tool_calls:
        return f"Sufficient observations collected across quantitative and news metrics; synthesizing final report for {ticker}."

    tool_names = [tc["name"] for tc in response.tool_calls]

    # Check if last observation was a failure requiring graceful recovery
    last_obs = observations[-1] if observations else {}
    if last_obs.get("has_error"):
        failed_tool = last_obs.get("tool_name", "Tool")
        return f"Observed failure/empty output from {failed_tool}; replanning alternative strategy by calling {', '.join(tool_names)}."

    if len(tool_names) > 1:
        return f"Observed state; replanning parallel data collection: invoking {', '.join(tool_names)} simultaneously."

    target_tool = tool_names[0]
    if target_tool == "get_price_data":
        return f"Evaluating market foundation; retrieving technical price indicators for {ticker}."
    elif target_tool == "calculate_volatility":
        return f"Price trend observed; re-planning to compute annualized return volatility for {ticker}."
    elif target_tool == "get_news" or target_tool == "llm_sentiment":
        return f"Price metrics observed; retrieving recent news and analyzing qualitative headline sentiment for {ticker}."
    elif target_tool == "web_search":
        return f"News sentiment observed; searching web intelligence for analyst commentary on {ticker}."

    return f"Re-planning research path: selecting tool(s) {', '.join(tool_names)} based on updated state."


def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM Agent Node (Replan & Act).

    Evaluates current state history and observations, logs new decision, and invokes LLM.
    """
    messages = list(state["messages"])
    ticker = state.get("ticker", "Equity")
    observations = state.get("observations") or []

    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=RESEARCH_AGENT_SYSTEM_PROMPT))

    llm = _get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    log_trace_event(
        event_type="AGENT",
        content=f"Inspecting updated state for {ticker} ({len(messages)} messages, {len(observations)} observations).",
        metadata={"ticker": ticker, "message_count": len(messages), "obs_count": len(observations)}
    )

    response: AIMessage = llm_with_tools.invoke(messages)

    decision_summary = _generate_replan_decision(response, observations, ticker)

    log_trace_event(
        event_type="AGENT DECISION",
        content=decision_summary,
        metadata={"tool_calls_count": len(response.tool_calls) if response.tool_calls else 0}
    )

    return {"messages": [response]}


def execute_tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Custom Tools Node (Observe).

    Executes requested tool_calls, records structured observation summaries in state,
    and logs TOOL CALL, TOOL RESULT, and UPDATED OBSERVATION trace events.
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

        log_trace_event(
            event_type="TOOL CALL",
            content=f"Invoking {tool_name} with arguments: {json.dumps(tool_args)}",
            metadata={"tool_name": tool_name, "args": tool_args, "call_id": call_id}
        )

        tool_obj = TOOLS_BY_NAME.get(tool_name)
        if tool_obj:
            try:
                result = tool_obj.invoke(tool_args)
                status = result.get("status", "success") if isinstance(result, dict) else "success"
                summary_str = json.dumps(result, default=str)
            except Exception as exc:
                status = "error"
                result = {"status": "error", "error": f"Tool execution failed: {str(exc)}"}
                summary_str = json.dumps(result)
        else:
            status = "error"
            result = {"status": "error", "error": f"Tool '{tool_name}' not found."}
            summary_str = json.dumps(result)

        # Log TOOL RESULT event
        log_trace_event(
            event_type="TOOL RESULT",
            content=summary_str,
            metadata={"tool_name": tool_name, "status": status, "call_id": call_id}
        )

        # Formulate observation summary
        obs_summary = _format_observation_summary(tool_name, result if isinstance(result, dict) else {})
        has_error = (status == "error") or (isinstance(result, dict) and result.get("count", 1) == 0)

        # Log UPDATED OBSERVATION event
        log_trace_event(
            event_type="UPDATED OBSERVATION",
            content=obs_summary,
            metadata={"tool_name": tool_name, "status": status, "has_error": has_error}
        )

        new_observations.append({
            "tool_name": tool_name,
            "status": status,
            "summary": obs_summary,
            "has_error": has_error
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
