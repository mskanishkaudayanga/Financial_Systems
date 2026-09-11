# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Optimize research_agent.py for parallel tool dispatch and strict non-retry budget rules to complete research in 2-3 turns', Date: 2026-09-11
"""
Autonomous Financial Research Agent Nodes.

Implements optimized single agent node and custom tool execution node with parallel
tool dispatch, strict non-retry budget rules, and trace logging.
"""

import json
from typing import Dict, Any, List
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
from src.schemas.agent_schemas import AgentState
from src.observability import log_trace_event

# Bind all 5 research tools
TOOLS = [get_price_data, get_news, calculate_volatility, llm_sentiment, web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

RESEARCH_AGENT_SYSTEM_PROMPT = """You are a Senior Financial Equity Research Analyst Agent.
Your task is to efficiently research an equity ticker to answer:
"Analyse the current financial health and market sentiment of [TICKER]. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy."

You have access to 5 specialized tools:
1. `get_price_data`: Returns OHLCV price trends, SMA20, SMA50, EMA20, RSI14, daily returns.
2. `get_news`: Retrieves recent financial news articles and headlines.
3. `calculate_volatility`: Computes 60 to 252-day annualized historical volatility.
4. `llm_sentiment`: Performs qualitative LLM sentiment analysis on news headlines.
5. `web_search`: Searches DuckDuckGo for analyst price targets or market commentary.

INDUSTRY EFFICIENCY & TOOL EXECUTION RULES:
- **PARALLEL DISPATCH**: In your VERY FIRST TURN, issue parallel tool calls for `get_price_data`, `calculate_volatility`, and `get_news` simultaneously.
- **NO RETRIES**: If `web_search` or `get_news` returns 0 results, DO NOT retry with alternative search queries. Accept the output and proceed immediately.
- **STRICT TURN BUDGET**: Complete all data collection in 1-2 tool turns, then immediately synthesize the final research report.
- Before calling tools, include a 1-sentence decision rationale explaining your parallel tool choices. Do NOT output private chain-of-thought.
- Synthesize a comprehensive final research report containing:
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


def _generate_concise_decision_summary(response: AIMessage, history: List[Any], ticker: str) -> str:
    """Generate a clean, 1-sentence decision summary without leaking private chain-of-thought."""
    if not response.tool_calls:
        return "Sufficient quantitative and qualitative data collected; synthesizing final report."

    tool_names = [tc["name"] for tc in response.tool_calls]

    content_str = str(response.content).strip() if response.content else ""
    if content_str and len(content_str) < 200 and "\n" not in content_str:
        return content_str

    if len(tool_names) > 1:
        return f"Executing parallel research dispatch for {ticker}: calling {', '.join(tool_names)} simultaneously."
    elif "get_price_data" in tool_names:
        return f"Evaluating market foundation; retrieving technical price indicators for {ticker}."
    elif "calculate_volatility" in tool_names:
        return f"Price history available; calculating annualized return volatility to quantify risk."
    elif "get_news" in tool_names or "llm_sentiment" in tool_names:
        return f"Retrieving recent news and analyzing qualitative headline sentiment for {ticker}."
    elif "web_search" in tool_names:
        return f"Executing single web search query for analyst commentary on {ticker}."
    else:
        return f"Calling tool(s) {', '.join(tool_names)} to collect remaining data."


def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM Agent Node.

    Evaluates current state messages, logs reasoning, and invokes LLM bound with research tools.
    """
    messages = list(state["messages"])
    ticker = state.get("ticker", "Equity")

    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=RESEARCH_AGENT_SYSTEM_PROMPT))

    llm = _get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    log_trace_event(
        event_type="AGENT",
        content=f"Inspecting research state for {ticker} ({len(messages)} state messages).",
        metadata={"ticker": ticker, "message_count": len(messages)}
    )

    response: AIMessage = llm_with_tools.invoke(messages)

    decision_summary = _generate_concise_decision_summary(response, messages, ticker)

    log_trace_event(
        event_type="AGENT DECISION",
        content=decision_summary,
        metadata={"tool_calls_count": len(response.tool_calls) if response.tool_calls else 0}
    )

    return {"messages": [response]}


def execute_tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Custom Tools Node with Execution Tracing.

    Executes requested tool_calls from the last AIMessage and logs TOOL CALL & TOOL RESULT events.
    """
    last_message = state["messages"][-1]
    tool_messages: List[ToolMessage] = []

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

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

        log_trace_event(
            event_type="TOOL RESULT",
            content=summary_str,
            metadata={"tool_name": tool_name, "status": status, "call_id": call_id}
        )

        tool_messages.append(
            ToolMessage(
                content=summary_str,
                tool_call_id=call_id,
                name=tool_name
            )
        )

    return {"messages": tool_messages}
