# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create src/agents/research_agent.py implementing autonomous research agent node and custom tool execution node with trace logging', Date: 2026-09-11
"""
Autonomous Financial Research Agent Nodes.

Implements the single agent node and custom tool node for autonomous financial research
with integrated trace logging (AGENT, TOOL CALL, TOOL RESULT, AGENT DECISION).
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
from src.schemas.agent_schemas import AgentState, FinancialReportSchema
from src.observability import log_trace_event

# 1. Bind all 5 research tools
TOOLS = [get_price_data, get_news, calculate_volatility, llm_sentiment, web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

RESEARCH_AGENT_SYSTEM_PROMPT = """You are a Senior Financial Equity Research Analyst Agent.
Your objective is to conduct an end-to-end, data-driven research analysis on an equity ticker to answer:
"Analyse the current financial health and market sentiment of [TICKER]. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy."

You have access to 5 specialized tools:
1. `get_price_data`: Returns OHLCV price trends, SMA20, SMA50, EMA20, RSI14, daily returns.
2. `get_news`: Retrieves recent financial news articles and headlines.
3. `calculate_volatility`: Computes 60 to 252-day annualized historical volatility.
4. `llm_sentiment`: Performs qualitative LLM sentiment analysis on news headlines.
5. `web_search`: Searches DuckDuckGo for analyst price targets, market commentary, or SEC filing notes.

RESEARCH GUIDELINES:
- Do NOT guess data. You MUST use your tools to gather empirical evidence.
- You have total autonomy to decide which tools to call and in what order based on missing information.
- For example, you might fetch price data first, calculate volatility, retrieve news, run headline sentiment analysis, and search for analyst reports.
- Once you have gathered sufficient quantitative and qualitative evidence, present a comprehensive report structured with:
  1. Financial Health & Price Trend Summary
  2. Top Three Share Price Risks (with explicit empirical supporting evidence for each)
  3. Data-Driven Hedge Strategy Recommendation (e.g. collar options, protective put, or inverse allocation)

Be thorough, precise, and objective.
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


def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM Agent Node.

    Evaluates current state messages, logs reasoning, and invokes LLM bound with research tools.
    """
    messages = list(state["messages"])
    ticker = state.get("ticker", "Equity")

    # Ensure system prompt is present at head of message list
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=RESEARCH_AGENT_SYSTEM_PROMPT))

    llm = _get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    # Log AGENT reasoning start
    log_trace_event(
        event_type="AGENT",
        content=f"Evaluating research state for ticker '{ticker}'. Processing {len(messages)} history messages.",
        metadata={"ticker": ticker, "message_count": len(messages)}
    )

    response: AIMessage = llm_with_tools.invoke(messages)

    # Log AGENT DECISION based on output
    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        log_trace_event(
            event_type="AGENT DECISION",
            content=f"Decided to call {len(response.tool_calls)} tool(s): {', '.join(tool_names)}.",
            metadata={"tool_calls": response.tool_calls}
        )
    else:
        log_trace_event(
            event_type="AGENT DECISION",
            content="Gathered sufficient research data. Synthesizing final research report.",
            metadata={"finish_reason": "synthesis"}
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

        # Log TOOL CALL event
        log_trace_event(
            event_type="TOOL CALL",
            content=f"Invoking {tool_name} with parameters: {json.dumps(tool_args)}",
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

        tool_messages.append(
            ToolMessage(
                content=summary_str,
                tool_call_id=call_id,
                name=tool_name
            )
        )

    return {"messages": tool_messages}
