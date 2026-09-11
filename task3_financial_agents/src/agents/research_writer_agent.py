# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create src/agents/research_writer_agent.py for Agent B Research Writer with restricted tools [get_news, web_search] and DataBrief context integration', Date: 2026-09-11
"""
Agent B: Qualitative Research Writer Agent.

Specialized sub-agent responsible for qualitative financial research, headline parsing,
analyst commentary retrieval via web search, and synthesizing the final equity research report.
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
from src.schemas.agent_schemas import AgentState
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
- You have access ONLY to 2 qualitative tools:
  1. `get_news`: Retrieves recent financial news articles and headlines.
  2. `web_search`: Searches DuckDuckGo for analyst price targets, earnings guidance, or commentary.
- You do NOT have access to `get_price_data` or `calculate_volatility`. Do NOT attempt to call price data or volatility tools directly.

INPUT CONTEXT:
- You receive Agent A's structured DataBrief containing validated quantitative metrics (current price, moving averages, RSI14, volatility, sentiment score).

WORKFLOW & SYNTHESIS RULES:
- Inspect Agent A's DataBrief in graph state.
- Autonomously use `get_news` and `web_search` to gather qualitative context, recent corporate announcements, and analyst risk notes.
- Compare your qualitative findings with Agent A's quantitative findings.
- Synthesize a comprehensive final equity research report containing:
  1. Financial Health & Price Trend Summary (combining DataBrief metrics & qualitative news context)
  2. Top Three Share Price Risks over the next 90 days (with explicit empirical supporting evidence for each)
  3. Data-Driven Hedge Strategy Recommendation (e.g. collar options, protective puts, inverse allocation).
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

    Receives Agent A's DataBrief from state, logs trace, and invokes LLM bound STRICTLY to RESEARCH_WRITER_TOOLS.
    """
    messages = list(state["messages"])
    ticker = state.get("ticker", "Equity")
    data_brief = state.get("data_brief") or {}

    # Ensure system prompt and DataBrief context are present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=RESEARCH_WRITER_SYSTEM_PROMPT))

    # Log receipt of Agent A's DataBrief if first turn for Agent B
    if data_brief and not any("[Agent B Received DataBrief]" in str(m.content) for m in messages):
        brief_summary = (
            f"[Agent B Received DataBrief]\n"
            f"Ticker: {data_brief.get('ticker')}, Current Price: ${data_brief.get('current_price')}, "
            f"Volatility: {data_brief.get('volatility')}%, Sentiment: {data_brief.get('sentiment_label')} ({data_brief.get('sentiment_score')}).\n"
            f"Quantitative Observations: {json.dumps(data_brief.get('quantitative_observations', []))}"
        )
        messages.append(HumanMessage(content=f"Agent A DataBrief Input:\n{brief_summary}"))
        log_trace_event(
            event_type="AGENT",
            content=f"[Agent B: Research Writer] Received Agent A DataBrief payload for {ticker}. Beginning qualitative analysis.",
            metadata={"agent": "ResearchWriter", "ticker": ticker, "data_brief": data_brief}
        )

    llm = _get_llm()
    # Enforce tool restriction by binding ONLY allowed qualitative tools
    llm_with_tools = llm.bind_tools(RESEARCH_WRITER_TOOLS)

    response: AIMessage = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        decision_str = f"[Agent B: Research Writer] Dispatching qualitative tools: {', '.join(tool_names)}."
    else:
        decision_str = f"[Agent B: Research Writer] Qualitative research complete. Synthesizing final equity report."

    log_trace_event(
        event_type="AGENT DECISION",
        content=decision_str,
        metadata={"agent": "ResearchWriter", "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0}
    )

    return {"messages": [response]}


def research_writer_tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent B (Research Writer) Tool Execution Node.

    Executes requested tool calls from RESEARCH_WRITER_TOOLS and logs trace events.
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

        # Security check: Ensure requested tool is allowed for Agent B
        if tool_name not in RESEARCH_WRITER_TOOLS_BY_NAME:
            status = "error"
            summary_str = json.dumps({"status": "error", "error": f"Access Denied: Tool '{tool_name}' is not allowed for Agent B."})
            obs_str = f"Agent B Access Denied: Attempted to call unauthorized quantitative tool '{tool_name}'."
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
