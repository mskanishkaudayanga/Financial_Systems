# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update src/schemas/agent_schemas.py with ObservationRecord model and observations channel in AgentState for Observe-Replan-Act cycle', Date: 2026-09-11
"""
Agent State and Research Report Schemas.

Defines the explicit LangGraph state container (AgentState) and Pydantic schemas
for structured final report generation (FinancialReportSchema).
"""

from typing import List, Optional, Sequence, TypedDict, Annotated, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def add_observations(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reducer function to append new observation records to state."""
    if not left:
        return list(right)
    if not right:
        return list(left)
    return list(left) + list(right)


class ObservationRecord(BaseModel):
    """Structured observation item recorded after tool execution."""

    tool_name: str = Field(description="Name of the executed tool")
    status: str = Field(description="Execution status ('success', 'error', 'fallback')")
    summary: str = Field(description="Concise 1-sentence observation summary")
    has_error: bool = Field(default=False, description="True if tool failed or returned 0 items")


class AgentState(TypedDict):
    """
    Explicit Graph State for the Autonomous Financial Research Agent.

    Attributes:
        messages: Accumulated sequence of conversation and tool execution messages.
        ticker: Equity ticker symbol under research (e.g. 'AAPL').
        research_question: Full user research prompt.
        observations: Accumulated sequence of structured tool observation summaries.
        final_report: Markdown or structured report string produced by synthesis.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    ticker: str
    research_question: str
    observations: Annotated[List[Dict[str, Any]], add_observations]
    final_report: Optional[str]


# ---------------------------------------------------------------------------
# Structured Report Schemas
# ---------------------------------------------------------------------------

class RiskItem(BaseModel):
    """Structured representation of a single equity risk factor."""

    risk_title: str = Field(
        description="Concise title of the risk factor (e.g. 'Margin Compression from Semiconductor Supply Delays')"
    )
    description: str = Field(
        description="Detailed explanation of the risk mechanism over the next 90 days"
    )
    supporting_evidence: List[str] = Field(
        description="Bullet points of concrete empirical evidence gathered from tools (price, volatility, news, sentiment, search)"
    )


class FinancialReportSchema(BaseModel):
    """Structured output schema for the comprehensive financial research report."""

    ticker: str = Field(description="Equity ticker symbol analyzed")
    financial_health_summary: str = Field(
        description="Comprehensive summary of financial health, price trend momentum (SMA20/50, RSI14), and historical volatility"
    )
    top_three_risks: List[RiskItem] = Field(
        description="Top three distinct risks to share price over the next 90 days with supporting evidence"
    )
    hedge_strategy_recommendation: str = Field(
        description="Detailed, data-driven hedge strategy recommendation (e.g. collar options, protective puts, inverse ETF allocation)"
    )
