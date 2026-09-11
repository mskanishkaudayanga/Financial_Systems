# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update src/schemas/agent_schemas.py adding ClarificationRequest and ClarificationResponse Pydantic schemas and clarification state fields to AgentState', Date: 2026-09-11
"""
Agent State and Research Report Schemas.

Defines the explicit LangGraph state container (AgentState), Pydantic schemas
for structured report generation (FinancialReportSchema), Agent A DataBrief,
and structured Clarification Request/Response models for inter-agent critique loops.
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


# ---------------------------------------------------------------------------
# Clarification / Critique Loop Schemas
# ---------------------------------------------------------------------------

class ClarificationRequest(BaseModel):
    """Structured Clarification Request issued by Agent B (Research Writer) to Agent A."""

    ticker: str = Field(description="Equity ticker symbol under investigation (e.g. 'AAPL')")
    request_id: str = Field(default="req_1", description="Unique request identifier")
    specific_question: str = Field(
        description="Specific quantitative calculation or metric requested (e.g. 'Calculate the percentage distance between current price and SMA50')"
    )
    metric_type: str = Field(
        description="Category of metric requested (e.g. 'percentage_distance_sma50', 'sma20_sma50_spread', 'volatility_ratio')"
    )


class ClarificationResponse(BaseModel):
    """Structured Clarification Response returned by Agent A (Data Analyst) to Agent B."""

    request_id: str = Field(description="Matching request identifier")
    ticker: str = Field(description="Equity ticker symbol")
    metric_name: str = Field(description="Name of calculated metric")
    calculated_value: float = Field(description="Numerical result of calculation")
    formatted_result: str = Field(description="Human-readable result summary (e.g. 'Current price $224.50 is +4.27% above SMA50 ($215.30)')")
    supporting_details: Dict[str, Any] = Field(description="Raw supporting values used in calculation")


class AgentState(TypedDict):
    """
    Explicit Graph State for Financial Research Agents.

    Attributes:
        messages: Accumulated sequence of conversation and tool execution messages.
        ticker: Equity ticker symbol under research (e.g. 'AAPL').
        research_question: Full user research prompt.
        observations: Accumulated sequence of structured tool observation summaries.
        data_brief: Dedicated structured handoff payload produced by Agent A for Agent B.
        clarification_request: Structured ClarificationRequest issued by Agent B to Agent A.
        clarification_response: Structured ClarificationResponse returned by Agent A to Agent B.
        clarification_count: Counter guard enforcing max 1 clarification loop between agents.
        final_report: Markdown or structured report string produced by synthesis.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    ticker: str
    research_question: str
    observations: Annotated[List[Dict[str, Any]], add_observations]
    data_brief: Optional[Dict[str, Any]]
    clarification_request: Optional[Dict[str, Any]]
    clarification_response: Optional[Dict[str, Any]]
    clarification_count: int
    final_report: Optional[str]


# ---------------------------------------------------------------------------
# Task 3B Agent A: Structured Data Brief Schema
# ---------------------------------------------------------------------------

class DataBrief(BaseModel):
    """
    Structured Quantitative Data Brief produced by Agent A (Data Analyst).
    """

    ticker: str = Field(description="Equity ticker symbol analyzed (e.g. 'AAPL')")
    current_price: float = Field(description="Most recent closing price in USD")
    relevant_technical_indicators: Dict[str, Any] = Field(
        description="Key technical indicators dictionary containing latest_sma20, latest_sma50, latest_ema20, latest_rsi14, latest_daily_return"
    )
    volatility: Optional[float] = Field(
        default=None, description="Annualized historical volatility percentage (e.g. 21.4 for 21.4%)"
    )
    sentiment_score: Optional[float] = Field(
        default=None, description="Quantitative headline sentiment score ranging from -1.0 (bearish) to +1.0 (bullish)"
    )
    sentiment_label: Optional[str] = Field(
        default=None, description="Qualitative sentiment classification label ('Bullish', 'Bearish', 'Neutral')"
    )
    quantitative_observations: List[str] = Field(
        description="Bullet points of key quantitative insights and numerical findings derived from market tools"
    )


# ---------------------------------------------------------------------------
# Task 3A / 3B Structured Report Schemas
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
