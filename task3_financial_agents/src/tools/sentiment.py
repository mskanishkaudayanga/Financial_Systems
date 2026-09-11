# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Convert llm_sentiment into a LangChain @tool with LLMSentimentArgs Pydantic input schema and explicit LLM tool description', Date: 2026-09-11
"""
LLM News Sentiment Tool.

Performs qualitative sentiment analysis across financial headlines using an LLM,
returning structured score (-1.0 to 1.0), label (Bullish/Bearish/Neutral), confidence,
and reasoning with defensive fallback handling.
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from src.config import config
from src.schemas.tools_schemas import SentimentResult, SentimentOutput, LLMSentimentArgs


@tool(args_schema=LLMSentimentArgs)
def llm_sentiment(headlines: List[str]) -> Dict[str, Any]:
    """Perform qualitative LLM sentiment analysis on a list of financial news headlines. Returns a sentiment score between -1.0 (bearish) and +1.0 (bullish), qualitative label ('Bullish', 'Bearish', 'Neutral'), confidence score, and rationale. Use this tool after retrieving news headlines to assess net market sentiment."""
    # 1. Validate inputs
    if not headlines or not isinstance(headlines, list):
        return SentimentOutput(
            status="error",
            headlines_count=0,
            sentiment=SentimentResult(
                score=0.0,
                label="Neutral",
                confidence=0.0,
                reasoning="Empty or invalid headlines list provided."
            ),
            error="Headlines input must be a non-empty list of strings."
        ).model_dump()

    cleaned_headlines = [str(h).strip() for h in headlines if h and str(h).strip()]

    if not cleaned_headlines:
        return SentimentOutput(
            status="error",
            headlines_count=0,
            sentiment=SentimentResult(
                score=0.0,
                label="Neutral",
                confidence=0.0,
                reasoning="No valid text headlines remaining after whitespace cleaning."
            ),
            error="Headlines list contained no valid non-empty text strings."
        ).model_dump()

    # 2. Check for configured LLM API Key
    if not config.LLM_API_KEY or config.LLM_API_KEY.strip() == "" or config.LLM_API_KEY == "your_api_key_here":
        return SentimentOutput(
            status="fallback",
            headlines_count=len(cleaned_headlines),
            sentiment=SentimentResult(
                score=0.0,
                label="Neutral",
                confidence=0.0,
                reasoning="LLM API key not configured in environment; returning neutral fallback sentiment."
            ),
            error="Missing or unconfigured LLM_API_KEY."
        ).model_dump()

    # 3. Call LLM with Structured Output
    try:
        llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            openai_api_key=config.LLM_API_KEY,
            openai_api_base=config.LLM_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            max_retries=1,
            timeout=15.0
        )

        structured_llm = llm.with_structured_output(SentimentResult)

        system_prompt = SystemMessage(
            content=(
                "You are an expert financial sentiment analyst. "
                "Analyze the provided news headlines for a company or market asset. "
                "Output a net sentiment score between -1.0 (extremely bearish) and +1.0 (extremely bullish), "
                "a qualitative label ('Bullish', 'Bearish', or 'Neutral'), "
                "a confidence rating between 0.0 and 1.0, and a concise 1-2 sentence rationale."
            )
        )

        formatted_headlines = "\n".join([f"- {h}" for h in cleaned_headlines])
        user_prompt = HumanMessage(
            content=f"Analyze financial sentiment for the following news headlines:\n{formatted_headlines}"
        )

        result: SentimentResult = structured_llm.invoke([system_prompt, user_prompt])

        return SentimentOutput(
            status="success",
            headlines_count=len(cleaned_headlines),
            sentiment=result
        ).model_dump()

    except Exception as exc:
        # Catch network, quota, rate-limit, or parsing errors gracefully
        return SentimentOutput(
            status="fallback",
            headlines_count=len(cleaned_headlines),
            sentiment=SentimentResult(
                score=0.0,
                label="Neutral",
                confidence=0.0,
                reasoning=f"LLM sentiment call failed due to runtime exception: {str(exc)}"
            ),
            error=str(exc)
        ).model_dump()
