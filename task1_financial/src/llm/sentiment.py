"""
Per-headline financial news sentiment analysis using LLM and structured Pydantic validation.

Orchestrates sentiment extraction, schema validation, fallback handling,
and confidence-weighted sentiment score aggregation.
"""

import json
import logging
import re
from typing import Dict, List, Optional

from src.llm.client import LLMClient, LLMClientError
from src.prompts.sentiment import SENTIMENT_SYSTEM_PROMPT, SENTIMENT_USER_PROMPT_TEMPLATE
from src.schemas.models import AggregatedSentiment, HeadlineSentiment, OverallSentimentType

logger = logging.getLogger(__name__)


def extract_json_payload(text: str) -> str:
    """
    Extract clean JSON string from raw LLM output text, removing markdown code fences.

    Args:
        text: Raw LLM output text.

    Returns:
        str: Cleaned JSON string.
    """
    cleaned = text.strip()

    # Remove markdown code fences like ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback search for first '{' and last '}'
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def create_fallback_headline_sentiment(headline: str, reason: str = "Analysis unavailable due to LLM error.") -> HeadlineSentiment:
    """
    Construct a safe default fallback HeadlineSentiment object on failure.

    Args:
        headline: News headline text.
        reason: Explanation of failure or fallback.

    Returns:
        HeadlineSentiment: Neutral fallback object with 0.0 confidence.
    """
    clean_h = headline if (headline and headline.strip()) else "Unknown Headline"
    return HeadlineSentiment(
        headline=clean_h.strip(),
        sentiment="neutral",
        confidence=0.0,
        brief_reason=reason
    )


def analyze_headline_sentiment(
    headline: str,
    ticker: str,
    client: Optional[LLMClient] = None
) -> HeadlineSentiment:
    """
    Analyze sentiment for a single news headline using LLM with strict Pydantic validation.

    Args:
        headline: Financial news headline text.
        ticker: Target stock ticker symbol.
        client: Optional LLMClient instance (instantiates default if None).

    Returns:
        HeadlineSentiment: Validated sentiment object (or fallback on error).
    """
    if not headline or not isinstance(headline, str) or not headline.strip():
        return create_fallback_headline_sentiment("Invalid Headline", "Headline text is empty or invalid.")

    llm_client = client if client is not None else LLMClient()

    system_prompt = SENTIMENT_SYSTEM_PROMPT.format(ticker=ticker.upper())
    user_prompt = SENTIMENT_USER_PROMPT_TEMPLATE.format(ticker=ticker.upper(), headline=headline)

    try:
        raw_response = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)
        json_str = extract_json_payload(raw_response)
        parsed_dict = json.loads(json_str)

        if not isinstance(parsed_dict, dict):
            raise ValueError("Parsed JSON payload is not a dictionary.")

        # Ensure headline key is set to input headline
        parsed_dict["headline"] = headline.strip()

        # Validate against Pydantic schema
        validated = HeadlineSentiment.model_validate(parsed_dict)
        return validated

    except (LLMClientError, json.JSONDecodeError, ValueError, Exception) as exc:
        logger.warning(f"Sentiment analysis failed for headline '{headline[:30]}...': {exc}")
        return create_fallback_headline_sentiment(
            headline=headline,
            reason=f"Analysis fallback due to processing error: {exc}"
        )


def calculate_weighted_sentiment_score(results: List[HeadlineSentiment]) -> float:
    """
    Calculate confidence-weighted sentiment score across a list of headline results.

    Mathematical Definition:
        Weighted_Score = sum(Sentiment_Value_i * Confidence_i) / sum(Confidence_i)

        where Sentiment_Value:
            "positive" -> +1.0
            "negative" -> -1.0
            "neutral"  -> 0.0

    Args:
        results: List of validated HeadlineSentiment objects.

    Returns:
        float: Bounded score between -1.0 and +1.0 (rounded to 4 decimal places).
    """
    if not results:
        return 0.0

    value_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    total_weighted_val = 0.0
    total_confidence = 0.0

    for item in results:
        val = value_map.get(item.sentiment, 0.0)
        total_weighted_val += val * item.confidence
        total_confidence += item.confidence

    if total_confidence == 0.0:
        return 0.0

    score = total_weighted_val / total_confidence
    # Clamp score to [-1.0, 1.0] for safety
    score = max(-1.0, min(1.0, score))

    return round(score, 4)


def analyze_batch_sentiment(
    headlines: List[Dict[str, str]],
    ticker: str,
    client: Optional[LLMClient] = None
) -> AggregatedSentiment:
    """
    Analyze a batch of news headlines and produce an aggregated sentiment model.

    Args:
        headlines: List of normalized headline dictionaries (containing 'headline').
        ticker: Target stock ticker.
        client: Optional LLMClient instance.

    Returns:
        AggregatedSentiment: Aggregated sentiment summary model.
    """
    if not headlines:
        return AggregatedSentiment(
            total_headlines=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            weighted_sentiment_score=0.0,
            overall_label="NEUTRAL",
            headline_results=[]
        )

    headline_results: List[HeadlineSentiment] = []
    pos_count = 0
    neg_count = 0
    neu_count = 0

    for item in headlines:
        h_text = item.get("headline", "") if isinstance(item, dict) else str(item)
        res = analyze_headline_sentiment(headline=h_text, ticker=ticker, client=client)
        headline_results.append(res)

        if res.sentiment == "positive":
            pos_count += 1
        elif res.sentiment == "negative":
            neg_count += 1
        else:
            neu_count += 1

    weighted_score = calculate_weighted_sentiment_score(headline_results)

    # Classify overall sentiment label from weighted score threshold
    if weighted_score > 0.15:
        overall: OverallSentimentType = "POSITIVE"
    elif weighted_score < -0.15:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"

    return AggregatedSentiment(
        total_headlines=len(headline_results),
        positive_count=pos_count,
        negative_count=neg_count,
        neutral_count=neu_count,
        weighted_sentiment_score=weighted_score,
        overall_label=overall,
        headline_results=headline_results,
    )
