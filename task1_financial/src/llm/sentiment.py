"""
Per-headline financial news sentiment analysis using LLM and structured Pydantic validation.

Orchestrates sentiment extraction, schema validation, fallback handling,
and confidence-weighted sentiment score aggregation.
Supports single-call batch analysis to evaluate all headlines in a single LLM request.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Union

from src.llm.client import LLMClient, LLMClientError
from src.prompts.sentiment import (
    BATCH_SENTIMENT_SYSTEM_PROMPT,
    BATCH_SENTIMENT_USER_PROMPT_TEMPLATE,
    SENTIMENT_SYSTEM_PROMPT,
    SENTIMENT_USER_PROMPT_TEMPLATE,
)
from src.schemas.models import AggregatedSentiment, HeadlineSentiment, OverallSentimentType

logger = logging.getLogger(__name__)


def extract_json_payload(text: str) -> str:
    """
    Extract clean JSON string from raw LLM output text, removing markdown code fences.
    Handles both JSON objects {...} and JSON arrays [...].

    Args:
        text: Raw LLM output text.

    Returns:
        str: Cleaned JSON string.
    """
    cleaned = text.strip()

    # Remove markdown code fences like ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback search for first '[' or '{' and last ']' or '}'
    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")

    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        end_arr = cleaned.rfind("]")
        if end_arr > start_arr:
            return cleaned[start_arr : end_arr + 1]

    if start_obj != -1:
        end_obj = cleaned.rfind("}")
        if end_obj > start_obj:
            return cleaned[start_obj : end_obj + 1]

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

    system_prompt = SENTIMENT_SYSTEM_PROMPT.replace("{ticker}", ticker.upper())
    user_prompt = SENTIMENT_USER_PROMPT_TEMPLATE.format(ticker=ticker.upper(), headline=headline)

    try:
        raw_response = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)
        json_str = extract_json_payload(raw_response)
        parsed_dict = json.loads(json_str)

        if not isinstance(parsed_dict, dict):
            raise ValueError("Parsed JSON payload is not a dictionary.")

        parsed_dict["headline"] = headline.strip()
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
    score = max(-1.0, min(1.0, score))

    return round(score, 4)


def analyze_batch_sentiment(
    headlines: List[Union[Dict[str, str], str]],
    ticker: str,
    client: Optional[LLMClient] = None
) -> AggregatedSentiment:
    """
    Analyze a batch of news headlines in a SINGLE LLM API call and return an aggregated sentiment model.

    Optimizes performance and token budget by sending all headlines in one prompt.

    Args:
        headlines: List of normalized headline dicts or strings.
        ticker: Target stock ticker.
        client: Optional LLMClient instance.

    Returns:
        AggregatedSentiment: Aggregated sentiment summary model containing validated results.
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

    # Extract headline text strings
    headline_texts: List[str] = []
    for item in headlines:
        if isinstance(item, dict):
            h_text = item.get("headline", "")
        else:
            h_text = str(item)
        if h_text and h_text.strip():
            headline_texts.append(h_text.strip())

    if not headline_texts:
        return AggregatedSentiment(
            total_headlines=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            weighted_sentiment_score=0.0,
            overall_label="NEUTRAL",
            headline_results=[]
        )

    llm_client = client if client is not None else LLMClient()

    # Format all headlines into a single prompt string
    formatted_headlines_list = "\n".join([f"{idx+1}. \"{h}\"" for idx, h in enumerate(headline_texts)])
    system_prompt = BATCH_SENTIMENT_SYSTEM_PROMPT.replace("{ticker}", ticker.upper())
    user_prompt = BATCH_SENTIMENT_USER_PROMPT_TEMPLATE.format(
        ticker=ticker.upper(),
        headlines_formatted=formatted_headlines_list
    )

    batch_parsed_map: Dict[str, dict] = {}
    try:
        raw_response = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)
        json_str = extract_json_payload(raw_response)
        parsed_array = json.loads(json_str)

        if isinstance(parsed_array, list):
            for item in parsed_array:
                if isinstance(item, dict) and "headline" in item:
                    h_key = str(item["headline"]).strip().lower()
                    batch_parsed_map[h_key] = item

    except (LLMClientError, json.JSONDecodeError, ValueError, Exception) as exc:
        logger.warning(f"Batch LLM sentiment analysis failed: {exc}. Falling back to per-item processing.")

    # Validate each headline against batch response or fallback
    headline_results: List[HeadlineSentiment] = []
    pos_count = 0
    neg_count = 0
    neu_count = 0

    for h_text in headline_texts:
        h_key = h_text.lower()
        validated: Optional[HeadlineSentiment] = None

        if h_key in batch_parsed_map:
            raw_item = batch_parsed_map[h_key]
            raw_item["headline"] = h_text
            try:
                validated = HeadlineSentiment.model_validate(raw_item)
            except Exception as val_exc:
                logger.warning(f"Validation failed for batch item '{h_text[:20]}': {val_exc}")

        if validated is None:
            # Fallback to neutral if batch item missing or invalid
            validated = create_fallback_headline_sentiment(
                headline=h_text,
                reason="Fallback neutral model generated for headline."
            )

        headline_results.append(validated)

        if validated.sentiment == "positive":
            pos_count += 1
        elif validated.sentiment == "negative":
            neg_count += 1
        else:
            neu_count += 1

    weighted_score = calculate_weighted_sentiment_score(headline_results)

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
