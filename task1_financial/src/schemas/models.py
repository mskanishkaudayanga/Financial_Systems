"""
Pydantic data models for structured output validation.

Defines strict validation schemas for headline sentiment outputs and aggregated sentiment analysis.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator

# Allowed sentiment classifications
SentimentType = Literal["positive", "negative", "neutral"]
OverallSentimentType = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]


class HeadlineSentiment(BaseModel):
    """
    Validated sentiment model for a single financial news headline.
    """

    headline: str = Field(..., min_length=1, description="Financial news headline text.")
    sentiment: SentimentType = Field(..., description="Sentiment classification: positive, negative, or neutral.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score bounded between 0.0 and 1.0.")
    brief_reason: str = Field(..., min_length=1, description="Brief justification for the sentiment classification.")

    @field_validator("headline", "brief_reason")
    @classmethod
    def check_non_empty_string(cls, val: str) -> str:
        """Ensure string fields contain non-whitespace characters."""
        if not val or not val.strip():
            raise ValueError("String field cannot be empty or whitespace only.")
        return val.strip()

    @field_validator("sentiment")
    @classmethod
    def normalize_sentiment_label(cls, val: str) -> str:
        """Normalize sentiment string to lowercase."""
        cleaned = str(val).strip().lower()
        if cleaned not in ("positive", "negative", "neutral"):
            raise ValueError("Sentiment must be 'positive', 'negative', or 'neutral'.")
        return cleaned


class AggregatedSentiment(BaseModel):
    """
    Aggregated sentiment summary model computed over a collection of headlines.
    """

    total_headlines: int = Field(..., ge=0, description="Total number of evaluated headlines.")
    positive_count: int = Field(..., ge=0, description="Number of positive headlines.")
    negative_count: int = Field(..., ge=0, description="Number of negative headlines.")
    neutral_count: int = Field(..., ge=0, description="Number of neutral headlines.")
    weighted_sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Confidence-weighted aggregate sentiment score in [-1.0, 1.0]."
    )
    overall_label: OverallSentimentType = Field(..., description="Overall aggregate sentiment classification.")
    headline_results: List[HeadlineSentiment] = Field(
        default_factory=list, description="List of validated individual headline sentiment results."
    )
