"""
Pydantic data models for API responses, sentiments, and recommendations.
"""
from pydantic import BaseModel
from typing import Optional, List

class SentimentResponse(BaseModel):
    score: float
    reasoning: str
    confidence: float

class RecommendationResponse(BaseModel):
    action: str  # BUY, SELL, HOLD
    summary: str
