"""
Prompt templates for trading signal recommendations.
"""

RECOMMENDATION_PROMPT_TEMPLATE = """
Based on technical indicators:
{technical_summary}

And news sentiment score:
{sentiment_summary}

Provide a recommendation (BUY, SELL, HOLD) and detailed justification.
"""
