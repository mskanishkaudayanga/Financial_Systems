"""
Prompt templates for financial sentiment analysis.
"""

SENTIMENT_PROMPT_TEMPLATE = """
Analyze the sentiment of the following financial news item for stock {ticker}:

"{headline}"

Provide a score from -1.0 (extremely negative) to +1.0 (extremely positive) and explain your reasoning.
"""
