# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Convert web_search into a LangChain @tool with WebSearchArgs Pydantic input schema and explicit LLM tool description', Date: 2026-09-11
"""
Web Search Tool.

Performs web search queries via DuckDuckGo Search (DDGS) to retrieve analyst commentary,
SEC filings, or market intelligence, returning normalized structured outputs.
"""

from typing import Dict, Any, List
from duckduckgo_search import DDGS
from langchain_core.tools import tool

from src.schemas.tools_schemas import WebSearchOutput, SearchResultItem, WebSearchArgs


@tool(args_schema=WebSearchArgs)
def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web via DuckDuckGo for financial news, analyst price targets, market commentary, or company intelligence. Use this tool when official market news or price metrics are insufficient and broader web research is required."""
    # 1. Validate inputs
    if not query or not isinstance(query, str) or not query.strip():
        return WebSearchOutput(
            status="error",
            query=str(query),
            count=0,
            error="Search query must be a non-empty string."
        ).model_dump()

    cleaned_query = query.strip()

    if not isinstance(max_results, int) or max_results <= 0:
        return WebSearchOutput(
            status="error",
            query=cleaned_query,
            count=0,
            error="Parameter 'max_results' must be a positive integer."
        ).model_dump()

    # 2. Execute search via DDGS
    try:
        results_list: List[SearchResultItem] = []
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(cleaned_query, max_results=max_results))

        if raw_results and isinstance(raw_results, list):
            for item in raw_results:
                if not isinstance(item, dict):
                    continue

                title = item.get("title") or "No Title"
                snippet = item.get("body") or item.get("snippet") or title
                url = item.get("href") or item.get("url") or f"https://duckduckgo.com/?q={cleaned_query}"

                results_list.append(
                    SearchResultItem(
                        title=str(title).strip(),
                        snippet=str(snippet).strip(),
                        url=str(url).strip()
                    )
                )

        return WebSearchOutput(
            status="success",
            query=cleaned_query,
            count=len(results_list),
            results=results_list
        ).model_dump()

    except Exception as exc:
        return WebSearchOutput(
            status="error",
            query=cleaned_query,
            count=0,
            error=f"Web search failed due to exception: {str(exc)}"
        ).model_dump()
