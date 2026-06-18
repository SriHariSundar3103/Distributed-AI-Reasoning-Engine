"""
Web Search Tool

Integrates with Tavily API for AI-optimized web search.
Includes rate limiting, retry logic, and result structuring.
"""

from typing import Any

from langchain_core.tools import tool
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchError(Exception):
    """Custom exception for search failures."""
    pass


def get_tavily_client() -> TavilyClient:
    """Get configured Tavily client."""
    settings = get_settings()
    return TavilyClient(api_key=settings.tavily_api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True
)
def _execute_search(client: TavilyClient, query: str, max_results: int) -> list[dict[str, Any]]:
    """Execute search with retry logic."""
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=False
        )
        return response.get("results", [])
    except Exception as e:
        logger.error("Search API error", query=query, error=str(e))
        raise SearchError(f"Search failed: {str(e)}") from e


@tool
def web_search(query: str) -> list[dict[str, Any]]:
    """
    Search the web for information about a topic.
    
    Args:
        query: The search query string
        
    Returns:
        List of search results with url, title, and content snippet
    """
    settings = get_settings()
    logger.info("Executing web search", query=query)
    
    try:
        client = get_tavily_client()
        raw_results = _execute_search(
            client=client,
            query=query,
            max_results=settings.max_search_results
        )
        
        # Structure results consistently
        results = []
        for item in raw_results:
            results.append({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "score": item.get("score", 0.0),
            })
        
        logger.info("Search completed", query=query, result_count=len(results))
        return results
        
    except SearchError:
        raise
    except Exception as e:
        logger.error("Unexpected search error", query=query, error=str(e))
        raise SearchError(f"Search failed unexpectedly: {str(e)}") from e


def search_multiple_queries(queries: list[str]) -> dict[str, list[dict[str, Any]]]:
    """
    Execute multiple search queries and aggregate results.
    
    Args:
        queries: List of search query strings
        
    Returns:
        Dictionary mapping query to its results
    """
    all_results = {}
    
    for query in queries:
        try:
            results = web_search.invoke(query)
            all_results[query] = results
        except SearchError as e:
            logger.warning("Query failed, continuing", query=query, error=str(e))
            all_results[query] = []
    
    return all_results
