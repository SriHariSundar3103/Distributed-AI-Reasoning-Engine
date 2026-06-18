"""
Document Fetch Tool

Fetches and extracts text content from web pages.
Uses httpx for async HTTP and BeautifulSoup for HTML parsing.
"""

import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FetchError(Exception):
    """Custom exception for fetch failures."""
    pass


# User agent to avoid blocks
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True
)
def _fetch_url(url: str, timeout: int) -> str:
    """Fetch URL content with retry logic."""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(
            url,
            headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        return response.text


def _extract_text(html: str, max_length: int = 8000) -> str:
    """
    Extract clean text from HTML content.
    
    Args:
        html: Raw HTML string
        max_length: Maximum characters to return (to limit token usage)
        
    Returns:
        Cleaned text content
    """
    soup = BeautifulSoup(html, "lxml")
    
    # Remove script, style, nav, footer elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        element.decompose()
    
    # Get text
    text = soup.get_text(separator="\n", strip=True)
    
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    
    # Truncate to limit tokens
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"
    
    return text


@tool
def fetch_document(url: str) -> dict[str, str]:
    """
    Fetch and extract text content from a URL.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        Dictionary with url, title, and extracted content
    """
    settings = get_settings()
    logger.info("Fetching document", url=url)
    
    try:
        html = _fetch_url(url, timeout=settings.http_timeout)
        
        # Extract title
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"
        
        # Extract main content
        content = _extract_text(html)
        
        logger.info("Document fetched", url=url, content_length=len(content))
        
        return {
            "url": url,
            "title": title,
            "content": content
        }
        
    except httpx.HTTPStatusError as e:
        logger.warning("HTTP error fetching document", url=url, status=e.response.status_code)
        raise FetchError(f"HTTP {e.response.status_code}: {url}") from e
        
    except httpx.TimeoutException:
        logger.warning("Timeout fetching document", url=url)
        raise FetchError(f"Timeout fetching: {url}")
        
    except Exception as e:
        logger.error("Failed to fetch document", url=url, error=str(e))
        raise FetchError(f"Failed to fetch {url}: {str(e)}") from e


def fetch_multiple_urls(urls: list[str]) -> list[dict[str, str]]:
    """
    Fetch multiple URLs, continuing on individual failures.
    
    Args:
        urls: List of URLs to fetch
        
    Returns:
        List of successfully fetched documents
    """
    documents = []
    
    for url in urls:
        try:
            doc = fetch_document.invoke(url)
            documents.append(doc)
        except FetchError as e:
            logger.warning("Skipping failed URL", url=url, error=str(e))
            continue
    
    return documents
