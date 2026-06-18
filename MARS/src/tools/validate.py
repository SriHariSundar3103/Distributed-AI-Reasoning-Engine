"""
Source Validation Tool

Combines heuristic checks (domain reputation, recency, duplication) with
LLM-assisted relevance scoring to balance determinism and semantic judgment.
"""

import json
import re
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Domain reputation tiers for heuristic scoring
DOMAIN_REPUTATION = {
    "high": [".edu", ".gov", ".org", "docs.", "documentation.", "official"],
    "medium": ["github.com", "stackoverflow.com", "medium.com", "dev.to", 
               "arxiv.org", "wikipedia.org", "mdn.", "microsoft.com", 
               "google.com", "aws.amazon.com", "cloud.google.com"],
    "low": ["reddit.com", "quora.com", "blogspot.", "wordpress.com"]
}


def calculate_domain_score(url: str) -> float:
    """
    Calculate credibility score based on domain reputation.
    
    Returns:
        Score between 0.3 and 0.95 based on domain type
    """
    url_lower = url.lower()
    parsed = urlparse(url_lower)
    domain = parsed.netloc
    
    # Check high reputation domains
    for pattern in DOMAIN_REPUTATION["high"]:
        if pattern in domain or pattern in url_lower:
            return 0.9
    
    # Check medium reputation domains
    for pattern in DOMAIN_REPUTATION["medium"]:
        if pattern in domain:
            return 0.7
    
    # Check low reputation domains
    for pattern in DOMAIN_REPUTATION["low"]:
        if pattern in domain:
            return 0.4
    
    # Default for unknown domains
    return 0.5


def check_recency_indicators(content: str) -> float:
    """
    Check for recency indicators in content.
    
    Returns:
        Boost factor (0.0 to 0.1) based on recency signals
    """
    recency_patterns = [
        r"202[4-6]",  # Recent years
        r"updated.*202[4-6]",
        r"last modified",
        r"published.*202[4-6]"
    ]
    
    for pattern in recency_patterns:
        if re.search(pattern, content.lower()):
            return 0.1
    
    return 0.0


def get_validation_llm():
    """Get configured LLM for validation (uses research model - fast/cheap)."""
    from src.utils.llm_factory import get_research_llm
    return get_research_llm(temperature=0.1)


VALIDATION_PROMPT = """You are a source credibility evaluator. Analyze the following source and rate it.

Source URL: {url}
Source Title: {title}
Content Preview: {content_preview}

Evaluate on these criteria:
1. Domain credibility (official docs, academic, reputable tech sites score higher)
2. Content relevance to the research topic: "{topic}"
3. Information freshness indicators
4. Technical depth and accuracy signals

Respond in this exact JSON format:
{{
    "credibility_score": <0.0-1.0>,
    "relevance_score": <0.0-1.0>,
    "source_type": "<official_docs|academic|tech_blog|news|forum|unknown>",
    "key_insights": ["<insight1>", "<insight2>"],
    "concerns": ["<concern1 if any>"],
    "recommendation": "<include|exclude|review>"
}}

Only output the JSON, no other text."""


@tool
def validate_source(url: str, title: str, content: str, topic: str) -> dict[str, Any]:
    """
    Validate a source's credibility and relevance using heuristics + LLM.
    
    Validation layers:
    1. Domain reputation scoring (deterministic)
    2. Recency indicator check (deterministic)  
    3. LLM semantic relevance (probabilistic)
    
    Args:
        url: Source URL
        title: Page title
        content: Extracted content
        topic: Research topic for relevance matching
        
    Returns:
        Validation result with scores and recommendation
    """
    logger.info("Validating source", url=url)
    
    # Layer 1: Deterministic domain reputation scoring
    domain_score = calculate_domain_score(url)
    
    # Layer 2: Recency boost
    recency_boost = check_recency_indicators(content)
    
    try:
        llm = get_validation_llm()
        
        # Truncate content for the prompt
        content_preview = content[:2000] if len(content) > 2000 else content
        
        prompt = VALIDATION_PROMPT.format(
            url=url,
            title=title,
            content_preview=content_preview,
            topic=topic
        )
        
        messages = [
            SystemMessage(content="You are a precise source evaluator. Output only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        # Layer 3: LLM semantic relevance scoring
        response = llm.invoke(messages)
        
        # Parse JSON response
        result_text = response.content.strip()
        
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        
        # Add source metadata
        result["url"] = url
        result["title"] = title
        result["validated"] = True
        
        # Add heuristic scores to result
        result["domain_score"] = domain_score
        result["recency_boost"] = recency_boost
        
        # Blend heuristic and LLM scores for final combined score
        # Formula: 30% domain heuristic + 10% LLM credibility + 60% LLM relevance + recency boost
        llm_credibility = result.get("credibility_score", 0.5)
        llm_relevance = result.get("relevance_score", 0.5)
        
        result["combined_score"] = min(1.0, (
            domain_score * 0.3 +
            llm_credibility * 0.1 +
            llm_relevance * 0.6 +
            recency_boost
        ))
        
        
        logger.info(
            "Source validated",
            url=url,
            combined_score=result["combined_score"],
            recommendation=result.get("recommendation")
        )
        
        return result
        
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse validation response", url=url, error=str(e))
        # Return default validation on parse failure
        return {
            "url": url,
            "title": title,
            "validated": False,
            "credibility_score": 0.5,
            "relevance_score": 0.5,
            "combined_score": 0.5,
            "source_type": "unknown",
            "key_insights": [],
            "concerns": ["Validation parsing failed"],
            "recommendation": "review"
        }
        
    except Exception as e:
        logger.error("Validation failed", url=url, error=str(e))
        return {
            "url": url,
            "title": title,
            "validated": False,
            "credibility_score": 0.3,
            "relevance_score": 0.3,
            "combined_score": 0.3,
            "source_type": "unknown",
            "key_insights": [],
            "concerns": [f"Validation error: {str(e)}"],
            "recommendation": "exclude"
        }


def validate_sources(
    sources: list[dict[str, str]],
    topic: str,
    min_score: float = 0.5
) -> list[dict[str, Any]]:
    """
    Validate multiple sources and filter by minimum score.
    
    Args:
        sources: List of source dicts with url, title, content
        topic: Research topic
        min_score: Minimum combined score to include
        
    Returns:
        List of validated sources meeting the threshold
    """
    validated = []
    seen_urls = set()
    
    for source in sources:
        url = source.get("url", "")
        
        # Skip banned sources
        banned_domains = ["bcg.com"]
        if any(domain in (url or "").lower() for domain in banned_domains):
            logger.warning("Skipping banned source", url=url)
            continue
        
        # Skip duplicates
        if url in seen_urls:
            logger.debug("Skipping duplicate URL", url=url)
            continue
        seen_urls.add(url)

        
        # Validate
        result = validate_source.invoke({
            "url": url,
            "title": source.get("title", ""),
            "content": source.get("content", ""),
            "topic": topic
        })
        
        # Log detailed validation results for debugging
        logger.info(
            "Validation result",
            url=url[:80],
            combined_score=result.get("combined_score"),
            domain_score=result.get("domain_score"),
            credibility_score=result.get("credibility_score"),
            relevance_score=result.get("relevance_score"),
            recency_boost=result.get("recency_boost"),
            recommendation=result.get("recommendation"),
            source_type=result.get("source_type"),
            min_score_threshold=min_score,
            will_include=result.get("combined_score", 0) >= min_score
        )
        
        # Filter by score
        if result.get("combined_score", 0) >= min_score:
            result["original_content"] = source.get("content", "")
            validated.append(result)
            logger.info("Source ACCEPTED", url=url[:80], score=result.get("combined_score"))
        else:
            logger.warning(
                "Source REJECTED - below threshold",
                url=url[:80],
                combined_score=result.get("combined_score"),
                min_score_threshold=min_score,
                gap=min_score - result.get("combined_score", 0),
                concerns=result.get("concerns", [])
            )
    
    # Sort by combined score
    validated.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    
    return validated
