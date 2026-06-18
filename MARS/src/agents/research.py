"""
Research Agent

Responsible for:
- Accepting technical topics as input
- Searching multiple sources (web, documentation)
- Extracting and validating information
- Handling rate limits and failures gracefully
"""

from typing import Any

from src.tools.search import web_search, search_multiple_queries, SearchError
from src.tools.fetch import fetch_document, fetch_multiple_urls, FetchError
from src.tools.validate import validate_sources
from src.utils.logger import AgentLogger

logger = AgentLogger("ResearchAgent")


def research_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Execute research phase: search, fetch, and validate sources.
    
    Args:
        state: Current workflow state containing topic and gaps
        
    Returns:
        Updated state with research_results and validated_sources
    """
    logger.start(topic=state.get("topic"), iteration=state.get("iterations", 0))
    
    try:
        # Build search queries from topic and identified gaps
        topic = state.get("topic", "")
        gaps = state.get("gaps", [])
        
        # Create search queries
        queries = [topic]
        
        # Add gap-focused queries if this is a subsequent iteration
        for gap in gaps[:3]:  # Limit to top 3 gaps
            queries.append(f"{topic} {gap}")
        
        logger.tool_call("web_search", queries=queries)
        
        # Execute searches
        all_search_results = []
        for query in queries:
            try:
                results = web_search.invoke(query)
                all_search_results.extend(results)
            except SearchError as e:
                logger.error(e, context="search", query=query)
                state["errors"].append(f"Search failed for '{query}': {str(e)}")
        
        if not all_search_results:
            logger.error(Exception("No search results"), context="search")
            state["status"] = "research_failed"
            state["errors"].append("No search results found for any query")
            return state
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_search_results:
            url = result.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        logger.tool_call("fetch_document", url_count=len(unique_results))
        
        # Fetch document content (limit to top results by score)
        sorted_results = sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)
        urls_to_fetch = [r["url"] for r in sorted_results[:8]]  # Limit fetches
        
        fetched_docs = []
        for url in urls_to_fetch:
            try:
                doc = fetch_document.invoke(url)
                fetched_docs.append(doc)
            except FetchError as e:
                logger.error(e, context="fetch", url=url)
                # Continue with other URLs
        
        if not fetched_docs:
            state["status"] = "research_failed"
            state["errors"].append("Failed to fetch any documents")
            return state
        
        logger.tool_call("validate_sources", doc_count=len(fetched_docs))
        
        # Validate sources
        validated = validate_sources(
            sources=fetched_docs,
            topic=topic,
            min_score=0.3  # Lowered from 0.4 to include more sources
        )
        
        # Log research results for debugging
        logger.log_state("search_results", {
            "total_found": len(all_search_results),
            "unique_urls": list(seen_urls)[:10],  # First 10 URLs
            "fetched_count": len(fetched_docs),
            "validated_count": len(validated)
        })
        
        # Log fetched document summaries
        for i, doc in enumerate(fetched_docs[:5]):  # First 5 docs
            logger.log_state(f"fetched_doc_{i}", {
                "url": doc.get("url", "unknown"),
                "title": doc.get("title", "untitled"),
                "content_preview": doc.get("content", "")[:500] if doc.get("content") else "No content"
            })
        
        # Log validated sources
        for i, source in enumerate(validated[:5]):  # First 5 validated
            logger.log_state(f"validated_source_{i}", {
                "url": source.get("url", "unknown"),
                "title": source.get("title", "untitled"),
                "credibility_score": source.get("credibility_score"),
                "key_insights": source.get("key_insights", [])
            })
        
        # Update state
        state["research_results"] = fetched_docs
        state["validated_sources"] = validated
        state["status"] = "research_complete"
        
        logger.complete(
            sources_found=len(all_search_results),
            sources_fetched=len(fetched_docs),
            sources_validated=len(validated)
        )
        
        return state
        
    except Exception as e:
        logger.error(e, context="research_agent")
        state["errors"].append(f"Research agent error: {str(e)}")
        state["status"] = "research_failed"
        return state
