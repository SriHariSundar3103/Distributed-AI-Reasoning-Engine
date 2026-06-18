"""
Analysis Agent

Responsible for:
- Synthesizing research findings
- Validating research quality and completeness
- Identifying gaps or contradictions
- Generating insights and recommendations
- Deciding whether more research is needed
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_settings
from src.utils.llm_factory import get_analysis_llm
from src.utils.logger import AgentLogger

logger = AgentLogger("AnalysisAgent")


ANALYSIS_SYSTEM_PROMPT = """You are a technical analyst specializing in synthesizing research findings.
Your task is to analyze gathered information and produce structured insights.
Be thorough, identify gaps, and provide actionable recommendations.
Always respond with valid JSON only."""


ANALYSIS_PROMPT_TEMPLATE = """Analyze the following research findings about: "{topic}"

## Validated Sources ({source_count} sources):
{sources_summary}

## Task:
1. Synthesize the key findings across all sources
2. Identify areas of consensus and contradiction
3. Assess the completeness of the research
4. Identify gaps that need more investigation
5. Generate actionable insights and recommendations

## Response Format (JSON only):
{{
    "summary": "A comprehensive 3-5 paragraph synthesis of all findings",
    "key_findings": [
        {{"finding": "...", "confidence": "high|medium|low", "sources": ["url1", "url2"]}},
        ...
    ],
    "consensus_areas": ["Areas where sources agree"],
    "contradictions": [
        {{"topic": "...", "viewpoint_a": "...", "viewpoint_b": "...", "sources": ["..."]}}
    ],
    "gaps": ["Specific topics or questions that need more research"],
    "insights": ["Non-obvious insights derived from the research"],
    "recommendations": ["Actionable recommendations based on findings"],
    "confidence_score": <0.0-1.0 overall confidence in research completeness>,
    "needs_more_research": <true|false>,
    "suggested_queries": ["If more research needed, specific queries to run"]
}}

Respond ONLY with the JSON object, no other text."""


def _format_sources_summary(validated_sources: list[dict[str, Any]]) -> str:
    """Format validated sources for the analysis prompt."""
    summaries = []
    
    for i, source in enumerate(validated_sources[:10], 1):  # Limit to top 10
        source_type = source.get("source_type", "unknown")
        credibility = source.get("credibility_score", 0)
        insights = source.get("key_insights", [])
        content = source.get("original_content", "")[:1500]  # Truncate content
        
        summary = f"""
### Source {i}: {source.get('title', 'Untitled')}
- URL: {source.get('url', 'N/A')}
- Type: {source_type} | Credibility: {credibility:.2f}
- Key Insights: {', '.join(insights) if insights else 'None extracted'}
- Content Preview:
{content}
"""
        summaries.append(summary)
    
    return "\n".join(summaries)


def analysis_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze research findings and determine if more research is needed.
    
    Args:
        state: Current workflow state with validated_sources
        
    Returns:
        Updated state with analysis results and potential gaps
    """
    logger.start(
        topic=state.get("topic"),
        iteration=state.get("iterations", 0),
        source_count=len(state.get("validated_sources", []))
    )
    
    try:
        validated_sources = state.get("validated_sources", [])
        topic = state.get("topic", "")
        
        if not validated_sources:
            logger.error(Exception("No validated sources"), context="analysis")
            state["analysis"] = {
                "summary": "Insufficient data for analysis",
                "confidence_score": 0.0,
                "needs_more_research": True,
                "gaps": ["No validated sources available"],
                "key_findings": [],
                "recommendations": []
            }
            state["gaps"] = ["No validated sources - research failed"]
            state["status"] = "analysis_complete"
            state["iterations"] = state.get("iterations", 0) + 1
            return state
        
        # Format sources for prompt
        sources_summary = _format_sources_summary(validated_sources)
        
        # Build analysis prompt
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            topic=topic,
            source_count=len(validated_sources),
            sources_summary=sources_summary
        )
        
        # Call LLM (uses per-agent model from config)
        llm = get_analysis_llm()
        messages = [
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        logger.tool_call("llm_analysis")
        
        # Log the prompt being sent
        logger.conversation_turn("human", prompt, turn_number=1)
        
        response = llm.invoke(messages)
        
        # Parse response
        result_text = response.content.strip()
        
        # Log the LLM response
        logger.llm_response(
            prompt_summary=f"Analysis of {topic} with {len(validated_sources)} sources",
            response_content=result_text,
            model="analysis_llm"
        )
        logger.conversation_turn("assistant", result_text, turn_number=2)
        
        # Handle markdown code blocks
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1])
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()
        
        try:
            analysis = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.error(e, context="json_parse", response_preview=result_text[:200])
            # Create minimal analysis on parse failure
            analysis = {
                "summary": "Analysis parsing failed, using source summaries directly.",
                "confidence_score": 0.5,
                "needs_more_research": False,
                "gaps": [],
                "key_findings": [],
                "recommendations": ["Review raw sources manually"],
                "parse_error": str(e)
            }
        
        # Update state
        state["analysis"] = analysis
        state["gaps"] = analysis.get("gaps", [])
        state["iterations"] = state.get("iterations", 0) + 1
        state["status"] = "analysis_complete"
        
        # Log final analysis state for debugging
        logger.log_state("analysis_result", {
            "summary": analysis.get("summary", "")[:500],
            "key_findings_count": len(analysis.get("key_findings", [])),
            "gaps": analysis.get("gaps", []),
            "confidence_score": analysis.get("confidence_score"),
            "needs_more_research": analysis.get("needs_more_research")
        })
        
        logger.complete(
            confidence_score=analysis.get("confidence_score", 0),
            gaps_found=len(analysis.get("gaps", [])),
            needs_more_research=analysis.get("needs_more_research", False)
        )
        
        return state
        
    except Exception as e:
        logger.error(e, context="analysis_agent")
        state["errors"].append(f"Analysis agent error: {str(e)}")
        state["analysis"] = {
            "summary": f"Analysis failed: {str(e)}",
            "confidence_score": 0.0,
            "needs_more_research": False,
            "gaps": [],
            "key_findings": [],
            "recommendations": []
        }
        state["iterations"] = state.get("iterations", 0) + 1
        state["status"] = "analysis_failed"
        return state
