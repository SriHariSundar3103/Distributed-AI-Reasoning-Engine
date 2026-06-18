"""
Report Generation Agent

Responsible for:
- Structuring findings into a coherent report
- Including citations and sources
- Generating executive summary
- Exporting to markdown and PDF
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_settings
from src.utils.llm_factory import get_report_llm
from src.utils.logger import AgentLogger
from src.utils.pdf_export import markdown_to_pdf

logger = AgentLogger("ReportAgent")


REPORT_SYSTEM_PROMPT = """You are a technical writer creating professional research reports.
Your reports are well-structured, clear, and actionable.
Always include proper citations and maintain an executive tone.
Output only the markdown content, no code blocks or extra formatting."""


REPORT_PROMPT_TEMPLATE = """Create a comprehensive technical analysis report on: "{topic}"

## Analysis Summary:
{analysis_summary}

## Key Findings:
{key_findings}

## Insights:
{insights}

## Recommendations:
{recommendations}

## Source Information:
{sources}

## Report Requirements:
1. Start with a compelling executive summary (2-3 paragraphs)
2. Include a "Key Findings" section with bullet points
3. Add a detailed "Analysis" section expanding on findings
4. Include a "Recommendations" section with actionable items
5. End with a "Sources" section listing all references with URLs
6. Use proper markdown formatting (headers, lists, bold for emphasis)
7. Include inline citations where appropriate [Source: URL]

## Report Structure:
# Technical Analysis Report: {topic}

## Executive Summary
[2-3 paragraph overview of findings and recommendations]

## Key Findings
[Bullet points of main discoveries]

## Detailed Analysis
[In-depth discussion of each finding]

## Recommendations
[Actionable items based on analysis]

## Areas for Further Research
[If any gaps were identified]

## Sources
[Numbered list of all sources with URLs]

---

Generate the complete report now:"""


def _format_key_findings(analysis: dict[str, Any]) -> str:
    """Format key findings for the prompt."""
    findings = analysis.get("key_findings", [])
    if not findings:
        return "No specific findings extracted."
    
    formatted = []
    for f in findings:
        if isinstance(f, dict):
            formatted.append(f"- {f.get('finding', '')} (Confidence: {f.get('confidence', 'unknown')})")
        else:
            formatted.append(f"- {f}")
    
    return "\n".join(formatted)


def _format_sources_list(validated_sources: list[dict[str, Any]]) -> str:
    """Format sources for the report."""
    sources = []
    for i, source in enumerate(validated_sources, 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "N/A")
        source_type = source.get("source_type", "unknown")
        credibility = source.get("credibility_score", 0)
        
        sources.append(f"{i}. [{title}]({url}) - Type: {source_type}, Credibility: {credibility:.2f}")
    
    return "\n".join(sources) if sources else "No sources available."


def report_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate the final research report.
    
    Args:
        state: Current workflow state with analysis and validated_sources
        
    Returns:
        Updated state with final_report in markdown
    """
    logger.start(topic=state.get("topic"), iteration=state.get("iterations", 0))
    
    try:
        topic = state.get("topic", "Unknown Topic")
        analysis = state.get("analysis", {})
        validated_sources = state.get("validated_sources", [])
        
        # Prepare prompt components
        analysis_summary = analysis.get("summary", "No analysis summary available.")
        key_findings = _format_key_findings(analysis)
        insights = "\n".join([f"- {i}" for i in analysis.get("insights", [])]) or "No insights extracted."
        recommendations = "\n".join([f"- {r}" for r in analysis.get("recommendations", [])]) or "No recommendations."
        sources = _format_sources_list(validated_sources)
        
        # Build prompt
        prompt = REPORT_PROMPT_TEMPLATE.format(
            topic=topic,
            analysis_summary=analysis_summary,
            key_findings=key_findings,
            insights=insights,
            recommendations=recommendations,
            sources=sources
        )
        
        # Call LLM (uses per-agent model from config)
        llm = get_report_llm()
        messages = [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        logger.tool_call("llm_report_generation")
        
        # Log the prompt being sent
        logger.conversation_turn("human", prompt, turn_number=1)
        
        response = llm.invoke(messages)
        report_md = response.content.strip()
        
        # Log the LLM response
        logger.llm_response(
            prompt_summary=f"Report generation for {topic}",
            response_content=report_md,
            model="report_llm"
        )
        logger.conversation_turn("assistant", report_md, turn_number=2)
        
        # Clean up any code block artifacts
        if report_md.startswith("```markdown"):
            report_md = report_md[11:]
        if report_md.startswith("```"):
            report_md = report_md[3:]
        if report_md.endswith("```"):
            report_md = report_md[:-3]
        
        # Add metadata footer
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_md += f"\n\n---\n*Report generated on {timestamp}*\n"
        report_md += f"*Research iterations: {state.get('iterations', 1)}*\n"
        report_md += f"*Sources analyzed: {len(validated_sources)}*\n"
        report_md += f"*Confidence score: {analysis.get('confidence_score', 'N/A')}*\n"
        
        # Save report files
        settings = get_settings()
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:50]
        timestamp_short = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"report_{safe_topic}_{timestamp_short}"
        
        # Save markdown
        md_path = output_dir / f"{base_name}.md"
        md_path.write_text(report_md, encoding="utf-8")
        logger.tool_call("save_markdown", path=str(md_path))
        
        # Generate PDF
        pdf_path = output_dir / f"{base_name}.pdf"
        try:
            markdown_to_pdf(report_md, str(pdf_path), title=f"Research Report: {topic}")
            logger.tool_call("generate_pdf", path=str(pdf_path))
        except Exception as pdf_error:
            logger.error(pdf_error, context="pdf_generation")
            state["errors"].append(f"PDF generation failed: {str(pdf_error)}")
        
        # Update state
        state["final_report"] = report_md
        state["report_paths"] = {
            "markdown": str(md_path),
            "pdf": str(pdf_path) if pdf_path.exists() else None
        }
        state["status"] = "done"
        
        logger.complete(
            markdown_path=str(md_path),
            pdf_path=str(pdf_path) if pdf_path.exists() else "failed",
            report_length=len(report_md)
        )
        
        return state
        
    except Exception as e:
        logger.error(e, context="report_agent")
        state["errors"].append(f"Report agent error: {str(e)}")
        state["final_report"] = f"# Report Generation Failed\n\nError: {str(e)}"
        state["status"] = "report_failed"
        return state
