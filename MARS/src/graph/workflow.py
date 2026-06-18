"""
LangGraph Workflow Module

Defines the multi-agent workflow orchestration using LangGraph.
Implements the research -> analysis -> (conditional) -> report flow.
"""

from typing import Literal

from langgraph.graph import StateGraph, END

from config.settings import get_settings
from src.agents.research import research_agent
from src.agents.analysis import analysis_agent
from src.agents.report import report_agent
from src.utils.logger import get_logger
from .state import GraphState, create_initial_state

logger = get_logger(__name__)


def should_continue(state: GraphState) -> Literal["research", "report"]:
    """
    Routing function to determine next step after analysis.
    
    Decision criteria:
    1. If max iterations reached -> go to report
    2. If confidence is high enough -> go to report
    3. If gaps exist and confidence is low -> go back to research
    
    Args:
        state: Current workflow state
        
    Returns:
        "research" to loop back or "report" to generate final output
    """
    settings = get_settings()
    
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", settings.max_iterations)
    analysis = state.get("analysis", {})
    confidence = analysis.get("confidence_score", 0)
    gaps = state.get("gaps", [])
    needs_more = analysis.get("needs_more_research", False)
    
    logger.info(
        "Routing decision",
        iterations=iterations,
        max_iterations=max_iterations,
        confidence=confidence,
        gaps_count=len(gaps),
        needs_more_research=needs_more
    )
    
    # Check iteration limit
    if iterations >= max_iterations:
        logger.info("Max iterations reached, proceeding to report")
        return "report"
    
    # Check if confidence is sufficient
    if confidence >= settings.confidence_threshold:
        logger.info("Confidence threshold met, proceeding to report")
        return "report"
    
    # Check if more research is explicitly not needed
    if not needs_more and not gaps:
        logger.info("No gaps identified, proceeding to report")
        return "report"
    
    # Need more research
    logger.info("Gaps identified, looping back to research", gaps=gaps[:3])
    return "research"


def create_workflow() -> StateGraph:
    """
    Build and compile the LangGraph workflow.
    
    Workflow structure:
        research -> analysis -> (conditional) -> research OR report -> END
    
    Returns:
        Compiled LangGraph workflow ready for execution
    """
    logger.info("Building workflow graph")
    
    # Create graph with state schema
    graph = StateGraph(GraphState)
    
    # Add nodes (agents)
    graph.add_node("research", research_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("report", report_agent)
    
    # Set entry point
    graph.set_entry_point("research")
    
    # Add edges
    graph.add_edge("research", "analysis")
    
    # Conditional routing after analysis
    graph.add_conditional_edges(
        "analysis",
        should_continue,
        {
            "research": "research",
            "report": "report",
        }
    )
    
    # Report leads to end
    graph.add_edge("report", END)
    
    # Compile the graph
    compiled = graph.compile()
    
    logger.info("Workflow graph compiled successfully")
    return compiled


def run_research(
    topic: str,
    max_iterations: int = 3,
    verbose: bool = True
) -> dict:
    """
    Execute the full research workflow for a given topic.
    
    Args:
        topic: Technical topic to research
        max_iterations: Maximum research-analysis iterations
        verbose: If True, log intermediate states
        
    Returns:
        Final state with research results and generated report
    """
    logger.info("Starting research workflow", topic=topic, max_iterations=max_iterations)
    
    # Create initial state
    initial_state = create_initial_state(
        topic=topic,
        max_iterations=max_iterations
    )
    
    # Build and run workflow
    workflow = create_workflow()
    
    try:
        # Execute with streaming for observability
        final_state = None
        
        for step in workflow.stream(initial_state):
            if verbose:
                # Log each step
                for node_name, node_state in step.items():
                    status = node_state.get("status", "unknown")
                    iterations = node_state.get("iterations", 0)
                    logger.info(
                        f"Workflow step: {node_name}",
                        status=status,
                        iteration=iterations
                    )
                    final_state = node_state
        
        if final_state is None:
            raise RuntimeError("Workflow produced no output")
        
        # Log completion
        logger.info(
            "Workflow completed",
            status=final_state.get("status"),
            total_iterations=final_state.get("iterations"),
            sources_used=len(final_state.get("validated_sources", [])),
            errors_count=len(final_state.get("errors", []))
        )
        
        return final_state
        
    except Exception as e:
        logger.error("Workflow execution failed", error=str(e))
        raise
