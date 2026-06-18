"""
State Definition Module

Defines the shared state structure for the multi-agent workflow.
Uses TypedDict for type-safe state management across agents.
"""

from typing import Any, TypedDict


class GraphState(TypedDict):
    """
    Shared state across all agents in the workflow.
    
    This TypedDict defines the contract for state that flows
    between agents in the LangGraph workflow.
    """
    
    # Input
    topic: str  # The technical topic to research
    
    # Research Phase
    research_results: list[dict[str, Any]]  # Raw fetched documents
    validated_sources: list[dict[str, Any]]  # Sources after validation
    
    # Analysis Phase
    analysis: dict[str, Any]  # Analysis output with findings, gaps, confidence
    gaps: list[str]  # Identified knowledge gaps
    
    # Control Flow
    iterations: int  # Current iteration count
    max_iterations: int  # Maximum allowed iterations
    
    # Output
    final_report: str  # Generated markdown report
    report_paths: dict[str, str]  # Paths to saved report files
    
    # Status & Errors
    status: str  # Current workflow status
    errors: list[str]  # Collected errors for debugging


def create_initial_state(
    topic: str,
    max_iterations: int = 3
) -> GraphState:
    """
    Create initial state for a new research workflow.
    
    Args:
        topic: The technical topic to research
        max_iterations: Maximum research-analysis loops
        
    Returns:
        Initialized GraphState ready for workflow execution
    """
    return GraphState(
        topic=topic,
        research_results=[],
        validated_sources=[],
        analysis={},
        gaps=[],
        iterations=0,
        max_iterations=max_iterations,
        final_report="",
        report_paths={},
        status="initialized",
        errors=[]
    )
