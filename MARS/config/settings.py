"""
Configuration Management Module

Centralized configuration using Pydantic Settings for type-safe environment management.
Supports loading from .env file and environment variables.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # NOTE:
    # The runtime environment for this project may block access to .env files.
    # Read keys exclusively from OS environment variables.
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore"
    )

    # API Keys
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key (optional if using Bedrock only)"
    )
    
    # Support common env-var naming.
    # With `case_sensitive=False`, both `GOOGLE_API_KEY` and `google_api_key` map here.
    tavily_api_key: str = Field(
        ...,
        description="Tavily API key for web search"
    )

    
    # AWS Configuration (for Bedrock Claude)
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock API calls"
    )
    
    # LLM Configuration - Model-agnostic per-agent selection
    # Default model used when agent-specific model not set
    llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Default LLM model for all agents"
    )
    
    # Per-agent model overrides (supports swapping to Claude, etc.)
    research_model: str = Field(
        default="gemini-2.0-flash",
        description="Model for Research Agent (fast, high throughput)"
    )
    analysis_model: str = Field(
        default="gemini-2.0-flash",
        description="Model for Analysis Agent (can swap to claude-4.5)"
    )
    report_model: str = Field(
        default="gemini-2.0-flash",
        description="Model for Report Agent (can swap to claude-4.5)"
    )
    
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for LLM responses"
    )
    
    # Workflow Configuration
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum research-analysis iterations"
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to skip additional research"
    )
    
    # Search Configuration
    max_search_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum search results per query"
    )
    
    # Timeouts (seconds)
    http_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="HTTP request timeout"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    # Output
    output_dir: str = Field(
        default="outputs",
        description="Directory for generated reports"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()
