"""
LLM Factory Module

Provides model-agnostic LLM instantiation supporting multiple providers:
- Google Gemini (via langchain-google-genai)
- AWS Bedrock Claude (via langchain-aws)

Enables per-agent model selection for optimal cost/quality tradeoffs.
"""

from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported model identifiers
ModelProvider = Literal["gemini", "bedrock"]



def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    agent_type: Optional[Literal["research", "analysis", "report"]] = None
) -> BaseChatModel:
    """
    Get an LLM instance based on model name or agent type.
    
    Supports:
    - Gemini models: gemini-2.0-flash, gemini-2.0-flash, etc.
    - Bedrock Claude: claude-4.5, anthropic.claude-4-5-sonnet, etc.
    
    Args:
        model_name: Specific model to use (overrides agent_type)
        temperature: Override default temperature
        agent_type: Agent requesting the LLM (uses per-agent config)
        
    Returns:
        Configured LLM instance
    """
    settings = get_settings()
    
    # Determine which model to use
    if model_name:
        selected_model = model_name
    elif agent_type:
        # Use per-agent configuration
        model_map = {
            "research": settings.research_model,
            "analysis": settings.analysis_model,
            "report": settings.report_model,
        }
        selected_model = model_map.get(agent_type, settings.llm_model)
    else:
        selected_model = settings.llm_model
    
    # Determine temperature
    temp = temperature if temperature is not None else settings.llm_temperature
    
    logger.debug("Creating LLM", model=selected_model, temperature=temp, agent=agent_type)
    
    # Route to appropriate provider
    if _is_bedrock_model(selected_model):
        return _create_bedrock_llm(selected_model, temp)
    else:
        return _create_gemini_llm(selected_model, temp)


def _is_bedrock_model(model_name: str) -> bool:
    """Check if model name indicates a Bedrock model."""
    bedrock_indicators = [
        "claude",
        "anthropic",
        "bedrock",
        "amazon",
    ]
    return any(indicator in model_name.lower() for indicator in bedrock_indicators)


def _create_gemini_llm(model_name: str, temperature: float) -> BaseChatModel:
    """Create a Google Gemini LLM instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    settings = get_settings()
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )


def _create_bedrock_llm(model_name: str, temperature: float) -> BaseChatModel:
    """
    Create an AWS Bedrock Claude LLM instance.
    
    Requires AWS credentials configured via:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - AWS credentials file (~/.aws/credentials)
    - IAM role (when running on AWS)
    """
    from langchain_aws import ChatBedrock
    
    settings = get_settings()
    
    # Map friendly names to Bedrock model IDs
    model_id_map = {
        "claude-4.5": "anthropic.claude-4-5-sonnet-20250120-v1:0",
        "claude-4": "anthropic.claude-4-sonnet-20250514-v1:0",
        "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    }
    
    # Resolve model ID
    model_id = model_id_map.get(model_name.lower(), model_name)
    
    logger.info("Creating Bedrock LLM", model_id=model_id, region=settings.aws_region)
    
    return ChatBedrock(
        model_id=model_id,
        region_name=settings.aws_region,
        model_kwargs={
            "temperature": temperature,
            "max_tokens": 4096,
        }
    )


# Convenience functions for each agent type
def get_research_llm(temperature: float = 0.1) -> BaseChatModel:
    """Get LLM configured for Research Agent (fast, validation-focused)."""
    return get_llm(agent_type="research", temperature=temperature)


def get_analysis_llm(temperature: float = 0.3) -> BaseChatModel:
    """Get LLM configured for Analysis Agent (reasoning-focused)."""
    return get_llm(agent_type="analysis", temperature=temperature)


def get_report_llm(temperature: float = 0.4) -> BaseChatModel:
    """Get LLM configured for Report Agent (creative writing)."""
    return get_llm(agent_type="report", temperature=temperature)

