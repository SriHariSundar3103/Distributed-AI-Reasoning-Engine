# ... existing imports ...
import json
import logging
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import structlog
from structlog.typing import Processor

_CURRENT_SESSION_DIR: Optional[Path] = None

def setup_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """
    Configure structured logging and session directory.
    """
    global _CURRENT_SESSION_DIR
    
    # Create session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _CURRENT_SESSION_DIR = Path("logs") / f"session_{timestamp}"
    _CURRENT_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set up standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    
    # Define processors
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Log session start
    logger = structlog.get_logger("system")
    logger.info("Session started", session_dir=str(_CURRENT_SESSION_DIR))


def get_logger(name: str, **context: Any) -> structlog.stdlib.BoundLogger:
    """Get a logger instance with optional context."""
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


def get_session_dir() -> Optional[Path]:
    """Get the current session directory path."""
    return _CURRENT_SESSION_DIR


class AgentLogger:
    """
    Specialized logger for agent operations with file-based state dumping.
    """
    
    def __init__(self, agent_name: str):
        self.logger = get_logger("agent", agent=agent_name)
        self.agent_name = agent_name
        self._step_counter = 0
    
    def _save_state_snapshot(self, event: str, data: dict[str, Any]) -> None:
        """Save agent state to session directory."""
        if not _CURRENT_SESSION_DIR:
            return
            
        try:
            self._step_counter += 1
            filename = f"{self._step_counter:03d}_{self.agent_name.lower()}_{event}.json"
            filepath = _CURRENT_SESSION_DIR / filename
            
            # Prepare data for serialization (handle non-serializable objects if needed)
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "agent": self.agent_name,
                "event": event,
                "data": data
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning("Failed to save state snapshot", error=str(e))

    def start(self, **context: Any) -> None:
        """Log execution start and save input state."""
        self.logger.info(f"{self.agent_name} started", **context)
        self._save_state_snapshot("start", context)
    
    def complete(self, **context: Any) -> None:
        """Log execution completion and save output state."""
        self.logger.info(f"{self.agent_name} completed", **context)
        self._save_state_snapshot("complete", context)
    
    def error(self, error: Exception, **context: Any) -> None:
        """Log error and save error state."""
        self.logger.error(
            f"{self.agent_name} failed",
            error=str(error),
            error_type=type(error).__name__,
            **context
        )
        self._save_state_snapshot("error", {"error": str(error), **context})
    
    def tool_call(self, tool_name: str, **context: Any) -> None:
        """Log tool usage."""
        self.logger.debug(f"Tool called: {tool_name}", **context)
        self._save_state_snapshot(f"tool_{tool_name}", context)
    
    def iteration(self, current: int, max_iter: int, **context: Any) -> None:
        """Log iteration progress."""
        self.logger.info(
            f"Iteration {current}/{max_iter}",
            iteration=current,
            max_iterations=max_iter,
            **context
        )
    
    def llm_response(self, prompt_summary: str, response_content: str, **context: Any) -> None:
        """Log LLM response content for debugging."""
        self.logger.info(
            f"{self.agent_name} LLM response",
            prompt_summary=prompt_summary[:200],  # Truncate prompt summary
            response_preview=response_content[:500] if response_content else "Empty",
            response_length=len(response_content) if response_content else 0,
            **context
        )
        self._save_state_snapshot("llm_response", {
            "prompt_summary": prompt_summary,
            "response_content": response_content,
            **context
        })
    
    def log_state(self, event_name: str, state_data: dict[str, Any], **context: Any) -> None:
        """Log arbitrary state data for debugging workflow state."""
        self.logger.debug(
            f"{self.agent_name} state: {event_name}",
            state_keys=list(state_data.keys()) if isinstance(state_data, dict) else "non-dict",
            **context
        )
        # Serialize state safely with truncation for large values
        safe_state = {}
        for key, value in state_data.items():
            if isinstance(value, str) and len(value) > 2000:
                safe_state[key] = f"{value[:2000]}... [truncated, total: {len(value)} chars]"
            elif isinstance(value, list) and len(value) > 20:
                safe_state[key] = f"[list with {len(value)} items, showing first 10: {value[:10]}]"
            else:
                safe_state[key] = value
        self._save_state_snapshot(f"state_{event_name}", safe_state)
    
    def conversation_turn(self, role: str, content: str, turn_number: int = 0, **context: Any) -> None:
        """Log a conversation turn (human/assistant message)."""
        self.logger.info(
            f"{self.agent_name} conversation turn",
            role=role,
            content_preview=content[:300] if content else "Empty",
            content_length=len(content) if content else 0,
            turn_number=turn_number,
            **context
        )
        self._save_state_snapshot(f"conversation_turn_{turn_number}_{role}", {
            "role": role,
            "content": content,
            "turn_number": turn_number,
            **context
        })
