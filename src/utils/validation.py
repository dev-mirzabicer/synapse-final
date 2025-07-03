import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError
from src.core.exceptions import StateError, ConfigurationError


def validate_user_input(user_input: str) -> str:
    """
    Validate and sanitize user input.

    Args:
        user_input: Raw user input string

    Returns:
        Cleaned and validated input

    Raises:
        ValueError: If input is invalid
    """
    if not user_input or not user_input.strip():
        raise ValueError("Input cannot be empty")

    # Remove excessive whitespace
    cleaned = re.sub(r"\s+", " ", user_input.strip())

    # Check length
    if len(cleaned) > 10000:
        raise ValueError("Input too long (max 10000 characters)")

    # Basic content validation (can be expanded)
    if len(cleaned) < 3:
        raise ValueError("Input too short (minimum 3 characters)")

    return cleaned


def validate_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate graph state structure.

    Args:
        state: State dictionary to validate

    Returns:
        Validated state

    Raises:
        StateError: If state is invalid
    """
    required_keys = ["messages", "active_tasks", "completed_tasks"]

    for key in required_keys:
        if key not in state:
            raise StateError(f"Missing required state key: {key}")

    # Validate messages
    if not isinstance(state["messages"], list):
        raise StateError("messages must be a list")

    # Validate task dictionaries
    if not isinstance(state["active_tasks"], dict):
        raise StateError("active_tasks must be a dictionary")

    if not isinstance(state["completed_tasks"], set):
        raise StateError("completed_tasks must be a set")

    return state


def validate_agent_response(response: str, agent_name: str) -> str:
    """
    Validate agent response.

    Args:
        response: Agent response to validate
        agent_name: Name of the agent that generated the response

    Returns:
        Validated response

    Raises:
        AgentError: If response is invalid
    """
    if not response or not response.strip():
        from src.core.exceptions import AgentError

        raise AgentError(agent_name, "Agent produced empty response")

    # Check for reasonable length
    if len(response) > 50000:
        from src.core.exceptions import AgentError

        raise AgentError(agent_name, "Agent response too long")

    return response.strip()
