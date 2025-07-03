"""Custom exception classes for the multi-agent system."""


class SynapseError(Exception):
    """Base exception for all synapse-related errors."""

    pass


class AgentError(SynapseError):
    """Raised when an agent encounters an error."""

    def __init__(self, agent_name: str, message: str, original_error: Exception = None):
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"Agent '{agent_name}': {message}")


class ToolError(SynapseError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str, original_error: Exception = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}': {message}")


class StateError(SynapseError):
    """Raised when state management fails."""

    pass


class ConfigurationError(SynapseError):
    """Raised when configuration is invalid."""

    pass
