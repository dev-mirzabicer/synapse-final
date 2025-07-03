from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum


class ModelProvider(str, Enum):
    """Supported model providers."""

    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AgentConfig(BaseModel):
    """Enhanced configuration for a single agent."""

    name: str = Field(
        ..., description="Unique name of the agent", min_length=1, max_length=50
    )

    llm_provider: ModelProvider = Field(
        ModelProvider.GOOGLE, description="The provider for the language model"
    )

    model_name: str = Field(
        "gemini-2.5-flash", description="The specific model name to use"
    )

    temperature: float = Field(
        0.7, description="Temperature for model generation", ge=0.0, le=2.0
    )

    max_tokens: Optional[int] = Field(
        None, description="Maximum tokens for model response", gt=0
    )

    system_prompt_template: str = Field(
        ..., description="System prompt template for the agent", min_length=10
    )

    tools: List[str] = Field(
        default_factory=list, description="List of tool names available to this agent"
    )

    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model configuration parameters"
    )

    retry_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_attempts": 3,
            "base_delay": 1.0,
            "max_delay": 10.0,
        },
        description="Retry configuration for this agent",
    )

    timeout_seconds: int = Field(
        60, description="Timeout for agent execution in seconds", gt=0
    )

    @validator("name")
    def validate_name(cls, v):
        """Validate agent name."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Agent name must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    @validator("tools")
    def validate_tools(cls, v):
        """Validate tool names."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate tools not allowed")
        return v

    class Config:
        use_enum_values = True


class TeamConfig(BaseModel):
    """Enhanced configuration for a team of agents."""

    team_name: str = Field(
        ..., description="Name of the team", min_length=1, max_length=100
    )

    team_description: str = Field(
        ..., description="Description of the team's purpose", min_length=10
    )

    agents: List[AgentConfig] = Field(
        ..., description="List of agent configurations", min_items=1
    )

    orchestrator_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "gemini-2.5-flash",
            "temperature": 0.0,
            "max_planning_iterations": 5,
        },
        description="Configuration for the orchestrator",
    )

    global_tools: List[str] = Field(
        default_factory=list, description="Tools available to all agents"
    )

    conversation_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_rounds": 20,
            "timeout_minutes": 30,
            "auto_save_frequency": 5,
        },
        description="Conversation flow configuration",
    )

    @validator("agents")
    def validate_unique_agent_names(cls, v):
        """Ensure all agent names are unique."""
        names = [agent.name for agent in v]
        if len(names) != len(set(names)):
            raise ValueError("All agent names must be unique")
        return v

    @validator("agents")
    def validate_orchestrator_not_in_agents(cls, v):
        """Ensure no agent is named 'orchestrator'."""
        for agent in v:
            if agent.name.lower() == "orchestrator":
                raise ValueError("Agent cannot be named 'orchestrator' (reserved name)")
        return v

    def get_agent_by_name(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def get_all_agent_names(self) -> List[str]:
        """Get list of all agent names."""
        return [agent.name for agent in self.agents]
