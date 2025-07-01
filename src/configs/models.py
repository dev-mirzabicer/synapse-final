from pydantic import BaseModel, Field
from typing import List, Dict, Any

class AgentConfig(BaseModel):
    """
    Pydantic model for a single agent's configuration.
    This structure ensures that all necessary parameters for creating an agent
    are present and correctly typed.
    """
    name: str = Field(..., description="The unique name of the agent.")
    llm_provider: str = Field("openai", description="The provider for the language model.")
    model_name: str = Field("gpt-4o", description="The specific model name to be used.")
    temperature: float = Field(0.7, description="The temperature for the language model's generation.")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the language model."
    )
    system_prompt_template: str = Field(
        ...,
        description="The template for the agent's system prompt. This will be formatted with team details."
    )
    tools: List[str] = Field(
        default_factory=list,
        description="A list of tool names that this agent has access to."
    )

class TeamConfig(BaseModel):
    """
    Pydantic model for a team's configuration.
    This model composes multiple AgentConfig objects to define a full team.
    """
    team_name: str = Field(..., description="The name of the team.")
    team_description: str = Field(..., description="A brief description of the team's purpose and instructions.")
    agents: List[AgentConfig] = Field(
        ...,
        description="A list of agent configurations that constitute the team."
    )