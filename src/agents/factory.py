"""LangGraph-based agent factory."""

import logging
from typing import Dict, List, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from tenacity import retry, stop_after_attempt, wait_exponential

from src.configs.models import AgentConfig, TeamConfig
from src.configs.settings import settings
from src.core.exceptions import AgentError, ConfigurationError
from src.core.logging import AgentLogger
from src.prompts.templates import AGENT_SYSTEM_PROMPT_TEMPLATE


class AgentFactory:
    """Factory for creating LangGraph agents."""

    def __init__(self):
        self.supported_tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        from src.agents.tools import get_default_tools

        for tool in get_default_tools():
            self.supported_tools[tool.name] = tool

    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self.supported_tools[tool.name] = tool

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def create_agent(self, agent_config: AgentConfig, team_config: TeamConfig) -> Any:
        """Create a LangGraph react agent."""
        logger = AgentLogger(agent_config.name)
        logger.info("Creating LangGraph agent", config=agent_config.dict())

        try:
            # Create LLM
            llm = self._create_llm(agent_config)

            # Get tools
            tools = self._get_agent_tools(agent_config, logger)

            # Create prompt
            prompt = self._create_agent_prompt(agent_config, team_config)

            # Create react agent
            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=prompt,
                checkpointer=None,  # Will be set at graph level
                debug=settings.log_level == "DEBUG",
            )

            logger.info("Successfully created LangGraph agent")
            return agent

        except Exception as e:
            logger.error("Failed to create agent", error=str(e))
            raise AgentError(agent_config.name, f"Failed to create agent: {e}", e)

    def _create_llm(self, agent_config: AgentConfig) -> ChatGoogleGenerativeAI:
        """Create the LLM for the agent."""
        return ChatGoogleGenerativeAI(
            model=agent_config.model_name,
            temperature=agent_config.temperature,
            api_key=settings.google_api_key,
            **agent_config.kwargs,
        )

    def _get_agent_tools(
        self, agent_config: AgentConfig, logger: AgentLogger
    ) -> List[BaseTool]:
        """Get tools for the agent."""
        tools = []
        for tool_name in agent_config.tools:
            if tool_name in self.supported_tools:
                tools.append(self.supported_tools[tool_name])
                logger.info("Added tool to agent", tool=tool_name)
            else:
                logger.warning("Tool not supported", tool=tool_name)
                raise ConfigurationError(f"Tool '{tool_name}' not supported")
        return tools

    def _create_agent_prompt(
        self, agent_config: AgentConfig, team_config: TeamConfig
    ) -> ChatPromptTemplate:
        """Create the agent prompt."""
        team_agent_names = [agent.name for agent in team_config.agents]
        formatted_system_prompt = AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=agent_config.name,
            agent_specific_prompt=agent_config.system_prompt_template,
            team_name=team_config.team_name,
            team_desc=team_config.team_description,
            team_agent_list=", ".join(team_agent_names),
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", formatted_system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
