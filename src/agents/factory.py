"""Enhanced LangGraph-based agent factory for creating subgraph agents."""

from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential

from src.configs.models import AgentConfig, TeamConfig
from src.configs.settings import settings
from src.core.exceptions import AgentError, ConfigurationError
from src.core.logging import AgentLogger
from src.prompts.templates import AGENT_SYSTEM_PROMPT_TEMPLATE
from src.graph.nodes import create_agent_subgraph


class AgentFactory:
    """Enhanced factory for creating LangGraph agent subgraphs."""

    def __init__(self):
        self.supported_tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        from src.agents.tools import (
            get_default_tools,
        )

        # Register all available tools
        for tool in get_default_tools():
            self.supported_tools[tool.name] = tool

    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self.supported_tools[tool.name] = tool

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def create_agent_subgraph(
        self, agent_config: AgentConfig, team_config: TeamConfig
    ) -> StateGraph:
        """
        Create a LangGraph agent subgraph with cursor-based personalization.

        Args:
            agent_config: Configuration for the agent
            team_config: Team configuration

        Returns:
            Compiled StateGraph for the agent
        """
        logger = AgentLogger(agent_config.name)
        logger.info("Creating LangGraph agent subgraph", config=agent_config.dict())

        try:
            # Create the underlying ReAct agent first
            react_agent = self._create_react_agent(agent_config, team_config)

            # Create agent subgraph
            subgraph = create_agent_subgraph(agent_config, team_config, react_agent)

            logger.info("Successfully created LangGraph agent subgraph")
            return subgraph

        except Exception as e:
            logger.error("Failed to create agent subgraph", error=str(e))
            raise AgentError(
                agent_config.name, f"Failed to create agent subgraph: {e}", e
            )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def create_react_agent(
        self, agent_config: AgentConfig, team_config: TeamConfig
    ) -> Any:
        """
        Create a traditional LangGraph ReAct agent (for backwards compatibility).

        Args:
            agent_config: Configuration for the agent
            team_config: Team configuration

        Returns:
            LangGraph ReAct agent
        """
        return self._create_react_agent(agent_config, team_config)

    def _create_react_agent(
        self, agent_config: AgentConfig, team_config: TeamConfig
    ) -> Any:
        """Create the underlying ReAct agent."""
        logger = AgentLogger(agent_config.name)

        try:
            # Create LLM
            llm = self._create_llm(agent_config)

            # Get tools based on agent type
            tools = self._get_agent_tools(agent_config, logger)

            # Create prompt
            prompt = self._create_agent_prompt(agent_config, team_config)

            # Create react agent
            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=prompt,
                checkpointer=None,  # Will be set at graph level
                # debug=settings.log_level == "DEBUG",
            )

            logger.info("Successfully created ReAct agent")
            return agent

        except Exception as e:
            logger.error("Failed to create ReAct agent", error=str(e))
            raise AgentError(agent_config.name, f"Failed to create ReAct agent: {e}", e)

    def create_orchestrator_runnable(self, team_config: TeamConfig):
        """
        Create the orchestrator runnable.

        Args:
            team_config: Team configuration

        Returns:
            Orchestrator runnable (LLM with tools)
        """
        logger = AgentLogger("orchestrator")
        logger.info("Creating orchestrator runnable")

        try:
            # Create LLM for orchestrator
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro", temperature=0, api_key=settings.google_api_key
            )

            # Get orchestrator tools
            tools = self._get_orchestrator_tools(logger)

            # Create prompt
            prompt = self._create_orchestrator_prompt(team_config)

            # Bind tools to LLM
            llm_with_tools = llm.bind_tools(tools)

            # Create runnable
            orchestrator_runnable = prompt | llm_with_tools

            logger.info("Successfully created orchestrator runnable")
            return orchestrator_runnable

        except Exception as e:
            logger.error("Failed to create orchestrator runnable", error=str(e))
            raise ConfigurationError(f"Failed to create orchestrator runnable: {e}")

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
        """Get tools for a regular agent (not orchestrator)."""
        from src.agents.tools import get_agent_tools

        # Start with agent-specific tools
        tools = []

        # Add tools from agent-specific tool set (includes finish_my_turn)
        default_agent_tools = get_agent_tools()
        for tool in default_agent_tools:
            if tool.name not in [t.name for t in tools]:  # Avoid duplicates
                tools.append(tool)
                logger.info("Added default tool to agent", tool=tool.name)

        # Add any additional tools specified in config
        for tool_name in agent_config.tools:
            if tool_name in self.supported_tools:
                tool = self.supported_tools[tool_name]
                if tool.name not in [t.name for t in tools]:  # Avoid duplicates
                    tools.append(tool)
                    logger.info("Added configured tool to agent", tool=tool_name)
            else:
                logger.warning("Tool not supported", tool=tool_name)
                raise ConfigurationError(f"Tool '{tool_name}' not supported")

        return tools

    def _get_orchestrator_tools(self, logger: AgentLogger) -> List[BaseTool]:
        """Get tools for the orchestrator."""
        from src.agents.tools import get_orchestrator_tools

        tools = get_orchestrator_tools()

        for tool in tools:
            logger.info("Added tool to orchestrator", tool=tool.name)

        return tools

    def _create_agent_prompt(
        self, agent_config: AgentConfig, team_config: TeamConfig
    ) -> ChatPromptTemplate:
        """Create the agent prompt with message prefix instructions."""
        from src.prompts.templates import AGENT_MESSAGE_PREFIX_INSTRUCTION

        team_agent_names = [agent.name for agent in team_config.agents]

        # Combine original prompt with message prefix instructions
        enhanced_system_prompt = (
            agent_config.system_prompt_template
            + "\n\n"
            + AGENT_MESSAGE_PREFIX_INSTRUCTION
        )

        formatted_system_prompt = AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=agent_config.name,
            agent_specific_prompt=enhanced_system_prompt,
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

    def _create_orchestrator_prompt(
        self, team_config: TeamConfig
    ) -> ChatPromptTemplate:
        """Create the orchestrator prompt."""
        from src.prompts.templates import ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE

        team_agent_names = [agent.name for agent in team_config.agents]
        formatted_prompt = ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE.format(
            team_name=team_config.team_name,
            team_desc=team_config.team_description,
            team_agent_list=", ".join(team_agent_names),
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", formatted_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
