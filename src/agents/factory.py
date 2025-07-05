"""Enhanced LangGraph-based agent factory with subgraph support."""

from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from tenacity import retry, stop_after_attempt, wait_exponential

from src.configs.models import AgentConfig, TeamConfig
from src.configs.settings import settings
from src.core.exceptions import AgentError, ConfigurationError
from src.core.logging import get_logger
from src.prompts.templates import ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE
from src.agents.tools import get_default_tools, get_agent_tools
from src.agents.agent_subgraph import AgentSubgraphBuilder

logger = get_logger(__name__)


class AgentFactory:
    """Enhanced factory for creating LangGraph agents and agent subgraphs."""

    def __init__(self):
        self.supported_tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        for tool in get_default_tools():
            self.supported_tools[tool.name] = tool

        # Also register agent-specific tools
        for tool in get_agent_tools():
            self.supported_tools[tool.name] = tool

    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self.supported_tools[tool.name] = tool

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def create_orchestrator_agent(self, team_config: TeamConfig) -> Any:
        """Create the orchestrator agent (traditional ReAct agent)."""
        logger.info("Creating orchestrator agent")

        try:
            # Create LLM for orchestrator
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
                api_key=settings.google_api_key,
            )

            # Get orchestrator tools (only assign_tasks)
            orchestrator_tools = [
                tool for tool in get_default_tools() if tool.name == "assign_tasks"
            ]

            # Create orchestrator prompt
            team_agent_names = [agent.name for agent in team_config.agents]
            formatted_system_prompt = ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE.format(
                team_name=team_config.team_name,
                team_desc=team_config.team_description,
                team_agent_list=", ".join(team_agent_names),
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", formatted_system_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            # Create react agent for orchestrator
            orchestrator_agent = create_react_agent(
                model=llm,
                tools=orchestrator_tools,
                prompt=prompt,
                debug=settings.log_level == "DEBUG",
            )

            logger.info("Successfully created orchestrator agent")
            return orchestrator_agent

        except Exception as e:
            logger.error("Failed to create orchestrator agent", error=str(e))
            raise AgentError("orchestrator", f"Failed to create orchestrator: {e}", e)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def create_agent_subgraph(
        self, agent_config: AgentConfig, team_config: TeamConfig
    ) -> Any:
        """Create an enhanced agent subgraph."""
        logger.info("Creating agent subgraph", agent=agent_config.name)

        try:
            # Validate agent tools
            self._validate_agent_tools(agent_config)

            # Create the agent subgraph builder
            subgraph_builder = AgentSubgraphBuilder(agent_config, team_config)

            # Build the subgraph
            agent_subgraph = subgraph_builder.build_agent_subgraph()

            logger.info("Successfully created agent subgraph", agent=agent_config.name)
            return agent_subgraph

        except Exception as e:
            logger.error(
                "Failed to create agent subgraph", agent=agent_config.name, error=str(e)
            )
            raise AgentError(
                agent_config.name, f"Failed to create agent subgraph: {e}", e
            )

    def _validate_agent_tools(self, agent_config: AgentConfig):
        """Validate that agent tools are supported."""
        available_tools = {tool.name: tool for tool in get_agent_tools()}

        for tool_name in agent_config.tools:
            if (
                tool_name not in available_tools
                and tool_name not in self.supported_tools
            ):
                logger.warning(
                    "Tool not supported for agent",
                    tool=tool_name,
                    agent=agent_config.name,
                )
                raise ConfigurationError(
                    f"Tool '{tool_name}' not supported for agent {agent_config.name}"
                )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def create_legacy_agent(
        self, agent_config: AgentConfig, team_config: TeamConfig
    ) -> Any:
        """Create a legacy ReAct agent (for backwards compatibility)."""
        logger.info("Creating legacy ReAct agent", agent=agent_config.name)

        try:
            # Create LLM
            llm = self._create_llm(agent_config)

            # Get tools
            tools = self._get_agent_tools(agent_config)

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

            logger.info("Successfully created legacy agent", agent=agent_config.name)
            return agent

        except Exception as e:
            logger.error(
                "Failed to create legacy agent", agent=agent_config.name, error=str(e)
            )
            raise AgentError(
                agent_config.name, f"Failed to create legacy agent: {e}", e
            )

    def _create_llm(self, agent_config: AgentConfig) -> ChatGoogleGenerativeAI:
        """Create the LLM for the agent."""
        return ChatGoogleGenerativeAI(
            model=agent_config.model_name,
            temperature=agent_config.temperature,
            api_key=settings.google_api_key,
            **agent_config.kwargs,
        )

    def _get_agent_tools(self, agent_config: AgentConfig) -> List[BaseTool]:
        """Get tools for the agent."""
        tools = []
        for tool_name in agent_config.tools:
            if tool_name in self.supported_tools:
                tools.append(self.supported_tools[tool_name])
                logger.info(
                    "Added tool to agent", tool=tool_name, agent=agent_config.name
                )
            else:
                logger.warning(
                    "Tool not supported", tool=tool_name, agent=agent_config.name
                )
                raise ConfigurationError(f"Tool '{tool_name}' not supported")
        return tools

    def _create_agent_prompt(
        self, agent_config: AgentConfig, team_config: TeamConfig
    ) -> ChatPromptTemplate:
        """Create the agent prompt."""
        from src.prompts.templates import ENHANCED_AGENT_SYSTEM_PROMPT_TEMPLATE

        team_agent_names = [agent.name for agent in team_config.agents]
        formatted_system_prompt = ENHANCED_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
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
