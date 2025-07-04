"""Enhanced graph builder with better error handling and monitoring."""

import json
from pathlib import Path
from functools import partial
from typing import Optional, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.configs.models import TeamConfig
from src.agents.factory import AgentFactory
from src.agents.tools import assign_tasks
from src.prompts.templates import ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE
from src.core.logging import get_logger
from src.core.exceptions import ConfigurationError
from src.configs.settings import settings
from .state import GroupChatState
from .nodes import orchestrator_node, create_agent_node, aggregator_node

logger = get_logger(__name__)


class GraphBuilder:
    """Enhanced graph builder with better error handling."""

    def __init__(self):
        self.agent_factory = AgentFactory()

    def build_graph(self, team_config_path: Path, checkpointer: Optional[Any] = None):
        """Build and compile the enhanced LangGraph."""
        logger.info("Building graph", config_path=str(team_config_path))

        try:
            # Load and validate configuration
            team_config = self._load_team_config(team_config_path)

            # Create orchestrator
            orchestrator_runnable = self._create_orchestrator(team_config)

            # Create agent runnables
            agent_runnables = self._create_agent_runnables(team_config)

            # Build the graph
            workflow = self._build_workflow(orchestrator_runnable, agent_runnables)

            # Compile with checkpointer
            if checkpointer is None:
                checkpointer = MemorySaver()

            graph = workflow.compile(
                checkpointer=checkpointer, debug=settings.log_level == "DEBUG"
            )

            logger.info("Graph compiled successfully")
            return graph

        except Exception as e:
            logger.error("Failed to build graph", error=str(e))
            raise ConfigurationError(f"Failed to build graph: {e}")

    def _load_team_config(self, config_path: Path) -> TeamConfig:
        """Load and validate team configuration."""
        try:
            config_data = json.loads(config_path.read_text())
            return TeamConfig(**config_data)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration file: {e}")

    def _create_orchestrator(self, team_config: TeamConfig):
        """Create the orchestrator runnable."""
        team_agent_names = [agent.name for agent in team_config.agents]
        formatted_prompt = ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE.format(
            team_name=team_config.team_name,
            team_desc=team_config.team_description,
            team_agent_list=", ".join(team_agent_names),
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", formatted_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0, api_key=settings.google_api_key
        )
        llm_with_tools = llm.bind_tools([assign_tasks])

        return prompt_template | llm_with_tools

    def _create_agent_runnables(self, team_config: TeamConfig):
        """Create all agent runnables."""
        agent_runnables = {}

        for agent_config in team_config.agents:
            try:
                agent = self.agent_factory.create_agent(agent_config, team_config)
                agent_runnables[agent_config.name] = agent
                logger.info("Created agent", name=agent_config.name)
            except Exception as e:
                logger.error(
                    "Failed to create agent", name=agent_config.name, error=str(e)
                )
                raise

        return agent_runnables

    def _build_workflow(self, orchestrator_runnable, agent_runnables):
        """Build the graph workflow."""
        workflow = StateGraph(GroupChatState)

        # Add nodes
        workflow.add_node(
            "orchestrator",
            partial(orchestrator_node, orchestrator_runnable=orchestrator_runnable),
        )
        workflow.add_node("aggregator", aggregator_node)

        for agent_name, agent_runnable in agent_runnables.items():
            workflow.add_node(agent_name, create_agent_node(agent_runnable, agent_name))

        # Add edges
        workflow.set_entry_point("orchestrator")

        # Orchestrator routing
        def orchestrator_router(state: GroupChatState):
            active_tasks = state.get("active_tasks", {})
            if not active_tasks:
                logger.info("No active tasks, ending")
                return END

            task_agents = list(active_tasks.keys())
            logger.info("Routing to agents", agents=task_agents)
            return task_agents

        workflow.add_conditional_edges("orchestrator", orchestrator_router)

        # Agent to aggregator edges
        for agent_name in agent_runnables.keys():
            workflow.add_edge(agent_name, "aggregator")

        # Aggregator routing
        def aggregator_router(state: GroupChatState):
            """Fixed aggregator routing that waits for all agents."""
            active_tasks = state.get("active_tasks", {})

            if not active_tasks:
                return END

            # Count agent responses since last orchestrator message
            agent_responses = set()

            # Look at messages in reverse to find agent responses
            for msg in reversed(state["messages"]):
                if hasattr(msg, "name"):
                    if msg.name == "orchestrator":
                        break  # Stop at last orchestrator message
                    elif msg.name in active_tasks:
                        agent_responses.add(msg.name)

            total_tasks = len(active_tasks)
            responses_received = len(agent_responses)

            logger.info(
                "Aggregator routing",
                total_tasks=total_tasks,
                responses_received=responses_received,
                responding_agents=list(agent_responses),
            )

            if responses_received >= total_tasks:
                logger.info("All agents responded, routing to orchestrator")
                return "orchestrator"
            else:
                logger.info(
                    f"Waiting for {total_tasks - responses_received} more responses"
                )
                return END

        workflow.add_conditional_edges(
            "aggregator", aggregator_router, {"orchestrator": "orchestrator", END: END}
        )

        return workflow


# Convenience function
def build_graph(team_config_path: Path, checkpointer: Optional[Any] = None):
    """Build graph using the enhanced builder."""
    builder = GraphBuilder()
    return builder.build_graph(team_config_path, checkpointer)
