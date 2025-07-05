"""Enhanced graph builder with agent subgraphs and sophisticated state management."""

import json
from pathlib import Path
from functools import partial
from typing import Optional, Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.configs.models import TeamConfig
from src.agents.factory import AgentFactory
from src.core.logging import get_logger
from src.core.exceptions import ConfigurationError
from src.configs.settings import settings
from src.graph.state import GroupChatState, AgentPersonalState
from src.graph.nodes import (
    orchestrator_node,
    create_agent_subgraph_node,
    aggregator_node,
    create_legacy_agent_node,
)

logger = get_logger(__name__)


class GraphBuilder:
    """Enhanced graph builder with agent subgraphs and improved state management."""

    def __init__(self, use_subgraphs: bool = True):
        self.agent_factory = AgentFactory()
        self.use_subgraphs = use_subgraphs

    def build_graph(self, team_config_path: Path, checkpointer: Optional[Any] = None):
        """Build and compile the enhanced LangGraph with agent subgraphs."""
        logger.info("Building enhanced graph", config_path=str(team_config_path))

        try:
            # Load and validate configuration
            team_config = self._load_team_config(team_config_path)

            # Create orchestrator
            orchestrator_runnable = self._create_orchestrator(team_config)

            # Create agent components (subgraphs or legacy agents)
            if self.use_subgraphs:
                agent_components = self._create_agent_subgraphs(team_config)
                logger.info("Using agent subgraphs for enhanced functionality")
            else:
                agent_components = self._create_legacy_agents(team_config)
                logger.info("Using legacy ReAct agents for backwards compatibility")

            # Build the main workflow
            workflow = self._build_enhanced_workflow(
                orchestrator_runnable, agent_components
            )

            # Compile with checkpointer
            if checkpointer is None:
                checkpointer = MemorySaver()

            graph = workflow.compile(
                checkpointer=checkpointer, debug=settings.log_level == "DEBUG"
            )

            logger.info("Enhanced graph compiled successfully")
            return graph

        except Exception as e:
            logger.error("Failed to build enhanced graph", error=str(e))
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
        return self.agent_factory.create_orchestrator_agent(team_config)

    def _create_agent_subgraphs(self, team_config: TeamConfig):
        """Create agent subgraphs for enhanced functionality."""
        agent_subgraphs = {}

        for agent_config in team_config.agents:
            try:
                agent_subgraph = self.agent_factory.create_agent_subgraph(
                    agent_config, team_config
                )
                agent_subgraphs[agent_config.name] = agent_subgraph
                logger.info("Created agent subgraph", name=agent_config.name)
            except Exception as e:
                logger.error(
                    "Failed to create agent subgraph",
                    name=agent_config.name,
                    error=str(e),
                )
                raise

        return agent_subgraphs

    def _create_legacy_agents(self, team_config: TeamConfig):
        """Create legacy ReAct agents for backwards compatibility."""
        legacy_agents = {}

        for agent_config in team_config.agents:
            try:
                legacy_agent = self.agent_factory.create_legacy_agent(
                    agent_config, team_config
                )
                legacy_agents[agent_config.name] = legacy_agent
                logger.info("Created legacy agent", name=agent_config.name)
            except Exception as e:
                logger.error(
                    "Failed to create legacy agent",
                    name=agent_config.name,
                    error=str(e),
                )
                raise

        return legacy_agents

    def _build_enhanced_workflow(self, orchestrator_runnable, agent_components):
        """Build the enhanced workflow with sophisticated routing."""
        workflow = StateGraph(GroupChatState)

        # Add orchestrator node
        workflow.add_node(
            "orchestrator",
            partial(orchestrator_node, orchestrator_runnable=orchestrator_runnable),
        )

        # Add aggregator node
        workflow.add_node("aggregator", aggregator_node)

        # Add agent nodes (subgraphs or legacy)
        for agent_name, agent_component in agent_components.items():
            if self.use_subgraphs:
                workflow.add_node(
                    agent_name, create_agent_subgraph_node(agent_component, agent_name)
                )
            else:
                workflow.add_node(
                    agent_name, create_legacy_agent_node(agent_component, agent_name)
                )

        # Set entry point
        workflow.set_entry_point("orchestrator")

        # Enhanced orchestrator routing
        def enhanced_orchestrator_router(state: GroupChatState):
            """Enhanced routing that handles agent state initialization."""
            active_tasks = state.get("active_tasks", {})

            logger.debug(
                "Orchestrator router called",
                active_tasks_count=len(active_tasks),
                active_tasks_keys=list(active_tasks.keys()),
            )

            if not active_tasks:
                logger.info("No active tasks, ending")
                return END

            # Initialize agent states for new agents
            agent_states = state.get("agent_states", {})
            for agent_name in active_tasks.keys():
                if agent_name not in agent_states:
                    agent_states[agent_name] = AgentPersonalState(agent_name=agent_name)
                    logger.info("Initialized state for new agent", agent=agent_name)

            task_agents = list(active_tasks.keys())
            logger.info("Routing to agents", agents=task_agents)
            return task_agents

        workflow.add_conditional_edges("orchestrator", enhanced_orchestrator_router)

        # Agent to aggregator edges
        for agent_name in agent_components.keys():
            workflow.add_edge(agent_name, "aggregator")

        # Enhanced aggregator routing
        def enhanced_aggregator_router(state: GroupChatState):
            """Enhanced aggregator routing with sophisticated completion tracking."""
            active_tasks = state.get("active_tasks", {})
            pending_responses = state.get("pending_agent_responses", set())

            if not active_tasks:
                logger.info("No active tasks remaining")
                return END

            # Check if all agents have responded
            all_responded = len(pending_responses) == 0

            logger.info(
                "Enhanced aggregator routing",
                total_tasks=len(active_tasks),
                pending_responses=len(pending_responses),
                all_responded=all_responded,
            )

            if all_responded:
                logger.info("All agents responded, routing to orchestrator")
                return "orchestrator"
            else:
                logger.info(f"Waiting for {len(pending_responses)} more responses")
                return END

        workflow.add_conditional_edges(
            "aggregator",
            enhanced_aggregator_router,
            {"orchestrator": "orchestrator", END: END},
        )

        return workflow

    def get_agent_state_summary(self, state: GroupChatState) -> Dict[str, Any]:
        """Get a summary of all agent states for debugging."""
        agent_states = state.get("agent_states", {})
        summary = {}

        for agent_name, agent_state in agent_states.items():
            summary[agent_name] = {
                "message_count": len(agent_state.personalized_messages),
                "private_message_count": len(agent_state.private_messages),
                "tool_call_count": len(agent_state.tool_call_history),
                "last_updated": agent_state.last_updated.isoformat(),
                "context_keys": list(agent_state.context.keys()),
            }

        return summary


# Convenience function
def build_graph(
    team_config_path: Path,
    checkpointer: Optional[Any] = None,
    use_subgraphs: bool = True,
):
    """Build graph using the enhanced builder."""
    builder = GraphBuilder(use_subgraphs=use_subgraphs)
    return builder.build_graph(team_config_path, checkpointer)
