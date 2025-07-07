"""Enhanced graph builder with subgraph integration and cursor-based state management."""

import json
from pathlib import Path
from typing import Optional, Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.configs.models import TeamConfig
from src.agents.factory import AgentFactory
from src.core.logging import get_logger
from src.core.exceptions import ConfigurationError
from src.configs.settings import settings
from .state import GroupChatState, get_state_debug_info
from .nodes import create_orchestrator_node, create_agent_node, aggregator_node

logger = get_logger(__name__)


def debug_node_wrapper(node_func, node_name):
    """
    Wrap node functions to debug their updates.

    This wrapper logs the state before and after each node execution,
    helping to track down state management issues.
    """

    def wrapped(state):
        # Log input state
        logger.debug(
            f"[{node_name}] PRE-EXECUTION state stats", **get_state_debug_info(state)
        )

        # Execute the node
        try:
            result = node_func(state)
        except Exception as e:
            logger.error(
                f"[{node_name}] Node execution failed",
                error=str(e),
                state_stats=get_state_debug_info(state),
            )
            raise

        # Log what the node is returning
        if result:
            updates_summary = {}
            if "messages" in result:
                updates_summary["new_messages"] = len(result.get("messages", []))
                for i, msg in enumerate(result["messages"][:3]):  # Log first 3
                    logger.debug(
                        f"[{node_name}] New message {i}",
                        msg_type=type(msg).__name__,
                        msg_id=getattr(msg, "message_id", "NO_ID")[:8],
                        content_preview=getattr(msg, "content", "")[:50],
                    )

            if "active_tasks" in result:
                updates_summary["active_tasks_update"] = list(
                    result["active_tasks"].keys()
                )

            if "completed_tasks" in result:
                updates_summary["completed_tasks_update"] = list(
                    result["completed_tasks"]
                )

            if "failed_tasks" in result:
                updates_summary["failed_tasks_update"] = list(result["failed_tasks"])

            if "round_number" in result:
                updates_summary["round_number"] = result["round_number"]

            logger.debug(f"[{node_name}] Updates being returned", **updates_summary)
        else:
            logger.debug(f"[{node_name}] No updates returned")

        return result

    # Preserve the original function name for LangGraph
    wrapped.__name__ = node_func.__name__
    return wrapped


class GraphBuilder:
    """Enhanced graph builder with subgraph integration and agent state management."""

    def __init__(self):
        self.agent_factory = AgentFactory()
        self._enable_debug = settings.log_level == "DEBUG"

    def build_graph(self, team_config_path: Path, checkpointer: Optional[Any] = None):
        """
        Build and compile the enhanced LangGraph with subgraphs.

        Args:
            team_config_path: Path to team configuration JSON
            checkpointer: Optional checkpointer (defaults to MemorySaver)

        Returns:
            Compiled LangGraph
        """
        logger.info(
            "Building enhanced graph with subgraphs", config_path=str(team_config_path)
        )

        try:
            # Load and validate configuration
            team_config = self._load_team_config(team_config_path)

            # Validate the configuration thoroughly
            if not self.validate_team_config(team_config):
                raise ConfigurationError("Team configuration validation failed")

            # Create orchestrator runnable
            orchestrator_runnable = self._create_orchestrator_runnable(team_config)

            # Create agent subgraphs
            agent_subgraphs = self._create_agent_subgraphs(team_config)

            # Build the workflow
            workflow = self._build_workflow(
                orchestrator_runnable, agent_subgraphs, team_config
            )

            # Compile with checkpointer
            if checkpointer is None:
                checkpointer = MemorySaver()

            graph = workflow.compile(
                checkpointer=checkpointer  # , debug=self._enable_debug
            )

            logger.info("Enhanced graph compiled successfully")

            # Log the graph structure for debugging
            if self._enable_debug:
                self._log_graph_structure(graph, team_config)

            return graph

        except Exception as e:
            logger.error("Failed to build enhanced graph", error=str(e))
            raise ConfigurationError(f"Failed to build enhanced graph: {e}")

    def _load_team_config(self, config_path: Path) -> TeamConfig:
        """Load and validate team configuration."""
        try:
            config_data = json.loads(config_path.read_text())
            return TeamConfig(**config_data)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration file: {e}")

    def _create_orchestrator_runnable(self, team_config: TeamConfig):
        """Create the orchestrator runnable."""
        logger.info("Creating orchestrator runnable")
        return self.agent_factory.create_orchestrator_runnable(team_config)

    def _create_agent_subgraphs(self, team_config: TeamConfig) -> Dict[str, Any]:
        """
        Create all agent subgraphs.

        Args:
            team_config: Team configuration

        Returns:
            Dictionary of agent subgraphs
        """
        agent_subgraphs = {}

        for agent_config in team_config.agents:
            try:
                logger.info("Creating agent subgraph", name=agent_config.name)

                # Create agent subgraph
                subgraph = self.agent_factory.create_agent_subgraph(
                    agent_config, team_config
                )
                agent_subgraphs[agent_config.name] = subgraph

                logger.info(
                    "Successfully created agent subgraph", name=agent_config.name
                )

            except Exception as e:
                logger.error(
                    "Failed to create agent subgraph",
                    name=agent_config.name,
                    error=str(e),
                )
                raise

        return agent_subgraphs

    def _build_workflow(
        self,
        orchestrator_runnable,
        agent_subgraphs: Dict[str, Any],
        team_config: TeamConfig,
    ):
        """
        Build the main graph workflow with subgraphs.

        Args:
            orchestrator_runnable: Orchestrator LLM runnable
            agent_subgraphs: Dictionary of agent subgraphs
            team_config: Team configuration

        Returns:
            StateGraph workflow
        """
        workflow = StateGraph(GroupChatState)

        # Create orchestrator node
        orchestrator_node = create_orchestrator_node(orchestrator_runnable)

        # Wrap nodes with debug logging if enabled
        if self._enable_debug:
            orchestrator_node = debug_node_wrapper(orchestrator_node, "orchestrator")
            aggregator_node_wrapped = debug_node_wrapper(aggregator_node, "aggregator")
        else:
            aggregator_node_wrapped = aggregator_node

        workflow.add_node("orchestrator", orchestrator_node)

        # Add aggregator node
        workflow.add_node("aggregator", aggregator_node_wrapped)

        # Add agent subgraph nodes
        for agent_name, agent_subgraph in agent_subgraphs.items():
            agent_node = create_agent_node(agent_subgraph, agent_name)

            # Wrap with debug if enabled
            if self._enable_debug:
                agent_node = debug_node_wrapper(agent_node, agent_name)

            workflow.add_node(agent_name, agent_node)
            logger.info("Added agent node to workflow", agent=agent_name)

        # Set entry point
        workflow.set_entry_point("orchestrator")

        # Orchestrator routing
        def orchestrator_router(state: GroupChatState):
            """Route from orchestrator based on active tasks."""
            active_tasks = state.get("active_tasks", {})

            if not active_tasks:
                logger.info("No active tasks, ending conversation")
                return END

            # Only route to agents that have pending tasks
            task_agents = []
            for agent_name, task_info in active_tasks.items():
                if task_info.status == "pending":
                    task_agents.append(agent_name)

            if not task_agents:
                logger.info("No pending tasks, ending conversation")
                return END

            logger.info(
                "Routing to agents", agents=task_agents, task_count=len(task_agents)
            )
            return task_agents

        workflow.add_conditional_edges("orchestrator", orchestrator_router)

        # Agent to aggregator edges
        for agent_name in agent_subgraphs.keys():
            workflow.add_edge(agent_name, "aggregator")

        # Aggregator routing
        def aggregator_router(state: GroupChatState):
            """Enhanced aggregator routing that waits for all agents."""
            active_tasks = state.get("active_tasks", {})
            completed_tasks = state.get("completed_tasks", set())
            failed_tasks = state.get("failed_tasks", set())

            if not active_tasks:
                logger.info("No active tasks, ending conversation")
                return END

            # Agents that were assigned a task in this round
            tasked_agents = set(active_tasks.keys())
            # Agents that have finished their task (successfully or not)
            finished_agents = completed_tasks | failed_tasks

            logger.info(
                "Aggregator routing decision",
                tasked_agents=list(tasked_agents),
                finished_agents=list(finished_agents),
                completed_count=len(completed_tasks),
                failed_count=len(failed_tasks),
            )

            # If all assigned agents have finished, route back to orchestrator
            if tasked_agents.issubset(finished_agents):
                logger.info(
                    "All agents have completed their tasks, routing back to orchestrator."
                )
                return "orchestrator"
            else:
                # Some agents are still running, so this branch should end.
                # Another agent's branch will eventually trigger the aggregator again.
                still_running = tasked_agents - finished_agents
                logger.info(
                    f"Waiting for other agents to complete: {list(still_running)}. Ending this branch."
                )
                return END

        workflow.add_conditional_edges(
            "aggregator", aggregator_router, {"orchestrator": "orchestrator", END: END}
        )

        logger.info("Workflow structure built successfully")
        return workflow

    def get_workflow_info(self, team_config: TeamConfig) -> Dict[str, Any]:
        """
        Get information about the workflow structure.

        Args:
            team_config: Team configuration

        Returns:
            Dictionary with workflow information
        """
        agent_names = [agent.name for agent in team_config.agents]

        return {
            "team_name": team_config.team_name,
            "agent_count": len(agent_names),
            "agent_names": agent_names,
            "orchestrator_model": team_config.orchestrator_config.get(
                "model", "gemini-2.5-flash"
            ),
            "workflow_nodes": ["orchestrator", "aggregator"] + agent_names,
            "max_rounds": team_config.conversation_config.get("max_rounds", 20),
            "timeout_minutes": team_config.conversation_config.get(
                "timeout_minutes", 30
            ),
        }

    def validate_team_config(self, team_config: TeamConfig) -> bool:
        """
        Validate team configuration for subgraph compatibility.

        Args:
            team_config: Team configuration to validate

        Returns:
            True if valid, raises ConfigurationError if invalid
        """
        try:
            # Check for required fields
            if not team_config.team_name:
                raise ConfigurationError("Team name is required")

            if not team_config.agents:
                raise ConfigurationError("At least one agent is required")

            # Check agent configurations
            agent_names = set()
            for agent in team_config.agents:
                if agent.name in agent_names:
                    raise ConfigurationError(f"Duplicate agent name: {agent.name}")
                agent_names.add(agent.name)

                if not agent.system_prompt_template:
                    raise ConfigurationError(
                        f"Agent {agent.name} missing system prompt"
                    )

            # Check for reserved names
            reserved_names = {"orchestrator", "aggregator", "user", "system"}
            conflicts = agent_names & reserved_names
            if conflicts:
                raise ConfigurationError(
                    f"Agents cannot use reserved names: {conflicts}"
                )

            logger.info("Team configuration validation passed")
            return True

        except Exception as e:
            logger.error("Team configuration validation failed", error=str(e))
            raise

    def _log_graph_structure(self, graph, team_config: TeamConfig):
        """Log the compiled graph structure for debugging."""
        try:
            logger.debug(
                "Graph structure",
                nodes=list(graph.nodes.keys()) if hasattr(graph, "nodes") else "N/A",
                agent_count=len(team_config.agents),
                team_name=team_config.team_name,
            )
        except Exception as e:
            logger.debug("Could not log graph structure", error=str(e))


# Convenience function
def build_graph(team_config_path: Path, checkpointer: Optional[Any] = None):
    """
    Build enhanced graph using the subgraph builder.

    Args:
        team_config_path: Path to team configuration
        checkpointer: Optional checkpointer

    Returns:
        Compiled LangGraph
    """
    builder = GraphBuilder()
    return builder.build_graph(team_config_path, checkpointer)
