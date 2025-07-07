"""Enhanced graph builder with subgraph integration and robust cursor-based state management."""

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
from .state import (
    GroupChatState,
    get_state_debug_info,
    get_task_completion_summary,
    apply_state_validation,
)
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
        Build the main graph workflow with subgraphs and ENHANCED routing logic.

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

        # ENHANCED Orchestrator routing
        def orchestrator_router(state: GroupChatState):
            """ENHANCED route from orchestrator based on TaskInfo status."""
            active_tasks = state.get("active_tasks", {})

            # Get comprehensive task status
            completion_summary = get_task_completion_summary(state)

            logger.info(
                "ORCHESTRATOR ROUTER - analyzing task assignment", **completion_summary
            )

            if not active_tasks:
                logger.info("No active tasks assigned, ending conversation")
                return END

            # Only route to agents that have pending or in_progress tasks
            agents_to_route = []
            for agent_name, task_info in active_tasks.items():
                if task_info.status in ["pending", "in_progress"]:
                    agents_to_route.append(agent_name)
                    logger.info(
                        f"Will route to agent {agent_name} (status: {task_info.status})",
                        task_preview=task_info.task_description[:100],
                    )
                else:
                    logger.info(
                        f"NOT routing to agent {agent_name} (status: {task_info.status})"
                    )

            if not agents_to_route:
                logger.info(
                    "No agents need to work (all finished), ending conversation"
                )
                return END

            logger.info(
                "ORCHESTRATOR ROUTER DECISION - routing to agents",
                agents=agents_to_route,
                agent_count=len(agents_to_route),
                total_tasks=len(active_tasks),
            )

            # Validate state before routing
            apply_state_validation(state, "before orchestrator routing")

            return agents_to_route

        workflow.add_conditional_edges("orchestrator", orchestrator_router)

        # Agent to aggregator edges
        for agent_name in agent_subgraphs.keys():
            workflow.add_edge(agent_name, "aggregator")

        # COMPLETELY REDESIGNED Aggregator routing based on TaskInfo status
        def aggregator_router(state: GroupChatState):
            """
            ENHANCED aggregator routing using TaskInfo status as authoritative source.

            This is the critical fix - instead of relying on fragile set logic,
            we examine TaskInfo objects directly to make routing decisions.
            """
            # Get comprehensive completion summary
            completion_summary = get_task_completion_summary(state)

            # Log detailed routing analysis
            logger.info(
                "AGGREGATOR ROUTER - comprehensive analysis", **completion_summary
            )

            # Early exit if no tasks
            if not completion_summary["has_tasks"]:
                logger.info("AGGREGATOR ROUTER: No active tasks, ending conversation")
                return END

            # The key insight: use TaskInfo status as the authoritative source
            active_tasks = state.get("active_tasks", {})

            # Detailed analysis for logging
            status_breakdown = {}
            for agent_name, task_info in active_tasks.items():
                status_breakdown[agent_name] = {
                    "status": task_info.status,
                    "is_finished": task_info.is_finished(),
                    "duration": task_info.get_duration(),
                    "task_preview": task_info.task_description[:50],
                }

            logger.info(
                "AGGREGATOR ROUTER - detailed status breakdown",
                status_breakdown=status_breakdown,
            )

            # Core routing logic based on completion_summary
            if completion_summary["all_finished"]:
                logger.info(
                    "AGGREGATOR ROUTER DECISION: ALL agents finished - routing to ORCHESTRATOR",
                    completed_agents=completion_summary["completed_agents"],
                    failed_agents=completion_summary["failed_agents"],
                    total_finished=completion_summary["finished_count"],
                    total_tasks=completion_summary["total_tasks"],
                )

                # Validate state before routing back to orchestrator
                validation_passed = apply_state_validation(
                    state, "before routing to orchestrator"
                )
                if not validation_passed:
                    logger.error(
                        "State validation failed before routing to orchestrator!"
                    )

                return "orchestrator"
            else:
                # Some agents still working
                still_working = [
                    name
                    for name, task in active_tasks.items()
                    if not task.is_finished()
                ]

                logger.info(
                    "AGGREGATOR ROUTER DECISION: Some agents still working - ENDING this branch",
                    still_working_agents=still_working,
                    still_working_count=len(still_working),
                    finished_count=completion_summary["finished_count"],
                    total_tasks=completion_summary["total_tasks"],
                    completion_percentage=completion_summary["completion_percentage"],
                )

                return END

        workflow.add_conditional_edges(
            "aggregator", aggregator_router, {"orchestrator": "orchestrator", END: END}
        )

        logger.info("Enhanced workflow structure built successfully")
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
            "routing_logic": "enhanced_taskinfo_based",
            "aggregation_mode": "batch_completion_processing",
            "state_validation": "enabled",
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
                "Enhanced graph structure",
                nodes=list(graph.nodes.keys()) if hasattr(graph, "nodes") else "N/A",
                agent_count=len(team_config.agents),
                team_name=team_config.team_name,
                routing_enhancements="TaskInfo-based routing enabled",
                aggregation_enhancements="Batch completion processing enabled",
                state_validation="Enabled throughout pipeline",
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
