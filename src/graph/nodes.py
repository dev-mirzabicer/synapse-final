"""Enhanced LangGraph nodes with subgraph-based orchestrator and aggregator."""

from typing import Callable, List, Dict, Any
from datetime import datetime
from functools import partial

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential

from src.configs.models import AgentConfig, TeamConfig

from .state import GroupChatState, TaskInfo, get_agent_state, get_state_debug_info
from .personalizer import personalizer
from .messages import (
    from_langchain_messages,
    OrchestratorMessage,
    SynapseToolMessage,
    has_finish_turn_marker,
)
from src.core.logging import get_logger
from src.core.exceptions import AgentError
from src.configs.settings import settings

logger = get_logger(__name__)


class OrchestratorSubgraph:
    """
    Orchestrator subgraph with cursor-based personalization but different completion logic.
    """

    def __init__(self, orchestrator_runnable):
        self.orchestrator_runnable = orchestrator_runnable
        self.logger = get_logger("orchestrator_subgraph")

    def create_subgraph(self) -> StateGraph:
        """
        Create the orchestrator subgraph.

        Returns:
            Compiled StateGraph for orchestrator
        """
        subgraph = StateGraph(GroupChatState)

        # Add nodes
        subgraph.add_node("update_personal_history", self._update_personal_history)
        subgraph.add_node("invoke_orchestrator", self._invoke_orchestrator)
        subgraph.add_node("process_response", self._process_response)

        # Set up flow
        subgraph.add_edge("update_personal_history", "invoke_orchestrator")
        subgraph.add_edge("invoke_orchestrator", "process_response")

        # Entry and exit points
        subgraph.set_entry_point("update_personal_history")
        subgraph.set_finish_point("process_response")

        return subgraph.compile()

    def _update_personal_history(self, state: GroupChatState) -> Dict[str, Any]:
        """Update orchestrator's personal history using cursor-based approach."""
        self.logger.info("Updating orchestrator personal history")

        try:
            personalizer.update_agent_history(state, "orchestrator")

            # Log statistics
            stats = personalizer.get_agent_history_stats(state, "orchestrator")
            self.logger.debug("Orchestrator history update stats", **stats)

            return {}

        except Exception as e:
            self.logger.error(
                "Failed to update orchestrator personal history", error=str(e)
            )
            raise AgentError(
                "orchestrator", f"Failed to update personal history: {e}", e
            )

    def _invoke_orchestrator(self, state: GroupChatState) -> Dict[str, Any]:
        """Invoke the orchestrator with updated personal history."""
        agent_state = get_agent_state(state, "orchestrator")

        try:
            self.logger.info(
                "Invoking orchestrator", round=state.get("round_number", 0)
            )

            # Invoke orchestrator with personal history
            result = self.orchestrator_runnable.invoke(
                {"messages": agent_state.personal_history}
            )

            # Store result for processing
            agent_state.temp_new_history = [
                result
            ]  # Orchestrator returns single message

            self.logger.info("Orchestrator invocation completed")

            return {}

        except Exception as e:
            self.logger.error("Orchestrator invocation failed", error=str(e))
            raise AgentError("orchestrator", f"Orchestrator invocation failed: {e}", e)

    def _process_response(self, state: GroupChatState) -> Dict[str, Any]:
        """Process orchestrator response and extract task assignments."""
        agent_state = get_agent_state(state, "orchestrator")

        try:
            if not agent_state.temp_new_history:
                self.logger.error("No orchestrator response to process")
                return {"messages": []}

            orchestrator_message = agent_state.temp_new_history[0]
            orchestrator_message.name = "orchestrator"

            if len(agent_state.temp_new_history) > 1:
                self.logger.warning(
                    "Orchestrator returned multiple messages, only processing the first"
                )

            # Process tool calls for task assignments
            active_tasks = {}
            messages_to_add = []

            # Add the orchestrator message as OrchestratorMessage
            synapse_orchestrator_msg = OrchestratorMessage(
                content=orchestrator_message.content,
                tool_calls=getattr(orchestrator_message, "tool_calls", []),
            )
            messages_to_add.append(synapse_orchestrator_msg)

            # Process tool calls
            if (
                hasattr(orchestrator_message, "tool_calls")
                and orchestrator_message.tool_calls
            ):
                for tool_call in orchestrator_message.tool_calls:
                    if tool_call["name"] == "assign_tasks":
                        tasks = tool_call["args"].get("tasks", {})

                        # Create TaskInfo objects
                        for agent_name, task_desc in tasks.items():
                            active_tasks[agent_name] = TaskInfo(
                                agent_name=agent_name,
                                task_description=task_desc,
                                assigned_at=datetime.now(),
                                status="pending",
                            )

                            # Get agent state and prepare for new task
                            task_agent_state = get_agent_state(state, agent_name)
                            task_agent_state.current_task = active_tasks[agent_name]
                            # Agent is ready to start a new task (not currently working)
                            task_agent_state.turn_finished = True
                            task_agent_state.continuation_attempts = 0
                            task_agent_state.temp_new_history = None

                        self.logger.info(
                            "Tasks assigned by orchestrator", tasks=list(tasks.keys())
                        )

                        # Create tool message
                        tool_message = SynapseToolMessage(
                            content=f"Tasks assigned: {', '.join(tasks.keys())}",
                            tool_call_id=tool_call["id"],
                            caller_agent="orchestrator",
                            tool_name="assign_tasks",
                        )
                        messages_to_add.append(tool_message)

            # Clean up temp state
            personalizer.reset_agent_temp_state(state, "orchestrator")

            # Build update dict - only include what's changing
            updates = {
                "messages": messages_to_add,
                "round_number": state.get("round_number", 0) + 1,
            }

            # Only update active_tasks if there are new ones
            if active_tasks:
                # Merge with existing active tasks
                current_active = state.get("active_tasks", {}).copy()
                current_active.update(active_tasks)
                updates["active_tasks"] = current_active

                # Clear completed/failed tasks when new tasks are assigned
                updates["completed_tasks"] = set()
                updates["failed_tasks"] = set()

            return updates

        except Exception as e:
            self.logger.error("Failed to process orchestrator response", error=str(e))
            # Clean up temp state even on error
            personalizer.reset_agent_temp_state(state, "orchestrator")

            # Create error message
            error_msg = OrchestratorMessage(
                content=f"I encountered an error: {str(e)}."
            )
            return {
                "messages": [error_msg],
                "error_count": state.get("error_count", 0) + 1,
            }


def create_orchestrator_node(orchestrator_runnable) -> Callable:
    """
    Create orchestrator node using subgraph pattern.

    Args:
        orchestrator_runnable: The orchestrator's LangChain runnable

    Returns:
        Orchestrator node function
    """
    orchestrator_subgraph_builder = OrchestratorSubgraph(orchestrator_runnable)
    orchestrator_subgraph = orchestrator_subgraph_builder.create_subgraph()

    def orchestrator_node(state: GroupChatState) -> Dict[str, Any]:
        """Orchestrator node that uses subgraph."""
        try:
            # Log state before invocation
            logger.debug("Orchestrator node input state", **get_state_debug_info(state))

            result = orchestrator_subgraph.invoke(state)

            # Log what we're returning
            if "messages" in result:
                logger.debug(
                    f"Orchestrator returning {len(result['messages'])} new messages"
                )

            return result
        except Exception as e:
            logger.error("Orchestrator subgraph failed", error=str(e))
            # Return error state
            error_msg = OrchestratorMessage(
                content=f"Orchestrator encountered an error: {str(e)}"
            )
            return {
                "messages": [error_msg],
                "error_count": state.get("error_count", 0) + 1,
            }

    return orchestrator_node


def create_agent_node(agent_subgraph: StateGraph, agent_name: str) -> Callable:
    """
    Create an agent node using the agent subgraph.

    Args:
        agent_subgraph: Compiled agent subgraph
        agent_name: Name of the agent

    Returns:
        Agent node function
    """

    def agent_node(state: GroupChatState) -> Dict[str, Any]:
        """Agent node that uses subgraph."""
        agent_logger = get_logger("agent_node").bind(agent_name=agent_name)
        agent_logger.info("Starting agent execution")

        try:
            # Get task info
            task_info = state["active_tasks"].get(agent_name)
            if not task_info:
                agent_logger.warning("No active task found")
                return {}

            # Update task status
            task_info.status = "in_progress"

            agent_logger.info(
                "Executing agent subgraph", task=task_info.task_description[:100]
            )

            # Log state before invocation
            agent_logger.debug("Agent node input state", **get_state_debug_info(state))

            # Execute agent subgraph
            result = agent_subgraph.invoke(state)

            # Log what we're returning
            if "messages" in result:
                agent_logger.debug(
                    f"Agent returning {len(result['messages'])} new messages"
                )

            agent_logger.info("Agent subgraph completed successfully")
            return result

        except Exception as e:
            agent_logger.error("Agent subgraph execution failed", error=str(e))

            # Update task status
            if task_info:
                task_info.status = "failed"
                task_info.error_message = str(e)

            # Create error message
            from .messages import AgentMessage

            error_message = AgentMessage(
                content=f"I encountered an error while working on the task: {str(e)}",
                agent_name=agent_name,
                is_private=False,
            )

            return {
                "messages": [error_message],
                "failed_tasks": {agent_name},
                "error_count": state.get("error_count", 0) + 1,
            }

    return agent_node


def aggregator_node(state: GroupChatState) -> Dict[str, Any]:
    """
    Enhanced aggregator node that properly tracks all agent completions.
    """
    logger.info("Starting aggregator")

    # Check if we have any new messages
    if not state.get("messages"):
        logger.warning("No messages to aggregate")
        return {}

    last_message = state["messages"][-1]

    # Determine sender name
    sender_name = None
    if hasattr(last_message, "agent_name"):
        sender_name = last_message.agent_name
    elif isinstance(last_message, OrchestratorMessage):
        sender_name = "orchestrator"

    if not sender_name:
        logger.debug("Could not determine sender name from last message")
        return {}

    # Skip aggregation for orchestrator messages (they handle their own state)
    if sender_name == "orchestrator":
        logger.debug("Skipping aggregation for orchestrator message")
        return {}

    # Update completion status for agent tasks
    completed_tasks = state.get("completed_tasks", set()).copy()
    failed_tasks = state.get("failed_tasks", set()).copy()
    active_tasks = state.get("active_tasks", {})

    if sender_name in active_tasks:
        task_info = active_tasks[sender_name]

        # Check if task failed based on error indicators
        has_error = False
        if hasattr(last_message, "is_private") and last_message.is_private:
            has_error = "error" in last_message.content.lower()

        # Check if this was an error message from the agent subgraph
        if sender_name in state.get("failed_tasks", set()):
            has_error = True

        # Check if the task was marked as failed
        if hasattr(task_info, "status") and task_info.status == "failed":
            has_error = True

        if has_error:
            failed_tasks.add(sender_name)
            task_info.status = "failed"
            logger.info("Agent task marked as failed", agent=sender_name)
        else:
            completed_tasks.add(sender_name)
            task_info.status = "completed"
            logger.info("Agent task marked as completed", agent=sender_name)

            # Reset the agent's turn state for next time
            agent_state = get_agent_state(state, sender_name)
            agent_state.reset_turn_state()

    total_tasks = len(active_tasks)
    completed_count = len(completed_tasks)
    failed_count = len(failed_tasks)

    logger.info(
        "Aggregator status",
        completed=completed_count,
        failed=failed_count,
        total=total_tasks,
        still_pending=total_tasks - completed_count - failed_count,
    )

    # Build updates - only return what's changing
    updates = {}

    if completed_tasks != state.get("completed_tasks", set()):
        updates["completed_tasks"] = completed_tasks

    if failed_tasks != state.get("failed_tasks", set()):
        updates["failed_tasks"] = failed_tasks

    # Update active_tasks with modified task info
    if active_tasks and any(task.status != "pending" for task in active_tasks.values()):
        # Return the full active_tasks dict since we modified task statuses
        updates["active_tasks"] = active_tasks

    return updates


class AgentSubgraph:
    """
    Creates and manages an agent subgraph with cursor-based personalization and turn management.
    """

    def __init__(self, agent_config: AgentConfig, team_config: TeamConfig, react_agent):
        self.agent_name = agent_config.name
        self.agent_config = agent_config
        self.team_config = team_config
        self.react_agent = react_agent
        self.logger = get_logger(f"agent_subgraph.{self.agent_name}")
        self.max_continuation_attempts = 3  # Prevent infinite loops

    def create_subgraph(self) -> StateGraph:
        """
        Create the agent subgraph with proper flow control.

        Returns:
            Compiled StateGraph for this agent
        """
        subgraph = StateGraph(GroupChatState)

        # Add nodes
        subgraph.add_node("update_personal_history", self._update_personal_history)
        subgraph.add_node("invoke_react_agent", self._invoke_react_agent)
        subgraph.add_node("check_turn_completion", self._check_turn_completion)
        subgraph.add_node("extract_new_messages", self._extract_new_messages)

        # Set up flow
        subgraph.add_edge("update_personal_history", "invoke_react_agent")
        subgraph.add_edge("invoke_react_agent", "check_turn_completion")

        # Conditional routing after turn check
        subgraph.add_conditional_edges(
            "check_turn_completion",
            self._route_after_turn_check,
            {"continue": "invoke_react_agent", "finished": "extract_new_messages"},
        )

        # Entry and exit points
        subgraph.set_entry_point("update_personal_history")
        subgraph.set_finish_point("extract_new_messages")

        return subgraph.compile()

    def _update_personal_history(self, state: GroupChatState) -> Dict[str, Any]:
        """
        Update agent's personal history using cursor-based approach.

        Args:
            state: Current global state

        Returns:
            Empty dict (state updated in-place via references)
        """
        self.logger.info("Updating personal history", agent=self.agent_name)

        try:
            # Use personalizer to update history based on cursor
            personalizer.update_agent_history(state, self.agent_name)

            # Log statistics for debugging
            stats = personalizer.get_agent_history_stats(state, self.agent_name)
            self.logger.debug("History update stats", **stats)

            return {}

        except Exception as e:
            self.logger.error("Failed to update personal history", error=str(e))
            raise AgentError(
                self.agent_name, f"Failed to update personal history: {e}", e
            )

    def _invoke_react_agent(self, state: GroupChatState) -> Dict[str, Any]:
        """
        Invoke the ReAct agent with updated personal history or continuation.

        Args:
            state: Current global state

        Returns:
            Empty dict (temp state updated in agent_state)
        """
        agent_state = get_agent_state(state, self.agent_name)

        try:
            # Determine which history to use
            if agent_state.has_temp_history():
                # Continue from temp history
                history_to_use = agent_state.temp_new_history
                self.logger.info(
                    "Continuing from temp history",
                    agent=self.agent_name,
                    temp_length=len(history_to_use),
                )
            else:
                # Start fresh with personal history
                history_to_use = agent_state.personal_history
                # Mark that we're starting work on this turn
                agent_state.start_new_turn()
                self.logger.info(
                    "Starting with personal history",
                    agent=self.agent_name,
                    history_length=len(history_to_use),
                )

            # Get current task for context
            task_info = state["active_tasks"].get(self.agent_name)
            if task_info:
                self.logger.debug(
                    "Agent task context",
                    agent=self.agent_name,
                    task=task_info.task_description[:100] + "...",
                )

            # Invoke ReAct agent
            result = self.react_agent.invoke({"messages": history_to_use})

            # Store complete new history in temp state
            agent_state.temp_new_history = result["messages"]

            self.logger.info(
                "ReAct agent invocation completed",
                agent=self.agent_name,
                result_length=len(result["messages"]),
            )

            return {}

        except Exception as e:
            self.logger.error(
                "ReAct agent invocation failed", agent=self.agent_name, error=str(e)
            )
            raise AgentError(self.agent_name, f"ReAct agent invocation failed: {e}", e)

    def _check_turn_completion(self, state: GroupChatState) -> Dict[str, Any]:
        """
        Check if agent called finish_my_turn, otherwise prompt to continue.

        Args:
            state: Current global state

        Returns:
            Dict with turn completion status stored in agent state
        """
        agent_state = get_agent_state(state, self.agent_name)

        if not agent_state.has_temp_history():
            self.logger.error("No temp history found during turn completion check")
            # Store completion status in agent state for routing
            agent_state.turn_finished = True
            return {}

        # Determine original length to extract new messages
        original_length = len(agent_state.personal_history)

        # Extract new messages for analysis
        new_messages = personalizer.extract_new_messages_from_result(
            agent_state.temp_new_history, original_length
        )

        self.logger.debug(
            "Checking turn completion",
            agent=self.agent_name,
            new_messages_count=len(new_messages),
            message_contents=[
                getattr(msg, "content", "NO_CONTENT")[:100] for msg in new_messages
            ],
        )

        # Check for [FINISH_TURN] marker in new messages
        has_finish_marker = False
        for i, msg in enumerate(new_messages):
            if (
                hasattr(msg, "content")
                and msg.content
                and not isinstance(msg, HumanMessage)
            ):
                content = str(msg.content)
                has_marker = has_finish_turn_marker(content)
                self.logger.debug(
                    f"Checking message {i} for finish marker",
                    agent=self.agent_name,
                    content_preview=content[:100],
                    has_marker=has_marker,
                )
                if has_marker:
                    has_finish_marker = True
                    self.logger.info(
                        "Found [FINISH_TURN] marker in message content",
                        agent=self.agent_name,
                        message_index=i,
                        content_preview=content[:50] + "..."
                        if len(content) > 50
                        else content,
                    )
                    break

        if has_finish_marker:
            self.logger.info("Agent finished turn successfully", agent=self.agent_name)
            # Store completion status in agent state
            agent_state.turn_finished = True
            agent_state.continuation_attempts = 0
            return {}

        # Log that no finish marker was found
        self.logger.info(
            "No [FINISH_TURN] marker found in messages",
            agent=self.agent_name,
            new_message_count=len(new_messages),
        )

        # Check continuation attempts to prevent infinite loops
        attempts = getattr(agent_state, "continuation_attempts", 0)
        if attempts >= self.max_continuation_attempts:
            self.logger.warning(
                "Max continuation attempts reached, forcing turn end",
                agent=self.agent_name,
                attempts=attempts,
            )
            # Force completion
            agent_state.turn_finished = True
            return {}

        # Agent hasn't finished turn, add continuation prompt
        self.logger.info(
            "Agent turn not finished, adding continuation prompt", agent=self.agent_name
        )

        continuation_msg = HumanMessage(
            content="[NOT_A_USER | System] To end your turn, add [FINISH_TURN] at the end of your message. Otherwise, you may continue working on your task.",
            additional_kwargs={"continuation": self.agent_name},
        )

        # Add continuation message to temp history
        agent_state.temp_new_history.append(continuation_msg)

        # Update continuation attempts in agent state
        agent_state.continuation_attempts = attempts + 1

        # Mark as not finished
        agent_state.turn_finished = False

        self.logger.debug(
            "Added continuation prompt", agent=self.agent_name, attempt=attempts + 1
        )

        return {}

    def _extract_new_messages(self, state: GroupChatState) -> Dict[str, Any]:
        """
        Extract and convert new messages for global history.

        Args:
            state: Current global state

        Returns:
            Dict with new messages for global state
        """
        agent_state = get_agent_state(state, self.agent_name)

        try:
            if not agent_state.has_temp_history():
                self.logger.warning("No temp history to extract from")
                return {"messages": []}

            # Calculate original length before invocation
            original_length = len(agent_state.personal_history)

            # Extract new messages
            # new_lc_messages = personalizer.extract_new_messages_from_result(
            #     agent_state.temp_new_history, original_length
            # )

            # Convert to SynapseMessages for global history
            new_synapse_messages = from_langchain_messages(
                agent_state.temp_new_history,  # Full history for context
                self.agent_name,
                original_length,  # Only extract new ones
            )

            # Clean up temp state and turn completion flags
            personalizer.reset_agent_temp_state(state, self.agent_name)

            # Reset agent completion state
            agent_state.turn_finished = True  # Ensure it's marked as finished
            agent_state.continuation_attempts = 0  # Reset for next time

            self.logger.info(
                "Successfully extracted new messages and completed turn",
                agent=self.agent_name,
                new_messages_count=len(new_synapse_messages),
            )

            # Log message types for debugging
            message_types = [type(msg).__name__ for msg in new_synapse_messages]
            self.logger.debug(
                "New message types", agent=self.agent_name, types=message_types
            )

            return {"messages": new_synapse_messages}

        except Exception as e:
            self.logger.error(
                "Failed to extract new messages", agent=self.agent_name, error=str(e)
            )
            # Clean up temp state even on error
            personalizer.reset_agent_temp_state(state, self.agent_name)

            # Reset agent state
            agent_state.turn_finished = True
            agent_state.continuation_attempts = 0

            raise AgentError(self.agent_name, f"Failed to extract new messages: {e}", e)

    def _route_after_turn_check(self, state: GroupChatState) -> str:
        """
        Route after turn completion check using agent's internal state.

        Args:
            state: Current global state

        Returns:
            "continue" or "finished"
        """
        agent_state = get_agent_state(state, self.agent_name)

        # Check the agent's turn completion status
        turn_finished = getattr(agent_state, "turn_finished", True)

        self.logger.debug(
            "Routing decision after turn check",
            agent=self.agent_name,
            turn_finished=turn_finished,
            continuation_attempts=getattr(agent_state, "continuation_attempts", 0),
        )

        if turn_finished:
            self.logger.info(
                "Agent turn finished, routing to extract messages",
                agent=self.agent_name,
            )
            return "finished"
        else:
            self.logger.info(
                "Agent turn not finished, routing to continue", agent=self.agent_name
            )
            return "continue"


def create_agent_subgraph(
    agent_config: AgentConfig, team_config: TeamConfig, react_agent
) -> StateGraph:
    """
    Create an agent subgraph.

    Args:
        agent_config: Configuration for the agent
        team_config: Team configuration
        react_agent: The ReAct agent instance

    Returns:
        Compiled subgraph for the agent
    """
    subgraph_builder = AgentSubgraph(agent_config, team_config, react_agent)
    return subgraph_builder.create_subgraph()
