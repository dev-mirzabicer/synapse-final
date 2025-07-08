"""Enhanced LangGraph nodes with staged message processing and robust batch aggregation."""

from typing import Callable, Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

from src.configs.models import AgentConfig, TeamConfig

from .state import (
    GroupChatState,
    TaskInfo,
    get_agent_state,
    get_state_debug_info,
    get_task_completion_summary,
    apply_state_validation,
    MessageMerger,
)
from .personalizer import personalizer
from .messages import (
    from_langchain_messages,
    OrchestratorMessage,
    SynapseToolMessage,
    SynapseSystemMessage,
    has_finish_turn_marker,
)
from src.core.logging import get_logger
from src.core.exceptions import AgentError

logger = get_logger(__name__)


class OrchestratorSubgraph:
    """
    Orchestrator subgraph with direct global state updates and explicit task assignment.

    Key architectural changes:
    - Orchestrator messages go directly to global state (no pending queue)
    - Task assignments create explicit SynapseSystemMessage instances
    - Tool messages are properly scoped to orchestrator
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
        """
        Process orchestrator response with direct global state updates.

        This is the key architectural fix:
        - Orchestrator messages go directly to global state
        - Task assignments create explicit system messages
        - No pending message queue for orchestrator
        """
        agent_state = get_agent_state(state, "orchestrator")

        try:
            if not agent_state.temp_new_history:
                self.logger.error("No orchestrator response to process")
                # Return empty update - no messages to add
                return {"round_number": state.get("round_number", 0) + 1}

            orchestrator_message = agent_state.temp_new_history[0]
            orchestrator_message.name = "orchestrator"

            if len(agent_state.temp_new_history) > 1:
                self.logger.warning(
                    "Orchestrator returned multiple messages, only processing the first"
                )

            # Initialize collections for direct global state updates
            global_messages_to_add = []
            active_tasks = {}

            # Create the orchestrator message for global state
            synapse_orchestrator_msg = OrchestratorMessage(
                content=orchestrator_message.content,
                tool_calls=getattr(orchestrator_message, "tool_calls", []),
            )
            global_messages_to_add.append(synapse_orchestrator_msg)

            self.logger.info(
                "Processing orchestrator response with direct global state updates",
                has_tool_calls=bool(getattr(orchestrator_message, "tool_calls", [])),
                message_content_preview=orchestrator_message.content[:100],
            )

            # Process tool calls and create task assignments
            tool_calls = getattr(orchestrator_message, "tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call["name"] == "assign_tasks":
                        tasks = tool_call["args"].get("tasks", {})

                        self.logger.info(
                            "Processing task assignments",
                            task_count=len(tasks),
                            assigned_agents=list(tasks.keys()),
                        )

                        # Create TaskInfo objects and prepare agent states
                        for agent_name, task_desc in tasks.items():
                            active_tasks[agent_name] = TaskInfo(
                                agent_name=agent_name,
                                task_description=task_desc,
                                assigned_at=datetime.now(),
                                status="pending",
                            )

                            # Prepare agent state for new task
                            task_agent_state = get_agent_state(state, agent_name)
                            task_agent_state.current_task = active_tasks[agent_name]
                            task_agent_state.turn_finished = True  # Ready for new task
                            task_agent_state.continuation_attempts = 0
                            task_agent_state.temp_new_history = None

                            # Create explicit task assignment system message
                            task_assignment_msg = SynapseSystemMessage(
                                content=f"[TASK ASSIGNMENT] You have been assigned: {task_desc}",
                                target_agent=agent_name,
                            )
                            global_messages_to_add.append(task_assignment_msg)

                            self.logger.debug(
                                "Created task assignment",
                                agent=agent_name,
                                task_preview=task_desc[:50],
                            )

                        # Create private tool message for orchestrator (not visible to agents)
                        orchestrator_tool_message = SynapseToolMessage(
                            content=f"Tasks assigned: {', '.join(tasks.keys())}",
                            tool_call_id=tool_call["id"],
                            caller_agent="orchestrator",
                            tool_name="assign_tasks",
                        )
                        global_messages_to_add.append(orchestrator_tool_message)

                        self.logger.info(
                            "Task assignments processed",
                            assigned_agents=list(tasks.keys()),
                            global_messages_count=len(global_messages_to_add),
                        )

            # Clean up temp state
            personalizer.reset_agent_temp_state(state, "orchestrator")

            # Build direct global state updates (no pending messages)
            updates = {
                "messages": global_messages_to_add,  # Direct to global state
                "round_number": state.get("round_number", 0) + 1,
            }

            # Add active tasks if any were assigned
            if active_tasks:
                # Merge with existing active tasks
                current_active = state.get("active_tasks", {}).copy()
                current_active.update(active_tasks)
                updates["active_tasks"] = current_active

                # Clear completed/failed tasks when new tasks are assigned
                updates["completed_tasks"] = set()
                updates["failed_tasks"] = set()

            self.logger.info(
                "Orchestrator response processed with direct global state updates",
                global_messages_count=len(global_messages_to_add),
                new_tasks_count=len(active_tasks),
                update_keys=list(updates.keys()),
            )

            return updates

        except Exception as e:
            self.logger.error("Failed to process orchestrator response", error=str(e))
            # Clean up temp state even on error
            personalizer.reset_agent_temp_state(state, "orchestrator")

            # Create error message for direct global state
            error_msg = OrchestratorMessage(
                content=f"I encountered an error: {str(e)}."
            )
            return {
                "messages": [error_msg],  # Direct to global state
                "error_count": state.get("error_count", 0) + 1,
                "round_number": state.get("round_number", 0) + 1,
            }


def create_orchestrator_node(orchestrator_runnable) -> Callable:
    """
    Create orchestrator node using subgraph pattern with direct global state updates.

    Args:
        orchestrator_runnable: The orchestrator's LangChain runnable

    Returns:
        Orchestrator node function
    """
    orchestrator_subgraph_builder = OrchestratorSubgraph(orchestrator_runnable)
    orchestrator_subgraph = orchestrator_subgraph_builder.create_subgraph()

    def orchestrator_node(state: GroupChatState) -> Dict[str, Any]:
        """Orchestrator node that uses subgraph with direct global state updates."""
        try:
            # Log state before invocation
            logger.debug("Orchestrator node input state", **get_state_debug_info(state))

            # Execute subgraph - now returns direct global state updates
            result = orchestrator_subgraph.invoke(state)

            # Log what we're returning
            message_count = len(result.get("messages", []))
            if message_count > 0:
                logger.debug(
                    f"Orchestrator returning {message_count} messages for direct global state update"
                )

            return result
        except Exception as e:
            logger.error("Orchestrator subgraph failed", error=str(e))
            # Return error state with direct global updates
            error_msg = OrchestratorMessage(
                content=f"Orchestrator encountered an error: {str(e)}"
            )
            return {
                "messages": [error_msg],  # Direct to global state
                "error_count": state.get("error_count", 0) + 1,
                "round_number": state.get("round_number", 0) + 1,
            }

    return orchestrator_node


def create_agent_node(agent_subgraph: StateGraph, agent_name: str) -> Callable:
    """
    Create an agent node using the agent subgraph with staged message processing.

    Args:
        agent_subgraph: Compiled agent subgraph
        agent_name: Name of the agent

    Returns:
        Agent node function
    """

    def agent_node(state: GroupChatState) -> Dict[str, Any]:
        """Agent node that uses subgraph with staged message processing."""
        agent_logger = get_logger("agent_node").bind(agent_name=agent_name)
        agent_logger.info("Starting agent execution")

        try:
            # Get task info
            task_info = state["active_tasks"].get(agent_name)
            if not task_info:
                agent_logger.warning("No active task found")
                return {}

            # Update task status to in_progress
            task_info.mark_in_progress()

            agent_logger.info(
                "Executing agent subgraph", task=task_info.task_description[:100]
            )

            # Log state before invocation
            agent_logger.debug("Agent node input state", **get_state_debug_info(state))

            # Execute agent subgraph - this returns pending messages
            result = agent_subgraph.invoke(state)

            # Log what we're returning
            pending_count = len(result.get("pending_messages", []))
            if pending_count > 0:
                agent_logger.debug(f"Agent returning {pending_count} pending messages")

            agent_logger.info("Agent subgraph completed successfully")
            return result

        except Exception as e:
            agent_logger.error("Agent subgraph execution failed", error=str(e))

            # Mark task as failed in the task info
            if task_info:
                task_info.mark_failed(str(e))

            # Create error message for staged processing
            from .messages import AgentMessage

            error_message = AgentMessage(
                content=f"I encountered an error while working on the task: {str(e)}",
                agent_name=agent_name,
                is_private=False,
            )

            return {
                "pending_messages": [error_message],
                "pending_sources": {agent_name: [error_message.message_id]},
                "failed_tasks": {agent_name},
                "active_tasks": {agent_name: task_info} if task_info else {},
                "error_count": state.get("error_count", 0) + 1,
            }

    return agent_node


def aggregator_node(state: GroupChatState) -> Dict[str, Any]:
    """
    Enhanced aggregator node with atomic pending message merging and improved separation of concerns.

    This aggregator now has clearer separation:
    1. ALWAYS atomically merge pending messages (if any exist)
    2. Process task completion status updates
    3. Maintain state consistency throughout

    The routing logic is handled separately in the aggregator router.
    """
    logger.info("Starting aggregator with atomic pending message processing")

    # Phase 1: ATOMIC pending message merge (always happens if pending messages exist)
    updates = {}

    pending_messages = state.get("pending_messages", [])
    if pending_messages:
        pending_summary = MessageMerger.get_pending_summary(state)
        logger.info("Atomically merging pending messages", **pending_summary)

        # Atomic merge operation
        merge_updates = MessageMerger.merge_pending_messages(state)
        updates.update(merge_updates)

        logger.info(
            "Successfully merged pending messages atomically",
            merged_count=len(pending_messages),
            sources_processed=list(pending_summary.get("pending_sources", [])),
        )
    else:
        logger.debug("No pending messages to merge")

    # Phase 2: Process task completion status (existing logic, unchanged)
    completion_summary = get_task_completion_summary(state)
    logger.info("Aggregator - analyzing task completion status", **completion_summary)

    # Early exit if no tasks
    if not completion_summary["has_tasks"]:
        logger.info("No active tasks to process in aggregator")
        return updates

    active_tasks = state.get("active_tasks", {})

    # Process ALL task statuses in batch
    completed_agents = []
    failed_agents = []
    in_progress_agents = []
    pending_agents = []

    # Log detailed task analysis
    logger.info("Batch analysis - examining all task statuses:")
    for agent_name, task_info in active_tasks.items():
        status_summary = task_info.get_status_summary()
        logger.info(f"  {agent_name}: {status_summary}")

        if task_info.is_successful():
            completed_agents.append(agent_name)
        elif task_info.status == "failed":
            failed_agents.append(agent_name)
        elif task_info.is_in_progress():
            in_progress_agents.append(agent_name)
        else:  # pending
            pending_agents.append(agent_name)

    # Update completion tracking sets
    completed_tasks = set(completed_agents)
    failed_tasks = set(failed_agents)

    # Reset agent states for completed agents
    for agent_name in completed_agents:
        agent_state = get_agent_state(state, agent_name)
        agent_state.reset_turn_state()
        logger.info(f"Reset turn state for completed agent: {agent_name}")

    # Log comprehensive aggregation results
    logger.info(
        "Aggregator - batch processing results",
        total_tasks=len(active_tasks),
        completed_agents=completed_agents,
        completed_count=len(completed_agents),
        failed_agents=failed_agents,
        failed_count=len(failed_agents),
        in_progress_agents=in_progress_agents,
        in_progress_count=len(in_progress_agents),
        pending_agents=pending_agents,
        pending_count=len(pending_agents),
        all_finished=len(completed_agents) + len(failed_agents) == len(active_tasks),
    )

    # Build task completion updates
    current_completed = state.get("completed_tasks", set())
    current_failed = state.get("failed_tasks", set())

    if completed_tasks != current_completed:
        updates["completed_tasks"] = completed_tasks
        logger.info(
            f"Updating completed_tasks: {current_completed} -> {completed_tasks}"
        )

    if failed_tasks != current_failed:
        updates["failed_tasks"] = failed_tasks
        logger.info(f"Updating failed_tasks: {current_failed} -> {failed_tasks}")

    # Always return active_tasks to ensure task status updates are reflected
    if active_tasks:
        updates["active_tasks"] = active_tasks

    # Validate state consistency for the updates
    if updates:
        # Create a mock updated state for validation
        test_state = state.copy()
        test_state.update(updates)
        validation_passed = apply_state_validation(
            test_state, "after aggregator processing"
        )

        if not validation_passed:
            logger.error("Aggregator produced invalid state updates!")
        else:
            logger.debug("Aggregator state updates validated successfully")

    logger.info(
        "Aggregator completed with atomic message processing",
        update_keys=list(updates.keys()),
        updates_summary={
            "pending_messages_merged": "messages" in updates,
            "pending_queue_cleared": "pending_messages" in updates,
            "completed_tasks": list(updates.get("completed_tasks", [])),
            "failed_tasks": list(updates.get("failed_tasks", [])),
            "active_tasks_count": len(updates.get("active_tasks", {})),
        },
    )

    return updates


class AgentSubgraph:
    """
    Creates and manages an agent subgraph with staged message processing and enhanced completion tracking.

    This class remains largely unchanged, as the agent logic is working correctly.
    The key is that agents still return pending messages, which get processed by the aggregator.
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
        Create the agent subgraph with proper flow control and staged message processing.

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

        This should now properly include:
        - User messages
        - Orchestrator messages (now in global state)
        - Task assignment system messages (target_agent specific)

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
        Enhanced message extraction with staged processing and explicit task completion marking.

        This continues to use the pending message pattern for agents (only orchestrator uses direct updates).
        """
        agent_state = get_agent_state(state, self.agent_name)

        try:
            if not agent_state.has_temp_history():
                self.logger.warning("No temp history to extract from")
                return {"pending_messages": []}

            # Calculate original length before invocation
            original_length = len(agent_state.personal_history)

            # Convert to SynapseMessages for staged processing
            new_synapse_messages = from_langchain_messages(
                agent_state.temp_new_history,  # Full history for context
                self.agent_name,
                original_length,  # Only extract new ones
            )

            # CRITICAL FIX: Explicitly mark this agent's task as completed
            active_tasks = state.get("active_tasks", {})
            task_updates = {}

            if self.agent_name in active_tasks:
                task_info = active_tasks[self.agent_name]

                # Mark task as completed (this is the key fix!)
                task_info.mark_completed()
                task_updates[self.agent_name] = task_info

                self.logger.info(
                    "EXPLICITLY marked agent task as COMPLETED",
                    agent=self.agent_name,
                    task_status=task_info.status,
                    task_description=task_info.task_description[:100],
                    completion_time=task_info.completed_at.isoformat()
                    if task_info.completed_at
                    else None,
                )
            else:
                self.logger.warning(
                    f"Agent {self.agent_name} not found in active_tasks during completion"
                )

            # Clean up temp state and turn completion flags
            personalizer.reset_agent_temp_state(state, self.agent_name)

            # Reset agent completion state
            agent_state.turn_finished = True  # Ensure it's marked as finished
            agent_state.continuation_attempts = 0  # Reset for next time

            self.logger.info(
                "Successfully extracted messages with staged processing",
                agent=self.agent_name,
                pending_messages_count=len(new_synapse_messages),
                task_explicitly_marked_completed=self.agent_name in task_updates,
            )

            # Log message types for debugging
            message_types = [type(msg).__name__ for msg in new_synapse_messages]
            self.logger.debug(
                "Pending message types", agent=self.agent_name, types=message_types
            )

            # Build result with staged message processing
            result = {
                "pending_messages": new_synapse_messages,
                "pending_sources": {
                    self.agent_name: [msg.message_id for msg in new_synapse_messages]
                },
            }
            if task_updates:
                result["active_tasks"] = task_updates

            return result

        except Exception as e:
            self.logger.error(
                "Failed to extract new messages", agent=self.agent_name, error=str(e)
            )

            # ENHANCED ERROR HANDLING: Mark task as failed with staged processing
            active_tasks = state.get("active_tasks", {})
            task_updates = {}
            failed_tasks = set()

            if self.agent_name in active_tasks:
                task_info = active_tasks[self.agent_name]
                task_info.mark_failed(str(e))
                task_updates[self.agent_name] = task_info
                failed_tasks.add(self.agent_name)

                self.logger.error(
                    "Marked agent task as FAILED due to extraction error",
                    agent=self.agent_name,
                    error_message=str(e),
                    task_status=task_info.status,
                )

            # Clean up temp state even on error
            personalizer.reset_agent_temp_state(state, self.agent_name)

            # Reset agent state
            agent_state.turn_finished = True
            agent_state.continuation_attempts = 0

            # Return error state with staged processing
            result = {
                "pending_messages": [],
                "pending_sources": {},
                "error_count": state.get("error_count", 0) + 1,
            }
            if task_updates:
                result["active_tasks"] = task_updates
            if failed_tasks:
                result["failed_tasks"] = failed_tasks

            return result

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
    Create an agent subgraph with staged message processing.

    Args:
        agent_config: Configuration for the agent
        team_config: Team configuration
        react_agent: The ReAct agent instance

    Returns:
        Compiled subgraph for the agent
    """
    subgraph_builder = AgentSubgraph(agent_config, team_config, react_agent)
    return subgraph_builder.create_subgraph()
