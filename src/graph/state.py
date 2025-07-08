"""Enhanced state management for the multi-agent system with staged message processing and robust task completion tracking."""

import operator
from typing import List, Dict, Set, Annotated, Optional, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from datetime import datetime

from .messages import SynapseMessage
from src.core.logging import get_logger

logger = get_logger(__name__)


class TaskInfo(BaseModel):
    """Enhanced information about a task assigned to an agent with explicit completion tracking."""

    agent_name: str
    task_description: str
    assigned_at: datetime
    status: str = "pending"  # pending, in_progress, completed, failed
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def mark_in_progress(self) -> None:
        """Mark task as in progress with validation and logging."""
        if self.status not in ["pending"]:
            logger.warning(
                "Task status transition may be invalid",
                agent=self.agent_name,
                from_status=self.status,
                to_status="in_progress",
            )

        self.status = "in_progress"
        logger.info(
            "Task marked as in progress",
            agent=self.agent_name,
            task_description=self.task_description[:100],
        )

    def mark_completed(self) -> None:
        """Explicitly mark task as completed with validation and logging."""
        if self.status in ["completed", "failed"]:
            logger.warning(
                "Attempting to mark already finished task as completed",
                agent=self.agent_name,
                current_status=self.status,
                task_description=self.task_description[:100],
            )
            return

        previous_status = self.status
        self.status = "completed"
        self.completed_at = datetime.now()

        logger.info(
            "Task explicitly marked as completed",
            agent=self.agent_name,
            previous_status=previous_status,
            task_description=self.task_description[:100],
            completion_time=self.completed_at.isoformat(),
        )

    def mark_failed(self, error_message: str) -> None:
        """Explicitly mark task as failed with error tracking."""
        if self.status in ["completed", "failed"]:
            logger.warning(
                "Attempting to mark already finished task as failed",
                agent=self.agent_name,
                current_status=self.status,
                new_error=error_message[:100],
            )
            return

        previous_status = self.status
        self.status = "failed"
        self.error_message = error_message
        self.completed_at = datetime.now()
        self.retry_count += 1

        logger.error(
            "Task explicitly marked as failed",
            agent=self.agent_name,
            previous_status=previous_status,
            error_message=error_message[:200],
            retry_count=self.retry_count,
            completion_time=self.completed_at.isoformat(),
        )

    def is_finished(self) -> bool:
        """Check if task is finished (completed or failed)."""
        return self.status in ["completed", "failed"]

    def is_pending(self) -> bool:
        """Check if task is still pending."""
        return self.status == "pending"

    def is_in_progress(self) -> bool:
        """Check if task is currently in progress."""
        return self.status == "in_progress"

    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.status == "completed"

    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds if completed."""
        if self.completed_at:
            return (self.completed_at - self.assigned_at).total_seconds()
        return None

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary for debugging."""
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "assigned_at": self.assigned_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration_seconds": self.get_duration(),
            "retry_count": self.retry_count,
            "has_error": bool(self.error_message),
            "error_preview": self.error_message[:100] if self.error_message else None,
            "task_preview": self.task_description[:100],
        }


class AgentInternalState(BaseModel):
    """Internal state for each agent including personalized history and cursor."""

    agent_name: str
    personal_history: List[BaseMessage] = Field(default_factory=list)
    cursor: int = 0  # Position in global history that's been personalized
    current_task: Optional[TaskInfo] = None
    temp_new_history: Optional[List[BaseMessage]] = (
        None  # Temporary storage during multi-turn
    )
    turn_finished: bool = True  # Whether the agent has finished its current turn
    continuation_attempts: int = 0  # Number of continuation attempts for current task

    class Config:
        arbitrary_types_allowed = True

    def reset_temp_history(self):
        """Clear temporary history after processing."""
        self.temp_new_history = None

    def has_temp_history(self) -> bool:
        """Check if there's temporary history to continue from."""
        return self.temp_new_history is not None and len(self.temp_new_history) > 0

    def reset_turn_state(self):
        """Reset turn-related state for new task."""
        self.turn_finished = True  # Agent starts in finished state
        self.continuation_attempts = 0
        self.reset_temp_history()

    def start_new_turn(self):
        """Mark the start of a new turn."""
        self.turn_finished = False  # Now working on a task
        self.continuation_attempts = 0

    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary for debugging."""
        return {
            "agent_name": self.agent_name,
            "cursor_position": self.cursor,
            "personal_history_length": len(self.personal_history),
            "has_temp_history": self.has_temp_history(),
            "temp_history_length": len(self.temp_new_history)
            if self.temp_new_history
            else 0,
            "turn_finished": self.turn_finished,
            "continuation_attempts": self.continuation_attempts,
            "has_current_task": self.current_task is not None,
            "current_task_status": self.current_task.status
            if self.current_task
            else None,
        }


def add_messages_without_duplicates(
    existing: List[SynapseMessage], new: List[SynapseMessage]
) -> List[SynapseMessage]:
    """
    Add only truly new messages, preventing duplicates.

    This custom reducer ensures messages are never duplicated in the global state.
    It uses message_id for deduplication, which is guaranteed to be unique.
    """
    if not new:
        return existing

    # Create a set of existing message IDs for O(1) lookup
    existing_ids = {msg.message_id for msg in existing}

    # Track order of addition for debugging
    added_messages = []

    # Only add messages that aren't already in the list
    for msg in new:
        if msg.message_id not in existing_ids:
            added_messages.append(msg)
            existing_ids.add(msg.message_id)  # Update the set for subsequent checks
        else:
            logger.warning(
                f"Prevented duplicate message: {type(msg).__name__} "
                f"with id {msg.message_id[:8]}... content: {getattr(msg, 'content', 'N/A')[:50]}..."
            )

    if added_messages:
        logger.debug(
            f"Adding {len(added_messages)} new messages to state. "
            f"Total will be: {len(existing) + len(added_messages)}"
        )

    return existing + added_messages


def merge_pending_sources(
    existing: Dict[str, List[str]], new: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Merge pending sources dictionaries, tracking which agent added which messages.

    Args:
        existing: Current pending sources mapping
        new: New pending sources to merge

    Returns:
        Merged dictionary with combined source tracking
    """
    if not new:
        return existing

    merged = existing.copy()

    for agent_name, message_ids in new.items():
        if agent_name in merged:
            # Combine message IDs, avoiding duplicates
            existing_ids = set(merged[agent_name])
            new_ids = [msg_id for msg_id in message_ids if msg_id not in existing_ids]
            merged[agent_name] = merged[agent_name] + new_ids

            if new_ids:
                logger.debug(
                    f"Merged pending sources for {agent_name}: added {len(new_ids)} new message IDs"
                )
        else:
            merged[agent_name] = message_ids.copy()
            logger.debug(
                f"Added new pending sources for {agent_name}: {len(message_ids)} message IDs"
            )

    return merged


def merge_active_tasks(
    existing: Dict[str, TaskInfo], new: Dict[str, TaskInfo]
) -> Dict[str, TaskInfo]:
    """
    Merge active tasks dictionaries, with new values overwriting existing ones.
    Includes validation and logging for task state transitions.
    """
    if not new:
        return existing

    # Create a copy to avoid modifying the existing dict
    merged = existing.copy()

    # Track changes for logging
    updates = []
    additions = []

    for agent_name, new_task in new.items():
        if agent_name in merged:
            old_task = merged[agent_name]
            if old_task.status != new_task.status:
                updates.append(
                    {
                        "agent": agent_name,
                        "old_status": old_task.status,
                        "new_status": new_task.status,
                        "task_preview": new_task.task_description[:50],
                    }
                )
        else:
            additions.append(
                {
                    "agent": agent_name,
                    "status": new_task.status,
                    "task_preview": new_task.task_description[:50],
                }
            )

        merged[agent_name] = new_task

    # Log significant changes
    if updates:
        logger.info("Task status updates in merge", updates=updates)
    if additions:
        logger.info("New tasks added in merge", additions=additions)

    return merged


def merge_agent_states(
    existing: Dict[str, AgentInternalState], new: Dict[str, AgentInternalState]
) -> Dict[str, AgentInternalState]:
    """
    Merge agent states dictionaries, with new values overwriting existing ones.
    This ensures agent state updates are properly reflected.
    """
    if not new:
        return existing

    # Create a copy to avoid modifying the existing dict
    merged = existing.copy()
    merged.update(new)
    return merged


def take_last_or_max(a: Any, b: Any) -> Any:
    """Reducer for numeric values, takes the max. For others, takes the last value."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return max(a, b)
    return b


def unique(a: str, b: str) -> str:
    """Reducer that checks for uniqueness of strings."""
    if a == b:
        return a
    elif a is None:
        return b
    elif b is None:
        return a
    # Raise an error if values are not unique
    logger.error(f"Values are not unique: {a} != {b}")
    raise ValueError(f"Values are not unique: {a} != {b}")


class MessageMerger:
    """Utility class for managing staged message processing and merging."""

    @staticmethod
    def merge_pending_messages(state) -> Dict[str, Any]:
        """
        Atomically merge pending messages into main history.

        Args:
            state: Current global state

        Returns:
            Updates dictionary to apply to state
        """
        pending = state.get("pending_messages", [])

        if not pending:
            logger.debug("No pending messages to merge")
            return {}

        pending_sources = state.get("pending_sources", {})

        logger.info(
            "Merging pending messages to main history",
            pending_count=len(pending),
            current_main_count=len(state.get("messages", [])),
            sources=list(pending_sources.keys()),
        )

        # Log detailed source breakdown for debugging
        for agent_name, message_ids in pending_sources.items():
            logger.debug(
                f"Merging {len(message_ids)} messages from {agent_name}",
                message_ids=[msg_id[:8] for msg_id in message_ids],
            )

        # Return updates to clear pending and move to main
        return {
            "messages": pending,  # Will be merged by add_messages_without_duplicates
            "pending_messages": [],  # Clear pending queue
            "pending_sources": {},  # Clear source tracking
        }

    @staticmethod
    def should_merge_pending(state, context: str) -> bool:
        """
        Determine if pending messages should be merged at this point.

        Args:
            state: Current global state
            context: Execution context for merge decision

        Returns:
            True if pending messages should be merged
        """
        merge_contexts = [
            "before_aggregation",
            "after_agent_completion",
            "before_orchestrator_routing",
        ]

        has_pending = len(state.get("pending_messages", [])) > 0
        is_merge_context = context in merge_contexts

        if has_pending and is_merge_context:
            logger.debug(
                f"Should merge pending messages at context: {context}",
                pending_count=len(state.get("pending_messages", [])),
            )
            return True

        return False

    @staticmethod
    def get_pending_summary(state) -> Dict[str, Any]:
        """
        Get summary of pending messages for debugging.

        Args:
            state: Current global state

        Returns:
            Summary dictionary with pending message statistics
        """
        pending = state.get("pending_messages", [])
        sources = state.get("pending_sources", {})

        return {
            "pending_count": len(pending),
            "pending_sources": list(sources.keys()),
            "messages_by_source": {
                agent: len(msg_ids) for agent, msg_ids in sources.items()
            },
            "pending_message_types": [type(msg).__name__ for msg in pending],
        }


class GroupChatState(TypedDict):
    """Enhanced state for the group chat application with staged message processing."""

    messages: Annotated[List[SynapseMessage], add_messages_without_duplicates]
    """Complete global conversation history using SynapseMessages with deduplication."""

    pending_messages: Annotated[List[SynapseMessage], add_messages_without_duplicates]
    """Staging area for new messages before they're merged into main history."""

    pending_sources: Annotated[Dict[str, List[str]], merge_pending_sources]
    """Track which agent added which pending messages (for debugging and control)."""

    agent_states: Annotated[Dict[str, AgentInternalState], merge_agent_states]
    """Internal states for each agent (including orchestrator) with personalized histories."""

    active_tasks: Annotated[Dict[str, TaskInfo], merge_active_tasks]
    """Currently active tasks by agent name with enhanced tracking."""

    completed_tasks: Annotated[Set[str], operator.or_]
    """Agent names that completed their tasks (union operation for sets)."""

    failed_tasks: Annotated[Set[str], operator.or_]
    """Agent names whose tasks failed (union operation for sets)."""

    round_number: Annotated[int, take_last_or_max]
    """Current orchestration round."""

    error_count: Annotated[int, operator.add]
    """Total number of errors encountered."""

    context: Annotated[Dict[str, Any], operator.or_]
    """Additional context for agents."""

    conversation_id: Annotated[Optional[str], unique]
    """Unique identifier for this conversation."""


def initialize_agent_states(agent_names: List[str]) -> Dict[str, AgentInternalState]:
    """
    Initialize internal states for all agents including orchestrator.

    Args:
        agent_names: List of agent names (should include 'orchestrator')

    Returns:
        Dictionary mapping agent names to their internal states
    """
    agent_states = {}

    # Always include orchestrator
    if "orchestrator" not in agent_names:
        agent_names = ["orchestrator"] + agent_names

    for agent_name in agent_names:
        agent_states[agent_name] = AgentInternalState(agent_name=agent_name)

    return agent_states


def get_agent_state(state: GroupChatState, agent_name: str) -> AgentInternalState:
    """
    Get agent's internal state, creating if it doesn't exist.

    Args:
        state: The global state
        agent_name: Name of the agent

    Returns:
        Agent's internal state
    """
    if agent_name not in state["agent_states"]:
        state["agent_states"][agent_name] = AgentInternalState(agent_name=agent_name)

    agent_state = state["agent_states"][agent_name]

    # Ensure all required attributes exist (for backward compatibility)
    if not hasattr(agent_state, "turn_finished"):
        agent_state.turn_finished = True
    if not hasattr(agent_state, "continuation_attempts"):
        agent_state.continuation_attempts = 0

    return agent_state


def update_agent_cursor(
    state: GroupChatState, agent_name: str, new_cursor: int
) -> None:
    """
    Update agent's cursor position.

    Args:
        state: The global state
        agent_name: Name of the agent
        new_cursor: New cursor position
    """
    agent_state = get_agent_state(state, agent_name)
    agent_state.cursor = new_cursor


def get_unprocessed_messages(
    state: GroupChatState, agent_name: str
) -> List[SynapseMessage]:
    """
    Get messages that haven't been processed by the agent yet.

    Args:
        state: The global state
        agent_name: Name of the agent

    Returns:
        List of unprocessed messages
    """
    agent_state = get_agent_state(state, agent_name)
    global_messages = state["messages"]

    return global_messages[agent_state.cursor :]


def mark_messages_processed(state: GroupChatState, agent_name: str, count: int) -> None:
    """
    Mark a number of messages as processed by advancing the cursor.

    Args:
        state: The global state
        agent_name: Name of the agent
        count: Number of messages to mark as processed
    """
    agent_state = get_agent_state(state, agent_name)
    agent_state.cursor += count


def create_initial_state(
    user_message: str, agent_names: List[str], conversation_id: str
) -> GroupChatState:
    """
    Create initial state for a new conversation with staged message processing.

    Args:
        user_message: Initial user message
        agent_names: List of agent names
        conversation_id: Unique conversation identifier

    Returns:
        Initial GroupChatState with staged processing support
    """
    from .messages import SynapseHumanMessage

    initial_message = SynapseHumanMessage(content=user_message)

    initial_state: GroupChatState = {
        "messages": [initial_message],
        "pending_messages": [],  # Empty staging area
        "pending_sources": {},  # Empty source tracking
        "agent_states": initialize_agent_states(agent_names),
        "active_tasks": {},
        "completed_tasks": set(),
        "failed_tasks": set(),
        "round_number": 0,
        "error_count": 0,
        "context": {},
        "conversation_id": conversation_id,
    }

    return initial_state


def validate_task_consistency(state: GroupChatState) -> List[str]:
    """
    Validate task state consistency and return any errors found.

    This is critical for ensuring the robustness of concurrent execution.

    Args:
        state: The global state to validate

    Returns:
        List of validation error strings (empty if valid)
    """
    errors = []

    active_tasks = state.get("active_tasks", {})
    completed_tasks = state.get("completed_tasks", set())
    failed_tasks = state.get("failed_tasks", set())

    # Validate completed_tasks consistency
    for agent_name in completed_tasks:
        if agent_name not in active_tasks:
            errors.append(
                f"Agent {agent_name} in completed_tasks but not in active_tasks"
            )
        elif not active_tasks[agent_name].is_finished():
            errors.append(
                f"Agent {agent_name} in completed_tasks but TaskInfo status is {active_tasks[agent_name].status}"
            )
        elif not active_tasks[agent_name].is_successful():
            errors.append(
                f"Agent {agent_name} in completed_tasks but TaskInfo status is {active_tasks[agent_name].status} (not completed)"
            )

    # Validate failed_tasks consistency
    for agent_name in failed_tasks:
        if agent_name not in active_tasks:
            errors.append(f"Agent {agent_name} in failed_tasks but not in active_tasks")
        elif active_tasks[agent_name].status != "failed":
            errors.append(
                f"Agent {agent_name} in failed_tasks but TaskInfo status is {active_tasks[agent_name].status}"
            )

    # Check for impossible overlaps
    overlap = completed_tasks & failed_tasks
    if overlap:
        errors.append(f"Agents in both completed AND failed: {overlap}")

    # Validate TaskInfo status consistency
    for agent_name, task_info in active_tasks.items():
        if task_info.is_successful() and agent_name not in completed_tasks:
            errors.append(
                f"TaskInfo shows {agent_name} completed but not in completed_tasks set"
            )
        if task_info.status == "failed" and agent_name not in failed_tasks:
            errors.append(
                f"TaskInfo shows {agent_name} failed but not in failed_tasks set"
            )

    # Validate agent states exist for active tasks
    agent_states = state.get("agent_states", {})
    for agent_name in active_tasks:
        if agent_name not in agent_states:
            errors.append(
                f"Active task for agent {agent_name} but no agent state found"
            )

    return errors


def apply_state_validation(state: GroupChatState, context: str = "") -> bool:
    """
    Apply state validation and log any issues found.

    Args:
        state: The global state to validate
        context: Context string for logging (e.g., "after aggregator")

    Returns:
        True if validation passed, False if errors found
    """
    errors = validate_task_consistency(state)

    if errors:
        logger.error(
            f"State consistency errors detected {context}",
            errors=errors,
            error_count=len(errors),
        )
        return False
    else:
        logger.debug(f"State consistency validation passed {context}")
        return True


def validate_state_consistency(state: GroupChatState) -> bool:
    """
    Validate that the state is consistent (for debugging).

    Args:
        state: The state to validate

    Returns:
        True if state is consistent, False otherwise
    """
    try:
        # Check for message duplicates in main messages
        message_ids = [msg.message_id for msg in state["messages"]]
        if len(message_ids) != len(set(message_ids)):
            logger.error(
                f"Duplicate messages detected in main history: "
                f"{len(message_ids)} total, {len(set(message_ids))} unique"
            )
            return False

        # Check for duplicates in pending messages
        pending_ids = [msg.message_id for msg in state.get("pending_messages", [])]
        if len(pending_ids) != len(set(pending_ids)):
            logger.error(
                f"Duplicate messages detected in pending queue: "
                f"{len(pending_ids)} total, {len(set(pending_ids))} unique"
            )
            return False

        # Check agent states consistency
        for agent_name, agent_state in state["agent_states"].items():
            # Cursor should not exceed message count
            if agent_state.cursor > len(state["messages"]):
                logger.error(
                    f"Agent {agent_name} cursor ({agent_state.cursor}) "
                    f"exceeds message count ({len(state['messages'])})"
                )
                return False

        # Check task consistency using enhanced validation
        task_errors = validate_task_consistency(state)
        if task_errors:
            logger.error(f"Task consistency errors: {task_errors}")
            return False

        return True

    except Exception as e:
        logger.error(f"State validation error: {e}")
        return False


def get_state_debug_info(state: GroupChatState) -> Dict[str, Any]:
    """Get detailed debug information about the state."""
    message_ids = [msg.message_id for msg in state["messages"]]
    unique_ids = set(message_ids)

    # Enhanced task status information
    active_tasks = state.get("active_tasks", {})
    task_status_summary = {}
    for agent_name, task_info in active_tasks.items():
        task_status_summary[f"{agent_name}_task_status"] = {
            "status": task_info.status,
            "is_finished": task_info.is_finished(),
            "duration": task_info.get_duration(),
            "retry_count": task_info.retry_count,
        }

    # Pending message information
    pending_summary = MessageMerger.get_pending_summary(state)

    return {
        "total_messages": len(state["messages"]),
        "unique_messages": len(unique_ids),
        "duplicate_count": len(message_ids) - len(unique_ids),
        "pending_messages_count": pending_summary["pending_count"],
        "pending_sources": pending_summary["pending_sources"],
        "agent_cursors": {
            name: agent_state.get_state_summary()
            for name, agent_state in state["agent_states"].items()
        },
        "active_task_count": len(state["active_tasks"]),
        "active_tasks": list(state["active_tasks"].keys()),
        "task_status_details": task_status_summary,
        "completed_count": len(state["completed_tasks"]),
        "completed_tasks": list(state["completed_tasks"]),
        "failed_count": len(state["failed_tasks"]),
        "failed_tasks": list(state["failed_tasks"]),
        "round_number": state.get("round_number", 0),
        "error_count": state.get("error_count", 0),
        "validation_passed": apply_state_validation(state, "(debug info generation)"),
    }


def deduplicate_messages(state: GroupChatState) -> None:
    """
    Remove duplicate messages from state in-place.

    This is a safety function that should not be needed if the state
    reducers are working correctly, but provides a fallback.
    """
    # Deduplicate main messages
    seen_ids = set()
    unique_messages = []
    removed_count = 0

    for msg in state["messages"]:
        if msg.message_id not in seen_ids:
            seen_ids.add(msg.message_id)
            unique_messages.append(msg)
        else:
            removed_count += 1
            logger.warning(
                f"Removing duplicate message from main history: {type(msg).__name__} "
                f"id={msg.message_id[:8]}... content={getattr(msg, 'content', 'N/A')[:50]}..."
            )

    if removed_count > 0:
        logger.warning(f"Removed {removed_count} duplicate messages from main history")
        state["messages"] = unique_messages

    # Deduplicate pending messages
    seen_pending_ids = set()
    unique_pending = []
    removed_pending_count = 0

    for msg in state.get("pending_messages", []):
        if msg.message_id not in seen_pending_ids:
            seen_pending_ids.add(msg.message_id)
            unique_pending.append(msg)
        else:
            removed_pending_count += 1
            logger.warning(
                f"Removing duplicate message from pending queue: {type(msg).__name__} "
                f"id={msg.message_id[:8]}... content={getattr(msg, 'content', 'N/A')[:50]}..."
            )

    if removed_pending_count > 0:
        logger.warning(
            f"Removed {removed_pending_count} duplicate messages from pending queue"
        )
        state["pending_messages"] = unique_pending


def get_task_completion_summary(state: GroupChatState) -> Dict[str, Any]:
    """
    Get a comprehensive summary of task completion status.

    This is useful for debugging concurrent execution and routing decisions.

    Args:
        state: The global state

    Returns:
        Dictionary with detailed completion information
    """
    active_tasks = state.get("active_tasks", {})

    if not active_tasks:
        return {"total_tasks": 0, "has_tasks": False, "completion_status": "no_tasks"}

    # Categorize tasks by status
    completed_agents = [
        name for name, task in active_tasks.items() if task.is_successful()
    ]
    failed_agents = [
        name for name, task in active_tasks.items() if task.status == "failed"
    ]
    in_progress_agents = [
        name for name, task in active_tasks.items() if task.is_in_progress()
    ]
    pending_agents = [name for name, task in active_tasks.items() if task.is_pending()]

    total_count = len(active_tasks)
    finished_count = len(completed_agents) + len(failed_agents)

    # Determine overall completion status
    if finished_count == 0:
        completion_status = "none_finished"
    elif finished_count == total_count:
        completion_status = "all_finished"
    else:
        completion_status = "partially_finished"

    return {
        "total_tasks": total_count,
        "has_tasks": True,
        "completion_status": completion_status,
        "completed_count": len(completed_agents),
        "failed_count": len(failed_agents),
        "in_progress_count": len(in_progress_agents),
        "comp_pending_count": len(pending_agents),
        "finished_count": finished_count,
        "completion_percentage": (finished_count / total_count * 100)
        if total_count > 0
        else 0,
        "completed_agents": completed_agents,
        "failed_agents": failed_agents,
        "in_progress_agents": in_progress_agents,
        "pending_agents": pending_agents,
        "all_finished": finished_count == total_count,
        "ready_for_orchestrator": finished_count == total_count and total_count > 0,
    }
