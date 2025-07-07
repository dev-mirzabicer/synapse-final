"""Enhanced state management for the multi-agent system with cursor-based personalization."""

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
    """Information about a task assigned to an agent."""

    agent_name: str
    task_description: str
    assigned_at: datetime
    status: str = "pending"  # pending, in_progress, completed, failed
    error_message: Optional[str] = None
    retry_count: int = 0


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


def merge_active_tasks(
    existing: Dict[str, TaskInfo], new: Dict[str, TaskInfo]
) -> Dict[str, TaskInfo]:
    """
    Merge active tasks dictionaries, with new values overwriting existing ones.
    """
    if not new:
        return existing

    # Create a copy to avoid modifying the existing dict
    merged = existing.copy()
    merged.update(new)
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


class GroupChatState(TypedDict):
    """Enhanced state for the group chat application with agent-specific states."""

    messages: Annotated[List[SynapseMessage], add_messages_without_duplicates]
    """Complete global conversation history using SynapseMessages with deduplication."""

    agent_states: Annotated[Dict[str, AgentInternalState], merge_agent_states]
    """Internal states for each agent (including orchestrator) with personalized histories."""

    active_tasks: Annotated[Dict[str, TaskInfo], merge_active_tasks]
    """Currently active tasks by agent name."""

    completed_tasks: Annotated[Set[str], operator.or_]
    """Agent names that completed their tasks (union operation for sets)."""

    failed_tasks: Annotated[Set[str], operator.or_]
    """Agent names whose tasks failed (union operation for sets)."""

    round_number: int
    """Current orchestration round."""

    error_count: int
    """Total number of errors encountered."""

    context: Dict[str, Any]
    """Additional context for agents."""

    conversation_id: Optional[str]
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
    Create initial state for a new conversation.

    Args:
        user_message: Initial user message
        agent_names: List of agent names
        conversation_id: Unique conversation identifier

    Returns:
        Initial GroupChatState
    """
    from .messages import SynapseHumanMessage

    initial_message = SynapseHumanMessage(content=user_message)

    initial_state: GroupChatState = {
        "messages": [initial_message],
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


def validate_state_consistency(state: GroupChatState) -> bool:
    """
    Validate that the state is consistent (for debugging).

    Args:
        state: The state to validate

    Returns:
        True if state is consistent, False otherwise
    """
    try:
        # Check for message duplicates
        message_ids = [msg.message_id for msg in state["messages"]]
        if len(message_ids) != len(set(message_ids)):
            logger.error(
                f"Duplicate messages detected in state: "
                f"{len(message_ids)} total, {len(set(message_ids))} unique"
            )
            # Find duplicates for debugging
            seen = set()
            duplicates = set()
            for msg_id in message_ids:
                if msg_id in seen:
                    duplicates.add(msg_id)
                seen.add(msg_id)
            logger.error(
                f"Duplicate message IDs: {list(duplicates)[:5]}"
            )  # Show first 5
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

            # Personal history should not have duplicates
            personal_ids = []
            for msg in agent_state.personal_history:
                if hasattr(msg, "id") and msg.id:
                    personal_ids.append(msg.id)

            if len(personal_ids) != len(set(personal_ids)):
                logger.error(
                    f"Agent {agent_name} has duplicate messages in personal history: "
                    f"{len(personal_ids)} total, {len(set(personal_ids))} unique"
                )
                return False

        # Check task consistency
        for agent_name in state["active_tasks"]:
            if agent_name not in state["agent_states"]:
                logger.error(f"Active task for non-existent agent: {agent_name}")
                return False

        # Completed and failed tasks should not overlap
        overlap = state["completed_tasks"] & state["failed_tasks"]
        if overlap:
            logger.error(f"Tasks in both completed and failed: {overlap}")
            return False

        # All agents in completed/failed should have been in active_tasks
        all_task_agents = (
            set(state["active_tasks"].keys())
            | state["completed_tasks"]
            | state["failed_tasks"]
        )
        for agent_name in state["completed_tasks"] | state["failed_tasks"]:
            if agent_name not in state["agent_states"]:
                logger.error(
                    f"Completed/failed task for non-existent agent: {agent_name}"
                )
                return False

        return True

    except Exception as e:
        logger.error(f"State validation error: {e}")
        return False


def get_state_debug_info(state: GroupChatState) -> Dict[str, Any]:
    """Get detailed debug information about the state."""
    message_ids = [msg.message_id for msg in state["messages"]]
    unique_ids = set(message_ids)

    return {
        "total_messages": len(state["messages"]),
        "unique_messages": len(unique_ids),
        "duplicate_count": len(message_ids) - len(unique_ids),
        "agent_cursors": {
            name: {
                "cursor": agent_state.cursor,
                "personal_history_length": len(agent_state.personal_history),
                "turn_finished": agent_state.turn_finished,
                "has_task": agent_state.current_task is not None,
                "continuation_attempts": agent_state.continuation_attempts,
            }
            for name, agent_state in state["agent_states"].items()
        },
        "active_task_count": len(state["active_tasks"]),
        "active_tasks": list(state["active_tasks"].keys()),
        "completed_count": len(state["completed_tasks"]),
        "completed_tasks": list(state["completed_tasks"]),
        "failed_count": len(state["failed_tasks"]),
        "failed_tasks": list(state["failed_tasks"]),
        "round_number": state.get("round_number", 0),
        "error_count": state.get("error_count", 0),
    }


def deduplicate_messages(state: GroupChatState) -> None:
    """
    Remove duplicate messages from state in-place.

    This is a safety function that should not be needed if the state
    reducers are working correctly, but provides a fallback.
    """
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
                f"Removing duplicate message: {type(msg).__name__} "
                f"id={msg.message_id[:8]}... content={getattr(msg, 'content', 'N/A')[:50]}..."
            )

    if removed_count > 0:
        logger.warning(f"Removed {removed_count} duplicate messages from state")
        state["messages"] = unique_messages
