"""Custom message system for Synapse multi-agent framework."""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    ToolMessage as LCToolMessage,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class SynapseMessage(BaseModel):
    """Base class for all Synapse messages."""

    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    class Config:
        arbitrary_types_allowed = True

    def __hash__(self):
        """Make messages hashable based on their unique ID."""
        return hash(self.message_id)

    def __eq__(self, other):
        """Messages are equal if they have the same ID."""
        if not isinstance(other, SynapseMessage):
            return False
        return self.message_id == other.message_id


class OrchestratorMessage(SynapseMessage):
    """Message from the orchestrator agent."""

    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


class SynapseHumanMessage(SynapseMessage):
    """Message from the human user."""

    pass


class AgentMessage(SynapseMessage):
    """Message from an agent."""

    agent_name: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    is_private: bool = False  # True if message started with [PRIVATE]


class SynapseSystemMessage(SynapseMessage):
    """System message that appears in chat history (not the main system prompt)."""

    target_agent: Optional[str] = (
        None  # If None, visible to all; if set, only to target agent
    )


class SynapseToolMessage(SynapseMessage):
    """Tool execution result message."""

    tool_call_id: str
    caller_agent: str
    tool_name: str
    is_error: bool = False


def to_langchain_message(
    synapse_msg: SynapseMessage, for_agent: str
) -> Optional[BaseMessage]:
    """
    Convert SynapseMessage to LangChain BaseMessage for agent invocation.

    Args:
        synapse_msg: The SynapseMessage to convert
        for_agent: The agent this message is being personalized for

    Returns:
        LangChain BaseMessage or None if agent shouldn't see this message
    """
    # Create a unique ID for the LangChain message based on the Synapse message ID
    # This helps with deduplication in agent personal histories
    lc_msg_id = f"{synapse_msg.message_id}-{for_agent}"

    if isinstance(synapse_msg, OrchestratorMessage):
        if for_agent == "orchestrator":
            # Orchestrator's own messages appear as AIMessage
            return AIMessage(
                content=synapse_msg.content, name="orchestrator", id=lc_msg_id
            )
        else:
            # Orchestrator messages appear as HumanMessage with prefix for other agents
            content = f"[NOT_A_USER | AGENT orchestrator]\n\n{synapse_msg.content}"
            return HumanMessage(content=content, name="orchestrator", id=lc_msg_id)

    elif isinstance(synapse_msg, SynapseHumanMessage):
        # User messages appear with USER prefix
        content = f"[USER]\n\n{synapse_msg.content}"
        return HumanMessage(content=content, name="user", id=lc_msg_id)

    elif isinstance(synapse_msg, AgentMessage):
        if synapse_msg.agent_name == for_agent:
            # Agent's own messages appear as AIMessage
            return AIMessage(
                content=synapse_msg.content, name=synapse_msg.agent_name, id=lc_msg_id
            )
        else:
            # Other agents' messages appear as HumanMessage with prefix
            content = f"[NOT_A_USER | AGENT {synapse_msg.agent_name}]\n\n{synapse_msg.content}"
            return HumanMessage(
                content=content, name=synapse_msg.agent_name, id=lc_msg_id
            )

    elif isinstance(synapse_msg, SynapseSystemMessage):
        # System messages targeted to specific agent or all agents
        if synapse_msg.target_agent is None or synapse_msg.target_agent == for_agent:
            content = f"[NOT_A_USER | SYSTEM]\n\n{synapse_msg.content}"
            return HumanMessage(content=content, name="system", id=lc_msg_id)
        else:
            # Agent shouldn't see this targeted system message
            return None

    elif isinstance(synapse_msg, SynapseToolMessage):
        # Tool messages only visible to the calling agent
        if synapse_msg.caller_agent == for_agent:
            return LCToolMessage(
                content=synapse_msg.content,
                tool_call_id=synapse_msg.tool_call_id,
                name=synapse_msg.tool_name,
                id=lc_msg_id,
            )
        else:
            # Other agents don't see tool messages
            return None

    # Fallback - shouldn't happen
    logger.warning(
        f"Unknown message type in to_langchain_message: {type(synapse_msg).__name__}"
    )
    return None


def from_langchain_messages(
    lc_messages: List[BaseMessage],
    agent_name: str,
    original_personal_history_length: int = 0,
) -> List[SynapseMessage]:
    """
    Convert new LangChain messages from agent back to SynapseMessages.

    Args:
        lc_messages: Full message history from agent invocation
        agent_name: Name of the agent that produced these messages
        original_personal_history_length: Length of agent's history before invocation

    Returns:
        List of SynapseMessages representing only the NEW messages
    """
    # Extract only new messages - this is critical to avoid duplicates
    if len(lc_messages) <= original_personal_history_length:
        # No new messages
        return []

    new_messages = lc_messages[original_personal_history_length:]
    synapse_messages = []

    # Debug logging
    logger.debug(
        "Converting LangChain messages to SynapseMessages",
        agent=agent_name,
        total_messages=len(lc_messages),
        original_length=original_personal_history_length,
        new_messages_count=len(new_messages),
    )

    for i, msg in enumerate(new_messages):
        try:
            # Check for continuation message marker
            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get(
                "continuation"
            ):
                target_agent = msg.additional_kwargs["continuation"]
                synapse_msg = SynapseSystemMessage(
                    content=msg.content, target_agent=target_agent
                )
                synapse_messages.append(synapse_msg)
                logger.debug(
                    f"Converted continuation message {i} to SynapseSystemMessage",
                    agent=agent_name,
                )
                continue

            if isinstance(msg, AIMessage):
                # Determine if message is private based on [PRIVATE] prefix
                content = msg.content
                is_private = content.startswith("[PRIVATE]")

                if is_private:
                    # Remove the [PRIVATE] prefix from content
                    content = content[9:].strip()  # Remove '[PRIVATE]' prefix
                elif content.startswith("[PUBLIC]"):
                    # Remove the [PUBLIC] prefix from content
                    content = content[8:].strip()  # Remove '[PUBLIC]' prefix
                    is_private = False
                else:
                    # No prefix, default to public
                    is_private = False

                # Check for and remove [FINISH_TURN] marker
                if has_finish_turn_marker(content):
                    content = remove_finish_turn_marker(content)
                    # Log the marker removal for debugging
                    logger.debug(
                        "Removed [FINISH_TURN] marker from agent message",
                        agent=agent_name,
                        original_content=msg.content[:100],
                        cleaned_content=content[:100],
                    )

                synapse_msg = AgentMessage(
                    content=content,
                    agent_name=agent_name,
                    tool_calls=getattr(msg, "tool_calls", []),
                    is_private=is_private,
                )
                synapse_messages.append(synapse_msg)
                logger.debug(
                    f"Converted AI message {i} to AgentMessage", agent=agent_name
                )

            elif isinstance(msg, LCToolMessage):
                # Tool messages are always private to the calling agent
                synapse_msg = SynapseToolMessage(
                    content=msg.content,
                    tool_call_id=msg.tool_call_id,
                    caller_agent=agent_name,
                    tool_name=getattr(msg, "name", "unknown_tool"),
                    is_error="error" in msg.content.lower(),
                )
                synapse_messages.append(synapse_msg)
                logger.debug(
                    f"Converted tool message {i} to SynapseToolMessage",
                    agent=agent_name,
                )

            else:
                # Log unexpected message types
                logger.warning(
                    f"Unexpected message type in from_langchain_messages: {type(msg).__name__}",
                    agent=agent_name,
                    message_index=i,
                )

        except Exception as e:
            logger.error(
                f"Error converting message {i}",
                agent=agent_name,
                error=str(e),
                message_type=type(msg).__name__,
            )
            # Skip this message rather than failing entirely
            continue

    logger.debug(
        "Message conversion completed",
        agent=agent_name,
        converted_count=len(synapse_messages),
        message_types=[type(msg).__name__ for msg in synapse_messages],
    )

    return synapse_messages


def has_finish_turn_marker(content: str) -> bool:
    """
    Check if a message content ends with [FINISH_TURN] marker.

    Args:
        content: Message content to check

    Returns:
        True if message ends with [FINISH_TURN], False otherwise
    """
    if not content:
        return False

    # Check if content ends with [FINISH_TURN] (case insensitive, allowing whitespace)
    content_stripped = content.strip()

    # Check for exact match
    if content_stripped.upper().endswith("[FINISH_TURN]"):
        return True

    # Also check for common variations with punctuation
    variations = [
        "[FINISH_TURN].",
        "[FINISH_TURN]!",
        "[FINISH_TURN]?",
        "[FINISH_TURN],",
        "[FINISH_TURN];",
        "[FINISH_TURN]:",
    ]

    for variation in variations:
        if content_stripped.upper().endswith(variation):
            logger.debug(
                f"Found finish turn marker with punctuation: {variation}",
                content_preview=content_stripped[-20:],
            )
            return True

    return False


def remove_finish_turn_marker(content: str) -> str:
    """
    Remove [FINISH_TURN] marker from message content.

    Args:
        content: Message content that may contain [FINISH_TURN]

    Returns:
        Content with [FINISH_TURN] marker removed
    """
    if not content:
        return content

    content_stripped = content.strip()

    # Remove exact match
    if content_stripped.upper().endswith("[FINISH_TURN]"):
        result = content_stripped[:-13].strip()  # Remove '[FINISH_TURN]'
        logger.debug(
            "Removed finish turn marker", original=content[:50], result=result[:50]
        )
        return result

    # Remove variations with punctuation
    variations = [
        ("[FINISH_TURN].", 14),
        ("[FINISH_TURN]!", 14),
        ("[FINISH_TURN]?", 14),
        ("[FINISH_TURN],", 14),
        ("[FINISH_TURN];", 14),
        ("[FINISH_TURN]:", 14),
    ]

    content_upper = content_stripped.upper()
    for marker, length in variations:
        if content_upper.endswith(marker):
            result = content_stripped[:-length].strip()
            logger.debug(
                "Removed finish turn marker with punctuation",
                marker=marker,
                original=content[:50],
                result=result[:50],
            )
            return result

    return content


def has_finish_turn_in_messages(messages: List) -> bool:
    """
    Check if any message in a list contains [FINISH_TURN] marker.

    Args:
        messages: List of LangChain message objects

    Returns:
        True if any message contains [FINISH_TURN] marker, False otherwise
    """
    for message in messages:
        if hasattr(message, "content") and has_finish_turn_marker(message.content):
            return True

    return False


def should_agent_see_message(
    agent_name: str, msg: Union[BaseMessage, SynapseMessage]
) -> bool:
    """
    Determine if an agent should see a specific message based on visibility rules.

    Args:
        agent_name: Name of the agent
        msg: The message to check

    Returns:
        True if agent should see the message, False otherwise
    """
    if isinstance(msg, (SynapseHumanMessage, OrchestratorMessage)):
        # User and orchestrator messages are always visible
        return True

    elif isinstance(msg, AgentMessage):
        if msg.agent_name == agent_name:
            # Agents always see their own messages
            return True
        elif msg.is_private:
            # Private messages from other agents are not visible
            return False
        else:
            # Public messages from other agents are visible
            return True

    elif isinstance(msg, SynapseSystemMessage):
        # System messages are visible if not targeted or targeted to this agent
        return msg.target_agent is None or msg.target_agent == agent_name

    elif isinstance(msg, SynapseToolMessage):
        # Tool messages only visible to the calling agent
        return msg.caller_agent == agent_name

    # Default to not visible for unknown message types
    logger.warning(
        f"Unknown message type in should_agent_see_message: {type(msg).__name__}"
    )
    return False


def get_message_summary(msg: SynapseMessage) -> str:
    """
    Get a summary of a message for logging/debugging.

    Args:
        msg: The message to summarize

    Returns:
        String summary of the message
    """
    msg_type = type(msg).__name__
    content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content

    if isinstance(msg, AgentMessage):
        visibility = "private" if msg.is_private else "public"
        return f"{msg_type}({msg.agent_name}, {visibility}): {content_preview}"
    elif isinstance(msg, SynapseToolMessage):
        return f"{msg_type}({msg.caller_agent}->{msg.tool_name}): {content_preview}"
    elif isinstance(msg, SynapseSystemMessage):
        target = msg.target_agent or "all"
        return f"{msg_type}(target={target}): {content_preview}"
    else:
        return f"{msg_type}: {content_preview}"


def validate_message_list(messages: List[SynapseMessage]) -> List[str]:
    """
    Validate a list of messages and return any issues found.

    Args:
        messages: List of messages to validate

    Returns:
        List of validation error strings (empty if valid)
    """
    errors = []
    seen_ids = set()

    for i, msg in enumerate(messages):
        # Check for duplicate IDs
        if msg.message_id in seen_ids:
            errors.append(f"Message {i} has duplicate ID: {msg.message_id}")
        seen_ids.add(msg.message_id)

        # Check for empty content
        if not msg.content.strip():
            errors.append(f"Message {i} ({type(msg).__name__}) has empty content")

        # Check for required fields
        if isinstance(msg, AgentMessage) and not msg.agent_name:
            errors.append(f"AgentMessage {i} missing agent_name")
        elif isinstance(msg, SynapseToolMessage):
            if not msg.tool_call_id:
                errors.append(f"SynapseToolMessage {i} missing tool_call_id")
            if not msg.caller_agent:
                errors.append(f"SynapseToolMessage {i} missing caller_agent")

    return errors
