"""Cursor-based message personalization for agent histories."""

from typing import List
from langchain_core.messages import BaseMessage

from .messages import to_langchain_message, should_agent_see_message
from .state import GroupChatState, get_agent_state
from src.core.logging import get_logger

logger = get_logger(__name__)


class MessagePersonalizer:
    """Handles cursor-based personalization of message histories for agents."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def update_agent_history(
        self, state: GroupChatState, agent_name: str
    ) -> List[BaseMessage]:
        """
        Update agent's personal history using cursor-based approach.

        Args:
            state: The global state
            agent_name: Name of the agent to update history for

        Returns:
            Complete personalized history ready for LLM invocation
        """
        agent_state = get_agent_state(state, agent_name)
        global_messages = state["messages"]

        self.logger.debug(
            "Updating agent history",
            agent=agent_name,
            cursor=agent_state.cursor,
            total_messages=len(global_messages),
            current_history_length=len(agent_state.personal_history),
        )

        # Get unprocessed messages starting from cursor
        unprocessed_messages = global_messages[agent_state.cursor :]

        # Process each new message
        for i, synapse_msg in enumerate(unprocessed_messages):
            current_position = agent_state.cursor + i

            # Check if agent should see this message
            if should_agent_see_message(agent_name, synapse_msg):
                # Convert to LangChain message with personalization
                lc_message = to_langchain_message(synapse_msg, agent_name)

                if lc_message is not None:
                    agent_state.personal_history.append(lc_message)
                    self.logger.debug(
                        "Added message to agent history",
                        agent=agent_name,
                        message_type=type(synapse_msg).__name__,
                        position=current_position,
                    )
                else:
                    self.logger.debug(
                        "Message conversion returned None",
                        agent=agent_name,
                        message_type=type(synapse_msg).__name__,
                        position=current_position,
                    )
            else:
                self.logger.debug(
                    "Agent should not see message",
                    agent=agent_name,
                    message_type=type(synapse_msg).__name__,
                    position=current_position,
                )

        # Update cursor to mark all messages as processed
        new_cursor = agent_state.cursor + len(unprocessed_messages)
        agent_state.cursor = new_cursor

        self.logger.info(
            "Agent history updated",
            agent=agent_name,
            new_cursor=new_cursor,
            history_length=len(agent_state.personal_history),
            processed_messages=len(unprocessed_messages),
        )

        return agent_state.personal_history

    def get_personalized_history_for_invocation(
        self, state: GroupChatState, agent_name: str
    ) -> List[BaseMessage]:
        """
        Get the complete personalized history for agent invocation.

        This first updates the history with any new messages, then returns
        the complete history ready for LLM invocation.

        Args:
            state: The global state
            agent_name: Name of the agent

        Returns:
            Complete personalized history for agent invocation
        """
        # First update the history with any new messages
        updated_history = self.update_agent_history(state, agent_name)

        # Return the complete updated history
        return updated_history

    def prepare_history_for_continuation(
        self, state: GroupChatState, agent_name: str, continuation_message: BaseMessage
    ) -> List[BaseMessage]:
        """
        Prepare history for agent continuation by adding a continuation message.

        Args:
            state: The global state
            agent_name: Name of the agent
            continuation_message: Message to add for continuation

        Returns:
            History ready for continuation invocation
        """
        agent_state = get_agent_state(state, agent_name)

        # Start with temp_new_history if it exists, otherwise use personal_history
        if agent_state.has_temp_history():
            history = agent_state.temp_new_history.copy()
            self.logger.debug(
                "Using temp history for continuation",
                agent=agent_name,
                temp_history_length=len(history),
            )
        else:
            history = agent_state.personal_history.copy()
            self.logger.debug(
                "Using personal history for continuation",
                agent=agent_name,
                personal_history_length=len(history),
            )

        # Add the continuation message
        history.append(continuation_message)

        self.logger.info(
            "Prepared history for continuation",
            agent=agent_name,
            final_length=len(history),
        )

        return history

    def extract_new_messages_from_result(
        self, result_messages: List[BaseMessage], original_length: int
    ) -> List[BaseMessage]:
        """
        Extract only the new messages from agent invocation result.

        Args:
            result_messages: Complete message history from agent invocation
            original_length: Length of history before invocation

        Returns:
            Only the new messages produced by the agent
        """
        new_messages = result_messages[original_length:]

        self.logger.debug(
            "Extracted new messages",
            total_messages=len(result_messages),
            original_length=original_length,
            new_messages_count=len(new_messages),
        )

        return new_messages

    def reset_agent_temp_state(self, state: GroupChatState, agent_name: str) -> None:
        """
        Reset agent's temporary state after successful completion.

        Args:
            state: The global state
            agent_name: Name of the agent
        """
        agent_state = get_agent_state(state, agent_name)
        agent_state.reset_temp_history()

        self.logger.debug("Reset agent temp state", agent=agent_name)

    def get_agent_history_stats(self, state: GroupChatState, agent_name: str) -> dict:
        """
        Get statistics about agent's history for debugging.

        Args:
            state: The global state
            agent_name: Name of the agent

        Returns:
            Dictionary with history statistics
        """
        agent_state = get_agent_state(state, agent_name)

        return {
            "agent_name": agent_name,
            "cursor_position": agent_state.cursor,
            "personal_history_length": len(agent_state.personal_history),
            "global_messages_count": len(state["messages"]),
            "has_temp_history": agent_state.has_temp_history(),
            "temp_history_length": len(agent_state.temp_new_history)
            if agent_state.temp_new_history
            else 0,
            "unprocessed_messages": len(state["messages"]) - agent_state.cursor,
        }


# Global personalizer instance
personalizer = MessagePersonalizer()
