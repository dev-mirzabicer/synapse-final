import operator
from typing import List, Dict, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class GroupChatState(TypedDict):
    """
    Represents the complete state of the group chat application.

    This state is passed between all nodes in the graph.
    """

    messages: Annotated[List[BaseMessage], operator.add]
    """The list of all messages in the conversation, serving as the shared history."""

    active_tasks: Dict[str, str]
    """
    A dictionary mapping an agent's name to its currently assigned task.
    The presence of an agent's name as a key indicates it is currently working.
    e.g., {'researcher': 'Find the latest news on AI.'}
    This is cleared at the end of each turn.
    """
