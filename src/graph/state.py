"""Enhanced state management for the multi-agent system."""

import operator
from typing import List, Dict, Set, Annotated, Optional, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from datetime import datetime


class TaskInfo(BaseModel):
    """Information about a task assigned to an agent."""

    agent_name: str
    task_description: str
    assigned_at: datetime
    status: str = "pending"  # pending, in_progress, completed, failed
    error_message: Optional[str] = None
    retry_count: int = 0


class GroupChatState(TypedDict):
    """Enhanced state for the group chat application."""

    messages: Annotated[List[BaseMessage], operator.add]
    """Complete conversation history."""

    active_tasks: Dict[str, TaskInfo]
    """Currently active tasks by agent name."""

    completed_tasks: Set[str]
    """Agent names that completed their tasks this round."""

    failed_tasks: Set[str]
    """Agent names whose tasks failed this round."""

    round_number: int
    """Current orchestration round."""

    error_count: int
    """Total number of errors encountered."""

    context: Dict[str, Any]
    """Additional context for agents."""

    conversation_id: Optional[str]
    """Unique identifier for this conversation."""
