"""Enhanced state management for the multi-agent system with per-agent state tracking."""

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


class AgentPersonalState(BaseModel):
    """Personal state maintained for each agent."""

    agent_name: str
    personalized_messages: List[BaseMessage] = []
    private_messages: List[BaseMessage] = []
    tool_call_history: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {}
    last_updated: datetime = datetime.now()

    class Config:
        arbitrary_types_allowed = True


class GroupChatState(TypedDict):
    """Enhanced state for the group chat application with per-agent state tracking."""

    messages: Annotated[List[BaseMessage], operator.add]
    """Complete conversation history (public messages only)."""

    agent_states: Dict[str, AgentPersonalState]
    """Personal state for each agent including personalized message history."""

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

    last_orchestrator_round: int
    """Track when orchestrator last assigned tasks."""

    pending_agent_responses: Set[str]
    """Track which agents still need to respond."""
