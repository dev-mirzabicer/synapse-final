"""Enhanced LangGraph nodes with error handling and logging."""

import logging
from typing import Callable, List, Dict, Any, Union
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from tenacity import retry, stop_after_attempt, wait_exponential

from .state import GroupChatState, TaskInfo
from src.core.exceptions import AgentError, StateError
from src.core.logging import get_logger
from src.configs.settings import settings

logger = get_logger(__name__)


def _personalize_history(state: GroupChatState, agent_name: str) -> List[BaseMessage]:
    """Create personalized message history for an agent."""
    personalized_messages = []

    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.name == agent_name:
            personalized_messages.append(msg)
        elif isinstance(msg, AIMessage):
            content = f"[AGENT {msg.name}]\n\n{msg.content}"
            personalized_messages.append(HumanMessage(content=content, name=msg.name))
        elif isinstance(msg, HumanMessage):
            content = f"[USER]\n\n{msg.content}"
            personalized_messages.append(
                HumanMessage(content=content, name=msg.name or "user")
            )
        else:
            personalized_messages.append(msg)

    return personalized_messages


@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def orchestrator_node(state: GroupChatState, orchestrator_runnable) -> Dict[str, Any]:
    """Enhanced orchestrator node with error handling."""
    logger.info("Starting orchestrator", round=state.get("round_number", 0))

    try:
        # Personalize history
        personalized_history = _personalize_history(state, "orchestrator")

        # Invoke orchestrator
        orchestrator_message = orchestrator_runnable.invoke(
            {"messages": personalized_history}
        )
        orchestrator_message.name = "orchestrator"

        # Process tool calls
        active_tasks = {}
        tool_message = None

        if orchestrator_message.tool_calls:
            tool_call = orchestrator_message.tool_calls[0]
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

                logger.info("Tasks assigned", tasks=list(tasks.keys()))
                tool_message = ToolMessage(
                    content=f"Tasks assigned: {', '.join(tasks.keys())}",
                    tool_call_id=tool_call["id"],
                )

        messages_to_add = [orchestrator_message]
        if tool_message:
            messages_to_add.append(tool_message)

        return {
            "messages": messages_to_add,
            "active_tasks": active_tasks,
            "completed_tasks": set(),
            "failed_tasks": set(),
            "round_number": state.get("round_number", 0) + 1,
        }

    except Exception as e:
        logger.error("Orchestrator error", error=str(e))
        error_message = AIMessage(
            content=f"I encountered an error: {str(e)}. Let me try to help you anyway.",
            name="orchestrator",
        )
        return {
            "messages": [error_message],
            "error_count": state.get("error_count", 0) + 1,
        }


def create_agent_node(agent_runnable: Any, agent_name: str) -> Callable:
    """Create an enhanced agent node with error handling."""

    def agent_node(state: GroupChatState) -> Dict[str, Any]:
        agent_logger = get_logger("agent").bind(agent_name=agent_name)
        agent_logger.info("Starting agent execution")

        try:
            # Get task info
            task_info = state["active_tasks"].get(agent_name)
            if not task_info:
                agent_logger.warning("No active task found")
                return {}

            # Update task status
            task_info.status = "in_progress"

            # Personalize history
            personalized_history = _personalize_history(state, agent_name)

            # Add task message
            task_message = HumanMessage(
                content=f"[TASK]\n\n{task_info.task_description}",
            )
            final_history = personalized_history + [task_message]

            # Execute agent with timeout
            agent_logger.info("Executing agent", task=task_info.task_description[:100])

            result = agent_runnable.invoke({"messages": final_history})

            # Create response message
            response_message = AIMessage(
                content=str(result["messages"][-1].content), name=agent_name
            )

            agent_logger.info("Agent completed successfully")
            return {"messages": [response_message]}

        except Exception as e:
            agent_logger.error("Agent execution failed", error=str(e))

            # Update task status
            if task_info:
                task_info.status = "failed"
                task_info.error_message = str(e)

            # Create error message
            error_message = AIMessage(
                content=f"I encountered an error while working on the task: {str(e)}",
                name=agent_name,
            )

            return {
                "messages": [error_message],
                "failed_tasks": {agent_name},
                "error_count": state.get("error_count", 0) + 1,
            }

    return agent_node


def aggregator_node(state: GroupChatState) -> Dict[str, Any]:
    """Enhanced aggregator node that properly tracks all agent completions."""
    logger.info("Starting aggregator")

    last_message = state["messages"][-1]
    sender_name = last_message.name

    # Update completion status
    completed_tasks = state.get("completed_tasks", set()).copy()
    failed_tasks = state.get("failed_tasks", set()).copy()
    active_tasks = state.get("active_tasks", {})

    if sender_name in active_tasks:
        task_info = active_tasks[sender_name]
        if task_info.status == "failed" or "error" in last_message.content.lower():
            failed_tasks.add(sender_name)
            task_info.status = "failed"
            logger.info("Agent task failed", agent=sender_name)
        else:
            completed_tasks.add(sender_name)
            task_info.status = "completed"
            logger.info("Agent task completed", agent=sender_name)

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

    return {
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        # Ensure task info is updated
        "active_tasks": active_tasks,
    }
