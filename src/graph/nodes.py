"""Enhanced LangGraph nodes with sophisticated state management - IMPROVED VERSION."""

from typing import Callable, List, Dict, Any, Union
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.graph.state import GroupChatState, TaskInfo, AgentPersonalState
from src.core.logging import get_logger
from src.configs.settings import settings

logger = get_logger(__name__)


def _safe_get_attr(obj: Union[Dict, Any], attr: str, default=None):
    """Safely get attribute from either dict or object."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)


def _update_agent_personalized_history(
    state: GroupChatState, agent_name: str, new_messages: List[BaseMessage]
) -> AgentPersonalState:
    """Update an agent's personalized message history incrementally."""
    agent_states = state.get("agent_states", {})

    if agent_name not in agent_states:
        agent_states[agent_name] = AgentPersonalState(agent_name=agent_name)

    agent_state = agent_states[agent_name]

    # Personalize only the new messages
    personalized_new_messages = []
    for msg in new_messages:
        # Handle both dict and message object formats
        msg_name = _safe_get_attr(msg, "name")
        msg_content = _safe_get_attr(msg, "content", "")

        if isinstance(msg, AIMessage) and msg_name == agent_name:
            # Agent sees its own AI messages as they are
            personalized_new_messages.append(msg)
        elif isinstance(msg, AIMessage):
            # Agent sees other agents' AI messages as HumanMessages
            content = f"[NOT_A_USER | AGENT {msg_name}]\n\n{msg_content}"
            personalized_new_messages.append(
                HumanMessage(content=content, name=msg_name)
            )
        elif isinstance(msg, HumanMessage):
            # Agent sees user and system messages as HumanMessages with prefixes
            if msg_name == "user":
                content = f"[ACTUAL_USER]\n\n{msg_content}"
            else:
                content = f"[NOT_A_USER | SYSTEM]\n\n{msg_content}"
            personalized_new_messages.append(
                HumanMessage(content=content, name=msg_name or "system")
            )
        # IMPORTANT: ToolMessages and other types are intentionally ignored here.
        # They are handled within the nodes that call them and added to
        # the specific agent's personal state directly, not via this function.

    # Add to existing personalized history
    agent_state.personalized_messages.extend(personalized_new_messages)
    agent_state.last_updated = datetime.now()

    return agent_state


def _ensure_all_agents_have_updated_state(
    state: GroupChatState,
) -> Dict[str, AgentPersonalState]:
    """Ensure all agents have the latest personalized message history before they execute."""
    agent_states = state.get("agent_states", {})
    current_messages = state.get("messages", [])
    active_tasks = state.get("active_tasks", {})

    # For each agent that has an active task, ensure they have up-to-date state
    for agent_name in active_tasks.keys():
        if agent_name not in agent_states:
            agent_states[agent_name] = AgentPersonalState(agent_name=agent_name)

        agent_state = agent_states[agent_name]

        # Calculate how many messages this agent is missing
        existing_count = len(agent_state.personalized_messages)

        if len(current_messages) > existing_count:
            # Get the new messages since this agent's last update
            new_messages = current_messages[existing_count:]
            updated_agent_state = _update_agent_personalized_history(
                state, agent_name, new_messages
            )
            agent_states[agent_name] = updated_agent_state
            logger.debug(
                "Updated agent state before execution",
                agent=agent_name,
                new_messages_count=len(new_messages),
                total_messages=len(updated_agent_state.personalized_messages),
            )

    return agent_states


@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def orchestrator_node(state: GroupChatState, orchestrator_runnable) -> Dict[str, Any]:
    """Enhanced orchestrator node with improved state management."""
    logger.info("Starting orchestrator", round=state.get("round_number", 0))

    try:
        # Get current agent states
        agent_states = state.get("agent_states", {})

        # Update orchestrator's personalized history incrementally
        current_messages = state.get("messages", [])

        # If orchestrator state doesn't exist, create it
        if "orchestrator" not in agent_states:
            agent_states["orchestrator"] = AgentPersonalState(agent_name="orchestrator")

        orchestrator_state = agent_states["orchestrator"]

        # Find new messages since last update
        existing_count = len(orchestrator_state.personalized_messages)
        if len(current_messages) > existing_count:
            new_messages = current_messages[existing_count:]
            updated_orchestrator_state = _update_agent_personalized_history(
                state, "orchestrator", new_messages
            )
            agent_states["orchestrator"] = updated_orchestrator_state
            orchestrator_state = updated_orchestrator_state

        # Invoke orchestrator with personalized history
        result = orchestrator_runnable.invoke(
            {"messages": orchestrator_state.personalized_messages}
        )

        # Extract the new message more robustly
        orchestrator_message = None
        if isinstance(result, dict):
            if "messages" in result and result["messages"]:
                result_messages = result["messages"]
                input_length = len(orchestrator_state.personalized_messages)
                if len(result_messages) > input_length:
                    orchestrator_message = result_messages[input_length]
                else:
                    orchestrator_message = result_messages[-1]
            else:
                orchestrator_message = result
        else:
            orchestrator_message = result

        if orchestrator_message is None:
            logger.error("Could not extract orchestrator message from result")
            raise Exception("Could not extract orchestrator message from result")

        if isinstance(orchestrator_message, dict):
            if "content" in orchestrator_message:
                from langchain_core.messages import AIMessage

                tool_calls = orchestrator_message.get("tool_calls", [])
                orchestrator_message = AIMessage(
                    content=orchestrator_message["content"],
                    name="orchestrator",
                    tool_calls=tool_calls,
                )
            else:
                orchestrator_message["name"] = "orchestrator"
        else:
            orchestrator_message.name = "orchestrator"

        # Add the orchestrator's public message to its own history
        orchestrator_state.personalized_messages.append(orchestrator_message)

        tool_calls = _safe_get_attr(orchestrator_message, "tool_calls", [])
        active_tasks = {}

        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = _safe_get_attr(tool_call, "name")
            tool_args = _safe_get_attr(tool_call, "args", {})
            tool_id = _safe_get_attr(tool_call, "id")

            if tool_name == "assign_tasks":
                tasks = tool_args.get("tasks", {})
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
                    tool_call_id=tool_id,
                )
                # Add the tool result ONLY to the orchestrator's personal history
                orchestrator_state.personalized_messages.append(tool_message)

        agent_states["orchestrator"] = orchestrator_state

        return {
            "messages": [orchestrator_message],  # Return only the public message
            "agent_states": agent_states,
            "active_tasks": active_tasks,
            "completed_tasks": set(),
            "failed_tasks": set(),
            "round_number": state.get("round_number", 0) + 1,
            "last_orchestrator_round": state.get("round_number", 0) + 1,
            "pending_agent_responses": set(active_tasks.keys())
            if active_tasks
            else set(),
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


def create_agent_subgraph_node(agent_subgraph: Any, agent_name: str) -> Callable:
    """Create an enhanced agent node that uses the agent subgraph."""

    def agent_subgraph_node(state: GroupChatState) -> Dict[str, Any]:
        agent_logger = get_logger("agent_subgraph").bind(agent_name=agent_name)
        agent_logger.info("Starting agent subgraph execution")

        try:
            # This function ensures the agent has the latest messages before running.
            updated_agent_states = _ensure_all_agents_have_updated_state(state)
            agent_state = updated_agent_states[agent_name]

            task_info = state["active_tasks"].get(agent_name)
            if not task_info:
                agent_logger.warning("No active task found, skipping.")
                return {}

            # Prepare the input for the stateful subgraph.
            # Note: The subgraph only needs its own message history and the current task.
            subgraph_input = {
                "messages": agent_state.personalized_messages.copy(),
                "current_task": task_info.task_description,
            }

            # Execute the agent's subgraph
            agent_logger.info(
                f"Executing agent subgraph with task: {task_info.task_description[:100]}"
            )
            subgraph_result = agent_subgraph.invoke(subgraph_input)

            # The subgraph's final state is the result. We extract the public messages.
            public_messages = subgraph_result.get("public_messages", [])
            agent_logger.info(
                f"Agent '{agent_name}' produced {len(public_messages)} public message(s)."
            )

            # The node's only job is to return the public messages to be added to the main chat.
            # LangGraph will automatically append this list to the main `messages` state.
            return {"messages": public_messages}

        except Exception as e:
            agent_logger.error("Agent subgraph execution failed", error=str(e))
            error_message = AIMessage(
                content=f"I encountered an error while working on my task: {str(e)}",
                name=agent_name,
            )
            return {"messages": [error_message]}

    return agent_subgraph_node


def aggregator_node(state: GroupChatState) -> Dict[str, Any]:
    """
    Checks which agents have responded by looking at the newly added messages
    and updates their task status accordingly.
    """
    logger.info("Starting aggregator to update task statuses.")

    # Find the index of the orchestrator's last message. We only care about
    # new messages that came *after* the last task assignment.
    last_orchestrator_msg_idx = -1
    for i in range(len(state["messages"]) - 1, -1, -1):
        if state["messages"][i].name == "orchestrator":
            last_orchestrator_msg_idx = i
            break

    # Get the list of new messages since the last orchestrator turn.
    new_messages = state["messages"][last_orchestrator_msg_idx + 1 :]

    # Identify the agents who responded by looking at the 'name' attribute of the new messages.
    responded_agents = {msg.name for msg in new_messages if isinstance(msg, AIMessage)}

    logger.debug(f"Aggregator identified responses from agents: {responded_agents}")

    if not responded_agents:
        logger.warning("Aggregator ran but no new agent messages were found.")
        return {}

    # Get the current state of tasks and pending responses.
    active_tasks = state.get("active_tasks", {}).copy()
    pending_responses = state.get("pending_agent_responses", set()).copy()

    # For each agent that responded, mark their task as completed and remove them
    # from the set of agents we are waiting on.
    for agent_name in responded_agents:
        if agent_name in active_tasks:
            logger.info(f"Marking task for agent '{agent_name}' as completed.")
            active_tasks[agent_name].status = "completed"
            pending_responses.discard(agent_name)

    # Return the updated task and pending response state to be merged into the graph.
    return {
        "active_tasks": active_tasks,
        "pending_agent_responses": pending_responses,
    }


def create_legacy_agent_node(agent_runnable: Any, agent_name: str) -> Callable:
    """Create a legacy agent node (for backwards compatibility)."""

    def legacy_agent_node(state: GroupChatState) -> Dict[str, Any]:
        """Legacy agent node that maintains backwards compatibility."""
        agent_logger = get_logger("legacy_agent").bind(agent_name=agent_name)
        agent_logger.info("Starting legacy agent execution")

        try:
            # Use the same state update pattern as subgraph nodes
            updated_agent_states = _ensure_all_agents_have_updated_state(state)

            if agent_name not in updated_agent_states:
                updated_agent_states[agent_name] = AgentPersonalState(
                    agent_name=agent_name
                )

            agent_state = updated_agent_states[agent_name]

            # Get task info
            task_info = state["active_tasks"].get(agent_name)
            if not task_info:
                agent_logger.warning("No active task found")
                return {"agent_states": updated_agent_states}

            # Update task status
            task_info.status = "in_progress"

            # Add task message to personalized history
            task_message = HumanMessage(
                content=f"[NOT_A_USER | TASK ASSIGNED TO YOU BY THE ORCHESTRATOR]\n\n{task_info.task_description}",
            )
            final_history = agent_state.personalized_messages + [task_message]

            # Execute agent
            agent_logger.info(
                "Executing legacy agent",
                task=task_info.task_description[:100],
                input_messages_count=len(final_history),
            )

            result = agent_runnable.invoke({"messages": final_history})

            # Create response message
            if result and "messages" in result and result["messages"]:
                last_result_message = result["messages"][-1]
                content = _safe_get_attr(last_result_message, "content", "No response")
            else:
                content = "No response received"

            response_message = AIMessage(content=str(content), name=agent_name)

            # Update agent's personalized history
            agent_state.personalized_messages = final_history + [response_message]
            agent_state.last_updated = datetime.now()
            updated_agent_states[agent_name] = agent_state

            agent_logger.info("Legacy agent completed successfully")
            return {
                "messages": [response_message],
                "agent_states": updated_agent_states,
            }

        except Exception as e:
            agent_logger.error("Legacy agent execution failed", error=str(e))

            # Update task status
            if "task_info" in locals() and task_info:
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
                "agent_states": updated_agent_states
                if "updated_agent_states" in locals()
                else state.get("agent_states", {}),
            }

    return legacy_agent_node
