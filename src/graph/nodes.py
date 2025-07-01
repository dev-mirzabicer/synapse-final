import logging
from typing import Callable, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain.agents import AgentExecutor

from .state import GroupChatState

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def _personalize_history(state: GroupChatState, agent_name: str) -> List[BaseMessage]:
    """
    Creates a personalized message history for a specific agent.

    This function transforms the shared message history into a view tailored
    for a given agent, marking messages from other agents and the user
    with special prefixes for clarity.
    """
    personalized_messages = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.name == agent_name:
            # The agent's own messages are kept as is
            personalized_messages.append(msg)
        elif isinstance(msg, AIMessage):
            # Messages from other agents (including orchestrator) are marked
            content = f"[NOT_AN_USER | AGENT {msg.name}]\n\n{msg.content}"
            personalized_messages.append(HumanMessage(content=content, name=msg.name))
        elif isinstance(msg, HumanMessage):
            # Messages from the actual user are marked
            content = f"[ACTUAL_USER]\n\n{msg.content}"
            personalized_messages.append(
                HumanMessage(content=content, name=msg.name or "user")
            )
        else:
            # Other message types (like ToolMessage) are included as is
            personalized_messages.append(msg)
    return personalized_messages


def orchestrator_node(state: GroupChatState, orchestrator_runnable) -> GroupChatState:
    """
    The central planning node for the group chat.

    It personalizes the history for the orchestrator, invokes the LLM with the
    task-assignment tool, and updates the state with new tasks and its own message.
    """
    logger.info("---ORCHESTRATOR---")

    # Personalize history for the orchestrator
    personalized_history = _personalize_history(state, "orchestrator")

    # The orchestrator runnable is an LLM with the `assign_tasks` tool bound
    response = orchestrator_runnable.invoke({"messages": personalized_history})

    # The response is the orchestrator's message, which may contain tool calls
    orchestrator_message = response

    active_tasks = {}
    tool_message = None

    if orchestrator_message.tool_calls:
        # for tool_call in orchestrator_message.tool_calls:
        if len(orchestrator_message.tool_calls) > 1:
            logger.warning(
                "Orchestrator returned multiple tool calls. Only the first one will be processed."
            )
        tool_call = orchestrator_message.tool_calls[0]
        if tool_call["name"] == "assign_tasks":
            tasks = tool_call["args"].get("tasks", {})
            active_tasks.update(tasks)
            logger.info(f"Orchestrator assigned tasks: {tasks}")
        else:
            logger.warning(
                f"Orchestrator called an unexpected tool: {tool_call['name']}. Expected 'assign_tasks'."
            )
        tool_message = ToolMessage(
            content=f"Tasks assigned: {', '.join(active_tasks.keys())}",
            tool_call_id=tool_call["id"],
        )

    messages_to_add = [orchestrator_message]

    if tool_message:
        messages_to_add.append(tool_message)

    return {
        "messages": messages_to_add,
        "active_tasks": active_tasks,
        "completed_tasks": set(),  # Reset for the new round
    }


def create_agent_node(agent_runnable: AgentExecutor, agent_name: str) -> Callable:
    """
    Factory function that creates a graph node for a specific agent.

    The returned node function will:
    1. Personalize the message history for its agent.
    2. Invoke the agent with its specific task.
    3. Return the agent's response as a new message.
    """

    def agent_node(state: GroupChatState) -> dict:
        logger.info(f"---AGENT: {agent_name}---")

        task = state["active_tasks"].get(agent_name)
        if task is None:
            logger.warning(f"Agent {agent_name} was invoked without an active task.")
            return {}

        # Personalize the history for this agent
        personalized_history = _personalize_history(state, agent_name)

        task_message = HumanMessage(
            content=f"[ORCHESTRATOR'S TASK FOR YOU]\n\n{task}",
        )

        final_history = personalized_history + [task_message]

        logger.info(f"Task for {agent_name}: {task}")

        # Invoke the agent executor with the personalized history and task
        result = agent_runnable.invoke({"messages": final_history})

        response_message = AIMessage(content=str(result["output"]), name=agent_name)

        logger.info(
            f"Agent {agent_name} finished. Response: {response_message.content[:100]}..."
        )
        return {"messages": [response_message]}

    return agent_node


def aggregator_node(state: GroupChatState) -> dict:
    """
    Synchronizes the parallel agent executions.
    """
    last_message = state["messages"][-1]
    sender_name = last_message.name

    logger.info(f"---AGGREGATOR---")
    logger.info(f"Received response from: {sender_name}")

    completed_tasks = state.get("completed_tasks", set())
    completed_tasks.add(sender_name)

    logger.info(f"Completed tasks: {len(completed_tasks)}/{len(state['active_tasks'])}")

    return {"completed_tasks": completed_tasks}
