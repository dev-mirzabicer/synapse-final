import logging
from typing import List, Dict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain.agents import AgentExecutor
from langgraph.func import task

from src.graph.state import GroupChatState

# Configure logging
logger = logging.getLogger(__name__)


def _personalize_history(
    messages: List[BaseMessage], agent_name: str
) -> List[BaseMessage]:
    """
    Creates a personalized message history for a specific agent.

    This function transforms the shared message history into a view tailored
    for a given agent, marking messages from other agents and the user
    with special prefixes for clarity, as defined in the system prompts.
    """
    personalized_messages = []
    for msg in messages:
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


@task
async def run_orchestrator(state: GroupChatState, orchestrator_runnable) -> Dict:
    """
    Runs the orchestrator task to plan the next steps.

    Args:
        state: The current state of the group chat.
        orchestrator_runnable: The compiled LangChain runnable for the orchestrator.

    Returns:
        A dictionary with updates for the GroupChatState.
    """
    logger.info("---TASK: Running Orchestrator---")
    # The orchestrator does not need a personalized history as it sees the raw log.
    response = await orchestrator_runnable.ainvoke({"messages": state["messages"]})

    # The response is the orchestrator's message, which may contain tool calls
    orchestrator_message = response
    active_tasks = {}
    tool_messages = []

    if orchestrator_message.tool_calls:
        for tool_call in orchestrator_message.tool_calls:
            if tool_call["name"] == "assign_tasks":
                tasks = tool_call["args"].get("tasks", {})
                active_tasks.update(tasks)
                tool_messages.append(
                    ToolMessage(
                        content=f"Tasks successfully assigned to: {', '.join(tasks.keys())}",
                        tool_call_id=tool_call["id"],
                    )
                )
                logger.info(f"Orchestrator assigned tasks: {tasks}")
            else:
                logger.warning(
                    f"Orchestrator called an unexpected tool: {tool_call['name']}"
                )
    else:
        logger.info("Orchestrator did not assign any tasks.")

    messages_to_add = [orchestrator_message] + tool_messages
    return {"messages": messages_to_add, "active_tasks": active_tasks}


@task
async def run_agent(
    agent_name: str,
    task_description: str,
    messages: List[BaseMessage],
    agent_runnable: AgentExecutor,
) -> AIMessage:
    """
    Runs a single agent task.

    Args:
        agent_name: The name of the agent to run.
        task_description: The specific task assigned by the orchestrator.
        messages: The full, shared message history.
        agent_runnable: The compiled LangChain runnable for the specific agent.

    Returns:
        An AIMessage containing the agent's response.
    """
    logger.info(f"---TASK: Running Agent '{agent_name}'---")

    # Personalize the history for this agent
    personalized_history = _personalize_history(messages, agent_name)

    # Add the specific task as the final message for the agent
    task_message = HumanMessage(
        content=f"[ORCHESTRATOR'S TASK FOR YOU]\n\n{task_description}"
    )
    final_input = {"messages": personalized_history + [task_message]}

    logger.info(f"Task for '{agent_name}': {task_description}")

    # Invoke the agent executor
    result = await agent_runnable.ainvoke(final_input)

    # Format the response
    response_content = str(result.get("output", "No output provided."))
    response_message = AIMessage(content=response_content, name=agent_name)

    logger.info(f"Agent '{agent_name}' finished. Response: {response_content[:150]}...")
    return response_message
