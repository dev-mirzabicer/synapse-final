import asyncio
import logging
import json
from pathlib import Path

from langgraph.func import entrypoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.configs.models import TeamConfig
from src.agents.factory import create_agent_runnable
from src.agents.tools import assign_tasks
from src.prompts.templates import ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE
from .state import GroupChatState
from .tasks import run_orchestrator, run_agent

# Configure logging
logger = logging.getLogger(__name__)


def build_graph(team_config_path: Path, checkpointer=None):
    """
    Builds and compiles the LangGraph workflow using the Functional API.
    """
    logger.info(f"Loading team configuration from: {team_config_path}")
    config_data = json.loads(team_config_path.read_text())
    team_config = TeamConfig(**config_data)

    # --- Create Runnables ---
    # Orchestrator Runnable
    team_agent_names = [agent.name for agent in team_config.agents]
    formatted_orchestrator_prompt = ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE.format(
        team_name=team_config.team_name,
        team_desc=team_config.team_description,
        team_agent_list=", ".join(team_agent_names),
    )
    orchestrator_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", formatted_orchestrator_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    orchestrator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    orchestrator_llm_with_tools = orchestrator_llm.bind_tools([assign_tasks])
    orchestrator_runnable = orchestrator_prompt_template | orchestrator_llm_with_tools

    # Agent Runnables
    agent_runnables = {
        agent_config.name: create_agent_runnable(agent_config, team_config)
        for agent_config in team_config.agents
    }

    # --- Define Workflow using @entrypoint ---
    @entrypoint(checkpointer=checkpointer)
    async def group_chat_workflow(initial_state: GroupChatState):
        """
        The main workflow for the multi-agent group chat. This function now
        contains a loop to handle multiple rounds of agent interactions for a
        single user query.
        """
        # The state for the current turn, starting with the user's input.
        # We will be returning only the *new* messages generated in this turn.
        turn_messages = []

        # The full history that grows with each step of the internal loop.
        current_full_history = initial_state["messages"]

        while True:
            # 1. Run the orchestrator to get a plan
            orchestrator_result = await run_orchestrator(
                {"messages": current_full_history},
                orchestrator_runnable=orchestrator_runnable,
            )

            new_orchestrator_messages = orchestrator_result["messages"]
            turn_messages.extend(new_orchestrator_messages)
            current_full_history = current_full_history + new_orchestrator_messages

            # 2. Check if the orchestrator assigned any tasks
            if not (active_tasks := orchestrator_result.get("active_tasks")):
                # If no tasks, the internal loop is over.
                logger.info("Orchestrator has no more tasks. Ending turn.")
                break

            # 3. Execute agent tasks in parallel
            agent_invocations = [
                run_agent(
                    agent_name=name,
                    task_description=task,
                    messages=current_full_history,
                    agent_runnable=agent_runnables[name],
                )
                for name, task in active_tasks.items()
            ]
            agent_responses = await asyncio.gather(*agent_invocations)

            # 4. Add agent responses to the history for the next loop iteration
            turn_messages.extend(agent_responses)
            current_full_history = current_full_history + agent_responses

        # The loop has finished, return all messages generated during this turn.
        return {"messages": turn_messages}

    logger.info("Graph construction complete.")
    return group_chat_workflow
