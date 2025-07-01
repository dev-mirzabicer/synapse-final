import logging
import json
from functools import partial
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.configs.models import TeamConfig
from src.agents.factory import create_agent_runnable
from src.agents.tools import assign_tasks
from src.prompts.templates import ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE
from .state import GroupChatState
from .nodes import orchestrator_node, create_agent_node, aggregator_node

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def build_graph(team_config_path: Path):
    """
    Builds and compiles the LangGraph for the multi-agent group chat.
    """
    # 1. Load configurations
    logger.info(f"Loading team configuration from: {team_config_path}")
    config_data = json.loads(team_config_path.read_text())
    team_config = TeamConfig(**config_data)

    # 2. Create the Orchestrator's LLM with its specific tool
    team_agent_names = [agent.name for agent in team_config.agents]
    formatted_orchestrator_prompt = ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE.format(
        team_name=team_config.team_name,
        team_desc=team_config.team_description,
        team_agent_list=", ".join(team_agent_names),
    )

    # Use a ChatPromptTemplate for robust and clear prompt construction
    orchestrator_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", formatted_orchestrator_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Create the orchestrator LLM with the assign_tasks tool bound
    orchestrator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    orchestrator_llm_with_tools = orchestrator_llm.bind_tools([assign_tasks])

    # Chain the LLM with the tool to create a runnable
    orchestrator_runnable = orchestrator_prompt_template | orchestrator_llm_with_tools

    # 3. Create agent runnables
    agent_runnables = {
        agent_config.name: create_agent_runnable(agent_config, team_config)
        for agent_config in team_config.agents
    }

    # 4. Define the graph
    workflow = StateGraph(GroupChatState)

    # 5. Add nodes to the graph
    # Orchestrator
    workflow.add_node(
        "orchestrator",
        partial(orchestrator_node, orchestrator_runnable=orchestrator_runnable),
    )
    # Aggregator
    workflow.add_node("aggregator", aggregator_node)
    # Agent nodes
    for agent_name, agent_runnable in agent_runnables.items():
        workflow.add_node(agent_name, create_agent_node(agent_runnable, agent_name))

    # 6. Define the edges and control flow
    workflow.set_entry_point("orchestrator")

    # Conditional edge from the orchestrator
    def orchestrator_router(state: GroupChatState):
        if not state.get("active_tasks"):
            logger.info("Orchestrator decided to end the turn.")
            return END
        else:
            logger.info(
                f"Orchestrator routing to agents: {list(state['active_tasks'].keys())}"
            )
            return list(state["active_tasks"].keys())

    workflow.add_conditional_edges("orchestrator", orchestrator_router)

    # Edges from all agents to the aggregator
    for agent_name in agent_runnables.keys():
        workflow.add_edge(agent_name, "aggregator")

    # Conditional edge from the aggregator
    def aggregator_router(state: GroupChatState):
        active_tasks = state.get("active_tasks", {})
        completed_tasks = state.get("completed_tasks", set())
        if completed_tasks == set(active_tasks.keys()):
            logger.info("All tasks complete. Routing back to orchestrator.")
            return "orchestrator"
        else:
            logger.info("Tasks still in progress. Waiting.")
            return END

    workflow.add_conditional_edges(
        "aggregator", aggregator_router, {"orchestrator": "orchestrator", END: END}
    )

    # 7. Compile the graph
    logger.info("Compiling the graph.")
    graph = workflow.compile()
    logger.info("Graph compiled successfully.")

    return graph
