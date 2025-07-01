import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from langchain_core.prompts import MessagesPlaceholder

from src.configs.models import AgentConfig, TeamConfig
from src.prompts.templates import AGENT_SYSTEM_PROMPT_TEMPLATE

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# A mapping from tool names to their implementation
# This allows us to dynamically add tools to agents based on their config
SUPPORTED_TOOLS = {
    "tavily_search_results_json": TavilySearch(max_results=3)
    # Add other tools here as they are created
}


def create_agent_runnable(
    agent_config: AgentConfig, team_config: TeamConfig
) -> AgentExecutor:
    """
    Factory function to create a LangChain AgentExecutor (a runnable) for a given agent.

    Args:
        agent_config: The configuration for the specific agent to be created.
        team_config: The configuration for the team, used to provide context to the agent.

    Returns:
        An AgentExecutor instance ready to be used as a node in the LangGraph.
    """
    logger.info(f"Creating agent runnable for: {agent_config.name}")

    # 1. Instantiate the LLM
    llm = ChatGoogleGenerativeAI(
        model=agent_config.model_name,
        temperature=agent_config.temperature,
        **agent_config.kwargs,
    )

    # 2. Format the system prompt with team and agent details
    team_agent_names = [agent.name for agent in team_config.agents]
    formatted_system_prompt = AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        agent_name=agent_config.name,
        agent_specific_prompt=agent_config.system_prompt_template,
        team_name=team_config.team_name,
        team_desc=team_config.team_description,
        team_agent_list=", ".join(team_agent_names),
    )

    # 3. Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", formatted_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 4. Instantiate and gather the tools for this agent
    agent_tools = []
    for tool_name in agent_config.tools:
        if tool_name in SUPPORTED_TOOLS:
            agent_tools.append(SUPPORTED_TOOLS[tool_name])
            logger.info(f"Added tool '{tool_name}' to agent '{agent_config.name}'")
        else:
            logger.warning(
                f"Tool '{tool_name}' is not supported and will be skipped for agent '{agent_config.name}'."
            )

    # 5. Create the agent itself using the robust `create_tool_calling_agent`
    agent = create_tool_calling_agent(llm, agent_tools, prompt)

    # 6. Create the AgentExecutor, which is the final runnable
    # The handle_parsing_errors flag provides additional robustness
    agent_executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        verbose=True,  # Set to True for detailed agent step logging
        handle_parsing_errors=True,
    )

    logger.info(f"Successfully created agent executor for '{agent_config.name}'")
    return agent_executor
