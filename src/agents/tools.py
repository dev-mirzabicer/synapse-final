from langchain_core.tools import tool
from typing import Dict, Annotated

@tool
def assign_tasks(tasks: Annotated[Dict[str, str], "A dictionary where keys are the names of agents and values are their assigned tasks. This tool must be called to assign tasks to one or more agents."]) -> str:
    """
    Assigns one or more tasks to specific agents.

    Args:
        tasks: A dictionary where keys are the exact names of the agents
               and values are the detailed task descriptions for each agent.
               Example: {"researcher": "Find the latest papers on LLM agents.", "software_engineer": "Find the best LLM providers and utility libraries"}
    """
    # This tool's primary purpose is to be called by the LLM.
    # The logic of processing these tasks will be handled in the orchestrator node
    # by inspecting the tool call in the AI message.
    # This function's return value is for confirming the tool call was received.
    agent_names = ", ".join(tasks.keys())
    return f"Tasks successfully assigned to the following agents: {agent_names}."

# We can add more tools here in the future, for example, for agents to use.
# For now, we'll just define the one for the orchestrator.
# Example agent tool:
# from tavily import TavilyClient
# tavily = TavilyClient(api_key="YOUR_TAVILY_API_KEY")
# @tool
# def tavily_search(query: str) -> str:
#     """Performs a search using Tavily."""
#     return tavily.search(query)