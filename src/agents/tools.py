from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from typing import Dict, Annotated, List, Any
from datetime import datetime
from src.core.logging import get_logger

logger = get_logger(__name__)


@tool
def assign_tasks(
    tasks: Annotated[
        Dict[str, str],
        "A dictionary where keys are agent names and values are task descriptions",
    ],
) -> str:
    """
    Assigns tasks to specific agents with enhanced validation and logging.

    Args:
        tasks: Dictionary mapping agent names to their task descriptions.
               Example: {"researcher": "Find latest AI papers", "writer": "Summarize findings"}

    Returns:
        Confirmation message with task assignment details.
    """
    if not tasks:
        logger.warning("Empty task assignment attempted")
        return "No tasks provided for assignment."

    logger.info(
        "Tasks assigned",
        agent_count=len(tasks),
        agents=list(tasks.keys()),
        timestamp=datetime.now().isoformat(),
    )

    agent_names = ", ".join(tasks.keys())
    task_summaries = [f"{agent}: {desc[:50]}..." for agent, desc in tasks.items()]

    return (
        f"Tasks successfully assigned to {len(tasks)} agents: {agent_names}. "
        f"Task summaries: {'; '.join(task_summaries)}"
    )


@tool
def search_web(
    query: Annotated[str, "Search query for web search"],
    max_results: Annotated[int, "Maximum number of results to return"] = 3,
) -> str:
    """
    Enhanced web search tool with error handling and result formatting.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Formatted search results or error message
    """
    try:
        logger.info("Web search initiated", query=query, max_results=max_results)

        search_tool = TavilySearch(max_results=max_results)
        results = search_tool.invoke(query)

        if not results:
            return f"No results found for query: {query}"

        logger.info(
            "Web search completed",
            query=query,
            result_count=len(results) if isinstance(results, list) else 1,
        )

        return f"Search results for '{query}':\n{results}"

    except Exception as e:
        logger.error("Web search failed", query=query, error=str(e))
        return f"Search failed for query '{query}': {str(e)}"


def get_default_tools() -> List[Any]:
    """Get the list of default tools available to agents."""
    return [
        assign_tasks,
        search_web,
        # Add more tools here as needed
    ]
