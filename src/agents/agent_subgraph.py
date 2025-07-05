"""Enhanced agent subgraph system for sophisticated agent behavior - FINAL FIXED VERSION."""

from typing import Dict, List, Any, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict
from datetime import datetime

from src.configs.models import AgentConfig, TeamConfig
from src.configs.settings import settings
from src.core.logging import get_logger
from src.core.exceptions import AgentError
from src.agents.tools import get_agent_tools
from src.prompts.templates import ENHANCED_AGENT_SYSTEM_PROMPT_TEMPLATE

logger = get_logger(__name__)


def _safe_get_attr(obj: Union[Dict, Any], attr: str, default=None):
    """Safely get attribute from either dict or object."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)


class AgentSubgraphState(TypedDict):
    """State for individual agent subgraphs."""

    messages: List[BaseMessage]
    """Personal message history for this agent."""

    public_messages: List[BaseMessage]
    """Messages marked for public sharing."""

    private_messages: List[BaseMessage]
    """Messages kept private to this agent."""

    tool_calls: List[Dict[str, Any]]
    """History of tool calls made by this agent."""

    current_task: Optional[str]
    """Current task assigned to this agent."""

    turn_finished: bool
    """Whether the agent has finished their turn."""

    continue_working: bool
    """Whether the agent should continue working."""

    agent_name: str
    """Name of this agent."""

    task_added: bool
    """Whether the task message has been added to prevent duplicates."""


class AgentSubgraphBuilder:
    """Builder for creating enhanced agent subgraphs."""

    def __init__(self, agent_config: AgentConfig, team_config: TeamConfig):
        self.agent_config = agent_config
        self.team_config = team_config
        self.logger = get_logger(f"agent_subgraph.{agent_config.name}")

    def build_agent_subgraph(self) -> Any:
        """Build a sophisticated agent subgraph."""
        try:
            # Create the state graph for this agent
            workflow = StateGraph(AgentSubgraphState)

            # Add nodes
            workflow.add_node("initialize_turn", self._initialize_turn_node)
            workflow.add_node("agent_work", self._create_agent_work_node())
            workflow.add_node("process_response", self._process_response_node)
            workflow.add_node("check_completion", self._check_completion_node)

            # Define edges
            workflow.set_entry_point("initialize_turn")
            workflow.add_edge("initialize_turn", "agent_work")
            workflow.add_edge("agent_work", "process_response")
            workflow.add_edge("process_response", "check_completion")

            # Conditional edge from check_completion
            workflow.add_conditional_edges(
                "check_completion",
                self._should_continue_working,
                {"continue": "agent_work", "finish": END},
            )

            # Compile the subgraph
            agent_subgraph = workflow.compile()
            self.logger.info("Agent subgraph compiled successfully")
            return agent_subgraph

        except Exception as e:
            self.logger.error("Failed to build agent subgraph", error=str(e))
            raise AgentError(
                self.agent_config.name, f"Failed to build subgraph: {e}", e
            )

    def _initialize_turn_node(self, state: AgentSubgraphState) -> Dict[str, Any]:
        """Initialize the agent's turn."""
        self.logger.info("Initializing agent turn", agent=self.agent_config.name)

        return {
            "turn_finished": False,
            "continue_working": True,
            "agent_name": self.agent_config.name,
            "public_messages": [],
            "private_messages": [],
            "task_added": False,  # NEW: Track whether task has been added
        }

    def _create_agent_work_node(self):
        """Create the core agent work node using ReAct pattern."""

        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model=self.agent_config.model_name,
            temperature=self.agent_config.temperature,
            api_key=settings.google_api_key,
            **self.agent_config.kwargs,
        )

        # Get tools (including finish_my_turn)
        available_tools = {tool.name: tool for tool in get_agent_tools()}
        tools = []

        # Always add finish_my_turn tool
        if "finish_my_turn" in available_tools:
            tools.append(available_tools["finish_my_turn"])

        # Add configured tools
        for tool_name in self.agent_config.tools:
            if (
                tool_name in available_tools and tool_name != "finish_my_turn"
            ):  # Avoid duplicates
                tools.append(available_tools[tool_name])

        # Create enhanced prompt
        team_agent_names = [agent.name for agent in self.team_config.agents]
        formatted_system_prompt = ENHANCED_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=self.agent_config.name,
            agent_specific_prompt=self.agent_config.system_prompt_template,
            team_name=self.team_config.team_name,
            team_desc=self.team_config.team_description,
            team_agent_list=", ".join(team_agent_names),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", formatted_system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Create ReAct agent
        react_agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=prompt,
            debug=settings.log_level == "DEBUG",
        )

        def agent_work_node(state: AgentSubgraphState) -> Dict[str, Any]:
            """Execute the agent's work using ReAct pattern."""
            try:
                self.logger.info("Agent starting work", agent=self.agent_config.name)

                # Start with existing messages
                current_messages = state.get("messages", []).copy()
                task_added = state.get("task_added", False)

                # FIXED: Only add task message once, not every iteration
                if not task_added and state.get("current_task"):
                    task_message = HumanMessage(
                        content=f"[NOT_A_USER | TASK ASSIGNED TO YOU BY THE ORCHESTRATOR]\n\n{state['current_task']}"
                    )
                    current_messages.append(task_message)
                    task_added = True
                    self.logger.debug("Added task message to agent history")

                # FIXED: Check if we've already called finish_my_turn in previous iterations
                # by looking for finish_my_turn tool calls in existing messages
                already_finished = False
                for msg in current_messages:
                    tool_calls = _safe_get_attr(msg, "tool_calls", [])
                    for tool_call in tool_calls:
                        if _safe_get_attr(tool_call, "name") == "finish_my_turn":
                            already_finished = True
                            break
                    if already_finished:
                        break

                if already_finished:
                    # Agent already called finish_my_turn, don't invoke ReAct again
                    self.logger.info("Agent already finished, skipping ReAct execution")
                    return {
                        "messages": current_messages,
                        "task_added": task_added,
                        "latest_response": [],  # No new response
                        "turn_finished": True,  # Mark as finished
                    }

                # Execute the ReAct agent
                result = react_agent.invoke({"messages": current_messages})

                # Extract ALL new messages from the ReAct execution
                all_result_messages = result.get("messages", [])
                input_message_count = len(current_messages)

                # Get only the NEW messages generated by the agent
                new_messages = (
                    all_result_messages[input_message_count:]
                    if len(all_result_messages) > input_message_count
                    else []
                )

                if not new_messages:
                    # Fallback if no new messages
                    new_messages = [
                        AIMessage(
                            content="I need to continue thinking about this task."
                        )
                    ]

                # Extract tool calls from ALL new messages
                tool_calls = []
                turn_finished = False

                for msg in new_messages:
                    msg_tool_calls = _safe_get_attr(msg, "tool_calls", [])
                    if msg_tool_calls:
                        tool_calls.extend(
                            [
                                {
                                    "tool_name": _safe_get_attr(tc, "name", "unknown"),
                                    "args": _safe_get_attr(tc, "args", {}),
                                    "timestamp": datetime.now().isoformat(),
                                }
                                for tc in msg_tool_calls
                            ]
                        )

                        # Check if finish_my_turn was called
                        for tc in msg_tool_calls:
                            if _safe_get_attr(tc, "name") == "finish_my_turn":
                                turn_finished = True
                                self.logger.info(
                                    "Agent called finish_my_turn, marking turn as finished"
                                )
                                break

                self.logger.info(
                    "Agent work completed",
                    agent=self.agent_config.name,
                    turn_finished=turn_finished,
                )

                return {
                    "messages": current_messages + new_messages,  # Complete history
                    "tool_calls": state.get("tool_calls", []) + tool_calls,
                    "latest_response": new_messages,  # Pass ALL new messages
                    "task_added": task_added,  # Track task addition
                    "turn_finished": turn_finished,  # NEW: Set turn_finished based on tool calls
                }

            except Exception as e:
                self.logger.error(
                    "Agent work failed", agent=self.agent_config.name, error=str(e)
                )
                error_message = AIMessage(
                    content=f"I encountered an error: {str(e)}. Let me try to continue anyway.",
                    name=self.agent_config.name,
                )
                return {
                    "messages": state.get("messages", []) + [error_message],
                    "latest_response": [error_message],
                    "continue_working": True,
                    "task_added": state.get("task_added", False),
                    "turn_finished": False,
                }

        return agent_work_node

    def _process_response_node(self, state: AgentSubgraphState) -> Dict[str, Any]:
        """Process ALL agent responses, maintaining order and categorizing as public/private."""
        latest_responses = state.get("latest_response", [])
        if not latest_responses:
            # No new responses, preserve existing turn_finished state
            return {"turn_finished": state.get("turn_finished", False)}

        # Ensure it's always a list (backward compatibility)
        if not isinstance(latest_responses, list):
            latest_responses = [latest_responses]

        # Process each message in order
        public_messages = state.get("public_messages", [])
        private_messages = state.get("private_messages", [])

        # FIXED: Get turn_finished from the agent_work node or check tool calls
        turn_finished = state.get("turn_finished", False)

        # If not already set, check tool calls in the latest responses
        if not turn_finished:
            for response in latest_responses:
                response_tool_calls = _safe_get_attr(response, "tool_calls", [])
                if response_tool_calls:
                    for tool_call in response_tool_calls:
                        tool_name = _safe_get_attr(tool_call, "name")
                        if tool_name == "finish_my_turn":
                            turn_finished = True
                            break
                if turn_finished:
                    break

        for response in latest_responses:
            # We only want to process AI responses for public/private chat, not tool results.
            if not isinstance(response, AIMessage):
                continue

            # Handle both dict and message object formats
            content = _safe_get_attr(response, "content", "")

            # Process message content for PUBLIC/PRIVATE prefixes
            if content.startswith("[PUBLIC]"):
                clean_content = content.replace("[PUBLIC]", "", 1).strip()
                public_message = AIMessage(
                    content=clean_content, name=self.agent_config.name
                )
                public_messages.append(public_message)
            elif content.startswith("[PRIVATE]"):
                clean_content = content.replace("[PRIVATE]", "", 1).strip()
                private_message = AIMessage(
                    content=clean_content, name=self.agent_config.name
                )
                private_messages.append(private_message)
            else:
                # Default to public if no prefix
                public_message = AIMessage(content=content, name=self.agent_config.name)
                public_messages.append(public_message)

        return {
            "public_messages": public_messages,
            "private_messages": private_messages,
            "turn_finished": turn_finished,
        }

    def _check_completion_node(self, state: AgentSubgraphState) -> Dict[str, Any]:
        """Check if the agent should continue working or finish."""
        turn_finished = state.get("turn_finished", False)

        if turn_finished:
            continue_working = False
            self.logger.info(
                "Agent turn completed, stopping work", agent=self.agent_config.name
            )
        else:
            # Agent should continue working if they haven't explicitly finished
            continue_working = True
            self.logger.debug("Agent continuing work", agent=self.agent_config.name)

        return {"continue_working": continue_working}

    def _should_continue_working(self, state: AgentSubgraphState) -> str:
        """Determine if agent should continue working or finish."""
        should_continue = state.get("continue_working", False)
        self.logger.debug(
            f"Routing decision: SHOULD_CONTINUE={should_continue}",
            agent=self.agent_config.name,
        )
        return "continue" if should_continue else "finish"
