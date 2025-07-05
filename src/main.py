"""Enhanced main application with agent subgraphs and sophisticated state management."""

import dotenv

dotenv.load_dotenv(".env")

import sys
from pathlib import Path
import argparse
from typing import Optional, Union, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph
from src.graph.state import GroupChatState, AgentPersonalState
from src.core.logging import configure_logging, get_logger
from src.core.exceptions import SynapseError
from src.configs.settings import settings

# Configure logging
configure_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    json_logs=settings.json_logs,
)

logger = get_logger(__name__)


def _safe_get_attr(obj: Union[dict, Any], attr: str, default=None):
    """Safely get attribute from either dict or object."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)


class ConversationManager:
    """Enhanced conversation manager with agent state tracking."""

    def __init__(self, team_config_path: Path, use_subgraphs: bool = True):
        self.team_config_path = team_config_path
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.graph = None
        self.checkpointer = MemorySaver()
        self.use_subgraphs = use_subgraphs
        self._initialize_graph()

    def _initialize_graph(self):
        """Initialize the enhanced graph with error handling."""
        try:
            logger.info(
                "Initializing enhanced graph",
                config=str(self.team_config_path),
                subgraphs=self.use_subgraphs,
            )
            self.graph = build_graph(
                self.team_config_path,
                self.checkpointer,
                use_subgraphs=self.use_subgraphs,
            )
            logger.info("Enhanced graph initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize enhanced graph", error=str(e))
            raise SynapseError(f"Failed to initialize graph: {e}")

    def process_message(self, user_input: str) -> str:
        """Process a user message with enhanced state management."""
        try:
            # Create initial state with agent state tracking
            initial_state: GroupChatState = {
                "messages": [HumanMessage(content=user_input, name="user")],
                "agent_states": {},
                "active_tasks": {},
                "completed_tasks": set(),
                "failed_tasks": set(),
                "round_number": 0,
                "error_count": 0,
                "context": {},
                "conversation_id": self.conversation_id,
                "last_orchestrator_round": 0,
                "pending_agent_responses": set(),
            }

            # Configure for conversation
            config = {
                "configurable": {"thread_id": self.conversation_id},
                "recursion_limit": settings.max_iterations,
            }

            logger.info(
                "Processing message with enhanced system",
                input_length=len(user_input),
                conversation_id=self.conversation_id,
                subgraphs=self.use_subgraphs,
            )

            # Invoke enhanced graph
            final_state = self.graph.invoke(initial_state, config)

            # --- MODIFIED RESPONSE EXTRACTION ---
            # Get all messages generated during this run
            initial_message_count = len(initial_state.get("messages", []))
            new_messages = final_state["messages"][initial_message_count:]

            # Filter for agent responses, excluding tool calls and empty messages
            agent_responses = []
            for msg in new_messages:
                # We want to display AIMessages from agents (not the orchestrator)
                # that have actual content.
                if (
                    isinstance(msg, AIMessage)
                    and _safe_get_attr(msg, "name") != "orchestrator"
                    and _safe_get_attr(msg, "content", "").strip()
                ):
                    content = _safe_get_attr(msg, "content", "")
                    # The agent's raw response might have the [PUBLIC] prefix; we clean it for display.
                    if content.startswith("[PUBLIC]"):
                        content = content.replace("[PUBLIC]", "", 1).strip()

                    # We don't show private thoughts to the user.
                    if not content.startswith("[PRIVATE]"):
                        # Prepend the agent's name for clarity in the chat.
                        agent_responses.append(
                            f"{_safe_get_attr(msg, 'name')}: {content}"
                        )

            if not agent_responses:
                response = "The team finished their work, but there are no new messages to display."
            else:
                response = "\n".join(agent_responses)
            # --- END MODIFICATION ---

            # Log enhanced statistics
            agent_states = final_state.get("agent_states", {})
            logger.info(
                "Message processed successfully with enhanced system",
                response_length=len(response),
                round_count=final_state.get("round_number", 0),
                error_count=final_state.get("error_count", 0),
                agent_count=len(agent_states),
                total_agent_messages=sum(
                    len(state.personalized_messages) for state in agent_states.values()
                ),
                total_private_messages=sum(
                    len(state.private_messages) for state in agent_states.values()
                ),
                total_tool_calls=sum(
                    len(state.tool_call_history) for state in agent_states.values()
                ),
            )

            return response

        except Exception as e:
            logger.error("Failed to process message with enhanced system", error=str(e))
            return f"I encountered an error: {str(e)}. Please try again."

    def get_conversation_state(self) -> dict:
        """Get the current conversation state for debugging."""
        try:
            config = {"configurable": {"thread_id": self.conversation_id}}
            state = self.graph.get_state(config)

            if state and state.values:
                return {
                    "message_count": len(state.values.get("messages", [])),
                    "agent_states": {
                        name: {
                            "personalized_messages": len(
                                agent_state.personalized_messages
                            ),
                            "private_messages": len(agent_state.private_messages),
                            "tool_calls": len(agent_state.tool_call_history),
                            "last_updated": agent_state.last_updated.isoformat(),
                        }
                        for name, agent_state in state.values.get(
                            "agent_states", {}
                        ).items()
                    },
                    "active_tasks": len(state.values.get("active_tasks", {})),
                    "round_number": state.values.get("round_number", 0),
                }
            return {"status": "No state available"}
        except Exception as e:
            logger.error("Failed to get conversation state", error=str(e))
            return {"error": str(e)}

    def reset_conversation(self):
        """Reset the conversation state."""
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info("Conversation reset", new_id=self.conversation_id)


def run_interactive_chat(team_config_path: Path, use_subgraphs: bool = True):
    """Run enhanced interactive chat session."""
    try:
        conversation_manager = ConversationManager(team_config_path, use_subgraphs)

        system_type = (
            "Enhanced (with Agent Subgraphs)"
            if use_subgraphs
            else "Legacy (ReAct Agents)"
        )

        print("-" * 80)
        print(f"🤖 Multi-Agent Group Chat - {system_type}")
        print("Commands:")
        print("  'quit' or 'exit' - End the session")
        print("  'state' - Show conversation state")
        print("  'reset' - Reset conversation")
        print("-" * 80)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "state":
                    state = conversation_manager.get_conversation_state()
                    print(f"\n📊 Conversation State:")
                    for key, value in state.items():
                        print(f"  {key}: {value}")
                    continue

                if user_input.lower() == "reset":
                    conversation_manager.reset_conversation()
                    print("🔄 Conversation reset!")
                    continue

                if not user_input:
                    continue

                print("\n🤔 Processing with enhanced system...")
                response = conversation_manager.process_message(user_input)
                print(f"\n🤖 Assistant: {response}")

            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error("Unexpected error in enhanced chat loop", error=str(e))
                print(f"❌ An unexpected error occurred: {e}")

    except Exception as e:
        logger.error("Failed to start enhanced chat session", error=str(e))
        print(f"❌ Failed to start chat: {e}")
        sys.exit(1)


def main():
    """Enhanced main entry point with subgraph support."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Agent Group Chat using LangGraph v0.3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to team configuration JSON file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy ReAct agents instead of enhanced subgraphs",
    )

    args = parser.parse_args()

    # Validate config file
    if not args.config.is_file():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)

    # Override log level if provided
    if args.log_level:
        import os

        os.environ["LOG_LEVEL"] = args.log_level

    use_subgraphs = not args.legacy

    logger.info(
        "Starting enhanced application",
        config_file=str(args.config),
        log_level=settings.log_level,
        system_type="subgraphs" if use_subgraphs else "legacy",
    )

    try:
        run_interactive_chat(args.config, use_subgraphs)
    except KeyboardInterrupt:
        logger.info("Enhanced application interrupted by user")
    except Exception as e:
        logger.error("Enhanced application failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
