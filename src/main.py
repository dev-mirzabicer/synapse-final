"""Enhanced main application with better error handling and monitoring."""

import dotenv

dotenv.load_dotenv(".env")

import sys
from pathlib import Path
import argparse
from typing import Optional
from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph
from src.graph.state import GroupChatState
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


class ConversationManager:
    """Manages conversation state and logging."""

    def __init__(self, team_config_path: Path):
        self.team_config_path = team_config_path
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.graph = None
        self.checkpointer = MemorySaver()
        self._initialize_graph()

    def _initialize_graph(self):
        """Initialize the graph with error handling."""
        try:
            logger.info("Initializing graph", config=str(self.team_config_path))
            self.graph = build_graph(self.team_config_path, self.checkpointer)
            logger.info("Graph initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize graph", error=str(e))
            raise SynapseError(f"Failed to initialize graph: {e}")

    def process_message(self, user_input: str) -> str:
        """Process a user message and return response."""
        try:
            # Create initial state
            initial_state: GroupChatState = {
                "messages": [HumanMessage(content=user_input, name="user")],
                "active_tasks": {},
                "completed_tasks": set(),
                "failed_tasks": set(),
                "round_number": 0,
                "error_count": 0,
                "context": {},
                "conversation_id": self.conversation_id,
            }

            # Configure for conversation
            config = {
                "configurable": {"thread_id": self.conversation_id},
                "recursion_limit": settings.max_iterations,
            }

            logger.info(
                "Processing message",
                input_length=len(user_input),
                conversation_id=self.conversation_id,
            )

            # Invoke graph
            final_state = self.graph.invoke(initial_state, config)

            # Extract response
            final_message = final_state["messages"][-1]
            response = final_message.content

            logger.info(
                "Message processed successfully",
                response_length=len(response),
                round_count=final_state.get("round_number", 0),
                error_count=final_state.get("error_count", 0),
            )

            return response

        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            return f"I encountered an error: {str(e)}. Please try again."


def run_interactive_chat(team_config_path: Path):
    """Run interactive chat session."""
    try:
        conversation_manager = ConversationManager(team_config_path)

        print("-" * 80)
        print("🤖 Multi-Agent Group Chat (LangGraph v0.3)")
        print("Type 'quit' or 'exit' to end the session")
        print("-" * 80)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                print("\n🤔 Thinking...")
                response = conversation_manager.process_message(user_input)
                print(f"\n🤖 Assistant: {response}")

            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error("Unexpected error in chat loop", error=str(e))
                print(f"❌ An unexpected error occurred: {e}")

    except Exception as e:
        logger.error("Failed to start chat session", error=str(e))
        print(f"❌ Failed to start chat: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Group Chat using LangGraph v0.3",
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

    args = parser.parse_args()

    # Validate config file
    if not args.config.is_file():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)

    # Override log level if provided
    if args.log_level:
        import os

        os.environ["LOG_LEVEL"] = args.log_level

    logger.info(
        "Starting application",
        config_file=str(args.config),
        log_level=settings.log_level,
    )

    try:
        run_interactive_chat(args.config)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Application failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
