"""Enhanced main application with cursor-based state management and subgraph support."""

import dotenv

dotenv.load_dotenv(".env")

import sys
from pathlib import Path
import argparse
from typing import Optional, Dict, Any
from datetime import datetime
import time

from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph
from src.graph.state import (
    create_initial_state,
    validate_state_consistency,
    get_state_debug_info,
    deduplicate_messages,
)
from src.graph.messages import SynapseHumanMessage
from src.core.logging import configure_logging, get_logger
from src.core.exceptions import SynapseError
from src.configs.settings import settings
from src.configs.models import TeamConfig
import json

# Configure logging
configure_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    json_logs=settings.json_logs,
)

logger = get_logger(__name__)


class ConversationManager:
    """Enhanced conversation manager with cursor-based state management."""

    def __init__(self, team_config_path: Path):
        self.team_config_path = team_config_path
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.graph = None
        self.checkpointer = MemorySaver()
        self.team_config = None
        self._enable_debug = settings.log_level == "DEBUG"
        self._initialize_graph()

    def _initialize_graph(self):
        """Initialize the enhanced graph with error handling."""
        try:
            logger.info(
                "Initializing enhanced graph", config=str(self.team_config_path)
            )

            # Load team config for agent names
            self._load_team_config()

            # Build the enhanced graph
            self.graph = build_graph(self.team_config_path, self.checkpointer)

            logger.info("Enhanced graph initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize enhanced graph", error=str(e))
            raise SynapseError(f"Failed to initialize enhanced graph: {e}")

    def _load_team_config(self):
        """Load team configuration."""
        try:
            config_data = json.loads(self.team_config_path.read_text())
            self.team_config = TeamConfig(**config_data)
            logger.info(
                "Team configuration loaded", team_name=self.team_config.team_name
            )
        except Exception as e:
            raise SynapseError(f"Failed to load team configuration: {e}")

    def process_message(self, user_input: str) -> str:
        """Process a user message and return response."""
        start_time = time.time()

        try:
            # Get agent names for state initialization
            agent_names = [agent.name for agent in self.team_config.agents]

            # Create initial state with enhanced state management
            initial_state = create_initial_state(
                user_message=user_input,
                agent_names=agent_names,
                conversation_id=self.conversation_id,
            )

            # Validate initial state
            if not validate_state_consistency(initial_state):
                logger.warning("Initial state validation failed - attempting to fix")
                # Try to fix by deduplicating
                deduplicate_messages(initial_state)

                if not validate_state_consistency(initial_state):
                    logger.error("Could not fix initial state consistency")

            # Log initial state if debug enabled
            if self._enable_debug:
                logger.debug(
                    "Initial state debug info", **get_state_debug_info(initial_state)
                )

            # Configure for conversation
            config = {
                "configurable": {"thread_id": self.conversation_id},
                "recursion_limit": settings.max_iterations,
            }

            logger.info(
                "Processing message with enhanced system",
                input_length=len(user_input),
                conversation_id=self.conversation_id,
                agent_count=len(agent_names),
                initial_message_count=len(initial_state["messages"]),
            )

            # Invoke enhanced graph
            final_state = self.graph.invoke(initial_state, config)

            # Apply deduplication as a safety measure
            pre_dedup_count = len(final_state.get("messages", []))
            deduplicate_messages(final_state)
            post_dedup_count = len(final_state.get("messages", []))

            if pre_dedup_count != post_dedup_count:
                logger.warning(
                    f"Deduplication removed {pre_dedup_count - post_dedup_count} messages"
                )

            # Validate final state
            if not validate_state_consistency(final_state):
                logger.warning("Final state validation failed")
                # Log detailed debug info
                if self._enable_debug:
                    logger.debug(
                        "Final state debug info after issues",
                        **get_state_debug_info(final_state),
                    )

            # Extract response from final state
            response = self._extract_response(final_state)

            # Calculate execution time
            execution_time = time.time() - start_time

            logger.info(
                "Message processed successfully with enhanced system",
                response_length=len(response),
                round_count=final_state.get("round_number", 0),
                error_count=final_state.get("error_count", 0),
                total_messages=len(final_state.get("messages", [])),
                unique_messages=len(
                    set(msg.message_id for msg in final_state.get("messages", []))
                ),
                execution_time_seconds=f"{execution_time:.2f}",
            )

            # Log final state debug info
            if self._enable_debug:
                logger.debug(
                    "Final state debug info", **get_state_debug_info(final_state)
                )

            return response

        except Exception as e:
            logger.error(
                "Failed to process message with enhanced system",
                error=str(e),
                execution_time_seconds=f"{time.time() - start_time:.2f}",
            )
            return f"I encountered an error: {str(e)}. Please try again."

    def _extract_response(self, final_state) -> str:
        """
        Extract the final response from the enhanced state.

        Args:
            final_state: The final state from graph execution

        Returns:
            String response for the user
        """
        messages = final_state.get("messages", [])

        if not messages:
            return "I didn't generate any response. Please try again."

        # Strategy: Find the last substantive public message
        # Priority order:
        # 1. Last public agent message (not private)
        # 2. Last orchestrator message with content
        # 3. Any last message with content

        # Iterate backwards through messages
        for message in reversed(messages):
            # Skip system messages intended for specific agents
            if hasattr(message, "target_agent") and message.target_agent:
                continue

            # Skip private agent messages
            if hasattr(message, "is_private") and message.is_private:
                continue

            # Skip tool messages (they're internal)
            if hasattr(message, "tool_call_id"):
                continue

            # Check if message has meaningful content
            if hasattr(message, "content") and message.content:
                content = message.content.strip()

                # Skip empty or system-only messages
                if not content or content.startswith("[NOT_A_USER"):
                    continue

                # Found a good message
                return content

        # Fallback: try to find any orchestrator message
        for message in reversed(messages):
            if hasattr(message, "content") and isinstance(message, type(messages[0])):
                if "orchestrator" in str(type(message).__name__).lower():
                    if message.content:
                        return message.content

        # Last resort: return the last message content if it exists
        last_message = messages[-1]
        if hasattr(last_message, "content") and last_message.content:
            return last_message.content

        return "I completed the task but didn't generate a visible response."

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation."""
        stats = {
            "conversation_id": self.conversation_id,
            "team_name": self.team_config.team_name if self.team_config else "Unknown",
            "agent_count": len(self.team_config.agents) if self.team_config else 0,
            "agent_names": [agent.name for agent in self.team_config.agents]
            if self.team_config
            else [],
            "graph_initialized": self.graph is not None,
            "checkpointer_type": type(self.checkpointer).__name__,
            "debug_enabled": self._enable_debug,
        }

        return stats


def run_interactive_chat(team_config_path: Path):
    """Run enhanced interactive chat session."""
    try:
        conversation_manager = ConversationManager(team_config_path)

        # Display startup information
        stats = conversation_manager.get_conversation_stats()

        print("-" * 80)
        print("🤖 Enhanced Multi-Agent Group Chat (Synapse)")
        print(f"Team: {stats['team_name']}")
        print(f"Agents: {stats['agent_count']} ({', '.join(stats['agent_names'])})")
        print(f"Conversation ID: {stats['conversation_id']}")
        print(f"Debug Mode: {'Enabled' if stats['debug_enabled'] else 'Disabled'}")
        print("Features: Cursor-based state, Subgraphs, Message deduplication")
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

                # Special commands
                if user_input.lower() == "/stats":
                    print("\n📊 Conversation Statistics:")
                    for (
                        key,
                        value,
                    ) in conversation_manager.get_conversation_stats().items():
                        print(f"  {key}: {value}")
                    continue

                print("\n🤔 Processing with enhanced system...")
                response = conversation_manager.process_message(user_input)
                print(f"\n🤖 Response: {response}")

            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error("Unexpected error in enhanced chat loop", error=str(e))
                print(f"❌ An unexpected error occurred: {e}")

    except Exception as e:
        logger.error("Failed to start enhanced chat session", error=str(e))
        print(f"❌ Failed to start enhanced chat: {e}")
        sys.exit(1)


def validate_config_file(config_path: Path) -> bool:
    """
    Validate the team configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        True if valid, False otherwise
    """
    try:
        if not config_path.is_file():
            print(f"❌ Configuration file not found: {config_path}")
            return False

        # Try to load and validate the configuration
        config_data = json.loads(config_path.read_text())
        team_config = TeamConfig(**config_data)

        print(f"✅ Configuration valid: {team_config.team_name}")
        print(f"   Agents: {', '.join(agent.name for agent in team_config.agents)}")

        # Additional validation
        agent_names = [agent.name for agent in team_config.agents]
        if len(agent_names) != len(set(agent_names)):
            print("❌ Duplicate agent names detected!")
            return False

        # Check for reserved names
        reserved_names = {"orchestrator", "aggregator", "user", "system"}
        conflicts = set(agent_names) & reserved_names
        if conflicts:
            print(f"❌ Agents using reserved names: {conflicts}")
            return False

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Enhanced main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Agent Group Chat using LangGraph with Subgraphs",
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
        "--validate-only",
        action="store_true",
        help="Only validate configuration file and exit",
    )
    parser.add_argument(
        "--test-message",
        type=str,
        help="Run with a single test message and exit",
    )

    args = parser.parse_args()

    # Override log level if provided
    if args.log_level:
        import os

        os.environ["LOG_LEVEL"] = args.log_level

        # Reconfigure logging with new level
        configure_logging(
            log_level=args.log_level,
            log_file=settings.log_file,
            json_logs=settings.json_logs,
        )

    # Validate configuration file
    if not validate_config_file(args.config):
        sys.exit(1)

    if args.validate_only:
        print("✅ Configuration validation complete")
        return

    logger.info(
        "Starting enhanced application",
        config_file=str(args.config),
        log_level=settings.log_level,
    )

    try:
        # Test mode
        if args.test_message:
            print(f"🧪 Running test with message: {args.test_message}")
            conversation_manager = ConversationManager(args.config)
            response = conversation_manager.process_message(args.test_message)
            print(f"🤖 Response: {response}")
            return

        # Interactive mode
        run_interactive_chat(args.config)

    except KeyboardInterrupt:
        logger.info("Enhanced application interrupted by user")
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error("Enhanced application failed", error=str(e))
        print(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
