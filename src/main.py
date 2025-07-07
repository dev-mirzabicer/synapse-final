"""Enhanced main application with staged message processing, cursor-based state management, and robust concurrent execution."""

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
    get_task_completion_summary,
    apply_state_validation,
    MessageMerger,
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
    """Enhanced conversation manager with staged message processing and robust concurrent execution."""

    def __init__(self, team_config_path: Path):
        self.team_config_path = team_config_path
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.graph = None
        self.checkpointer = MemorySaver()
        self.team_config = None
        self._enable_debug = settings.log_level == "DEBUG"
        self._enable_validation = True  # Enhanced validation
        self._initialize_graph()

    def _initialize_graph(self):
        """Initialize the enhanced graph with staged message processing."""
        try:
            logger.info(
                "Initializing enhanced graph with staged message processing support",
                config=str(self.team_config_path),
            )

            # Load team config for agent names
            self._load_team_config()

            # Build the enhanced graph with staged processing
            self.graph = build_graph(self.team_config_path, self.checkpointer)

            logger.info(
                "Enhanced graph initialized successfully",
                features=[
                    "staged_message_processing",
                    "pending_message_queue",
                    "batch_message_merging",
                    "cursor_based_state",
                    "subgraph_architecture",
                    "batch_completion_processing",
                    "explicit_task_tracking",
                    "comprehensive_state_validation",
                    "enhanced_error_handling",
                ],
            )

        except Exception as e:
            logger.error("Failed to initialize enhanced graph", error=str(e))
            raise SynapseError(f"Failed to initialize enhanced graph: {e}")

    def _load_team_config(self):
        """Load team configuration."""
        try:
            config_data = json.loads(self.team_config_path.read_text())
            self.team_config = TeamConfig(**config_data)
            logger.info(
                "Team configuration loaded",
                team_name=self.team_config.team_name,
                agent_count=len(self.team_config.agents),
                agents=[agent.name for agent in self.team_config.agents],
            )
        except Exception as e:
            raise SynapseError(f"Failed to load team configuration: {e}")

    def process_message(self, user_input: str) -> str:
        """Process a user message and return response with enhanced staged processing monitoring."""
        start_time = time.time()

        # Enhanced execution metrics with staged processing awareness
        execution_metrics = {
            "start_time": start_time,
            "validation_checkpoints": 0,
            "state_consistency_checks": 0,
            "concurrent_execution_detected": False,
            "task_completion_events": [],
            "pending_message_events": [],
            "message_merge_events": [],
        }

        try:
            # Get agent names for state initialization
            agent_names = [agent.name for agent in self.team_config.agents]

            # Create initial state with staged message processing support
            initial_state = create_initial_state(
                user_message=user_input,
                agent_names=agent_names,
                conversation_id=self.conversation_id,
            )

            # Enhanced initial state validation
            if self._enable_validation:
                validation_passed = apply_state_validation(
                    initial_state, "initial state creation"
                )
                execution_metrics["validation_checkpoints"] += 1

                if not validation_passed:
                    logger.warning(
                        "Initial state validation failed - attempting to fix"
                    )
                    deduplicate_messages(initial_state)

                    # Re-validate after fix attempt
                    if not apply_state_validation(
                        initial_state, "after deduplication fix"
                    ):
                        logger.error("Could not fix initial state consistency")

            # Log initial state if debug enabled
            if self._enable_debug:
                debug_info = get_state_debug_info(initial_state)
                logger.debug("Initial state debug info", **debug_info)

            # Configure for conversation
            config = {
                "configurable": {"thread_id": self.conversation_id},
                "recursion_limit": settings.max_iterations,
            }

            logger.info(
                "Processing message with ENHANCED staged message processing system",
                input_length=len(user_input),
                conversation_id=self.conversation_id,
                agent_count=len(agent_names),
                initial_message_count=len(initial_state["messages"]),
                initial_pending_count=len(initial_state.get("pending_messages", [])),
                enhanced_features_enabled=[
                    "staged_message_processing",
                    "pending_message_queue",
                    "batch_message_merging",
                    "batch_aggregation",
                    "explicit_completion_tracking",
                    "taskinfo_based_routing",
                    "state_validation",
                ],
            )

            # Invoke enhanced graph with staged processing monitoring
            final_state = self._invoke_graph_with_monitoring(
                initial_state, config, execution_metrics
            )

            # Enhanced post-execution validation and cleanup
            final_state = self._post_execution_validation(
                final_state, execution_metrics
            )

            # Extract response from final state
            response = self._extract_response(final_state)

            # Calculate execution time and log comprehensive metrics
            execution_time = time.time() - start_time
            execution_metrics["total_execution_time"] = execution_time

            self._log_execution_metrics(final_state, execution_metrics, response)

            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Failed to process message with enhanced staged processing system",
                error=str(e),
                execution_time_seconds=f"{execution_time:.2f}",
                execution_metrics=execution_metrics,
            )
            return f"I encountered an error: {str(e)}. Please try again."

    def _invoke_graph_with_monitoring(
        self, initial_state, config, execution_metrics
    ) -> Dict[str, Any]:
        """Invoke graph with enhanced monitoring for staged message processing and concurrent execution."""

        # Detect if concurrent execution is expected
        initial_completion = get_task_completion_summary(initial_state)
        if initial_completion["total_tasks"] > 1:
            execution_metrics["concurrent_execution_detected"] = True
            logger.info(
                "CONCURRENT EXECUTION DETECTED - enhanced monitoring enabled",
                task_count=initial_completion["total_tasks"],
                expected_agents=initial_completion.get("pending_agents", []),
            )

        # Monitor initial pending state
        initial_pending = MessageMerger.get_pending_summary(initial_state)
        execution_metrics["pending_message_events"].append(
            {"phase": "initial", "summary": initial_pending}
        )

        # Invoke enhanced graph with staged processing
        final_state = self.graph.invoke(initial_state, config)

        # Post-invocation analysis
        final_completion = get_task_completion_summary(final_state)
        execution_metrics["task_completion_events"] = final_completion

        final_pending = MessageMerger.get_pending_summary(final_state)
        execution_metrics["pending_message_events"].append(
            {"phase": "final", "summary": final_pending}
        )

        # Check if any pending messages remain (should be empty in final state)
        if final_pending["pending_count"] > 0:
            logger.warning(
                "Final state contains unmerged pending messages",
                pending_count=final_pending["pending_count"],
                pending_sources=final_pending["pending_sources"],
            )

        logger.info(
            "Graph invocation completed with staged processing",
            concurrent_execution_occurred=execution_metrics[
                "concurrent_execution_detected"
            ],
            final_completion_status=final_completion["completion_status"],
            final_pending_count=final_pending["pending_count"],
            **final_completion,
        )

        return final_state

    def _post_execution_validation(
        self, final_state, execution_metrics
    ) -> Dict[str, Any]:
        """Enhanced post-execution validation and cleanup with staged processing awareness."""

        # Apply deduplication as a safety measure for both main and pending messages
        pre_dedup_main = len(final_state.get("messages", []))
        pre_dedup_pending = len(final_state.get("pending_messages", []))

        deduplicate_messages(final_state)

        post_dedup_main = len(final_state.get("messages", []))
        post_dedup_pending = len(final_state.get("pending_messages", []))

        if pre_dedup_main != post_dedup_main:
            logger.warning(
                f"Post-execution deduplication removed {pre_dedup_main - post_dedup_main} main messages"
            )

        if pre_dedup_pending != post_dedup_pending:
            logger.warning(
                f"Post-execution deduplication removed {pre_dedup_pending - post_dedup_pending} pending messages"
            )

        # Enhanced final state validation
        if self._enable_validation:
            validation_passed = apply_state_validation(final_state, "final state")
            execution_metrics["validation_checkpoints"] += 1
            execution_metrics["final_validation_passed"] = validation_passed

            if not validation_passed:
                logger.warning("Final state validation failed")
                # Log detailed debug info for troubleshooting
                if self._enable_debug:
                    logger.debug(
                        "Final state debug info after validation failure",
                        **get_state_debug_info(final_state),
                    )
            else:
                logger.debug("Final state validation passed successfully")

        # Validate staged message processing completed properly
        self._validate_staged_processing_completion(final_state, execution_metrics)

        # Validate task completion consistency for concurrent execution
        if execution_metrics.get("concurrent_execution_detected", False):
            self._validate_concurrent_completion(final_state, execution_metrics)

        return final_state

    def _validate_staged_processing_completion(self, final_state, execution_metrics):
        """Validate that staged message processing completed properly."""

        pending_summary = MessageMerger.get_pending_summary(final_state)

        logger.info("STAGED PROCESSING VALIDATION", **pending_summary)

        # Check for unmerged pending messages
        if pending_summary["pending_count"] > 0:
            logger.warning(
                "STAGED PROCESSING ISSUE: Final state has unmerged pending messages",
                pending_count=pending_summary["pending_count"],
                pending_sources=pending_summary["pending_sources"],
            )
            execution_metrics["staged_processing_issue"] = "unmerged_pending_messages"
        else:
            logger.info("STAGED PROCESSING: All messages properly merged")
            execution_metrics["staged_processing_status"] = "completed_successfully"

    def _validate_concurrent_completion(self, final_state, execution_metrics):
        """Validate that concurrent task execution completed properly."""

        completion_summary = get_task_completion_summary(final_state)

        logger.info("CONCURRENT EXECUTION VALIDATION", **completion_summary)

        # Check for common concurrent execution issues
        if (
            completion_summary["has_tasks"]
            and completion_summary["completion_status"] == "none_finished"
        ):
            logger.error(
                "CONCURRENT EXECUTION ISSUE: Tasks were assigned but none finished",
                active_tasks=completion_summary.get("total_tasks", 0),
            )
        elif completion_summary["completion_status"] == "partially_finished":
            logger.warning(
                "CONCURRENT EXECUTION: Some tasks may not have completed",
                completion_percentage=completion_summary.get(
                    "completion_percentage", 0
                ),
                completed_agents=completion_summary.get("completed_agents", []),
                still_working=completion_summary.get("in_progress_agents", [])
                + completion_summary.get("pending_agents", []),
            )
        else:
            logger.info("CONCURRENT EXECUTION: All tasks completed successfully")

        execution_metrics["concurrent_completion_status"] = completion_summary[
            "completion_status"
        ]

    def _log_execution_metrics(self, final_state, execution_metrics, response):
        """Log comprehensive execution metrics with staged processing details."""

        logger.info(
            "ENHANCED message processing completed with staged processing",
            response_length=len(response),
            round_count=final_state.get("round_number", 0),
            error_count=final_state.get("error_count", 0),
            total_messages=len(final_state.get("messages", [])),
            total_pending_messages=len(final_state.get("pending_messages", [])),
            unique_messages=len(
                set(msg.message_id for msg in final_state.get("messages", []))
            ),
            execution_time_seconds=f"{execution_metrics['total_execution_time']:.2f}",
            validation_checkpoints=execution_metrics.get("validation_checkpoints", 0),
            concurrent_execution_detected=execution_metrics.get(
                "concurrent_execution_detected", False
            ),
            final_validation_passed=execution_metrics.get(
                "final_validation_passed", "not_run"
            ),
            concurrent_completion_status=execution_metrics.get(
                "concurrent_completion_status", "n/a"
            ),
            staged_processing_status=execution_metrics.get(
                "staged_processing_status", "n/a"
            ),
            pending_message_events_count=len(
                execution_metrics.get("pending_message_events", [])
            ),
        )

        # Log final state debug info if debug enabled
        if self._enable_debug:
            logger.debug(
                "Final execution state debug info with staged processing",
                **get_state_debug_info(final_state),
            )

    def _extract_response(self, final_state) -> str:
        """
        Extract the final response from the enhanced state with staged processing support.

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
                logger.debug(
                    "Extracted response from message",
                    message_type=type(message).__name__,
                    content_preview=content[:100],
                    agent_name=getattr(message, "agent_name", "unknown"),
                )
                return content

        # Fallback: try to find any orchestrator message
        for message in reversed(messages):
            if hasattr(message, "content") and isinstance(message, type(messages[0])):
                if "orchestrator" in str(type(message).__name__).lower():
                    if message.content:
                        logger.debug("Extracted response from orchestrator fallback")
                        return message.content

        # Last resort: return the last message content if it exists
        last_message = messages[-1]
        if hasattr(last_message, "content") and last_message.content:
            logger.debug("Extracted response from last message fallback")
            return last_message.content

        return "I completed the task but didn't generate a visible response."

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation with staged processing details."""
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
            "validation_enabled": self._enable_validation,
            "enhanced_features": [
                "staged_message_processing",
                "pending_message_queue",
                "batch_message_merging",
                "cursor_based_state_management",
                "batch_completion_processing",
                "explicit_task_tracking",
                "concurrent_execution_support",
                "comprehensive_state_validation",
                "enhanced_error_handling",
            ],
            "message_processing_architecture": "staged_with_pending_queue",
            "state_management_architecture": "cursor_based_with_validation",
        }

        return stats


def run_interactive_chat(team_config_path: Path):
    """Run enhanced interactive chat session with staged message processing and concurrent execution support."""
    try:
        conversation_manager = ConversationManager(team_config_path)

        # Display startup information
        stats = conversation_manager.get_conversation_stats()

        print("-" * 90)
        print(
            "🤖 ENHANCED Multi-Agent Group Chat (Synapse) - Staged Processing Edition"
        )
        print(f"Team: {stats['team_name']}")
        print(f"Agents: {stats['agent_count']} ({', '.join(stats['agent_names'])})")
        print(f"Conversation ID: {stats['conversation_id']}")
        print(f"Debug Mode: {'Enabled' if stats['debug_enabled'] else 'Disabled'}")
        print(f"Validation: {'Enabled' if stats['validation_enabled'] else 'Disabled'}")
        print(f"Message Processing: {stats['message_processing_architecture']}")
        print(f"State Management: {stats['state_management_architecture']}")
        print("🚀 ENHANCED FEATURES:")
        for feature in stats["enhanced_features"]:
            print(f"  ✓ {feature.replace('_', ' ').title()}")
        print("💡 Commands: 'quit'/'exit' to end, '/stats' for statistics")
        print("-" * 90)

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
                    print("\n📊 Enhanced Conversation Statistics:")
                    for (
                        key,
                        value,
                    ) in conversation_manager.get_conversation_stats().items():
                        if isinstance(value, list) and len(value) > 3:
                            print(
                                f"  {key}: [{', '.join(map(str, value[:3]))}, ... ({len(value)} total)]"
                            )
                        else:
                            print(f"  {key}: {value}")
                    continue

                print(
                    "\n🔄 Processing with ENHANCED staged message processing system..."
                )
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
    Validate the team configuration file with enhanced checks for staged processing.

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

        # Enhanced validation for staged processing and concurrent execution
        if len(agent_names) > 1:
            print(
                "✅ Multi-agent configuration detected - concurrent execution with staged processing enabled"
            )

        print(
            "✅ Enhanced validation passed - ready for robust staged message processing"
        )
        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Enhanced main entry point with staged message processing and concurrent execution support."""
    parser = argparse.ArgumentParser(
        description="ENHANCED Multi-Agent Group Chat using LangGraph with Staged Message Processing",
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
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        default=True,
        help="Enable enhanced state validation (default: enabled)",
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
        print("✅ Enhanced configuration validation complete")
        return

    logger.info(
        "Starting ENHANCED application with staged message processing",
        config_file=str(args.config),
        log_level=settings.log_level,
        validation_enabled=args.enable_validation,
        enhanced_features=[
            "staged_message_processing",
            "pending_message_queue",
            "batch_message_merging",
            "batch_completion_processing",
            "explicit_task_tracking",
            "taskinfo_based_routing",
            "comprehensive_state_validation",
            "concurrent_execution_monitoring",
        ],
    )

    try:
        # Test mode
        if args.test_message:
            print(
                f"🧪 Running enhanced test with staged processing: {args.test_message}"
            )
            conversation_manager = ConversationManager(args.config)
            response = conversation_manager.process_message(args.test_message)
            print(f"🤖 Enhanced Response: {response}")
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
