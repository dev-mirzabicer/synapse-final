import logging
from pathlib import Path
import argparse
from dotenv import load_dotenv
import os

# 1. Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    raise ValueError(
        "API keys for Google (GOOGLE_API_KEY) and Tavily (TAVILY_API_KEY) "
        "must be set in a .env file in the project root."
    )

from datetime import datetime
from typing import List

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage

from src.graph.builder import build_graph
from src.graph.state import GroupChatState

# --- Basic Configuration ---
# Configure top-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_message_for_log(msg: BaseMessage) -> str:
    """Formats a single message object into a detailed, readable string for logging."""
    log_entry = []

    if isinstance(msg, HumanMessage):
        sender = f"User ({msg.name or 'user'})"
        log_entry.append(f"### Sender: {sender}")
        log_entry.append(f"**Content:**\n{msg.content}")

    elif isinstance(msg, AIMessage):
        sender = f"Agent ({msg.name or 'AI'})"
        log_entry.append(f"### Sender: {sender}")
        if msg.content:
            log_entry.append(f"**Content:**\n{msg.content}")
        if msg.tool_calls:
            log_entry.append("\n**Tool Calls:**")
            for i, tool_call in enumerate(msg.tool_calls):
                log_entry.append(f"  - **Call {i+1}:**")
                log_entry.append(f"    - **Name:** `{tool_call['name']}`")
                log_entry.append(f"    - **Arguments:** `{tool_call['args']}`")
                log_entry.append(f"    - **ID:** `{tool_call['id']}`")

    elif isinstance(msg, ToolMessage):
        sender = f"Tool ({msg.name or 'tool'})"
        log_entry.append(f"### Sender: {sender}")
        log_entry.append(f"**Content:**\n{msg.content}")
        log_entry.append(f"**Tool Call ID:** `{msg.tool_call_id}`")

    else:
        log_entry.append(f"### Unknown Message Type: {type(msg).__name__}")
        log_entry.append(str(msg))

    return "\n".join(log_entry)


def log_conversation_turn(
    file_handle, turn_number: int, messages_in_turn: List[BaseMessage]
):
    """Writes a detailed log of a single conversation turn to the given file."""
    file_handle.write(f"\n\n---\n\n## Turn {turn_number}\n\n")
    for msg in messages_in_turn:
        file_handle.write(format_message_for_log(msg) + "\n\n")
    file_handle.flush()  # Ensure data is written to disk immediately


def main(team_config_path: Path):
    """
    The main function to run the multi-agent group chat application.

    Args:
        team_config_path: The path to the team's JSON configuration file.
    """

    # 2. Set up conversation log file
    log_dir = Path("conversations")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"conversation_{timestamp}.md"
    logger.info(f"Conversation will be logged to: {log_file_path}")

    # 3. Build the graph
    logger.info("Building the graph...")
    graph = build_graph(team_config_path)
    logger.info("Graph built successfully.")

    # 4. Start the interactive chat loop within a file-writing context
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# Conversation Log\n\n**Team:** `{team_config_path.stem}`\n")
        log_file.write(f"**Timestamp:** `{timestamp}`\n")

        current_state: GroupChatState = {
            "messages": [],
            "active_tasks": {},
            "completed_tasks": set(),
        }
        turn_count = 0

        logger.info("Starting interactive chat session. Type 'quit' or 'exit' to end.")
        print("-" * 80)
        print("Multi-Agent Group Chat is running. Start by typing your message.")
        print("-" * 80)

        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("Exiting chat session.")
                    break

                turn_count += 1
                user_message = HumanMessage(content=user_input, name="user")
                current_state["messages"] = [user_message]

                print("\n...Assistant is thinking...\n")
                final_state = graph.invoke(current_state)

                # Log the entire turn's messages to the file
                log_conversation_turn(log_file, turn_count, final_state["messages"])

                current_state = final_state

                final_response = final_state["messages"][-1]
                if final_response.name == "orchestrator":
                    print(f"[Orchestrator]: {final_response.content}")
                else:
                    print(
                        f"[{final_response.name or 'System'}]: {final_response.content}"
                    )

                print("-" * 80)

            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat session.")
                break
            except Exception:
                logger.exception("An unexpected error occurred during the chat loop.")
                log_file.write("\n\n**An unexpected error occurred. Ending session.**")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a multi-agent group chat application using LangGraph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the team configuration JSON file.",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at the specified path: {args.config}"
        )

    main(team_config_path=args.config)
