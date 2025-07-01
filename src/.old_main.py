import logging
from pathlib import Path
import uuid
from dotenv import load_dotenv
from IPython.display import Image

load_dotenv(".env")

from langchain_core.messages import HumanMessage

from src.graph.builder import build_graph
from src.graph.state import GroupChatState

# --- Configuration ---
# Load environment variables from .env file

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """
    The main entry point for running the multi-agent group chat application.
    """
    logger.info("Starting the multi-agent group chat application.")

    # Define the path to the team configuration
    config_path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "teams"
        / "research_team.json"
    )

    # Build the graph
    try:
        graph = build_graph(config_path)
    except Exception as e:
        logger.error(f"Failed to build the graph: {e}", exc_info=True)
        return

    # Save the graph structure image
    graph_image_path = Path(__file__).resolve().parent / "graph_structure.png"
    img = Image(graph.get_graph().draw_mermaid_png())
    with open(graph_image_path, "wb") as f:
        f.write(img.data)

    # --- Conversation Loop ---
    # A unique ID for the conversation thread.
    # While not used until we add a checkpointer, this is best practice.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # We will manually manage the history for this simple console application.
    full_history = []

    logger.info("Graph is ready. Starting conversation loop.")
    print("\n--- Multi-Agent Group Chat Initialized ---")
    print("Enter 'exit' or 'quit' to end the conversation.")

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break

            if not user_input.strip():
                continue

            # Construct the input for the graph
            # The state must match the GroupChatState TypedDict
            input_state = GroupChatState(
                messages=[HumanMessage(content=user_input)],
                active_tasks={},
                completed_tasks=set(),
            )

            # We use `invoke` here for simplicity in Phase 1. It runs the graph
            # until it reaches an END state and returns the final state.
            # This is ideal for verifying the core logic of state transitions.
            # We will upgrade to `stream` in a later phase for a real-time UX.
            final_state = graph.invoke(input_state, config=config)

            # The final message from the orchestrator is the user-facing response
            final_message = final_state["messages"][-1]

            print(f"\nOrchestrator: {final_message.content}")

            # Update the full history with the new messages
            full_history.extend(final_state["messages"])
            logger.info(f"Conversation updated. Total messages: {len(full_history)}")

        except KeyboardInterrupt:
            print("\nExiting chat due to interrupt.")
            with open("conversation_history.json", "w") as f:
                # Save the conversation history to a file
                import json

                json.dump(full_history, f, indent=2)
            break
        except Exception as e:
            logger.error(
                f"An error occurred during the conversation loop: {e}", exc_info=True
            )
            print("\nAn error occurred. Please check the logs. Restarting loop.")


if __name__ == "__main__":
    main()
