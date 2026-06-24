"""Interactive chat interface for the Hybrid RAG application."""

import asyncio
import logging

from config import AzureConfig
from hybrid_rag import HybridRAGApplication

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def interactive_chat() -> None:
    """Run an interactive chat session with the Hybrid RAG assistant."""
    config = AzureConfig.from_env()
    app = HybridRAGApplication(config)

    try:
        await app.initialize()

        print("=" * 70)
        print("Hybrid RAG Assistant - Interactive Chat")
        print("=" * 70)
        print("\nWelcome to the Hybrid RAG Assistant!")
        print("This assistant uses Azure AI Search index retrieval")
        print("with semantic ranking and optional vector hybrid search.\n")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 70)
        print()

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nAssistant: Thank you for using Hybrid RAG. Goodbye!")
                    break

                if not user_input:
                    print("Please enter a question or command.")
                    continue

                print("\nAssistant: ", end="", flush=True)
                async for chunk in app.query(user_input, stream=True):
                    print(chunk, end="", flush=True)
                print()

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Type 'exit' to quit gracefully.")
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(
                    f"\nError: {e}\n"
                    "Please check your configuration and try again."
                )

    finally:
        await app.close()
        print("\nSession ended. Goodbye!")


if __name__ == "__main__":
    asyncio.run(interactive_chat())
