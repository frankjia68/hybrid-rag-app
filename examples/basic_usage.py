"""Basic usage example of the Hybrid RAG application."""

import asyncio

from config import AzureConfig
from hybrid_rag import HybridRAGApplication


async def main():
    """Demonstrate basic usage of the Hybrid RAG application."""
    print("=" * 70)
    print("Basic Hybrid RAG Usage Example")
    print("=" * 70)
    print()

    config = AzureConfig.from_env()
    app = HybridRAGApplication(config)

    try:
        print("Initializing application...")
        await app.initialize()
        print("Application initialized\n")

        query = "What are the main topics in the search index?"
        print(f"Query: {query}")
        print("-" * 70)
        print("Response:")

        full_response = ""
        async for chunk in app.query(query, stream=True):
            print(chunk, end="", flush=True)
            full_response += chunk

        print("\n")
        print("=" * 70)
        print(f"Total response length: {len(full_response)} characters")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        await app.close()
        print("\nApplication closed.")


if __name__ == "__main__":
    asyncio.run(main())
