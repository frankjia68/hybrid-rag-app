"""Advanced hybrid search example for index-focused questions."""

import asyncio

from config import AzureConfig
from hybrid_rag import HybridRAGApplication


async def main():
    """Demonstrate queries suited to fast hybrid index retrieval."""
    print("=" * 70)
    print("Advanced Hybrid Search Examples")
    print("=" * 70)
    print()

    config = AzureConfig.from_env()
    app = HybridRAGApplication(config)

    try:
        await app.initialize()
        print("Application initialized\n")

        queries = [
            "Find the most relevant documents about the main recommendations.",
            "Compare the top indexed passages for this topic.",
            "Which source best supports the answer and why?",
            "What terms or entities appear most often in the relevant results?",
        ]

        for i, query in enumerate(queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 70)
            print("Hybrid RAG Response:")

            async for chunk in app.query(query, stream=True):
                print(chunk, end="", flush=True)

            print("\n")
            print("=" * 70)
            print()

    finally:
        await app.close()


if __name__ == "__main__":
    asyncio.run(main())
