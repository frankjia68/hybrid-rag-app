"""Example showing how to customize Hybrid RAG instructions for a domain."""

import asyncio

from agent_framework import Agent
from agent_framework.azure import AzureAISearchContextProvider
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential

from config import AzureConfig


class CustomDomainHybridRAGApplication:
    """Example of customizing Hybrid RAG for a specific domain."""

    def __init__(self, config: AzureConfig, domain: str = "Healthcare"):
        self.config = config
        self.domain = domain
        self.agent = None
        self.credential = None
        self.search_provider = None

    async def initialize(self):
        """Initialize with domain-specific instructions."""
        self.config.validate()
        self.credential = AzureCliCredential()

        self.search_provider = AzureAISearchContextProvider(
            source_id="search_provider",
            endpoint=self.config.search_endpoint,
            index_name=self.config.search_index_name,
            api_key=self.config.search_api_key,
            credential=self.credential if not self.config.search_api_key else None,
            mode="semantic",
            top_k=5,
            semantic_configuration_name=self.config.semantic_configuration_name,
            vector_field_name=None,
            embedding_function=None,
        )

        chat_client = FoundryChatClient(
            project_endpoint=self.config.foundry_project_endpoint,
            model=self.config.foundry_model,
            credential=self.credential,
        )

        instructions = (
            f"You are a specialized {self.domain} AI assistant using Azure AI "
            "Search hybrid retrieval. Ground every answer in retrieved index "
            "context, cite supporting sources, distinguish strong evidence from "
            "weak evidence, and state when the index does not contain enough "
            "information to answer safely."
        )

        self.agent = Agent(
            client=chat_client,
            name=f"{self.domain}HybridRAGAssistant",
            instructions=instructions,
            context_providers=[self.search_provider],
        )

    async def query(self, user_query: str):
        """Execute domain-specific query."""
        if not self.agent:
            raise RuntimeError("Agent not initialized")

        async for chunk in self.agent.run(user_query, stream=True):
            if chunk.text:
                yield chunk.text
            for content in chunk.contents:
                if content.annotations:
                    yield f"\n[Sources: {content.annotations}]"

    async def close(self):
        """Cleanup resources."""
        if self.search_provider and hasattr(self.search_provider, "__aexit__"):
            await self.search_provider.__aexit__(None, None, None)
        if self.credential:
            await self.credential.__aexit__(None, None, None)


async def main():
    """Example of domain-specific Hybrid RAG application."""
    print("=" * 70)
    print("Custom Domain Hybrid RAG Application")
    print("=" * 70)
    print()

    config = AzureConfig.from_env()
    app = CustomDomainHybridRAGApplication(config, domain="Healthcare")

    try:
        await app.initialize()
        print("Healthcare Hybrid RAG Assistant initialized\n")

        queries = [
            "What treatment options are described in the indexed documents?",
            "Compare the safety evidence in the most relevant indexed passages.",
        ]

        for i, query in enumerate(queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 70)

            async for chunk in app.query(query):
                print(chunk, end="", flush=True)

            print("\n")
            print("=" * 70)
            print()

    finally:
        await app.close()


if __name__ == "__main__":
    asyncio.run(main())
