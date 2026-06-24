"""Hybrid RAG application using Microsoft Agent Framework with Azure AI Search."""

import asyncio
import logging
from typing import AsyncGenerator, Awaitable, Callable

import aiohttp
from agent_framework import Agent
from agent_framework.azure import AzureAISearchContextProvider
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

from config import AzureConfig

# Load environment variables.
load_dotenv()

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EmbeddingFunction = Callable[[str], Awaitable[list[float]]]


class HybridRAGApplication:
    """Hybrid RAG application with Azure AI Search index retrieval."""

    def __init__(self, config: AzureConfig):
        """Initialize the Hybrid RAG application."""
        self.config = config
        self.agent: Agent | None = None
        self.search_provider: AzureAISearchContextProvider | None = None
        self.credential: AzureCliCredential | None = None

    async def initialize(self) -> None:
        """Initialize the agent and search provider."""
        logger.info("Initializing Hybrid RAG application...")

        self.config.validate()
        self.credential = AzureCliCredential()

        logger.info("Setting up Azure AI Search context provider (semantic mode)...")
        self.search_provider = await self._create_search_provider()

        logger.info("Creating Hybrid RAG agent...")
        self.agent = self._create_agent()

        logger.info("Hybrid RAG application initialized successfully")

    async def _create_search_provider(self) -> AzureAISearchContextProvider:
        """Create Azure AI Search context provider in semantic/hybrid mode."""
        embedding_function = self._create_embedding_function()

        return AzureAISearchContextProvider(
            source_id="hybrid_search_provider",
            endpoint=self.config.search_endpoint,
            index_name=self.config.search_index_name,
            api_key=self.config.search_api_key,
            credential=self.credential if not self.config.search_api_key else None,
            mode="semantic",
            top_k=5,
            semantic_configuration_name=self.config.semantic_configuration_name,
            vector_field_name=self.config.vector_field_name,
            embedding_function=embedding_function,
        )

    def _create_embedding_function(self) -> EmbeddingFunction | None:
        """Create an Azure OpenAI embedding function for vector hybrid search."""
        if not self.config.vector_field_name:
            return None

        endpoint = (
            self.config.azure_openai_endpoint
            or self.config.azure_openai_resource_url
        )
        deployment = self.config.azure_openai_embedding_deployment
        if not endpoint or not deployment:
            return None

        endpoint = endpoint.rstrip("/")
        api_version = "2024-02-01"
        url = (
            f"{endpoint}/openai/deployments/{deployment}/embeddings"
            f"?api-version={api_version}"
        )

        async def embed(text: str) -> list[float]:
            headers = {"Content-Type": "application/json"}
            if self.config.azure_openai_api_key:
                headers["api-key"] = self.config.azure_openai_api_key
            elif self.credential:
                token = await self.credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )
                headers["Authorization"] = f"Bearer {token.token}"

            payload = {
                "input": text,
                "model": self.config.azure_openai_embedding_model or deployment,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            return data["data"][0]["embedding"]

        return embed

    def _create_agent(self) -> Agent:
        """Create the Hybrid RAG agent."""
        chat_client = FoundryChatClient(
            project_endpoint=self.config.foundry_project_endpoint,
            model=self.config.foundry_model,
            credential=self.credential,
        )

        return Agent(
            client=chat_client,
            name="HybridRAGAssistant",
            instructions=(
                "You are an AI assistant using Azure AI Search hybrid retrieval. "
                "Answer questions using the retrieved index context, combine "
                "keyword, semantic, and vector-retrieved evidence when available, "
                "and cite sources from the search index. If the retrieved context "
                "does not support an answer, say what is missing."
            ),
            context_providers=[self.search_provider],
        )

    async def query(
        self,
        user_query: str,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """
        Execute a query against the Hybrid RAG system.

        Args:
            user_query: The user's question or prompt.
            stream: Whether to stream the response.

        Yields:
            Response chunks from the agent.
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        logger.info(f"Processing query: {user_query}")

        try:
            if stream:
                async for chunk in self.agent.run(user_query, stream=True):
                    if chunk.text:
                        yield chunk.text
                    for content in chunk.contents:
                        if content.annotations:
                            yield f"\n[Sources: {content.annotations}]"
            else:
                response = await self.agent.run(user_query, stream=False)
                if response.text:
                    yield response.text
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    async def close(self) -> None:
        """Clean up resources."""
        if self.search_provider and hasattr(self.search_provider, "__aexit__"):
            await self.search_provider.__aexit__(None, None, None)
        if self.credential:
            await self.credential.__aexit__(None, None, None)
        logger.info("Hybrid RAG application closed")


async def main() -> None:
    """Main function demonstrating the Hybrid RAG application."""
    config = AzureConfig.from_env()
    app = HybridRAGApplication(config)
    await app.initialize()

    sample_queries = [
        "What are the key topics covered in the search index?",
        "What are the most relevant documents for this topic?",
        "Summarize the most important findings from the indexed content.",
        "What recommendations appear in the indexed documents?",
        "Which source documents support the answer?",
    ]

    try:
        print("=" * 70)
        print("Hybrid RAG Application - Azure AI Search Index")
        print("=" * 70)
        print()

        for query in sample_queries:
            print(f"User Query: {query}")
            print("-" * 70)
            print("Agent Response: ", end="", flush=True)

            async for chunk in app.query(query, stream=True):
                print(chunk, end="", flush=True)

            print("\n")
            print("=" * 70)
            print()

    finally:
        await app.close()


if __name__ == "__main__":
    asyncio.run(main())
