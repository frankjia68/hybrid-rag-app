import unittest
from unittest.mock import Mock, patch

from config import AzureConfig
from hybrid_rag import HybridRAGApplication


class HybridRAGApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_search_provider_uses_semantic_mode_and_index(self):
        config = AzureConfig(
            foundry_project_endpoint="https://example-project.cognitiveservices.azure.com",
            foundry_model="gpt-4o",
            search_endpoint="https://example-search.search.windows.net",
            search_api_key="search-key",
            search_index_name="docs-index",
            knowledge_base_name=None,
            semantic_configuration_name="semantic-config",
            vector_field_name=None,
            azure_openai_endpoint=None,
            azure_openai_embedding_model=None,
            azure_openai_embedding_deployment=None,
            azure_openai_resource_url=None,
        )
        app = HybridRAGApplication(config)
        app.credential = Mock(name="credential")

        with patch("hybrid_rag.AzureAISearchContextProvider") as provider_class:
            provider = await app._create_search_provider()

        self.assertIs(provider, provider_class.return_value)
        provider_class.assert_called_once_with(
            source_id="hybrid_search_provider",
            endpoint="https://example-search.search.windows.net",
            index_name="docs-index",
            api_key="search-key",
            credential=None,
            mode="semantic",
            top_k=5,
            semantic_configuration_name="semantic-config",
            vector_field_name=None,
            embedding_function=None,
        )

    async def test_create_search_provider_uses_credential_when_api_key_absent(self):
        config = AzureConfig(
            foundry_project_endpoint="https://example-project.cognitiveservices.azure.com",
            foundry_model="gpt-4o",
            search_endpoint="https://example-search.search.windows.net",
            search_api_key=None,
            search_index_name="docs-index",
            knowledge_base_name=None,
            semantic_configuration_name=None,
            vector_field_name=None,
            azure_openai_endpoint=None,
            azure_openai_embedding_model=None,
            azure_openai_embedding_deployment=None,
            azure_openai_resource_url=None,
        )
        credential = Mock(name="credential")
        app = HybridRAGApplication(config)
        app.credential = credential

        with patch("hybrid_rag.AzureAISearchContextProvider") as provider_class:
            await app._create_search_provider()

        self.assertIs(provider_class.call_args.kwargs["credential"], credential)
        self.assertIsNone(provider_class.call_args.kwargs["api_key"])


if __name__ == "__main__":
    unittest.main()
