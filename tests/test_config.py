import os
import unittest
from unittest.mock import patch

from config import AzureConfig


class AzureConfigTests(unittest.TestCase):
    def test_from_env_loads_same_shared_settings_and_index(self):
        env = {
            "FOUNDRY_PROJECT_ENDPOINT": "https://example-project.cognitiveservices.azure.com",
            "FOUNDRY_MODEL": "gpt-4o-mini",
            "AZURE_SEARCH_ENDPOINT": "https://example-search.search.windows.net",
            "AZURE_SEARCH_API_KEY": "search-key",
            "AZURE_SEARCH_INDEX_NAME": "docs-index",
            "AZURE_SEARCH_KNOWLEDGE_BASE_NAME": "agentic-kb-not-used",
            "AZURE_SEARCH_SEMANTIC_CONFIGURATION": "semantic-config",
            "AZURE_SEARCH_VECTOR_FIELD_NAME": "contentVector",
            "AZURE_OPENAI_ENDPOINT": "https://example-openai.openai.azure.com",
            "AZURE_OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-large",
            "AZURE_OPENAI_RESOURCE_URL": "https://example-openai-resource.openai.azure.com",
        }

        with patch.dict(os.environ, env, clear=True):
            config = AzureConfig.from_env()

        self.assertEqual(config.foundry_project_endpoint, env["FOUNDRY_PROJECT_ENDPOINT"])
        self.assertEqual(config.foundry_model, env["FOUNDRY_MODEL"])
        self.assertEqual(config.search_endpoint, env["AZURE_SEARCH_ENDPOINT"])
        self.assertEqual(config.search_api_key, env["AZURE_SEARCH_API_KEY"])
        self.assertEqual(config.search_index_name, env["AZURE_SEARCH_INDEX_NAME"])
        self.assertEqual(
            config.knowledge_base_name,
            env["AZURE_SEARCH_KNOWLEDGE_BASE_NAME"],
        )
        self.assertEqual(
            config.semantic_configuration_name,
            env["AZURE_SEARCH_SEMANTIC_CONFIGURATION"],
        )
        self.assertEqual(config.vector_field_name, env["AZURE_SEARCH_VECTOR_FIELD_NAME"])
        self.assertEqual(config.azure_openai_endpoint, env["AZURE_OPENAI_ENDPOINT"])
        self.assertEqual(
            config.azure_openai_embedding_model,
            env["AZURE_OPENAI_EMBEDDING_MODEL"],
        )
        self.assertEqual(
            config.azure_openai_embedding_deployment,
            env["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        )
        self.assertEqual(
            config.azure_openai_resource_url,
            env["AZURE_OPENAI_RESOURCE_URL"],
        )

    def test_validate_requires_index_name_for_hybrid_search(self):
        config = AzureConfig(
            foundry_project_endpoint="https://example-project.cognitiveservices.azure.com",
            foundry_model="gpt-4o",
            search_endpoint="https://example-search.search.windows.net",
            search_api_key=None,
            search_index_name=None,
            knowledge_base_name=None,
            semantic_configuration_name=None,
            vector_field_name=None,
            azure_openai_endpoint=None,
            azure_openai_embedding_model=None,
            azure_openai_embedding_deployment=None,
            azure_openai_resource_url=None,
        )

        with self.assertRaisesRegex(ValueError, "AZURE_SEARCH_INDEX_NAME"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
