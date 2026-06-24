"""Configuration module for the Hybrid RAG application."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()


@dataclass
class AzureConfig:
    """Azure service configuration."""

    foundry_project_endpoint: str
    foundry_model: str
    search_endpoint: str
    search_api_key: str | None
    search_index_name: str | None
    knowledge_base_name: str | None
    semantic_configuration_name: str | None
    vector_field_name: str | None
    azure_openai_endpoint: str | None
    azure_openai_embedding_model: str | None
    azure_openai_embedding_deployment: str | None
    azure_openai_resource_url: str | None
    azure_openai_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Load configuration from environment variables."""
        embedding_model = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")
        embedding_deployment = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            embedding_model,
        )

        return cls(
            foundry_project_endpoint=os.environ.get(
                "FOUNDRY_PROJECT_ENDPOINT",
                "",
            ),
            foundry_model=os.environ.get("FOUNDRY_MODEL", "gpt-4o"),
            search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT", ""),
            search_api_key=os.environ.get("AZURE_SEARCH_API_KEY"),
            search_index_name=os.environ.get("AZURE_SEARCH_INDEX_NAME"),
            knowledge_base_name=os.environ.get(
                "AZURE_SEARCH_KNOWLEDGE_BASE_NAME",
            ),
            semantic_configuration_name=os.environ.get(
                "AZURE_SEARCH_SEMANTIC_CONFIGURATION",
            ),
            vector_field_name=os.environ.get("AZURE_SEARCH_VECTOR_FIELD_NAME"),
            azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_openai_embedding_model=embedding_model,
            azure_openai_embedding_deployment=embedding_deployment,
            azure_openai_resource_url=os.environ.get(
                "AZURE_OPENAI_RESOURCE_URL",
            ),
            azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        )

    def validate(self) -> None:
        """Validate required configuration."""
        required_fields = [
            ("FOUNDRY_PROJECT_ENDPOINT", self.foundry_project_endpoint),
            ("AZURE_SEARCH_ENDPOINT", self.search_endpoint),
            ("AZURE_SEARCH_INDEX_NAME", self.search_index_name),
        ]

        missing = [name for name, value in required_fields if not value]

        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}"
            )

        if self.vector_field_name:
            missing_embedding = [
                name
                for name, value in [
                    (
                        "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE_URL",
                        self.azure_openai_endpoint
                        or self.azure_openai_resource_url,
                    ),
                    (
                        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                        self.azure_openai_embedding_deployment,
                    ),
                ]
                if not value
            ]
            if missing_embedding:
                raise ValueError(
                    "Vector hybrid search requires embedding configuration: "
                    f"{', '.join(missing_embedding)}"
                )
