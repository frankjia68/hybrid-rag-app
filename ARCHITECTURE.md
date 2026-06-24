# Architecture Documentation

## System Overview

The Hybrid RAG Application is a peer to the Agentic RAG Application. It keeps the same Python layout, Azure Foundry model integration, interactive chat, and batch processing surfaces, but it uses an Azure AI Search index directly rather than an Azure AI Search Knowledge Base.

## Component Architecture

### 1. Configuration Layer (`config.py`)

`AzureConfig` loads shared Azure settings from environment variables:

```python
class AzureConfig:
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
    azure_openai_api_key: str | None
```

Required settings:

- `FOUNDRY_PROJECT_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_INDEX_NAME`

Compatibility settings loaded but not used by hybrid retrieval:

- `AZURE_SEARCH_KNOWLEDGE_BASE_NAME`
- `AZURE_OPENAI_RESOURCE_URL`

Optional vector hybrid settings:

- `AZURE_SEARCH_VECTOR_FIELD_NAME`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_MODEL`

### 2. Core Application (`hybrid_rag.py`)

`HybridRAGApplication` orchestrates:

- Azure CLI credential creation
- Azure AI Search context provider creation in `mode="semantic"`
- Optional embedding function creation for vector hybrid search
- Microsoft Agent Framework `Agent` creation
- Streaming and non-streaming query execution
- Resource cleanup

### 3. User Interface Layer

`interactive_chat.py` provides a streaming command-line chat loop.

`batch_processor.py` reads queries from `queries.json` or another JSON file and writes result objects to `results.json`.

## Data Flow

```text
User Query
    |
Agent.run(query)
    |
AzureAISearchContextProvider(mode="semantic")
    |-- Queries AZURE_SEARCH_INDEX_NAME
    |-- Applies semantic ranking when configured
    |-- Uses vector_field_name + embedding_function when configured
    |
FoundryChatClient
    |
Answer with citations
```

## Retrieval Behavior

Semantic mode is optimized for faster index-based retrieval. It is best for straightforward questions where latency and direct relevance matter more than multi-hop query planning.

When `AZURE_SEARCH_VECTOR_FIELD_NAME` is not set, the provider uses index retrieval with semantic ranking if the index supports it.

When `AZURE_SEARCH_VECTOR_FIELD_NAME` is set, `HybridRAGApplication` creates an Azure OpenAI embedding function using the configured embedding deployment. The context provider can then combine vector retrieval with keyword/semantic retrieval.

## Differences from Agentic RAG

| Area | Agentic RAG | Hybrid RAG |
| --- | --- | --- |
| Search provider mode | `agentic` | `semantic` |
| Search source | Knowledge Base or KB auto-created from index | Azure Search index |
| Best for | Complex multi-hop reasoning | Fast direct retrieval |
| Required search name | KB or index | Index |
| Token use | Higher | Lower |

## Extension Points

- Customize agent instructions in `HybridRAGApplication._create_agent`
- Tune `top_k` in `_create_search_provider`
- Set `AZURE_SEARCH_SEMANTIC_CONFIGURATION` for a specific semantic config
- Set `AZURE_SEARCH_VECTOR_FIELD_NAME` and embedding settings for true vector hybrid search

## Testing Strategy

The included tests verify:

- Environment loading preserves shared configuration settings
- Hybrid RAG requires `AZURE_SEARCH_INDEX_NAME`
- The search provider is created with `mode="semantic"`
- API key authentication disables credential passing
- RBAC mode passes the Azure credential when no API key is configured
