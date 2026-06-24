# Hybrid RAG Application with Microsoft Agent Framework

A production-ready Hybrid Retrieval-Augmented Generation (RAG) application built with Microsoft Agent Framework and Azure AI Search. This peer project mirrors the Agentic RAG application structure while using an Azure Search index directly through semantic/hybrid retrieval.

## Features

- Hybrid/semantic retrieval against an Azure AI Search index
- Optional vector + keyword hybrid retrieval when `AZURE_SEARCH_VECTOR_FIELD_NAME` is configured
- Azure AI Foundry chat model integration
- Interactive chat with streaming responses and source citations
- Batch processing from JSON query files
- Same configuration style and project structure as `agentic-rag-app`
- Async resource lifecycle and Azure CLI/RBAC authentication support

## Architecture

```text
User Query
    |
Microsoft Agent Framework
    |
Azure AI Search Context Provider (semantic mode)
    |-- Keyword search
    |-- Semantic ranking
    |-- Optional vector search with embeddings
    |
Foundry model response generation with citations
    |
User Response
```

## Prerequisites

- Azure subscription with:
  - Azure AI Foundry project with a deployed chat model
  - Azure AI Search service with a configured index
- Python 3.10+
- Azure CLI installed and authenticated with `az login`

## Quick Start

```bash
cd C:\work\projects\hybrid-rag-app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` with your Azure values:

```env
FOUNDRY_PROJECT_ENDPOINT=https://your-project-endpoint.cognitiveservices.azure.com
FOUNDRY_MODEL=gpt-4o
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX_NAME=your-search-index
AZURE_SEARCH_KNOWLEDGE_BASE_NAME=your-knowledge-base
AZURE_SEARCH_SEMANTIC_CONFIGURATION=default
```

For true vector + keyword hybrid retrieval, also configure:

```env
AZURE_SEARCH_VECTOR_FIELD_NAME=contentVector
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com
AZURE_OPENAI_RESOURCE_URL=https://your-openai-resource.openai.azure.com
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

Run the interactive chat:

```bash
python interactive_chat.py
```

Run batch processing:

```bash
python batch_processor.py
```

## Project Structure

```text
hybrid-rag-app/
|-- hybrid_rag.py                 # Main application class
|-- interactive_chat.py           # Interactive chat interface
|-- batch_processor.py            # Batch query processing
|-- config.py                     # Configuration management
|-- requirements.txt              # Python dependencies
|-- .env.example                  # Environment variables template
|-- queries.json                  # Example batch queries
|-- README.md                     # This file
|-- SETUP.md                      # Detailed setup guide
|-- ARCHITECTURE.md               # Architecture documentation
|-- tests/                        # Unit tests
`-- examples/
    |-- basic_usage.py
    |-- advanced_hybrid_search.py
    `-- custom_instructions.py
```

## Agentic vs Hybrid

The original `agentic-rag-app` uses `AzureAISearchContextProvider(mode="agentic")`, which relies on Knowledge Bases for query planning and multi-hop reasoning.

This peer app uses `AzureAISearchContextProvider(mode="semantic")`, which queries `AZURE_SEARCH_INDEX_NAME` directly for fast hybrid search with semantic ranking. `AZURE_SEARCH_KNOWLEDGE_BASE_NAME` is still loaded for configuration parity but is not used. If a vector field and embedding configuration are supplied, the provider can combine keyword and vector retrieval.

## Testing

```bash
python -m unittest discover -s tests -v
python -m py_compile hybrid_rag.py config.py interactive_chat.py batch_processor.py examples/basic_usage.py examples/advanced_hybrid_search.py examples/custom_instructions.py
```

## Security

- Supports Azure Search API key authentication
- Supports Azure CLI/RBAC authentication when no search API key is provided
- Keeps credentials in `.env`, which is ignored by git
- Supports Azure OpenAI API key for embeddings or Azure CLI token auth

## Troubleshooting

Missing `AZURE_SEARCH_INDEX_NAME` means the app cannot run. Hybrid RAG uses the index directly and does not use `AZURE_SEARCH_KNOWLEDGE_BASE_NAME`.

If vector search is configured, confirm the vector field exists in your Azure Search index and that the embedding deployment returns vectors with the same dimensions as the index field.
