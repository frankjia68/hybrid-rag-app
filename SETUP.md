# Detailed Setup Guide for Hybrid RAG Application

## Prerequisites

- Python 3.10 or higher
- Git
- Azure CLI
- Azure AI Foundry project with a deployed chat model
- Azure AI Search service with a populated index

## Azure Resources

### Azure AI Foundry

1. Open Azure AI Foundry.
2. Create or select a project.
3. Deploy a chat model such as `gpt-4o` or `gpt-4o-mini`.
4. Record the project endpoint and deployment name.

### Azure AI Search

1. Create or select an Azure AI Search service.
2. Create or select an index containing searchable content.
3. Optional: configure a semantic ranking configuration on the index.
4. Optional: configure a vector field if you want vector + keyword hybrid retrieval.

## Local Setup

```bash
cd C:\work\projects\hybrid-rag-app
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
az login
copy .env.example .env
```

Edit `.env`:

```env
FOUNDRY_PROJECT_ENDPOINT=https://your-project-endpoint.cognitiveservices.azure.com
FOUNDRY_MODEL=gpt-4o
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX_NAME=your-search-index
AZURE_SEARCH_KNOWLEDGE_BASE_NAME=your-knowledge-base
AZURE_SEARCH_SEMANTIC_CONFIGURATION=default
```

If using RBAC instead of a search API key, leave `AZURE_SEARCH_API_KEY` empty and grant your signed-in identity `Search Index Data Reader` on the Azure Search service.

## Optional Vector Hybrid Search

If your Azure Search index has a vector field, set:

```env
AZURE_SEARCH_VECTOR_FIELD_NAME=contentVector
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com
AZURE_OPENAI_RESOURCE_URL=https://your-openai-resource.openai.azure.com
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

Use `AZURE_OPENAI_API_KEY` for API key auth, or omit it to use Azure CLI/RBAC token auth.

The embedding model output dimensions must match the Azure Search vector field dimensions.

## Verification

Check local tests and syntax:

```bash
python -m unittest discover -s tests -v
python -m py_compile hybrid_rag.py config.py interactive_chat.py batch_processor.py examples/basic_usage.py examples/advanced_hybrid_search.py examples/custom_instructions.py
```

Check Azure authentication:

```bash
az account show
```

Run the app:

```bash
python interactive_chat.py
```

Run batch mode:

```bash
python batch_processor.py
```

## Troubleshooting

### Missing AZURE_SEARCH_INDEX_NAME

Hybrid RAG requires an Azure Search index. Set `AZURE_SEARCH_INDEX_NAME` in `.env`.

### Semantic configuration error

Confirm `AZURE_SEARCH_SEMANTIC_CONFIGURATION` matches a semantic configuration in your index. If your index has a default semantic configuration, you can leave this variable empty.

### Vector field error

Confirm `AZURE_SEARCH_VECTOR_FIELD_NAME` matches the vector field in the index. Also confirm the embedding deployment returns vectors with the same dimension as that field.

### Authentication error

Run:

```bash
az login
az account show
```

If using RBAC, confirm the signed-in identity has:

- Azure AI User on the Foundry project
- Search Index Data Reader on the Azure Search service
