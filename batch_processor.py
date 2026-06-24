"""Batch processing module for the Hybrid RAG application."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from config import AzureConfig
from hybrid_rag import HybridRAGApplication

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple queries in batch mode."""

    def __init__(self, config: AzureConfig):
        """Initialize the batch processor."""
        self.config = config
        self.app = HybridRAGApplication(config)

    async def process_queries_from_file(
        self,
        input_file: Path,
        output_file: Path,
    ) -> None:
        """Process queries from a JSON file and save results."""
        await self.app.initialize()

        try:
            with open(input_file, encoding="utf-8") as f:
                queries_data = json.load(f)

            queries = (
                queries_data
                if isinstance(queries_data, list)
                else queries_data.get("queries", [])
            )

            logger.info(f"Processing {len(queries)} queries...")

            results = []
            for i, query in enumerate(queries, 1):
                logger.info(f"Processing query {i}/{len(queries)}")

                query_text = query if isinstance(query, str) else query.get("query", "")
                if not query_text:
                    logger.warning(f"Skipping empty query at index {i}")
                    continue

                try:
                    response_text = ""
                    async for chunk in self.app.query(query_text, stream=False):
                        response_text += chunk

                    result = {
                        "query": query_text,
                        "response": response_text,
                        "status": "success",
                    }
                except Exception as e:
                    logger.error(f"Error processing query {i}: {e}")
                    result = {
                        "query": query_text,
                        "response": None,
                        "status": "error",
                        "error": str(e),
                    }

                results.append(result)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to {output_file}")

            successful = sum(1 for r in results if r["status"] == "success")
            failed = sum(1 for r in results if r["status"] == "error")
            print("\nBatch Processing Complete:")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Results saved to: {output_file}")

        finally:
            await self.app.close()

    async def process_queries_list(
        self,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """Process a list of queries and return results."""
        await self.app.initialize()

        try:
            results = []
            for i, query in enumerate(queries, 1):
                logger.info(f"Processing query {i}/{len(queries)}: {query}")

                try:
                    response_text = ""
                    async for chunk in self.app.query(query, stream=False):
                        response_text += chunk

                    result = {
                        "query": query,
                        "response": response_text,
                        "status": "success",
                    }
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    result = {
                        "query": query,
                        "response": None,
                        "status": "error",
                        "error": str(e),
                    }

                results.append(result)

            return results

        finally:
            await self.app.close()


async def main() -> None:
    """Main function for batch processing."""
    config = AzureConfig.from_env()
    processor = BatchProcessor(config)

    input_file = Path("queries.json")
    output_file = Path("results.json")

    if input_file.exists():
        logger.info(f"Processing queries from {input_file}")
        await processor.process_queries_from_file(input_file, output_file)
    else:
        logger.warning(
            f"{input_file} not found. "
            "Create it with a 'queries' field containing query strings."
        )
        example_queries = {
            "queries": [
                "What are the main topics in the search index?",
                "Which documents are most relevant to this question?",
                "What are the key recommendations?",
            ]
        }

        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(example_queries, f, indent=2)

        logger.info(f"Created example file: {input_file}")


if __name__ == "__main__":
    asyncio.run(main())
