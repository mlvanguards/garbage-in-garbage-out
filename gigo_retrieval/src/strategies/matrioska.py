from typing import Any, Dict, List, Optional

from openai import OpenAI
from qdrant_client import models

from colpali_rag.retrieval.strategies.base import (
    BaseQdrantRetriever,
    RetrievalConfig,
)


class MatrioskaRetriever(BaseQdrantRetriever):
    def __init__(
        self,
        config: RetrievalConfig,
        openai_client: OpenAI,
    ):
        super().__init__(config)
        self.openai_client = openai_client

    def _get_openai_embedding(
        self, text: str, model: str = "text-embedding-3-small", dimensions: int = 128
    ) -> List[float]:
        """Get OpenAI embeddings with specified model and dimensions."""
        text = text.replace("\n", " ")
        return (
            self.openai_client.embeddings.create(
                input=[text], model=model, dimensions=dimensions
            )
            .data[0]
            .embedding
        )

    def _get_matryoska_prefetch(
        self, query: str, limit_small_vector=100, limit_large_vector=50
    ):
        small_vector = self._get_openai_embedding(
            query, model="text-embedding-3-small", dimensions=128
        )
        large_vector = self._get_openai_embedding(
            query, model="text-embedding-3-large", dimensions=1024
        )
        matryoshka_prefetch = models.Prefetch(
            prefetch=[
                models.Prefetch(
                    prefetch=[
                        # The first prefetch operation retrieves 100 documents
                        # using the Matryoshka embeddings with the lowest
                        # dimensionality of 64.
                        models.Prefetch(
                            query=small_vector,
                            using="small-embedding",
                            limit=limit_small_vector,
                        ),
                    ],
                    # Then, the retrieved documents are re-ranked using the
                    # Matryoshka embeddings with the dimensionality of 128.
                    query=large_vector,
                    using="large-embedding",
                    limit=limit_large_vector,
                )
            ],
        )
        return matryoshka_prefetch

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        prefetch_limit: Optional[int] = 50,
        score_threshold: Optional[int] = 10,
        collection_name: str = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search using OpenAI's embedding models."""
        prefetch_limit = prefetch_limit or 50

        small_vector = self._get_openai_embedding(
            query, model="text-embedding-3-small", dimensions=128
        )
        large_vector = self._get_openai_embedding(
            query, model="text-embedding-3-large", dimensions=1024
        )

        results = self.client.query_points(
            collection_name=collection_name,
            prefetch=models.Prefetch(
                query=small_vector,
                using="small-embedding",
                limit=prefetch_limit,
            ),
            query=large_vector,
            using="large-embedding",
            limit=limit,
        )
        return self._format_results(results)
