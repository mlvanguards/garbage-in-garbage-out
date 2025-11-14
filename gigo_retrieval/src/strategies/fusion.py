from typing import List

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding
from openai import OpenAI
from qdrant_client import models

from colpali_rag.retrieval.strategies.base import (
    BaseQdrantRetriever,
    RetrievalConfig,
)


class FusionybridRetriever(BaseQdrantRetriever):
    def __init__(
        self,
        config: RetrievalConfig,
        openai_client: OpenAI,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
        colbert_model_name: str = "colbert-ir/colbertv2.0",
    ):
        super().__init__(config)
        self.openai_client = openai_client
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        self.colbert_model = LateInteractionTextEmbedding(model_name=colbert_model_name)

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

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        prefetch_limit=10,
        collection_name: str = None,
    ):
        # Get all embeddings
        small_vector = self._get_openai_embedding(
            query, model="text-embedding-3-small", dimensions=128
        )
        large_vector = self._get_openai_embedding(
            query, model="text-embedding-3-large", dimensions=1024
        )

        # Get dense and sparse embeddings
        dense_vector = next(self.dense_model.embed([query])).tolist()
        sparse_vector = next(self.sparse_model.embed([query]))
        colbert_vector = next(self.colbert_model.embed([query])).tolist()

        # First branch: Multi-stage Matryoshka pipeline using small and large embeddings
        matryoshka_prefetch = models.Prefetch(
            prefetch=[
                # First level: small embedding retrieval
                models.Prefetch(
                    query=small_vector,
                    using="small-embedding",
                    limit=100,
                )
            ],
            # Second level: large embedding reranking
            query=large_vector,
            using="large-embedding",
            limit=50,
        )

        # Second branch: Dense-Sparse fusion pipeline
        sparse_dense_prefetch = models.Prefetch(
            prefetch=[
                # Dense vector retrieval
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=100,
                ),
                # Sparse vector retrieval
                models.Prefetch(
                    query=models.SparseVector(**sparse_vector.as_object()),
                    using="sparse",
                    limit=25,
                ),
            ],
            # Combine using RRF fusion
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
        )

        # Final query with late interaction reranking
        results = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                matryoshka_prefetch,
                sparse_dense_prefetch,
            ],
            # Final reranking using late interaction model
            query=colbert_vector,
            using="colbert",
            with_payload=True,
            limit=limit,
        )

        # return results
        return self._format_results(results)

    def _format_results(self, results):
        """Format search results into a standard structure."""
        formatted_results = []
        for result in results.points:
            formatted_results.append(
                {"id": result.id, "score": result.score, "payload": result.payload}
            )
        return formatted_results


if __name__ == "__main__":
    from settings import get_settings

    config = get_settings()
    openai_client = OpenAI()
    collection_name = "hyperhuman"

    retrieval_config = RetrievalConfig(
        qdrant_host=config.QDRANT_HOST,
        qdrant_api_key=config.QDRANT_API_KEY,
        timeout=60,
    )

    retriever = FusionybridRetriever(
        config=retrieval_config, openai_client=openai_client
    )

    query = "Give me all the invoices from documents"
    results = retriever.retrieve(query=query, collection_name=collection_name)

    scores = [(result["score"], result["payload"]["name"]) for result in results]

    print(scores)
