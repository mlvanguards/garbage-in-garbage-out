from typing import Any, Dict, List, Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding
from qdrant_client import models

from colpali_rag.retrieval.strategies.base import (
    BaseQdrantRetriever,
    RetrievalConfig,
)


class HybridRetriever(BaseQdrantRetriever):
    def __init__(
        self,
        config: RetrievalConfig,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
        colbert_model_name: str = "colbert-ir/colbertv2.0",
    ):
        super().__init__(config)
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        self.colbert_model = LateInteractionTextEmbedding(model_name=colbert_model_name)

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        prefetch_limit: Optional[int] = 20,
        score_threshold: Optional[int] = 10,
        collection_name: str = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search with dense and sparse embeddings, reranked using ColBERT."""
        prefetch_limit = prefetch_limit or 20
        dense_vector = next(self.dense_model.embed(query)).tolist()
        sparse_vector = next(self.sparse_model.embed(query))
        colbert_vector = next(self.colbert_model.embed(query)).tolist()

        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vector.as_object()),
                using="sparse",
                limit=prefetch_limit,
            ),
        ]

        results = self.client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=colbert_vector,
            using="colbert",
            with_payload=True,
            limit=limit,
            score_threshold=score_threshold,
        )
        # return results
        return self._format_results(results)


if __name__ == "__main__":
    from settings import settings

    config = RetrievalConfig(
        qdrant_host=settings.QDRANT_URL,
        qdrant_api_key=settings.QDRANT_API_KEY,
    )
    retriever = HybridRetriever(config)

    query = "My front axle brakes will not release in my JLG 1055. How do I get them to release ?"
    results = retriever.retrieve(
        query=query, collection_name="service_manual_pages", score_threshold=4
    )
