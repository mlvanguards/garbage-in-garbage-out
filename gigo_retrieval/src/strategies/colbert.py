from typing import Any, Dict, List, Optional

from fastembed.late_interaction import LateInteractionTextEmbedding

from src.retrieval.strategies.base import (
    BaseQdrantRetriever,
    RetrievalConfig,
)


class ColbertRetriever(BaseQdrantRetriever):
    def __init__(
        self,
        config: RetrievalConfig,
        model_name: str = "colbert-ir/colbertv2.0",
    ):
        super().__init__(config)
        self.model = LateInteractionTextEmbedding(model_name=model_name)

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[int] = 10,
        prefetch_limit: Optional[int] = 20,
        collection_name: str = None,
    ) -> List[Dict[str, Any]]:
        """Perform search using ColBERT late interaction model."""
        query_vector = next(self.model.embed(query)).tolist()
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="colbert",
            with_payload=True,
            limit=limit,
            score_threshold=score_threshold,
        )
        return self._format_results(results)
