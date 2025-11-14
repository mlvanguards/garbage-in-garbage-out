from typing import Optional

from openai import OpenAI

from colpali_rag.retrieval.strategies.base import RetrievalConfig
from colpali_rag.retrieval.strategies.custom_qdrant.search.colbert import (
    ColbertRetriever,
)
from colpali_rag.retrieval.strategies.custom_qdrant.search.hybrid import (
    HybridRetriever,
)
from colpali_rag.retrieval.strategies.custom_qdrant.search.matrioska import (
    MatrioskaRetriever,
)
from colpali_rag.retrieval.types import SearchType

from enum import Enum


class SearchType(str, Enum):
    COLBERT = "colbert"
    HYBRID = "hybrid"
    MATRIOSKA = "matrioska"
    FUSION = "fusion"


class RetrieverFactory:
    @staticmethod
    def create_retriever(
        search_type: SearchType,
        config: RetrievalConfig,
        openai_client: Optional[OpenAI] = None,
    ):
        if search_type == SearchType.COLBERT:
            return ColbertRetriever(config=config)
        elif search_type == SearchType.HYBRID:
            return HybridRetriever(config=config)
        elif search_type == SearchType.MATRIOSKA:
            if not openai_client:
                raise ValueError("OpenAI client required for matrioska search")
            return MatrioskaRetriever(openai_client=openai_client, config=config)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
