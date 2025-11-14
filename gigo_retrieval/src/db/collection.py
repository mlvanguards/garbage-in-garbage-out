import logging
from typing import Any, Dict, List, Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class QdrantCollection:
    """
    Manages Qdrant collection operations including embeddings and indexing.
    
    This class handles:
    - Creating and managing Qdrant collections
    - Generating multiple types of embeddings (dense, sparse, ColBERT, OpenAI)
    - Creating and upserting points
    """

    def __init__(
        self,
        client: QdrantClient,
        name: str,
        openai_client: Optional[OpenAI] = None,
    ):
        """
        Initialize the Qdrant collection manager.
        
        Args:
            client: QdrantClient instance
            name: Name of the collection
            openai_client: Optional OpenAI client for embeddings
        """
        self.client = client
        self.name = name
        self.openai_client = openai_client or OpenAI()

        # Initialize embedding models
        logger.info("Initializing embedding models...")
        self.dense_embedding_model = TextEmbedding(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.sparse_embedding_model = SparseTextEmbedding(
            "Qdrant/bm42-all-minilm-l6-v2-attentions"
        )
        self.late_interaction_model = LateInteractionTextEmbedding(
            "colbert-ir/colbertv2.0"
        )
        logger.info("Embedding models initialized")


    def get_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.name)
            return {
                "collection_name": self.name,
                "points_count": collection_info.points_count,
                "vectors_config": collection_info.config.params.vectors,
                "sparse_vectors_config": collection_info.config.params.sparse_vectors,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}

