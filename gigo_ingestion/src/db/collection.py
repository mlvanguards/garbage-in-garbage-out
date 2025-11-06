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

    def exists(self) -> bool:
        """Check if the collection exists."""
        return self.client.collection_exists(self.name)

    def create(self, sample_text: str = "Sample text") -> None:
        """
        Create collection with vector configurations if it doesn't exist.
        
        Args:
            sample_text: Text to use for determining embedding dimensions
        """
        if self.exists():
            logger.info(f"Collection '{self.name}' already exists")
            return

        # Get sample embeddings to determine dimensions
        dense_embedding = list(self.dense_embedding_model.embed([sample_text]))[0]
        late_interaction_embedding = list(
            self.late_interaction_model.embed([sample_text])
        )[0]

        # Create collection with vector configurations
        self.client.create_collection(
            collection_name=self.name,
            vectors_config={
                "dense": models.VectorParams(
                    size=len(dense_embedding),
                    distance=models.Distance.COSINE,
                ),
                "colbert": models.VectorParams(
                    size=len(late_interaction_embedding[0]),
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
                "small-embedding": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    datatype=models.Datatype.FLOAT16,
                ),
                "large-embedding": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                    datatype=models.Datatype.FLOAT16,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    ),
                )
            },
        )
        logger.info(f"Created collection '{self.name}'")

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

    def _get_embeddings(self, text: str) -> Dict[str, Any]:
        """
        Generate all types of embeddings for a given text.
        
        Args:
            text: Text to embed
            
        Returns:
            Dictionary containing all embedding types
        """
        try:
            return {
                "dense": list(self.dense_embedding_model.embed([text]))[0].tolist(),
                "colbert": list(self.late_interaction_model.embed([text]))[0].tolist(),
                "small-embedding": self._get_small_embedding(text),
                "large-embedding": self._get_large_embedding(text),
                "sparse": self._create_sparse_vector(text),
            }
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def _get_small_embedding(self, text: str) -> List[float]:
        """Generate small embedding using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                input=text, model="text-embedding-3-small", dimensions=128
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting small embedding: {str(e)}")
            raise

    def _get_large_embedding(self, text: str) -> List[float]:
        """Generate large embedding using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                input=text, model="text-embedding-3-large", dimensions=1024
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting large embedding: {str(e)}")
            raise

    def _create_sparse_vector(self, text: str) -> models.SparseVector:
        """Create sparse vector from text."""
        try:
            embeddings = list(self.sparse_embedding_model.embed([text]))[0]

            if not (hasattr(embeddings, "indices") and hasattr(embeddings, "values")):
                raise ValueError("Invalid sparse embeddings format")

            return models.SparseVector(
                indices=embeddings.indices.tolist(), values=embeddings.values.tolist()
            )
        except Exception as e:
            logger.error(f"Error creating sparse vector: {str(e)}")
            raise

    def create_point(
        self, id: int, text: str, metadata: Dict[str, Any]
    ) -> models.PointStruct:
        """
        Create a single point with all vector types and metadata.
        
        Args:
            id: Point ID
            text: Text to embed
            metadata: Additional metadata to store
            
        Returns:
            PointStruct ready for upserting
        """
        try:
            vectors = self._get_embeddings(text)

            # Include the embedding text in payload for reference
            payload = {"embedding_text": text, **metadata}

            return models.PointStruct(id=id, vector=vectors, payload=payload)
        except Exception as e:
            logger.error(f"Error creating point for id {id}: {str(e)}")
            raise

    def upsert_points(self, points: List[models.PointStruct]) -> None:
        """
        Upsert a list of points to the collection.
        
        Args:
            points: List of points to upsert
        """
        try:
            self.client.upsert(collection_name=self.name, points=points)
            logger.debug(f"Upserted {len(points)} points to collection '{self.name}'")
        except Exception as e:
            logger.error(f"Error upserting points: {str(e)}")
            raise

    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.name)
            logger.info(f"Deleted collection '{self.name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
