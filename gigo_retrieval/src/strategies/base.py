from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient


@dataclass
class RetrievalConfig:
    qdrant_host: str
    qdrant_api_key: str
    timeout: int = 30

class RetrieverStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        pass


class BaseQdrantRetriever(ABC):
    def __init__(self, config: RetrievalConfig):
        """Initialize base retriever with Qdrant configuration."""
        self.config = config
        self.client = QdrantClient(
            url=self.config.qdrant_host,
            api_key=self.config.qdrant_api_key,
            timeout=self.config.timeout,
        )

    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        prefetch_limit: Optional[int] = None,
        score_threshold: Optional[int] = 10,
    ) -> List[Dict[str, Any]]:
        """Abstract method for retrieval implementation."""
        pass

    def _format_results(self, results) -> List[Dict[str, Any]]:
        """Format Qdrant results into standard format with enhanced metadata extraction."""
        formatted_results = []

        for point in results.points:
            payload = point.payload

            # Extract key metadata for easier access
            formatted_result = {
                "score": point.score,
                "id": point.id,
                "payload": payload,
                # Extract commonly used fields for convenience
                "text": payload.get("embedding_text", ""),
                "page_number": payload.get("page_number"),
                "document_title": payload.get("document_title"),
                "document_id": payload.get("document_id"),
                "section_title": payload.get("section_title"),
                "subsection_title": payload.get("subsection_title"),
                "manufacturer": payload.get("manufacturer"),
                "models_covered": payload.get("models_covered", []),
                "entities": payload.get("entities", []),
                "keywords": payload.get("keywords", []),
                "warnings": payload.get("warnings", []),
                "has_tables": payload.get("has_tables", False),
                "has_figures": payload.get("has_figures", False),
                "table_count": payload.get("table_count", 0),
                "figure_count": payload.get("figure_count", 0),
            }

            # Extract full page metadata if available
            if "full_page_metadata" in payload:
                full_metadata = payload["full_page_metadata"]
                formatted_result.update(
                    {
                        "page_visual_description": full_metadata.get(
                            "page_visual_description"
                        ),
                        "content_elements": full_metadata.get("content_elements", []),
                        "text_content": full_metadata.get("text_content"),
                        "text_file": full_metadata.get("text_file"),
                    }
                )

            formatted_results.append(formatted_result)

        return formatted_results