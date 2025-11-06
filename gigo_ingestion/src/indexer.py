"""
Document Indexer

This module provides high-level functionality for indexing page metadata into a Qdrant vector database.
It handles building embedding text from structured metadata and orchestrating the indexing process.
"""

import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from tqdm import tqdm

from src.db.collection import QdrantCollection

logger = logging.getLogger(__name__)


def build_embedding_text_from_page_metadata(metadata: dict) -> str:
    """
    Extract structured text from page metadata for embedding generation.
    
    This function creates a rich textual representation of a page by combining:
    - Document and section metadata
    - Content elements (text blocks, figures, tables)
    - Extracted entities, keywords, warnings, etc.
    
    Args:
        metadata: Page metadata dictionary containing document info, sections, and content elements
        
    Returns:
        Formatted text string suitable for embedding
    """
    doc = metadata.get("document_metadata", {})
    section = metadata.get("section", {})
    page_number = metadata.get("page_number", "")
    content_elements = metadata.get("content_elements", [])

    # Header
    header = [
        f"Document: {doc.get('document_title', '')} ({doc.get('manufacturer', '')}, Revision {doc.get('document_revision', '')})",
        f"Section: {section.get('section_number', '')} {section.get('section_title', '')}",
        f"Subsection: {section.get('subsection_number', '')} {section.get('subsection_title', '')}",
        f"Page: {page_number}",
    ]

    # Text content
    body = []
    for el in content_elements:
        el_type = el.get("type", "")
        title = el.get("title", "")
        summary = el.get("summary", "")
        text = ""
        if el_type == "text_block":
            text += f"Text Block: {title}\nSummary: {summary}\n"
        elif el_type == "figure":
            text += f"Figure: {title} – {summary}\n"
        elif el_type == "table":
            text += f"Table: {title} – {summary}\n"
        body.append(text.strip())

    # Include full text content if available
    if metadata.get("text_content"):
        body.append(f"Full Text Content:\n{metadata.get('text_content')}")

    # Entities and other metadata
    all_entities = set()
    all_keywords = set()
    all_warnings = set()
    all_contexts = set()
    all_models = set()

    for el in content_elements:
        all_entities.update(el.get("entities", []))
        all_keywords.update(el.get("keywords", []))
        all_warnings.update(el.get("warnings", []))
        all_contexts.update(el.get("application_context", []))
        all_models.update(el.get("model_applicability", []))

    tail = [
        f"Entities: {', '.join(sorted(all_entities))}" if all_entities else "",
        f"Warnings: {', '.join(sorted(all_warnings))}" if all_warnings else "",
        f"Keywords: {', '.join(sorted(all_keywords))}" if all_keywords else "",
        f"Model Applicability: {', '.join(sorted(all_models))}" if all_models else "",
        f"Context: {', '.join(sorted(all_contexts))}" if all_contexts else "",
    ]

    # Final embedding text
    full_text = "\n\n".join(part for part in (header + body + tail) if part)
    return full_text


class QdrantIndexer:
    """
    High-level orchestrator for indexing page metadata into Qdrant.
    
    This class handles:
    - Processing page metadata into structured documents
    - Managing the Qdrant collection
    - Batch indexing with progress tracking
    """

    def __init__(
        self,
        page_metadata_list: List[Dict[str, Any]],
        qdrant_client: QdrantClient,
        collection_name: str,
        batch_size: int = 4,
        openai_client: Optional[OpenAI] = None,
        include_full_metadata: bool = True,
    ):
        """
        Initialize the Qdrant indexer.
        
        Args:
            page_metadata_list: List of page metadata dictionaries to index
            qdrant_client: QdrantClient instance
            collection_name: Name of the Qdrant collection
            batch_size: Number of documents to process in each batch
            openai_client: Optional OpenAI client for embeddings
            include_full_metadata: Whether to include full page metadata in the index
        """
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.include_full_metadata = include_full_metadata

        # Initialize collection manager
        self.collection = QdrantCollection(
            client=qdrant_client,
            name=collection_name,
            openai_client=openai_client,
        )

        # Process page metadata into documents and metadata
        self._process_page_metadata(page_metadata_list)

        logger.info(f"QdrantIndexer initialized with {len(page_metadata_list)} pages")

    def _process_page_metadata(self, page_metadata_list: List[Dict[str, Any]]) -> None:
        """
        Extract text and metadata from page metadata.
        
        This method processes each page metadata dictionary and:
        - Generates embedding text
        - Creates unique IDs
        - Extracts structured metadata for filtering
        
        Args:
            page_metadata_list: List of page metadata dictionaries
        """
        self.ids = []
        self.documents = []
        self.metadata = []

        for i, page_metadata in enumerate(page_metadata_list):
            try:
                # Generate embedding text using your function
                embedding_text = build_embedding_text_from_page_metadata(page_metadata)

                # Create ID from document info and page number
                doc_info = page_metadata.get("document_metadata", {})
                page_num = page_metadata.get("page_number", i)
                doc_id = doc_info.get("document_id", "unknown")
                # Create a hash-based integer ID for Qdrant compatibility
                point_id_str = f"{doc_id}_page_{page_num}"
                point_id = (
                    hash(point_id_str) & 0x7FFFFFFFFFFFFFFF
                )  # Convert to positive 64-bit integer

                self.ids.append(point_id)
                self.documents.append(embedding_text)

                # Create structured metadata for easier filtering and retrieval
                structured_metadata = {
                    "page_number": page_metadata.get("page_number"),
                    "document_id": doc_info.get("document_id"),
                    "document_title": doc_info.get("document_title"),
                    "document_type": doc_info.get("document_type"),
                    "manufacturer": doc_info.get("manufacturer"),
                    "models_covered": doc_info.get("models_covered", []),
                    "section_number": page_metadata.get("section", {}).get(
                        "section_number"
                    ),
                    "section_title": page_metadata.get("section", {}).get(
                        "section_title"
                    ),
                    "subsection_number": page_metadata.get("section", {}).get(
                        "subsection_number"
                    ),
                    "subsection_title": page_metadata.get("section", {}).get(
                        "subsection_title"
                    ),
                    "has_tables": page_metadata.get("has_tables", False),
                    "has_figures": page_metadata.get("has_figures", False),
                    "table_count": page_metadata.get("table_count", 0),
                    "figure_count": page_metadata.get("figure_count", 0),
                    "text_block_count": page_metadata.get("text_block_count", 0),
                    "page_visual_description": page_metadata.get(
                        "page_visual_description"
                    ),
                }

                # Extract aggregated entities, keywords, etc. for easier filtering
                all_entities = set()
                all_keywords = set()
                all_warnings = set()
                all_contexts = set()
                all_models = set()
                all_component_types = set()

                for el in page_metadata.get("content_elements", []):
                    all_entities.update(el.get("entities", []))
                    all_keywords.update(el.get("keywords", []))
                    all_warnings.update(el.get("warnings", []))
                    all_contexts.update(el.get("application_context", []))
                    all_models.update(el.get("model_applicability", []))
                    if el.get("component_type"):
                        all_component_types.add(el.get("component_type"))

                structured_metadata.update(
                    {
                        "entities": list(all_entities),
                        "keywords": list(all_keywords),
                        "warnings": list(all_warnings),
                        "application_contexts": list(all_contexts),
                        "applicable_models": list(all_models),
                        "component_types": list(all_component_types),
                    }
                )

                # Include full metadata if requested (useful for reconstruction)
                if self.include_full_metadata:
                    structured_metadata["full_page_metadata"] = page_metadata

                self.metadata.append(structured_metadata)

            except Exception as e:
                logger.error(f"Error processing page metadata {i}: {str(e)}")
                # Use fallback values with integer ID
                self.ids.append(i)  # Use simple integer as fallback ID
                self.documents.append(f"Error processing page {i}")
                self.metadata.append({"error": str(e), "page_index": i})

    def create_collection(self) -> None:
        """Create collection with vector configurations if it doesn't exist."""
        sample_text = self.documents[0] if self.documents else "Sample text"
        self.collection.create(sample_text=sample_text)

    def index_documents(self) -> None:
        """
        Index all pages in batches with progress tracking.
        
        This method processes documents in batches and upserts them to Qdrant.
        It includes error handling and progress reporting.
        """
        total_docs = len(self.documents)
        processed = 0
        failed_batches = []

        with tqdm(total=total_docs, desc="Indexing pages") as pbar:
            while processed < total_docs:
                end_idx = min(processed + self.batch_size, total_docs)
                batch_ids = self.ids[processed:end_idx]
                batch_docs = self.documents[processed:end_idx]
                batch_metadata = self.metadata[processed:end_idx]

                try:
                    points = [
                        self.collection.create_point(id=id, text=doc, metadata=metadata)
                        for id, doc, metadata in zip(
                            batch_ids, batch_docs, batch_metadata
                        )
                    ]

                    self.collection.upsert_points(points)

                except Exception as e:
                    logger.error(
                        f"Error processing batch {processed}-{end_idx}: {str(e)}"
                    )
                    failed_batches.append((processed, end_idx))

                batch_size = end_idx - processed
                pbar.update(batch_size)
                processed = end_idx

        if failed_batches:
            logger.warning(
                f"Failed to process {len(failed_batches)} batches: {failed_batches}"
            )
        else:
            logger.info(f"Successfully processed all {total_docs} pages")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the indexed collection."""
        return self.collection.get_info()


# Example usage:
if __name__ == "__main__":
    import json
    from src.config import settings
    from src.db.manager import QDrantConnectionManager

    # Initialize connection manager
    connection_manager = QDrantConnectionManager()
    connection_manager.init()
    qdrant_client = connection_manager.get_client()

    # Load your page metadata files
    page_metadata_list = []

    for page_number in range(4, 566):
        try:
            page_metadata_list.append(
                json.load(
                    open(
                        f"/Users/vesaalexandru/Workspaces/cube/america/complex-rag/scratch/service_manual_long/page_{page_number}/context_metadata_page_{page_number}.json"
                    )
                )
            )
        except FileNotFoundError:
            print(f"Page {page_number} not found")
            continue

    # Initialize the indexer
    indexer = QdrantIndexer(
        page_metadata_list=page_metadata_list,
        qdrant_client=qdrant_client,
        collection_name="service_manual_pages",
        batch_size=1,
    )

    # Create collection and index documents
    indexer.create_collection()
    indexer.index_documents()
    print(json.dumps(indexer.get_collection_info(), indent=2, default=str))
    
    # Clean up
    connection_manager.close()
