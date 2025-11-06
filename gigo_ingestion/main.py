"""
Document Ingestion Pipeline

This module provides a complete end-to-end pipeline for ingesting PDF documents into a vector database.
The pipeline handles parsing, metadata extraction, content processing, and indexing.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.parser import DocumentParser
from src.metadata_extractors.page_context_extractor import PageContextMetadataExtractor
from src.processors.table import TableBatchProcessor
from src.processors.text_blocks import TextBatchProcessor
from src.processors.image import ImageProcessor, ImageBatchProcessor
from src.db import QDrantConnectionManager, QdrantCollection
from src.indexer import QdrantIndexer
from src.llm import LitellmClient
from src.utils import load_image_as_data_uri

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PathBuilder:
    """Helper class to manage file paths for processed documents."""
    
    def __init__(self, base_path: Path):
        """
        Initialize the path builder.
        
        Args:
            base_path: Base directory where processed document files are stored
        """
        self.base_path = Path(base_path)
    
    def get_page_dir(self, page_number: int) -> Path:
        """Get the directory for a specific page."""
        return self.base_path / f"page_{page_number}"
    
    def get_page_image(self, page_number: int) -> Path:
        """Get the path to the full page image."""
        return self.get_page_dir(page_number) / f"page_{page_number}_full.png"
    
    def get_tables_dir(self, page_number: int) -> Path:
        """Get the tables directory for a specific page."""
        return self.get_page_dir(page_number) / "tables"
    
    def get_images_dir(self, page_number: int) -> Path:
        """Get the images directory for a specific page."""
        return self.get_page_dir(page_number) / "images"
    
    def get_text_dir(self, page_number: int) -> Path:
        """Get the text directory for a specific page."""
        return self.get_page_dir(page_number) / "text"
    
    def get_context_metadata(self, page_number: int) -> Path:
        """Get the path to the context metadata file for a page."""
        return self.get_page_dir(page_number) / "context_metadata.json"
    
    def get_page_metadata(self, page_number: int) -> Path:
        """Get the path to the page metadata file."""
        return self.get_page_dir(page_number) / f"metadata_page_{page_number}.json"
    
    def get_all_page_numbers(self) -> List[int]:
        """Get all page numbers from the base directory."""
        page_dirs = [d for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith("page_")]
        return sorted([int(d.name.split("_")[1]) for d in page_dirs])


class DocumentPipeline:
    """
    End-to-end document ingestion pipeline.
    
    This pipeline orchestrates the following steps:
    1. PDF parsing and content extraction
    2. Page context metadata extraction
    3. Table metadata extraction and processing
    4. Text block metadata extraction and processing
    5. Image processing
    6. Vector database indexing
    """
    
    def __init__(
        self,
        output_dir: str = "data/processed",
        collection_name: str = "documents",
        llm_model: str = "openai/gpt-4o"
    ):
        """
        Initialize the document pipeline.
        
        Args:
            output_dir: Directory where processed documents will be stored
            collection_name: Name of the Qdrant collection to use
            llm_model: LLM model to use for metadata extraction
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.llm_model = llm_model
        
        # Initialize components
        logger.info("Initializing document pipeline...")
        self.parser = DocumentParser()
        self.llm_client = LitellmClient(model_name=llm_model)
        self.page_context_extractor = PageContextMetadataExtractor()
        self.table_processor = TableBatchProcessor()
        self.text_processor = TextBatchProcessor()
        self.image_processor = ImageBatchProcessor()
        
        # Initialize database connection
        self.db_manager = QDrantConnectionManager()
        
        logger.info("Document pipeline initialized successfully")
    
    def connect_database(self):
        """Connect to the vector database."""
        logger.info("Connecting to vector database...")
        self.db_manager.init()
        if not self.db_manager.connected():
            raise RuntimeError("Failed to connect to vector database")
        logger.info("Connected to vector database")
    
    def process_document(
        self,
        pdf_path: str,
        document_name: Optional[str] = None,
        extract_metadata: bool = True,
        index_to_db: bool = False,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a complete PDF document through the pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            document_name: Optional name for the document (defaults to filename)
            extract_metadata: Whether to extract rich metadata using LLM
            index_to_db: Whether to index the results to the vector database
            start_page: Optional starting page number (for partial processing)
            end_page: Optional ending page number (for partial processing)
            
        Returns:
            Dictionary containing processing statistics and results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        document_name = document_name or pdf_path.stem
        document_output_dir = self.output_dir / document_name

        logger.info(f"Processing document: {document_name}")
        logger.info(f"PDF path: {pdf_path}")
        logger.info(f"Output directory: {document_output_dir}")
        
        results = {
            "document_name": document_name,
            "pdf_path": str(pdf_path),
            "output_dir": str(document_output_dir),
            "pages_processed": 0,
            "tables_processed": 0,
            "text_blocks_processed": 0,
            "images_processed": 0,
            "errors": []
        }
        
        # Step 1: Parse the PDF
        try:
            self.parser.parse(str(pdf_path), str(document_output_dir))
            logger.info("PDF parsing complete")
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}", exc_info=True)
            results["errors"].append(f"PDF parsing failed: {str(e)}")
            return results
        
        # Step 2: Extract page context metadata
        if extract_metadata:
            try:
                metadata_stats = self._extract_page_metadata(
                    document_output_dir, 
                    start_page=start_page, 
                    end_page=end_page
                )
                results["pages_processed"] = metadata_stats["pages_processed"]
                logger.info(f"Page metadata extraction complete ({metadata_stats['pages_processed']} pages)")
            except Exception as e:
                logger.error(f"Error extracting page metadata: {e}", exc_info=True)
                results["errors"].append(f"Page metadata extraction failed: {str(e)}")
        
        # Step 3: Process tables
        try:
            table_stats = self.table_processor.process_all(
                start_page=start_page,
                end_page=end_page
            )
            results["tables_processed"] = table_stats["total_items"]
            logger.info(f"Table processing complete ({table_stats['total_items']} tables)")
        except Exception as e:
            logger.error(f"Error processing tables: {e}", exc_info=True)
            results["errors"].append(f"Table processing failed: {str(e)}")
        
        # Step 4: Process text blocks
        try:
            text_stats = self.text_processor.process_all(
                start_page=start_page,
                end_page=end_page
            )
            results["text_blocks_processed"] = text_stats["total_items"]
            logger.info(f"Text block processing complete ({text_stats['total_items']} blocks)")
        except Exception as e:
            logger.error(f"Error processing text blocks: {e}", exc_info=True)
            results["errors"].append(f"Text block processing failed: {str(e)}")
        
        # Step 5: Process images
        try:
            image_stats = self.image_processor.process_all(
                start_page=start_page,
                end_page=end_page
            )
            results["images_processed"] = image_stats["total_items"]
            logger.info(f"Image processing complete ({image_stats['total_items']} images)")
        except Exception as e:
            logger.error(f"Error processing images: {e}", exc_info=True)
            results["errors"].append(f"Image processing failed: {str(e)}")
        
        # Step 6: Index to database (if requested)
        if index_to_db:
            try:
                if not self.db_manager.connected():
                    self.connect_database()
                
                index_stats = self._index_to_database(
                    document_output_dir,
                    document_name,
                    start_page=start_page,
                    end_page=end_page
                )
                results["indexed_points"] = index_stats["total_points"]
                logger.info(f"Database indexing complete ({index_stats['total_points']} points)")
            except Exception as e:
                logger.error(f"Error indexing to database: {e}", exc_info=True)
                results["errors"].append(f"Database indexing failed: {str(e)}")
        
        # Final summary
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Document: {document_name}")
        logger.info(f"Pages processed: {results['pages_processed']}")
        logger.info(f"Tables processed: {results['tables_processed']}")
        logger.info(f"Text blocks processed: {results['text_blocks_processed']}")
        logger.info(f"Images processed: {results['images_processed']}")

        if index_to_db:
            logger.info(f"Points indexed: {results.get('indexed_points', 0)}")
        if results['errors']:
            logger.warning(f"Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                logger.warning(f"  - {error}")
        
        return results
    
    def _extract_page_metadata(
        self,
        document_dir: Path,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract rich metadata for all pages using context."""
        path_builder = PathBuilder(document_dir)
        page_numbers = path_builder.get_all_page_numbers()
        
        if start_page:
            page_numbers = [p for p in page_numbers if p >= start_page]
        if end_page:
            page_numbers = [p for p in page_numbers if p <= end_page]
        
        pages_processed = 0
        for page_number in page_numbers:
            try:
                logger.info(f"Extracting metadata for page {page_number}")
                
                # Extract context metadata
                context_metadata = self.page_context_extractor.extract(
                    page_number=page_number,
                    pdf_base_path=document_dir
                )
                
                # Save context metadata
                context_file = path_builder.get_context_metadata(page_number)
                with open(context_file, "w") as f:
                    json.dump(context_metadata, f, indent=2)
                
                pages_processed += 1
                logger.info(f"Page {page_number} metadata extracted")
                
            except Exception as e:
                logger.error(f"Error processing page {page_number}: {e}", exc_info=True)
        
        return {"pages_processed": pages_processed}
    
    def _index_to_database(
        self,
        document_dir: Path,
        document_name: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> Dict[str, Any]:
        """Index processed content to the vector database using QdrantIndexer."""
        path_builder = PathBuilder(document_dir)
        page_numbers = path_builder.get_all_page_numbers()
        
        if start_page:
            page_numbers = [p for p in page_numbers if p >= start_page]
        if end_page:
            page_numbers = [p for p in page_numbers if p <= end_page]
        
        # Load all page metadata
        page_metadata_list = []
        for page_number in page_numbers:
            try:
                context_file = path_builder.get_context_metadata(page_number)
                if not context_file.exists():
                    logger.warning(f"No context metadata for page {page_number}, skipping")
                    continue
                
                with open(context_file, "r") as f:
                    page_metadata = json.load(f)
                    # Add document name to metadata
                    page_metadata["document_metadata"] = page_metadata.get("document_metadata", {})
                    page_metadata["document_metadata"]["document_name"] = document_name
                    page_metadata_list.append(page_metadata)
                    
            except Exception as e:
                logger.error(f"Error loading metadata for page {page_number}: {e}", exc_info=True)
        
        if not page_metadata_list:
            logger.warning("No page metadata found to index")
            return {"total_points": 0}
        
        # Use QdrantIndexer for batch indexing
        logger.info(f"Initializing indexer with {len(page_metadata_list)} pages")
        indexer = QdrantIndexer(
            page_metadata_list=page_metadata_list,
            qdrant_client=self.db_manager.get_client(),
            collection_name=self.collection_name,
            batch_size=4,  # Process 4 pages at a time
            include_full_metadata=True,
        )
        
        # Create collection and index documents
        indexer.create_collection()
        indexer.index_documents()
        
        # Get collection info
        collection_info = indexer.get_collection_info()
        total_points = collection_info.get("points_count", 0)
        
        return {"total_points": total_points}
    
    def close(self):
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.close()
        logger.info("Pipeline resources cleaned up")


def main():
    """Example usage of the DocumentPipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF documents through the ingestion pipeline")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for processed files")
    parser.add_argument("--collection", default="documents", help="Qdrant collection name")
    parser.add_argument("--no-metadata", action="store_true", help="Skip metadata extraction")
    parser.add_argument("--index", action="store_true", help="Index to vector database")
    parser.add_argument("--start-page", type=int, help="Starting page number")
    parser.add_argument("--end-page", type=int, help="Ending page number")
    parser.add_argument("--model", default="openai/gpt-4o", help="LLM model to use")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentPipeline(
        output_dir=args.output_dir,
        collection_name=args.collection,
        llm_model=args.model
    )
    
    try:
        # Process the document
        results = pipeline.process_document(
            pdf_path=args.pdf_path,
            extract_metadata=not args.no_metadata,
            index_to_db=args.index,
            start_page=args.start_page,
            end_page=args.end_page
        )
        
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    finally:
        pipeline.close()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

