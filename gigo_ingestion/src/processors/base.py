from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):

    @abstractmethod
    def process(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    

class BaseBatchProcessor(ABC):

    @abstractmethod
    def process_batch(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def should_process_page(self, page_number: int, context_metadata: Dict[str, Any]) -> bool:
        """
        Determine if a page should be processed.

        Args:
            page_number: The page number
            context_metadata: The page's context metadata

        Returns:
            True if the page should be processed
        """
        pass

    @abstractmethod
    def get_items_to_process(self, page_number: int) -> List[Path]:
        """
        Get the list of items (files) to process for a page.

        Args:
            page_number: The page number

        Returns:
            List of file paths to process
        """
        pass

    @abstractmethod
    def process_item(
        self, item_path: Path, page_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single item (table, text block, etc.).

        Args:
            item_path: Path to the item file
            page_number: The page number

        Returns:
            Metadata dictionary, or None if processing failed
        """
        pass

    @abstractmethod
    def get_metadata_key(self) -> str:
        """
        Get the key used to store metadata in context_metadata.json.

        Returns:
            Metadata key (e.g., "table_metadata", "text_metadata")
        """
        pass

    def load_context_metadata(self, page_number: int) -> Optional[Dict[str, Any]]:
        """
        Load context metadata for a page.

        Args:
            page_number: The page number

        Returns:
            Context metadata dictionary, or None if not found
        """
        context_file = self.path_builder.get_context_metadata(page_number)
        if not context_file.exists():
            return None

        try:
            with open(context_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading context metadata for page {page_number}: {e}")
            return None

    def save_context_metadata(
        self, page_number: int, context_metadata: Dict[str, Any]
    ) -> bool:
        """
        Save context metadata for a page.

        Args:
            page_number: The page number
            context_metadata: The metadata to save

        Returns:
            True if successful, False otherwise
        """
        context_file = self.path_builder.get_context_metadata(page_number)

        try:
            with open(context_file, "w") as f:
                json.dump(context_metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving context metadata for page {page_number}: {e}")
            return False
    
    def process_page(self, page_number: int) -> int:
        """
        Process all items for a single page.

        Args:
            page_number: The page number to process

        Returns:
            Number of items successfully processed
        """
        # Load context metadata
        context_metadata = self.load_context_metadata(page_number)
        if not context_metadata:
            logger.warning(f"No context metadata for page {page_number}")
            return 0

        # Check if page should be processed
        if not self.should_process_page(page_number, context_metadata):
            logger.debug(f"Skipping page {page_number} (filter criteria not met)")
            return 0

        logger.info(f"Processing page {page_number}...")

        # Get items to process
        items = self.get_items_to_process(page_number)
        if not items:
            logger.warning(f"No items found for page {page_number}")
            return 0

        # Process each item
        metadata_list = []
        for item_path in sorted(items):
            item_name = item_path.stem
            logger.info(f"  Processing {item_name}...")

            try:
                item_metadata = self.process_item(item_path, page_number)
                if item_metadata:
                    metadata_list.append(item_metadata)
                    logger.info(f"Generated metadata for {item_name}")
                else:
                    logger.warning(f"No metadata generated for {item_name}")
            except Exception as e:
                logger.error(f"Error processing {item_name}: {e}")

        # Save metadata
        if metadata_list:
            context_metadata[self.get_metadata_key()] = metadata_list
            if self.save_context_metadata(page_number, context_metadata):
                self.logger.info(
                    f"Added {len(metadata_list)} item(s) to context metadata"
                )
            else:
                logger.error(f"Failed to save context metadata")

        return len(metadata_list)
    
    def process_all(
        self, start_page: Optional[int] = None, end_page: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process all pages in the document.

        Args:
            start_page: Optional starting page number (inclusive)
            end_page: Optional ending page number (inclusive)

        Returns:
            Summary statistics (total_pages, processed_pages, total_items)
        """
        page_numbers = self.path_builder.get_all_page_numbers()

        # Filter by range if specified
        if start_page:
            page_numbers = [p for p in page_numbers if p >= start_page]
        if end_page:
            page_numbers = [p for p in page_numbers if p <= end_page]

        logger.info(f"Found {len(page_numbers)} pages to process")

        total_items = 0
        processed_pages = 0

        for page_number in page_numbers:
            items_processed = self.process_page(page_number)
            if items_processed > 0:
                processed_pages += 1
                total_items += items_processed

        summary = {
            "total_pages": len(page_numbers),
            "processed_pages": processed_pages,
            "total_items": total_items,
        }

        logger.info(f"Processing complete: {summary}")
        return summary

