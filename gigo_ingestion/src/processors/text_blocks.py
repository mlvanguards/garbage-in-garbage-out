from typing import List, Optional, Dict, Any
from pathlib import Path
from src.processors.base import BaseBatchProcessor
from src.metadata_extractors.text_extractor import TextMetadataExtractor


class TextBatchProcessor(BaseBatchProcessor):
    """
    Batch processor for text blocks across multiple pages.

    This processor scans all pages, identifies those with text blocks,
    and extracts metadata for each text block found.
    """

    def should_process_page(
        self, page_number: int, context_metadata: Dict[str, Any]
    ) -> bool:
        """Process pages that have text blocks."""
        return context_metadata.get("has_text_blocks", False)

    def get_items_to_process(self, page_number: int) -> List[Path]:
        """Get all text files for a page."""
        text_dir = self.path_builder.get_text_dir(page_number)
        if not text_dir.exists():
            return []

        return list(text_dir.glob("page_*_text.txt"))

    def process_item(
        self, item_path: Path, page_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single text file.

        Args:
            item_path: Path to the text file
            page_number: The page number

        Returns:
            Text metadata dictionary, or None if processing failed
        """
        try:
            # Read text content
            with open(item_path, "r") as f:
                text_content = f.read()

            if not text_content.strip():
                self.logger.warning(f"Empty text file: {item_path.name}")
                return None

            # Get corresponding page image
            page_image = self.path_builder.get_page_image(page_number)
            if not page_image.exists():
                self.logger.warning(f"Page image not found for {item_path.name}")
                return None

            # Load image as data URI
            image_data_uri = self.loader.load_image_data_uri(page_image)
            if not image_data_uri:
                self.logger.warning(f"Could not load image for {item_path.name}")
                return None

            # Extract metadata
            extractor = TextMetadataExtractor()
            text_metadata = extractor.extract(
                text_content=text_content, image_data_uri=image_data_uri
            )

            # Add file identifiers
            text_metadata["text_id"] = item_path.stem
            text_metadata["text_file"] = str(item_path.name)

            return text_metadata

        except Exception as e:
            self.logger.error(f"Error processing text {item_path.name}: {e}")
            return None

    def get_metadata_key(self) -> str:
        """Text blocks are stored under 'text_metadata' key."""
        return "text_metadata"


if __name__ == "__main__":
    processor = TextBatchProcessor(Path("scratch/service_manual_long"))
    processor.process_all(start_page=1, end_page=10)