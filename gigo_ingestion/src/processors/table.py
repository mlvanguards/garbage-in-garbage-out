from typing import List, Optional, Dict, Any
from pathlib import Path

from src.processors.base import BaseProcessor, BaseBatchProcessor
from src.prompts.table_flattening import FLATTEN_TABLE_PROMPT
from src.llm import LitellmClient
from src.metadata_extractors.table_extractor import TableMetadataExtractor


class FlattenTableProcessor(BaseProcessor):

    def __init__(self):
        self._llm_client = LitellmClient()

    def process(self, content: str) -> str:
        resp = self._llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical assistant that transforms technical tables into compact but complete summaries. Your goal is to produce a single paragraph that retains all essential information from the table, so nothing is lost during this flattening process. Your outputs will be embedded into a vector index for retrieval.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": FLATTEN_TABLE_PROMPT.format(table_html=content),
                        },
                        {
                            "type": "text",
                            "text": f"<html><body>{content}</body></html>",
                        },
                    ],
                },
            ],
            response_format=None,
        )
        return resp.choices[0].message.content
    

class TableBatchProcessor(BaseBatchProcessor):
    """
    Batch processor for tables across multiple pages.

    This processor scans all pages, identifies those with tables,
    and extracts metadata for each table found.
    """

    def should_process_page(
        self, page_number: int, context_metadata: Dict[str, Any]
    ) -> bool:
        """Process pages that have tables."""
        return context_metadata.get("has_tables", False)

    def get_items_to_process(self, page_number: int) -> List[Path]:
        """Get all HTML table files for a page."""
        tables_dir = self.path_builder.get_tables_dir(page_number)
        if not tables_dir.exists():
            return []

        return list(tables_dir.glob("table-*.html"))

    def process_item(
        self, item_path: Path, page_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single table file.

        Args:
            item_path: Path to the HTML table file
            page_number: The page number

        Returns:
            Table metadata dictionary, or None if processing failed
        """
        try:
            # Read HTML content
            with open(item_path, "r") as f:
                html_content = f.read()

            # Get corresponding PNG file
            png_file = item_path.with_suffix(".png")
            if not png_file.exists():
                self.logger.warning(f"PNG file not found for {item_path.name}")
                return None

            # Load image as data URI
            image_data_uri = self.loader.load_image_data_uri(png_file)
            if not image_data_uri:
                self.logger.warning(f"Could not load image for {item_path.name}")
                return None

            # Extract metadata
            extractor = TableMetadataExtractor(self.litellm_client)
            table_metadata = extractor.extract(
                html_content=html_content, image_data_uri=image_data_uri
            )

            # Add file identifiers
            table_metadata["table_id"] = item_path.stem
            table_metadata["table_file"] = str(item_path.name)
            table_metadata["table_image"] = str(png_file.name)

            return table_metadata

        except Exception as e:
            self.logger.error(f"Error processing table {item_path.name}: {e}")
            return None

    def get_metadata_key(self) -> str:
        """Tables are stored under 'table_metadata' key."""
        return "table_metadata"
    
if __name__ == "__main__":
    processor = TableBatchProcessor(Path("scratch/service_manual_long"))
    processor.process_all(start_page=1, end_page=10)
