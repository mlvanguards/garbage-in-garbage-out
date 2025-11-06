from pathlib import Path
from typing import List, Dict, Any, Optional
import base64

from src.processors.base import BaseProcessor, BaseBatchProcessor
from src.prompts.image_interpretation import IMAGE_INTERPRETATION_PROMPT
from src.llm import LitellmClient

class ImageProcessor(BaseProcessor):
    """Process individual images with specific questions."""

    def __init__(self):
        self._llm_client = LitellmClient()

    def process(self, image_path: str, question: str) -> str:
        image_data_uri = self._encode_image(Path(image_path))

        response = self._llm_client.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_INTERPRETATION_PROMPT.format(question=question)},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                },
            ],
            response_format=None,
        )

        return response.choices[0].message.content

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_image}"


class ImageBatchProcessor(BaseBatchProcessor):
    """
    Batch processor for images across multiple pages.
    
    This processor scans all pages, identifies those with images,
    and extracts metadata for each image found.
    """

    def should_process_page(
        self, page_number: int, context_metadata: Dict[str, Any]
    ) -> bool:
        """Process pages that have images."""
        return context_metadata.get("has_figures", False)

    def get_items_to_process(self, page_number: int) -> List[Path]:
        """Get all image files for a page."""
        images_dir = self.path_builder.get_images_dir(page_number)
        if not images_dir.exists():
            return []

        return list(images_dir.glob("image-*.png"))

    def process_item(
        self, item_path: Path, page_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image file.

        Args:
            item_path: Path to the image file
            page_number: The page number

        Returns:
            Image metadata dictionary, or None if processing failed
        """
        try:
            # Load image as data URI
            image_data_uri = self.loader.load_image_data_uri(item_path)
            if not image_data_uri:
                self.logger.warning(f"Could not load image {item_path.name}")
                return None

            # Generate description using LLM
            prompt = "Describe this image in detail. What does it show? What are the key elements?"
            
            response = self.litellm_client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                        ],
                    },
                ],
                response_format=None,
            )

            description = response.choices[0].message.content

            # Create metadata
            image_metadata = {
                "image_id": item_path.stem,
                "image_file": str(item_path.name),
                "description": description,
            }

            return image_metadata

        except Exception as e:
            self.logger.error(f"Error processing image {item_path.name}: {e}")
            return None

    def get_metadata_key(self) -> str:
        """Images are stored under 'image_metadata' key."""
        return "image_metadata"