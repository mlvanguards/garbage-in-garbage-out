"""
Page context metadata extractor.

Extracts comprehensive metadata from a PDF page by analyzing it in context
with its neighboring pages (n-1, n, n+1).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.metadata_extractors.base import BaseMetadataExtractor
from src.llm import LitellmClient
from src.prompts.context_metadata import METADATA_PROMPT
from src.utils import load_image_as_data_uri, load_json_file, load_text_file

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """Data structure for a single page's information."""
    
    page_number: int
    image_data_uri: str
    metadata_content: str
    text_content: str

@dataclass
class PageContext:
    """Data structure containing n-1, n, and n+1 page information."""

    previous_page: Optional[PageData]
    current_page: PageData
    next_page: Optional[PageData]


class PageContextMetadataExtractor(BaseMetadataExtractor):
    """
    Extracts comprehensive metadata from a PDF page using surrounding context.
    
    This extractor analyzes the current page (n) along with its neighboring pages
    (n-1 and n+1) to provide rich, contextually-aware metadata including:
    - Document-level metadata
    - Section and subsection information
    - Content elements (tables, figures, text blocks)
    - Cross-page relationships
    - Within-page relationships
    """

    def __init__(self):
        """Initialize the page context metadata extractor."""
        self._llm_client = LitellmClient(model_name="openai/gpt-4o")

    def extract(self, page_number: int, pdf_base_path: Path) -> dict:
        """
        Extract comprehensive metadata for a page using its context.
        
        Args:
            page_number: The target page number to extract metadata for
            pdf_base_path: Base path to the processed PDF directory
                          (e.g., "data/processed_pdf")
        
        Returns:
            Dictionary containing structured metadata for the page
            
        Raises:
            FileNotFoundError: If required current page files are missing
            ValueError: If page_number is less than 1
        """
        if page_number < 1:
            raise ValueError("Page number must be at least 1")
        
        # Load page data for current, previous, and next pages
        page_n = self._load_page_data(page_number, pdf_base_path, required=True)
        previous_page = self._load_page_data(page_number - 1, pdf_base_path, required=False) if page_number > 1 else self._create_empty_page(page_number - 1)
        next_page = self._load_page_data(page_number + 1, pdf_base_path, required=False)

        content = self._build_prompt_content(
            page_n.image_data_uri,
            previous_page.image_data_uri,
            next_page.image_data_uri,
            page_n.metadata_content,
            previous_page.metadata_content,
            next_page.metadata_content,
            page_n.text_content,
            previous_page.text_content,
            next_page.text_content,
        )

        # Call LLM
        resp = self._llm_client.chat(
            messages=[{"role": "user", "content": content}],
            response_format=None,
            temperature=0.0,
            max_tokens=10000,
        )
        
        # Parse and return response
        metadata_str = resp.choices[0].message.content
        return self._parse_response(metadata_str)
    
    def _load_page_data(
        self, page_number: int, pdf_base_path: Path, required: bool = False
    ) -> PageData:
        """
        Load all data for a single page.
        
        Args:
            page_number: The page number to load
            pdf_base_path: Base path to the processed PDF directory
            required: If True, raise exception on missing files; if False, return empty data
            
        Returns:
            PageData object with loaded content
            
        Raises:
            FileNotFoundError: If required=True and files are missing
        """
        # Build paths
        page_dir = pdf_base_path / f"page_{page_number}"
        image_path = page_dir / f"page_{page_number}_full.png"
        metadata_path = page_dir / f"metadata_page_{page_number}.json"
        text_path = page_dir / "text" / f"page_{page_number}_text.txt"
        
        # Load files using utils functions
        try:
            image_data_uri = load_image_as_data_uri(image_path, required=required)
            metadata_content = load_json_file(metadata_path, required=required)
            text_content = load_text_file(text_path, required=required)
            
            return PageData(
                page_number=page_number,
                image_data_uri=image_data_uri,
                metadata_content=metadata_content,
                text_content=text_content,
            )
        except FileNotFoundError:
            if required:
                raise
            # Return empty page data if not required
            return self._create_empty_page(page_number)
    
    def _create_empty_page(self, page_number: int) -> PageData:
        """
        Create an empty PageData object for missing pages.
        
        Args:
            page_number: The page number
            
        Returns:
            PageData object with empty content
        """
        return PageData(
            page_number=page_number,
            image_data_uri="",
            metadata_content="{}",
            text_content="",
        )
    
    def _build_prompt_content(
        self,
        image_data_uri_n: str,
        image_data_uri_n_1: str,
        image_data_uri_n_plus_1: str,
        metadata_page_n: str,
        metadata_page_n_1: str,
        metadata_page_n_plus_1: str,
        page_n_text: str,
        page_n_1_text: str,
        page_n_plus_1_text: str,
    ) -> list:
        """
        Build the content array for the LLM request.

        Args:
            image_data_uri_n: Data URI for current page image
            image_data_uri_n_1: Data URI for previous page image
            image_data_uri_n_plus_1: Data URI for next page image
            metadata_page_n: Metadata for current page
            metadata_page_n_1: Metadata for previous page
            metadata_page_n_plus_1: Metadata for next page
            page_n_text: Text content for current page
            page_n_1_text: Text content for previous page
            page_n_plus_1_text: Text content for next page

        Returns:
            List of content items for the LLM
        """
        # Build the prompt with replacements
        prompt_text = (
            METADATA_PROMPT.replace("{metadata_page_n_1}", metadata_page_n_1)
            .replace("{metadata_page_n}", metadata_page_n)
            .replace("{metadata_page_n_plus_1}", metadata_page_n_plus_1)
            .replace("{page_n_1_text}", page_n_1_text)
            .replace("{page_n_text}", page_n_text)
            .replace("{page_n_plus_1_text}", page_n_plus_1_text)
        )

        # Build content array with text and images
        content = [{"type": "text", "text": prompt_text}]

        # Add images (only non-empty ones)
        if image_data_uri_n:
            content.append({"type": "image_url", "image_url": {"url": image_data_uri_n}})
        if image_data_uri_n_1:
            content.append({"type": "image_url", "image_url": {"url": image_data_uri_n_1}})
        if image_data_uri_n_plus_1:
            content.append(
                {"type": "image_url", "image_url": {"url": image_data_uri_n_plus_1}}
            )

        return content
    
    def _parse_response(self, metadata_str: str) -> dict:
        """Parse the LLM response, handling markdown code fences."""
        clean_json_str = (
            metadata_str.strip()
            .removeprefix("```json")
            .removesuffix("```")
            .strip()
        )
        return json.loads(clean_json_str)
