
import json

from typing import List, Optional, Dict, Any
from pathlib import Path
from src.metadata_extractors.base import BaseMetadataExtractor
from src.processors.base import BaseBatchProcessor
from src.llm.litellm_client import LitellmClient
from src.prompts import GENERATE_TEXT_METADATA_PROMPT
from src.schemas import TextMetadataResponse
from src.utils import encode_image_to_data_uri


class TextMetadataExtractor(BaseMetadataExtractor):

    def __init__(self):
        self._llm_client = LitellmClient(model_name="openai/gpt-4o")

    def extract(self, text_content: str, pdf_page: str) -> TextMetadataResponse:
        """Generate metadata for a text block."""
        image_data_uri = encode_image_to_data_uri(pdf_page)

        resp = self._llm_client.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GENERATE_TEXT_METADATA_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {
                            "type": "text",
                            "text": f"Text content:\n{text_content}",
                        },
                    ],
                },
            ],
            response_format=TextMetadataResponse,
            temperature=0.0,
        )
        metadata_resp_string = resp.choices[0].message.content
        metadata_resp_json = json.loads(metadata_resp_string)
        return metadata_resp_json
    