import json

from src.metadata_extractors.base import BaseMetadataExtractor
from src.llm.litellm_client import LitellmClient
from src.prompts.table_metadata import GENERATE_TABLE_METADATA_PROMPT
from src.schemas import TableMetadataResponse
from src.utils import encode_image_to_data_uri

class TableMetadataExtractor(BaseMetadataExtractor):

    def __init__(self):
        self._llm_client = LitellmClient(model_name="openai/gpt-4o")

    def extract(self, table_html: str, pdf_page: str) -> TableMetadataResponse:
        image_data_uri = encode_image_to_data_uri(pdf_page)

        resp = self._llm_client.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GENERATE_TABLE_METADATA_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {
                            "type": "text",
                            "text": f"<html><body>{table_html}</body></html>",
                        },
                    ],
                },
            ],
            response_format=TableMetadataResponse,
            temperature=0.0,
        )
        metadata_resp_string = resp.choices[0].message.content
        metadata_resp_json = json.loads(metadata_resp_string)
        return metadata_resp_json
