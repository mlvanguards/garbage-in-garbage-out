"""Prompt templates for various LLM tasks."""

from src.prompts.context_metadata import METADATA_PROMPT
from src.prompts.image_interpretation import IMAGE_INTERPRETATION_PROMPT
from src.prompts.table_flattening import FLATTEN_TABLE_PROMPT
from src.prompts.table_metadata import GENERATE_TABLE_METADATA_PROMPT

# TODO: GENERATE_TEXT_METADATA_PROMPT is referenced in text_extractor.py
# but doesn't exist yet. Need to create src/prompts/text_metadata.py

__all__ = [
    "METADATA_PROMPT",
    "IMAGE_INTERPRETATION_PROMPT",
    "FLATTEN_TABLE_PROMPT",
    "GENERATE_TABLE_METADATA_PROMPT",
]

