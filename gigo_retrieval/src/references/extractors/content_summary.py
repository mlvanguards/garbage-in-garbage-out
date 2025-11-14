"""Extractor for content_summary data source."""

from typing import Any, Dict, List, Tuple

from src.references.extractors.base import ReferenceExtractor
from src.references.extractors.content_elements import ContentElementsExtractor
from src.references.models import FigureReference, TableReference


class ContentSummaryExtractor(ReferenceExtractor):
    """Extracts figure references from content_summary."""

    def extract(
        self, result: Dict[str, Any], sub_question: str
    ) -> Tuple[List[TableReference], List[FigureReference]]:
        """Extract figures from content_summary."""
        figures = []
        content_summary = result.get("content_summary", {})
        summary_figures = content_summary.get("figures", [])

        for figure in summary_figures:
            if ContentElementsExtractor._is_valid_id(figure):
                figures.append(
                    FigureReference(
                        sub_question=sub_question,
                        label=figure,
                        page_number=result.get("page_number"),
                    )
                )

        return [], figures

