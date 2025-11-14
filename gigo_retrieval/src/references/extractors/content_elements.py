"""Extractor for content_elements data source."""

from typing import Any, Dict, List, Tuple

from src.references.extractors.base import ReferenceExtractor
from src.references.models import FigureReference, TableReference


class ContentElementsExtractor(ReferenceExtractor):
    """Extracts references from content_elements."""

    def extract(
        self, result: Dict[str, Any], sub_question: str
    ) -> Tuple[List[TableReference], List[FigureReference]]:
        """Extract tables and figures from content_elements."""
        tables = []
        figures = []
        content_elements = result.get("content_elements", [])

        for element in content_elements:
            element_type = element.get("type")

            if element_type == "table":
                element_id = element.get("element_id")
                if self._is_valid_id(element_id):
                    tables.append(
                        TableReference(
                            sub_question=sub_question,
                            element_id=element_id,
                            page_number=result.get("page_number"),
                        )
                    )

            elif element_type == "figure":
                figure_id = element.get("figure_id")
                if self._is_valid_id(figure_id):
                    figures.append(
                        FigureReference(
                            sub_question=sub_question,
                            label=figure_id,
                            page_number=result.get("page_number"),
                        )
                    )

        return tables, figures

    @staticmethod
    def _is_valid_id(identifier: Any) -> bool:
        """Check if identifier is valid (not None or 'None')."""
        return identifier and identifier != "None"

