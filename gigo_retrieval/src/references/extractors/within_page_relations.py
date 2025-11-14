"""Extractor for within_page_relations data source."""

from typing import Any, Dict, List, Tuple

from src.references.extractors.base import ReferenceExtractor
from src.references.extractors.content_elements import ContentElementsExtractor
from src.references.models import FigureReference, TableReference


class WithinPageRelationsExtractor(ReferenceExtractor):
    """Extracts figure references from within_page_relations."""

    def extract(
        self, result: Dict[str, Any], sub_question: str
    ) -> Tuple[List[TableReference], List[FigureReference]]:
        """Extract related figures from within_page_relations."""
        figures = []
        content_elements = result.get("content_elements", [])

        for element in content_elements:
            within_page_relations = element.get("within_page_relations", {})
            related_figures = within_page_relations.get("related_figures", [])

            for figure in related_figures:
                label = figure.get("label")
                if ContentElementsExtractor._is_valid_id(label):
                    figures.append(
                        FigureReference(
                            sub_question=sub_question,
                            label=label,
                            page_number=result.get("page_number"),
                        )
                    )

        return [], figures

