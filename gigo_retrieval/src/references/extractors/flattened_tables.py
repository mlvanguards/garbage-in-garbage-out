"""Extractor for flattened_tables data source."""

from typing import Any, Dict, List, Tuple

from src.references.extractors.base import ReferenceExtractor
from src.references.extractors.content_elements import ContentElementsExtractor
from src.references.models import FigureReference, TableReference


class FlattenedTablesExtractor(ReferenceExtractor):
    """Extracts table references from flattened_tables."""

    def extract(
        self, result: Dict[str, Any], sub_question: str
    ) -> Tuple[List[TableReference], List[FigureReference]]:
        """Extract tables from flattened_tables."""
        tables = []
        flattened_tables = result.get("flattened_tables", [])

        for table in flattened_tables:
            table_id = table.get("table_id")
            if ContentElementsExtractor._is_valid_id(table_id):
                tables.append(
                    TableReference(
                        sub_question=sub_question,
                        element_id=table_id,
                        page_number=result.get("page_number"),
                    )
                )

        return tables, []

