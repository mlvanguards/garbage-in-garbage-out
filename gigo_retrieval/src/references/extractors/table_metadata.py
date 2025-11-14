"""Extractor for table_metadata data source."""

from typing import Any, Dict, List, Tuple

from src.references.extractors.base import ReferenceExtractor
from src.references.extractors.content_elements import ContentElementsExtractor
from src.references.models import FigureReference, TableReference


class TableMetadataExtractor(ReferenceExtractor):
    """Extracts table references from table_metadata."""

    def extract(
        self, result: Dict[str, Any], sub_question: str
    ) -> Tuple[List[TableReference], List[FigureReference]]:
        """Extract tables from table_metadata."""
        tables = []
        table_metadata = result.get("table_metadata", [])

        for table in table_metadata:
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

