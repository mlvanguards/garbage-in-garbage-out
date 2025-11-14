"""Base extractor class for reference extraction."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from src.references.models import FigureReference, TableReference


class ReferenceExtractor(ABC):
    """Abstract base class for extracting references from result data."""

    @abstractmethod
    def extract(
        self, result: Dict[str, Any], sub_question: str
    ) -> Tuple[List[TableReference], List[FigureReference]]:
        """Extract tables and figures from a result."""
        pass

