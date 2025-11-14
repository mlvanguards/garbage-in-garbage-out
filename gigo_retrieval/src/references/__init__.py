"""Reference extraction, correlation, and deduplication modules."""

from src.references.extractor import extract_tables_and_figures_references
from src.references.models import FigureReference, References, TableReference

__all__ = [
    "TableReference",
    "FigureReference",
    "References",
    "extract_tables_and_figures_references",
]

