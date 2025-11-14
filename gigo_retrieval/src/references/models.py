"""Pydantic models for reference data structures."""

from typing import List, Optional

from pydantic import BaseModel


class TableReference(BaseModel):
    """Represents a table reference with metadata."""

    sub_question: str
    element_id: str
    page_number: Optional[int] = None
    png_file: Optional[str] = None
    html_file: Optional[str] = None


class FigureReference(BaseModel):
    """Represents a figure reference with metadata."""

    sub_question: str
    label: str
    page_number: Optional[int] = None
    png_file: Optional[str] = None


class References(BaseModel):
    """Container for all extracted references."""

    tables: List[TableReference] = []
    figures: List[FigureReference] = []

