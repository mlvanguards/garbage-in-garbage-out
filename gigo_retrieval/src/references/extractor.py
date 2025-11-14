"""Main extraction function using extractors."""

from pathlib import Path
from typing import Any, Dict, List

from src.references.extractors import (
    ContentElementsExtractor,
    ContentSummaryExtractor,
    FlattenedTablesExtractor,
    ReferenceExtractor,
    TableMetadataExtractor,
    WithinPageRelationsExtractor,
)
from src.references.models import FigureReference, References, TableReference


def correlate_references_with_files(
    references: References, scratch_path: str = "scratch/service_manual_long"
) -> References:
    """Correlate references with actual files on disk."""
    scratch_dir = Path(scratch_path)

    # Correlate tables
    for table in references.tables:
        if table.page_number and table.element_id:
            page_dir = scratch_dir / f"page_{table.page_number}"
            table_png_path = page_dir / "tables" / f"{table.element_id}.png"
            table_html_path = page_dir / "tables" / f"{table.element_id}.html"

            if table_png_path.exists():
                table.png_file = str(table_png_path)
            if table_html_path.exists():
                table.html_file = str(table_html_path)

    # Correlate figures
    for figure in references.figures:
        if figure.page_number and figure.label:
            page_dir = scratch_dir / f"page_{figure.page_number}"
            # Try different naming conventions
            possible_paths = [
                page_dir / "images" / f"{figure.label}.png",
                page_dir
                / "images"
                / f"image-{figure.page_number}-{figure.label.split('-')[-1]}.png",
            ]

            for figure_png_path in possible_paths:
                if figure_png_path.exists():
                    figure.png_file = str(figure_png_path)
                    break

    return references


def deduplicate_references(references: References) -> References:
    """Remove duplicate references based on unique identifiers."""
    # Deduplicate tables
    seen_tables = set()
    unique_tables = []
    for table in references.tables:
        key = f"{table.element_id}_{table.page_number}"
        if key not in seen_tables:
            seen_tables.add(key)
            unique_tables.append(table)

    # Deduplicate figures
    seen_figures = set()
    unique_figures = []
    for figure in references.figures:
        key = f"{figure.label}_{figure.page_number}"
        if key not in seen_figures:
            seen_figures.add(key)
            unique_figures.append(figure)

    return References(tables=unique_tables, figures=unique_figures)


def extract_tables_and_figures_references(
    relevant_points: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Extract all tables and figures as references from the relevant points structure.

    Args:
        relevant_points: Dictionary containing retrieved points with metadata

    Returns:
        Dictionary with 'tables' and 'figures' lists containing reference information
    """
    # Initialize extractors
    extractors: List[ReferenceExtractor] = [
        ContentElementsExtractor(),
        FlattenedTablesExtractor(),
        TableMetadataExtractor(),
        ContentSummaryExtractor(),
        WithinPageRelationsExtractor(),
    ]

    all_tables: List[TableReference] = []
    all_figures: List[FigureReference] = []

    # Process each sub-question's results
    for sub_question, results in relevant_points.items():
        for result in results:
            # Apply all extractors
            for extractor in extractors:
                tables, figures = extractor.extract(result, sub_question)
                all_tables.extend(tables)
                all_figures.extend(figures)

    # Create references object
    references = References(tables=all_tables, figures=all_figures)

    # Correlate with actual files on disk
    references = correlate_references_with_files(references)

    # Deduplicate references
    references = deduplicate_references(references)

    # Convert back to dict format for backward compatibility
    return {
        "tables": [table.model_dump() for table in references.tables],
        "figures": [figure.model_dump() for figure in references.figures],
    }

