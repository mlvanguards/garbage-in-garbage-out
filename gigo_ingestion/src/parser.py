from typing import List, Optional
import logging
import json

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc.document import DoclingDocument
from pathlib import Path

import fitz
from docling.datamodel.document import PictureItem, TextItem

logger = logging.getLogger(__name__)


class DocumentParser:

    def __init__(self, **kwargs):
        # Configure pipeline options
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.pipeline_options.ocr_options = EasyOcrOptions()
        self.pipeline_options.images_scale = 2
        self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.pipeline_options.generate_page_images = True
        self.pipeline_options.generate_picture_images = True

        # Initialize converter with pipeline options
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
        }
        self.converter = DocumentConverter(format_options=format_options, **kwargs)

    def _run_ocr(self, pdf_path: str, page_range: Optional[List[int]] = None) -> DoclingDocument:
        if page_range is None:
            result = self.converter.convert(pdf_path)
        else:
            result = self.converter.convert(pdf_path, page_range=page_range)
        return result.document

    def parse(self, file_path: str, output_dir: str, page_range: Optional[List[int]] = None):
        """
        Parse a PDF document and extract all content.
        
        Args:
            file_path: Path to the PDF file
            output_dir: Base directory for output
            page_range: Optional list of page numbers to process
        """
        with fitz.open(file_path) as pdf_doc:
            total_pages = pdf_doc.page_count
            logger.info(f"Total pages: {total_pages}")

        for page_num in range(1, total_pages + 1):
            logger.info(f"Processing page {page_num}")
            doc = self._run_ocr(
                file_path, page_range=[page_num, page_num]
            )

            doc_name = Path(file_path).stem
            output_path = Path(output_dir) / doc_name
            output_path.mkdir(parents=True, exist_ok=True)

            # Create page folder
            page_folder = output_path / f"page_{page_num}"
            tables_folder = page_folder / "tables"
            images_folder = page_folder / "images"
            text_folder = page_folder / "text"
            tables_folder.mkdir(parents=True, exist_ok=True)
            images_folder.mkdir(parents=True, exist_ok=True)
            text_folder.mkdir(parents=True, exist_ok=True)

            metadata = {
                "page_number": page_num,
                "page_image": f"page_{page_num}_full.png",
                "tables": [],
                "figures": [],
                "text_blocks": [],
            }

            # Save full page as image
            with fitz.open(file_path) as pdf_doc:
                page = pdf_doc[page_num - 1]  # 0-indexed
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                full_page_file = page_folder / f"page_{page_num}_full.png"
                pix.save(str(full_page_file))
                logger.info(f"Saved full page: {full_page_file}")

            # Export tables
            logger.info(f"Found {len(doc.tables)} tables on page {page_num}")
            page_table_idx = 0
            for table in doc.tables:
                page_table_idx += 1
                table_id = f"table-{page_num}-{page_table_idx}"
                metadata["tables"].append(table_id)

                logger.info(f"Exporting {table_id}")
                html = table.export_to_html(doc=doc)
                html_file = tables_folder / f"{table_id}.html"
                html_file.write_text(html)
                logger.info(f"Saved table HTML: {html_file}")

                img = table.get_image(doc)
                png_file = tables_folder / f"{table_id}.png"
                img.save(png_file, "PNG")
                logger.info(f"Saved table PNG: {png_file}")

            # Export figures
            # Export figures
            logger.info(f"Starting figure extraction for page {page_num}...")
            page_figure_idx = 0
            for item, _ in doc.iterate_items():
                if isinstance(item, PictureItem):
                    img = item.get_image(doc)
                    width, height = img.size
                    area = width * height

                    # Filter out icons/small images
                    if area < 400:
                        logger.info(
                            f"Skipping image on page {page_num} due to small size or icon-like shape."
                        )
                        continue

                    page_figure_idx += 1
                    image_id = f"image-{page_num}-{page_figure_idx}"
                    metadata["figures"].append(image_id)

                    logger.info(f"Exporting {image_id}")
                    img_file = images_folder / f"{image_id}.png"
                    img.save(img_file, "PNG")
                    logger.info(f"Saved image: {img_file}")

            # Export text blocks
            logger.info(f"Starting text extraction for page {page_num}...")
            page_text_blocks = []
            text_block_count = 0

            for item, _ in doc.iterate_items():
                if isinstance(item, TextItem):
                    text_block_count += 1
                    text_content = item.text if hasattr(item, "text") else str(item)
                    page_text_blocks.append(text_content)

            if page_text_blocks:
                unified_text = "\n\n".join(page_text_blocks)
                txt_file = text_folder / f"page_{page_num}_text.txt"
                txt_file.write_text(unified_text, encoding="utf-8")
                metadata["text_blocks"].append(txt_file.name)
                logger.info(
                    f"Saved unified text file: {txt_file} ({text_block_count} blocks)"
                )
            else:
                logger.info(f"No text blocks found on page {page_num}")

            # Save metadata for the page
            metadata_file = page_folder / f"metadata_page_{page_num}.json"
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved page metadata: {metadata_file}")

            logger.info(
                f"Page {page_num} complete! Exported {page_table_idx} tables, {page_figure_idx} figures, {text_block_count} text blocks"
            )

        logger.info("All pages processed successfully!")
        

if __name__ == "__main__":
    parser = DocumentParser()
    parser.parse("test.pdf", "output")
    print("Parsing complete")