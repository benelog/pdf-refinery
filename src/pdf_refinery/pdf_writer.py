"""Invisible text overlay for creating searchable PDFs."""

import fitz

from pdf_refinery.ocr_engine import OcrResult


def overlay_text_on_page(
    page: fitz.Page,
    ocr_results: list[OcrResult],
    image_width: int,
    image_height: int,
) -> int:
    """Overlay invisible text on a PDF page based on OCR results.

    Args:
        page: The PyMuPDF page to modify.
        ocr_results: OCR detection results with bounding boxes in pixel coordinates.
        image_width: Width of the rendered image in pixels.
        image_height: Height of the rendered image in pixels.

    Returns:
        Number of text blocks inserted.
    """
    page_rect = page.rect
    scale_x = page_rect.width / image_width
    scale_y = page_rect.height / image_height

    count = 0
    for result in ocr_results:
        bbox = result.bbox  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        # Calculate bounding box height in pixels for font size estimation
        top_y = min(bbox[0][1], bbox[1][1])
        bottom_y = max(bbox[2][1], bbox[3][1])
        bbox_height_px = bottom_y - top_y

        # Convert to PDF coordinates
        pdf_x = bbox[0][0] * scale_x
        pdf_y = bottom_y * scale_y  # insert_text uses baseline (bottom-left)
        font_size = bbox_height_px * scale_y * 0.85  # slight reduction for better fit

        if font_size < 1:
            continue

        page.insert_text(
            point=fitz.Point(pdf_x, pdf_y),
            text=result.text,
            fontsize=font_size,
            render_mode=3,  # invisible text
        )
        count += 1

    return count
