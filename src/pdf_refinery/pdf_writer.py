"""Invisible text overlay for creating searchable PDFs."""

import math

import fitz

from pdf_refinery.ocr_engine import OcrResult


def remove_text_layer(page: fitz.Page) -> bool:
    """Remove existing text from a PDF page, keeping images intact.

    Returns:
        True if text was found and removed, False otherwise.
    """
    text_dict = page.get_text("dict")
    has_text = False
    for block in text_dict["blocks"]:
        if block["type"] == 0:  # text block
            has_text = True
            page.add_redact_annot(fitz.Rect(block["bbox"]), fill=False)
    if has_text:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    return has_text


def overlay_text_on_page(
    page: fitz.Page,
    ocr_results: list[OcrResult],
    image_width: int,
    image_height: int,
) -> int:
    """Overlay invisible text on a PDF page based on OCR results.

    Uses the full polygon from OCR to handle rotated/skewed text accurately.

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
        # top-left, top-right, bottom-right, bottom-left

        # Convert polygon corners to PDF coordinates
        tl = fitz.Point(bbox[0][0] * scale_x, bbox[0][1] * scale_y)
        tr = fitz.Point(bbox[1][0] * scale_x, bbox[1][1] * scale_y)
        bl = fitz.Point(bbox[3][0] * scale_x, bbox[3][1] * scale_y)

        # Compute rotation angle from the top edge
        dx = tr.x - tl.x
        dy = tr.y - tl.y
        angle = math.degrees(math.atan2(dy, dx))

        # Height from left edge (top-left to bottom-left)
        height = math.sqrt((bl.x - tl.x) ** 2 + (bl.y - tl.y) ** 2)
        font_size = height * 0.85

        if font_size < 1:
            continue

        # Baseline origin: offset down from top-left along the left edge
        baseline = fitz.Point(
            tl.x + (bl.x - tl.x) * 0.85,
            tl.y + (bl.y - tl.y) * 0.85,
        )

        # Use morph for arbitrary rotation, snap to fixed angles for simple cases
        abs_angle = abs(angle)
        if abs_angle < 1:
            rotate_val = 0
            morph = None
        elif abs(abs_angle - 90) < 1:
            rotate_val = 90 if angle > 0 else 270
            morph = None
        elif abs(abs_angle - 180) < 1:
            rotate_val = 180
            morph = None
        else:
            rotate_val = 0
            morph = (baseline, fitz.Matrix(1, 0, 0, 1, 0, 0).prerotate(angle))

        page.insert_text(
            point=baseline,
            text=result.text,
            fontsize=font_size,
            render_mode=3,  # invisible text
            rotate=rotate_val,
            morph=morph,
        )
        count += 1

    return count
