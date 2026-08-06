"""Invisible text overlay for creating searchable PDFs."""

import math

import fitz

from pdf_refinery.fonts import FontResolver
from pdf_refinery.ocr_engine import OcrResult

# Approximate ratio of baseline position to total line height in typical fonts
BASELINE_RATIO = 0.85
ANGLE_SNAP_TOLERANCE = 1.0  # degrees


def has_text(page: fitz.Page) -> bool:
    """Return whether the page already carries extractable text."""
    return bool(page.get_text().strip())


def remove_text_layer(page: fitz.Page) -> bool:
    """Remove existing text from a PDF page, keeping images intact.

    Note that this erases *visible* glyphs too, so it is only safe on pages
    whose text is an invisible OCR layer. Use :func:`rasterize_page` when the
    text is part of the page's appearance.

    Returns:
        True if text was found and removed, False otherwise.
    """
    text_dict = page.get_text("dict")
    found = False
    for block in text_dict["blocks"]:
        if block["type"] == 0:  # text block
            found = True
            page.add_redact_annot(fitz.Rect(block["bbox"]), fill=False)
    if found:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    return found


def rasterize_page(page: fitz.Page, dpi: int = 300) -> None:
    """Flatten a page to a single image, preserving its appearance.

    Redaction alone would erase visible text along with the text layer, so the
    page is rendered first and the rendering is drawn back over the stripped
    page. This lets an OCR layer replace real text without losing content.
    """
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    remove_text_layer(page)
    page.insert_image(page.rect, pixmap=pixmap, overlay=True)


def overlay_text_on_page(
    page: fitz.Page,
    ocr_results: list[OcrResult],
    image_width: int,
    image_height: int,
    font_resolver: FontResolver | None = None,
) -> int:
    """Overlay invisible text on a PDF page based on OCR results.

    Uses the full polygon from OCR to handle rotated/skewed text accurately.

    Args:
        page: The PyMuPDF page to modify.
        ocr_results: OCR detection results with bounding boxes in pixel coordinates.
        image_width: Width of the rendered image in pixels.
        image_height: Height of the rendered image in pixels.
        font_resolver: Chooses a font per line so non-Latin text stays
            extractable. Defaults to built-in fonts selected by script.

    Returns:
        Number of text blocks inserted.
    """
    if font_resolver is None:
        font_resolver = FontResolver()
    page_rect = page.rect
    scale_x = page_rect.width / image_width
    scale_y = page_rect.height / image_height

    count = 0
    for result in ocr_results:
        bbox = result.bbox  # [top-left, top-right, bottom-right, bottom-left]

        tl = fitz.Point(bbox[0][0] * scale_x, bbox[0][1] * scale_y)
        tr = fitz.Point(bbox[1][0] * scale_x, bbox[1][1] * scale_y)
        bl = fitz.Point(bbox[3][0] * scale_x, bbox[3][1] * scale_y)

        dx = tr.x - tl.x
        dy = tr.y - tl.y
        angle = math.degrees(math.atan2(dy, dx))

        height = math.sqrt((bl.x - tl.x) ** 2 + (bl.y - tl.y) ** 2)
        font_size = height * BASELINE_RATIO

        if font_size < 1:
            continue

        baseline = fitz.Point(
            tl.x + (bl.x - tl.x) * BASELINE_RATIO,
            tl.y + (bl.y - tl.y) * BASELINE_RATIO,
        )

        abs_angle = abs(angle)
        if abs_angle < ANGLE_SNAP_TOLERANCE:
            rotate_val = 0
            morph = None
        elif abs(abs_angle - 90) < ANGLE_SNAP_TOLERANCE:
            rotate_val = 90 if angle > 0 else 270
            morph = None
        elif abs(abs_angle - 180) < ANGLE_SNAP_TOLERANCE:
            rotate_val = 180
            morph = None
        else:
            rotate_val = 0
            morph = (baseline, fitz.Matrix(1, 0, 0, 1, 0, 0).prerotate(angle))

        font = font_resolver.resolve(result.text)
        page.insert_text(
            point=baseline,
            text=result.text,
            fontsize=font_size,
            fontname=font.name,
            fontfile=font.file,
            render_mode=3,  # invisible text
            rotate=rotate_val,
            morph=morph,
        )
        count += 1

    return count
