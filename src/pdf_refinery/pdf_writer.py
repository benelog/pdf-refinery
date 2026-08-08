"""Invisible text overlay for creating searchable PDFs."""

import math
from typing import NamedTuple

import fitz

from pdf_refinery.fonts import FontResolver, span_metrics, text_length
from pdf_refinery.ocr_engine import OcrResult

ANGLE_SNAP_TOLERANCE = 1.0  # degrees

# Bounds on how far a line may be squeezed or stretched to match its detected
# box. Fitting the width is what keeps a line inside the page -- text drawn
# past the crop box is not extracted at all, so an unfitted overlay silently
# loses the tail of every line. The clamp only guards against a degenerate
# box: a ratio outside this range means the detection, not the text, is wrong,
# and distorting the line to match would misplace it rather than fix it.
MIN_WIDTH_SCALE = 0.1
MAX_WIDTH_SCALE = 10.0

# Scanned pages routinely carry a stray text layer that is not the body text: a
# page number stamped by a scanner, a running header, a watermark. Counting
# those as "already searchable" would leave the whole page unsearchable, so a
# page has to hold more than a caption's worth of text before it is skipped.
MIN_TEXT_CHARS = 100


def has_text(page: fitz.Page, min_chars: int = MIN_TEXT_CHARS) -> bool:
    """Return whether the page already carries enough text to leave it alone.

    Args:
        page: The page to inspect.
        min_chars: Characters, whitespace excluded, below which the page is
            treated as having no text layer worth keeping. Zero means any text
            at all counts.
    """
    text = "".join(page.get_text().split())
    return len(text) >= min_chars if min_chars else bool(text)


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


def _snap_angle(angle: float) -> float:
    """Round a near-square angle to the exact quarter turn.

    A page scanned straight still yields boxes off by hundredths of a degree.
    Left alone those accumulate into visibly tilted text for no reason.
    """
    quarter = round(angle / 90.0) * 90.0
    return quarter if abs(angle - quarter) < ANGLE_SNAP_TOLERANCE else angle


class OverlayStats(NamedTuple):
    """What an overlay did with the detections it was given.

    Attributes:
        inserted: Lines written into the text layer.
        too_small: Lines whose box was under a point tall and were dropped.
            Reported rather than discarded quietly, because a document that
            loses many of them is unsearchable in a way no error mentions.
    """

    inserted: int
    too_small: int


def overlay_text_on_page(
    page: fitz.Page,
    ocr_results: list[OcrResult],
    image_width: int,
    image_height: int,
    font_resolver: FontResolver | None = None,
) -> OverlayStats:
    """Overlay invisible text on a PDF page based on OCR results.

    Uses the full polygon from OCR to handle rotated/skewed text accurately,
    and squeezes each line horizontally to the width of its detected box. That
    fitting is not cosmetic: a font's natural advances have nothing to do with
    the scanned line's width, and an overrun line runs off the crop box, where
    ``get_text()`` cannot see it. On a Korean book scan, fitting the width took
    the extracted-text error rate from 16.4% to a fraction of that -- the
    recogniser had been right all along and the overlay was throwing the tail
    of every line away.

    Vertically it does the matching thing, sizing and placing each line so the
    span box a viewer reports coincides with the box OCR detected. Both numbers
    come from :func:`fonts.span_metrics` rather than a single ratio, because
    one ratio cannot suit two scripts and, for the built-in CJK fonts, does not
    even match what the font declares.

    Args:
        page: The PyMuPDF page to modify.
        ocr_results: OCR detection results with bounding boxes in pixel coordinates.
        image_width: Width of the rendered image in pixels.
        image_height: Height of the rendered image in pixels.
        font_resolver: Chooses a font per line so non-Latin text stays
            extractable. Defaults to built-in fonts selected by script.

    Returns:
        An :class:`OverlayStats` counting what was written and what was not.
    """
    if font_resolver is None:
        font_resolver = FontResolver()
    page_rect = page.rect
    scale_x = page_rect.width / image_width
    scale_y = page_rect.height / image_height

    count = 0
    too_small = 0
    for result in ocr_results:
        bbox = result.bbox  # [top-left, top-right, bottom-right, bottom-left]

        tl = fitz.Point(bbox[0][0] * scale_x, bbox[0][1] * scale_y)
        tr = fitz.Point(bbox[1][0] * scale_x, bbox[1][1] * scale_y)
        bl = fitz.Point(bbox[3][0] * scale_x, bbox[3][1] * scale_y)

        dx = tr.x - tl.x
        dy = tr.y - tl.y
        angle = math.degrees(math.atan2(dy, dx))
        box_width = math.hypot(dx, dy)

        height = math.sqrt((bl.x - tl.x) ** 2 + (bl.y - tl.y) ** 2)

        # Size and place the line so the span a viewer reports covers the
        # detected box exactly. Both numbers come from the font's measured
        # ascent and descent rather than a constant: a single ratio has to be
        # wrong for one script or the other, and for the built-in CJK fonts it
        # would be wrong against the declared metrics too. See
        # fonts.span_metrics.
        font = font_resolver.resolve(result.text)
        ascent, descent = span_metrics(font)
        font_size = height / (ascent + descent)
        baseline_ratio = ascent / (ascent + descent)

        if font_size < 1:
            # Too small to place meaningfully, and the text would be
            # unsearchable wherever it landed. Counted so the run can say so
            # rather than lose it in silence.
            too_small += 1
            continue

        baseline = fitz.Point(
            tl.x + (bl.x - tl.x) * baseline_ratio,
            tl.y + (bl.y - tl.y) * baseline_ratio,
        )

        natural_width = text_length(result.text, font, font_size)
        width_scale = box_width / natural_width if natural_width > 0 else 1.0
        width_scale = min(max(width_scale, MIN_WIDTH_SCALE), MAX_WIDTH_SCALE)

        # One transform for every orientation: the line is always drawn
        # horizontally and then scaled along itself and turned into place. The
        # ``rotate`` argument is deliberately not used -- it turns the line
        # after the fact, so a horizontal scale would then act across the line
        # instead of along it.
        fit = (fitz.Matrix(width_scale, 0, 0, 1, 0, 0)
               * fitz.Matrix(1, 0, 0, 1, 0, 0).prerotate(-_snap_angle(angle)))

        page.insert_text(
            point=baseline,
            text=result.text,
            fontsize=font_size,
            fontname=font.name,
            fontfile=font.file,
            render_mode=3,  # invisible text
            morph=(baseline, fit),
        )
        count += 1

    return OverlayStats(inserted=count, too_small=too_small)
