"""Invisible text overlay for creating searchable PDFs."""

import math
from typing import NamedTuple

from pdf_refinery.fonts import (
    LINE_ASCENT,
    LINE_DESCENT,
    encode,
    text_length,
    unsupported_chars,
)
from pdf_refinery.ocr_engine import OcrResult
from pdf_refinery.pdf_document import Matrix, Page

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

# Render mode 3 draws neither fill nor stroke: the text is there to be
# extracted and searched, not seen.
INVISIBLE = 3


def has_text(page: Page, min_chars: int = MIN_TEXT_CHARS) -> bool:
    """Return whether the page already carries enough text to leave it alone.

    Args:
        page: The page to inspect.
        min_chars: Characters, whitespace excluded, below which the page is
            treated as having no text layer worth keeping. Zero means any text
            at all counts.
    """
    text = "".join(page.text().split())
    return len(text) >= min_chars if min_chars else bool(text)


def _snap_angle(angle: float) -> float:
    """Round a near-square angle to the exact quarter turn.

    A page scanned straight still yields boxes off by hundredths of a degree.
    Left alone those accumulate into visibly tilted text for no reason.
    """
    quarter = round(angle / 90.0) * 90.0
    return quarter if abs(angle - quarter) < ANGLE_SNAP_TOLERANCE else angle


def _number(value: float) -> str:
    """A PDF numeric literal: no exponent, no trailing noise."""
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-", "-0") else text


class OverlayStats(NamedTuple):
    """What an overlay did with the detections it was given.

    Attributes:
        inserted: Lines written into the text layer.
        too_small: Lines whose box was under a point tall and were dropped.
            Reported rather than discarded quietly, because a document that
            loses many of them is unsearchable in a way no error mentions.
        dropped_chars: Characters the font could not encode; see
            :func:`fonts.unsupported_chars`. Empty for anything the
            recognisers in use here actually produce.
    """

    inserted: int
    too_small: int
    dropped_chars: frozenset[str] = frozenset()


def overlay_text_on_page(
    page: Page,
    ocr_results: list[OcrResult],
    image_width: int,
    image_height: int,
) -> OverlayStats:
    """Overlay invisible text on a PDF page based on OCR results.

    Uses the full polygon from OCR to handle rotated/skewed text accurately,
    and squeezes each line horizontally to the width of its detected box. That
    fitting is not cosmetic: a font's natural advances have nothing to do with
    the scanned line's width, and an overrun line runs off the crop box, where
    text extraction cannot see it. On a Korean book scan, fitting the width
    took the extracted-text error rate from 16.4% to a fraction of that -- the
    recogniser had been right all along and the overlay was throwing the tail
    of every line away.

    Vertically it does the matching thing, sizing and placing each line so the
    span box a viewer reports coincides with the box OCR detected. Because the
    glyphless font declares its own ascent and descent, sizing this way makes
    the two agree exactly rather than approximately -- the font is chosen to
    fit the box instead of the box being trusted to suit the font.

    Args:
        page: The page to modify.
        ocr_results: OCR detection results with bounding boxes in pixel coordinates.
        image_width: Width of the rendered image in pixels.
        image_height: Height of the rendered image in pixels.

    Returns:
        An :class:`OverlayStats` counting what was written and what was not.
    """
    page_width, page_height = page.size
    scale_x = page_width / image_width
    scale_y = page_height / image_height
    placement = page.placement

    # Where the baseline sits between the top and bottom of a detected box, and
    # how tall a line must be set for its span to fill that box. One font
    # writes every script, so both are the same for every line -- the fit is
    # exact rather than a per-script compromise.
    line_height = LINE_ASCENT + LINE_DESCENT
    baseline_ratio = LINE_ASCENT / line_height

    # Collected rather than written as they are found: a page with no
    # detections must not gain a font, a resource entry or a content stream.
    lines: list[tuple[Matrix, float, bytes]] = []
    too_small = 0
    dropped: set[str] = set()
    for result in ocr_results:
        bbox = result.bbox  # [top-left, top-right, bottom-right, bottom-left]

        tl_x, tl_y = bbox[0][0] * scale_x, bbox[0][1] * scale_y
        tr_x, tr_y = bbox[1][0] * scale_x, bbox[1][1] * scale_y
        bl_x, bl_y = bbox[3][0] * scale_x, bbox[3][1] * scale_y

        dx, dy = tr_x - tl_x, tr_y - tl_y
        box_width = math.hypot(dx, dy)
        height = math.hypot(bl_x - tl_x, bl_y - tl_y)

        font_size = height / line_height
        if font_size < 1:
            # Too small to place meaningfully, and the text would be
            # unsearchable wherever it landed. Counted so the run can say so
            # rather than lose it in silence.
            too_small += 1
            continue

        dropped |= unsupported_chars(result.text)
        codes = encode(result.text)
        if not codes:
            continue

        # Written flat and then transformed into place, all in one matrix: the
        # line runs along the box's top edge, is squeezed to the box's width,
        # and starts at the baseline an ascent below that edge. A separate
        # rotation applied afterwards would scale the line across itself
        # instead of along it, which is the bug this shape rules out.
        angle = math.radians(_snap_angle(math.degrees(math.atan2(dy, dx))))
        natural_width = text_length(result.text, font_size)
        width_scale = box_width / natural_width if natural_width > 0 else 1.0
        width_scale = min(max(width_scale, MIN_WIDTH_SCALE), MAX_WIDTH_SCALE)
        cos, sin = math.cos(angle), math.sin(angle)

        text_matrix = Matrix(
            width_scale * cos, width_scale * sin,
            sin, -cos,
            tl_x + (bl_x - tl_x) * baseline_ratio,
            tl_y + (bl_y - tl_y) * baseline_ratio,
        ).then(placement)
        lines.append((text_matrix, font_size, codes))

    if lines:
        name = page.add_font()
        drawn = "\n".join(
            f"{' '.join(_number(v) for v in matrix)} Tm "
            f"{name} {_number(size)} Tf <{codes.hex()}> Tj"
            for matrix, size, codes in lines
        )
        page.add_content(f"BT {INVISIBLE} Tr\n{drawn}\nET\n".encode("ascii"))

    return OverlayStats(
        inserted=len(lines), too_small=too_small, dropped_chars=frozenset(dropped)
    )
