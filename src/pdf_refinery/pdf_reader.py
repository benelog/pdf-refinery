"""PDF page extraction using PyMuPDF."""

from pathlib import Path

import fitz
import numpy as np


def open_pdf(path: Path) -> fitz.Document:
    """Open a PDF document."""
    return fitz.open(path)


def page_to_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    """Render a PDF page to a numpy array (RGB).

    Args:
        page: A PyMuPDF page object.
        dpi: Resolution for rendering.

    Returns:
        Numpy array of shape (height, width, 3) in RGB format.
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    )
    return img
