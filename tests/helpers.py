"""Plumbing shared by the PDF tests.

Input documents are built with PyMuPDF and outputs are read back with it.
That is the point rather than a convenience: nothing in the shipped code uses
MuPDF any more, so a document only PDFium can render, or a text layer only the
writer that produced it can decode, fails here instead of in whatever reader
the user actually has.
"""

import fitz
import numpy as np
import pypdfium2

from pdf_refinery.pdf_document import Document, open_pdf


def source_pdf(
    pages: int = 1,
    width: float = 612,
    height: float = 792,
    rotation: int = 0,
    cropbox: tuple[float, float, float, float] | None = None,
) -> bytes:
    """A blank PDF, as bytes."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=width, height=height)
        if rotation:
            page.set_rotation(rotation)
        if cropbox:
            page.set_cropbox(fitz.Rect(*cropbox))
    data = doc.tobytes()
    doc.close()
    return data


def source_pdf_with_text(
    lines: list[str],
    width: float = 612,
    height: float = 792,
    fontsize: float = 12,
    **insert,
) -> bytes:
    """A PDF whose page carries real, engine-written text."""
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    for index, line in enumerate(lines):
        page.insert_text(
            fitz.Point(50, 100 + index * (fontsize + 2)),
            line,
            fontsize=fontsize,
            **insert,
        )
    data = doc.tobytes()
    doc.close()
    return data


def opened(data: bytes) -> Document:
    """Open a PDF built above with the code under test."""
    return open_pdf(data)


def written(doc: Document, path) -> "fitz.Document":
    """Save ``doc``, close it, and hand back what an independent reader sees."""
    doc.save(path, final=True)
    doc.close()
    return fitz.open(path)


def spans(page: "fitz.Page") -> list[dict]:
    """Every text span on a page, as MuPDF reports it."""
    return [
        span
        for block in page.get_text("dict")["blocks"] if block["type"] == 0
        for line in block["lines"]
        for span in line["spans"]
    ]


def span_bounds(page: "fitz.Page") -> tuple[float, float, float, float]:
    """The rectangle every span on the page fits inside."""
    boxes = [span["bbox"] for span in spans(page)]
    assert boxes, "nothing was written"
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def pdfium_text(path) -> str:
    """What PDFium extracts, which is the engine most viewers are built on."""
    reader = pypdfium2.PdfDocument(path)
    try:
        textpage = reader[0].get_textpage()
        try:
            return textpage.get_text_bounded()
        finally:
            textpage.close()
    finally:
        reader.close()


def ink(image: np.ndarray) -> int:
    """How many dark pixels an image has, i.e. how much is drawn on it."""
    return int((image < 128).sum())
