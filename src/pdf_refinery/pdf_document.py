"""The document a run reads from and writes to.

Two libraries hold the same file because no one of them does both halves of
the job. PDFium, through ``pypdfium2``, rasterises a page and reads the text
already on it; QPDF, through ``pikepdf``, edits the object tree and writes the
result. Both are permissively licensed, which is the reason this module exists
at all -- see ``library-alternative.md``.

The rendering handle is opened from the file as it stood on disk and never
learns about an edit. That is not a limitation the pipeline has to work
around, it is the order it already works in: a page is rendered before
anything is written to it. :meth:`Page.to_image` says so where it matters.

Everything a caller places on a page is expressed in *page space*: the
coordinate system of the rendered image, with the origin at the top left and y
increasing downwards, measured in points. :attr:`Page.placement` is the one
place that knows how that relates to PDF user space, which is y-up, offset by
the crop box, and turned by ``/Rotate``.
"""

import io
import zlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pikepdf
import pypdfium2
from pikepdf import Array, Dictionary, Name, Stream

from pdf_refinery.fonts import FONTNAME, embed_font

# Name the rendered page is registered under when a page is flattened.
# ``add_resource`` picks another if the page already has one under this name,
# so it is a preference rather than a guarantee; the font's name is
# :data:`fonts.FONTNAME` for the same reason.
IMAGE_RESOURCE = "PdfRefineryPage"


class Matrix(NamedTuple):
    """An affine transform in PDF's own layout: ``[a b c d e f]``.

    Maps ``(x, y)`` to ``(a*x + c*y + e, b*x + d*y + f)``, which is what a
    ``cm`` or ``Tm`` operator takes, so a matrix built here can be written out
    without rearranging.
    """

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def then(self, other: "Matrix") -> "Matrix":
        """This transform followed by ``other``."""
        return Matrix(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
            self.e * other.a + self.f * other.c + other.e,
            self.e * other.b + self.f * other.d + other.f,
        )


def _normalised_box(box) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def open_pdf(source: Path | str | bytes) -> "Document":
    """Open a PDF for reading and editing."""
    return Document(source)


class Document:
    """A PDF open for rendering, text extraction and editing."""

    def __init__(self, source: Path | str | bytes):
        # Both handles read lazily, which is what keeps a book-length scan off
        # the heap. A checkpoint writes a sibling file and renames it into
        # place, so neither handle is reading the file being written.
        opened = io.BytesIO(source) if isinstance(source, bytes) else Path(source)
        self._pdf = pikepdf.Pdf.open(opened)
        self._renderer = pypdfium2.PdfDocument(
            source if isinstance(source, bytes) else opened
        )
        self._text_font: pikepdf.Object | None = None

    @property
    def pdf(self) -> pikepdf.Pdf:
        """The editable object tree."""
        return self._pdf

    @property
    def renderer(self) -> pypdfium2.PdfDocument:
        """The read-only view used for rasterising and text extraction."""
        return self._renderer

    @property
    def text_font(self) -> pikepdf.Object:
        """The invisible layer's font, embedded on first use.

        It lives on the document rather than at the call site because that is
        what keeps one copy of it in the file however many pages draw with it.
        """
        if self._text_font is None:
            self._text_font = embed_font(self._pdf)
        return self._text_font

    def __len__(self) -> int:
        return len(self._pdf.pages)

    def __getitem__(self, index: int) -> "Page":
        return Page(self, index)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def save(self, path: Path, final: bool = False) -> None:
        """Write the document out.

        Resources are swept only on the final save: it walks every page, which
        is wasted work on a checkpoint that will be superseded.
        """
        if final:
            self._pdf.remove_unreferenced_resources()
        self._pdf.save(
            path,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )

    def close(self) -> None:
        self._renderer.close()
        self._pdf.close()


class Page:
    """One page, addressed the same way in both libraries."""

    def __init__(self, document: Document, index: int):
        self._document = document
        self._index = index

    @property
    def document(self) -> Document:
        return self._document

    @property
    def index(self) -> int:
        return self._index

    @property
    def obj(self) -> pikepdf.Page:
        """The page as an editable object."""
        return self._document.pdf.pages[self._index]

    @property
    def _box(self) -> tuple[float, float, float, float]:
        """The area a reader shows, which is what gets rendered.

        The crop box clipped to the media box: a crop box outside it means
        nothing, and readers -- PDFium among them -- fall back to the media box
        rather than showing an empty page.
        """
        page = self.obj
        media = _normalised_box(page.mediabox)
        crop = _normalised_box(page.cropbox)
        x0, y0 = max(media[0], crop[0]), max(media[1], crop[1])
        x1, y1 = min(media[2], crop[2]), min(media[3], crop[3])
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else media

    @property
    def rotation(self) -> int:
        return int(self.obj.rotation) % 360 // 90 * 90

    @property
    def size(self) -> tuple[float, float]:
        """Width and height in page space, in points."""
        x0, y0, x1, y1 = self._box
        width, height = x1 - x0, y1 - y0
        return (height, width) if self.rotation in (90, 270) else (width, height)

    @property
    def placement(self) -> Matrix:
        """Maps page space onto PDF user space.

        Page space is what the caller sees: the rendered image, origin top
        left, y downwards. User space is what the file is written in: origin at
        the media box, y upwards, and then turned by ``/Rotate`` when shown.
        Each quarter turn sends the page-space axes to a different pair of user
        space ones, which is the whole of the table below.
        """
        x0, y0, x1, y1 = self._box
        return {
            0: Matrix(1, 0, 0, -1, x0, y1),
            90: Matrix(0, 1, 1, 0, x0, y0),
            180: Matrix(-1, 0, 0, 1, x1, y0),
            270: Matrix(0, -1, -1, 0, x1, y1),
        }[self.rotation]

    def to_image(self, dpi: int = 300) -> np.ndarray:
        """Render the page to an RGB array of shape ``(height, width, 3)``.

        Rendering reads the file as it was opened, so a page must be rendered
        before it is edited. Every caller here does; nothing renders a page
        twice.
        """
        page = self._document.renderer[self._index]
        bitmap = page.render(scale=dpi / 72.0, rev_byteorder=True)
        return bitmap.to_numpy()

    def text(self) -> str:
        """The text already on the page, as a reader would extract it."""
        textpage = self._document.renderer[self._index].get_textpage()
        try:
            return textpage.get_text_bounded()
        finally:
            textpage.close()

    # -- editing ----------------------------------------------------------

    def add_font(self) -> Name:
        """Reference the text layer's font from this page, returning its name."""
        return self.obj.add_resource(
            self._document.text_font, Name.Font, Name(f"/{FONTNAME}")
        )

    def add_content(self, content: bytes) -> None:
        """Draw ``content`` over whatever the page already draws.

        The existing content is wrapped in ``q``/``Q`` first. Without that, a
        page that leaves a transform in force -- and a scanned page often does,
        to place its image -- would apply it to the text as well and put every
        line somewhere else entirely.
        """
        page = self.obj
        page.contents_add(Stream(self._document.pdf, b"q\n"), prepend=True)
        page.contents_add(Stream(self._document.pdf, b"Q\n" + content))

    def replace_with_image(self, image: np.ndarray) -> None:
        """Discard everything on the page and draw ``image`` in its place.

        Used to make a page that already has text re-readable: rendering it and
        keeping only the rendering strips the text layer while preserving what
        the page looks like, which redaction of the text alone would not -- it
        would take the visible glyphs with it.

        The page keeps the size it had, but not its crop box or ``/Rotate``:
        the rendering is already cropped and already turned, so leaving either
        in place would apply it a second time. Both are overwritten rather
        than deleted, because either can be inherited from the page tree and a
        deleted key would simply expose the one above it.
        """
        pdf = self._document.pdf
        width, height = self.size
        rows, columns = image.shape[:2]

        xobject = Stream(pdf, b"")
        xobject.write(
            zlib.compress(np.ascontiguousarray(image).tobytes(), 6),
            filter=Name.FlateDecode,
        )
        xobject.Type = Name.XObject
        xobject.Subtype = Name.Image
        xobject.Width = columns
        xobject.Height = rows
        xobject.ColorSpace = Name.DeviceRGB
        xobject.BitsPerComponent = 8

        page = self.obj.obj
        page[Name.Contents] = pdf.make_indirect(Stream(
            pdf,
            f"q {width:g} 0 0 {height:g} 0 0 cm /{IMAGE_RESOURCE} Do Q\n".encode("ascii"),
        ))
        page[Name.Resources] = Dictionary(
            XObject=Dictionary(**{IMAGE_RESOURCE: pdf.make_indirect(xobject)})
        )
        page[Name.MediaBox] = Array([0, 0, width, height])
        page[Name.CropBox] = Array([0, 0, width, height])
        page[Name.Rotate] = 0
        if Name.Annots in page:
            del page[Name.Annots]
