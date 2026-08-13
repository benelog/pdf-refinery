"""Tests for the glyphless font the invisible text layer is written with."""

import fitz
import pytest

from pdf_refinery.fonts import (
    ADVANCE_EM,
    LINE_ASCENT,
    LINE_DESCENT,
    build_font,
    encode,
    font_file,
    text_length,
    unsupported_chars,
)
from pdf_refinery.ocr_engine import OcrResult
from pdf_refinery.text_overlay import overlay_text_on_page
from tests.helpers import opened, pdfium_text, source_pdf, spans, written

# One line per script the font is claimed to cover, plus the two the built-in
# fonts it replaced could not write at all.
SAMPLES = [
    "Hello World",
    "café naïve",
    "한글 검색 테스트",
    "日本語のテスト",
    "中文测试",
    "Привет мир",
    "สวัสดีครับ",
    "नमस्ते",
    "Ελληνικά",
    "①②③ ※ 〜 №",
]


LINE_HEIGHT = 25


def _fitted_box(text, top):
    """A box the shape a detector really returns for ``text``.

    Width matters: it is what the overlay squeezes the line to, and a box
    several times wider than the line it holds is not a case any scan
    produces. See :class:`TestStretchingALinePastItsBox`.
    """
    width = text_length(text, LINE_HEIGHT / (LINE_ASCENT + LINE_DESCENT))
    return [[50, top], [50 + width, top],
            [50 + width, top + LINE_HEIGHT], [50, top + LINE_HEIGHT]]


def _document_with(lines, path, page_size=(612, 792), boxes=None):
    """Write ``lines`` into a document through the real overlay and save it."""
    doc = opened(source_pdf(width=page_size[0], height=page_size[1]))
    results = [
        OcrResult(
            text=text,
            confidence=0.9,
            bbox=boxes[index] if boxes else _fitted_box(text, 40 + index * 30),
        )
        for index, text in enumerate(lines)
    ]
    overlay_text_on_page(doc[0], results, *page_size)
    return written(doc, path)


class TestTheFontIsWellFormed:
    def test_an_independent_engine_can_load_it(self):
        assert fitz.Font(fontbuffer=build_font()) is not None

    def test_declared_metrics_are_the_ones_the_overlay_places_by(self):
        font = fitz.Font(fontbuffer=build_font())
        assert font.ascender == pytest.approx(LINE_ASCENT, abs=1e-4)
        assert -font.descender == pytest.approx(LINE_DESCENT, abs=1e-4)

    def test_it_is_rebuilt_identically(self):
        # The font is generated, not shipped, so a document produced twice from
        # the same input has to be byte-identical in this respect too.
        assert build_font() == build_font()

    def test_it_is_built_once(self):
        assert font_file() is font_file()

    def test_it_is_two_glyphs(self):
        """The size of the whole approach, in one number.

        One glyph for ``.notdef`` and one for everything else; the mapping from
        65536 code points onto that one glyph is a ``CIDToGIDMap`` stream of
        repeated bytes rather than 65535 more glyphs and the table of ascending
        offsets that would come with them.
        """
        assert len(build_font()) < 1_000


class TestSpanMetrics:
    def test_the_span_a_reader_reports_matches_the_declared_metrics(self, tmp_path):
        """What the overlay sizes lines by must be what a reader gives back.

        This is the check the old code needed a runtime probe for: PyMuPDF's
        built-in CJK fonts behaved as 1.000/0.200 while declaring 1.043/-0.266,
        which put every Korean line 8% of its height out of place. A font we
        write ourselves has no such gap, and this test is what says so.
        """
        box = [[50, 100], [550, 100], [550, 200], [50, 200]]
        reader = _document_with(["Hxq"], tmp_path / "metrics.pdf", boxes=[box])

        # The overlay sets a line to the size that makes its ascent and
        # descent fill the detected box, so a box 100 points tall means this.
        size = 100 / (LINE_ASCENT + LINE_DESCENT)
        span = spans(reader[0])[0]
        baseline = span["origin"][1]
        assert (baseline - span["bbox"][1]) / size == pytest.approx(LINE_ASCENT, abs=0.01)
        assert (span["bbox"][3] - baseline) / size == pytest.approx(LINE_DESCENT, abs=0.01)
        reader.close()


class TestTextLength:
    def test_width_is_the_character_count_times_a_fixed_advance(self):
        assert text_length("abcd", 10) == pytest.approx(4 * ADVANCE_EM * 10)

    def test_empty_text_has_no_width(self):
        assert text_length("", 12) == 0.0

    def test_it_counts_only_what_gets_written(self):
        # An unencodable character is not drawn, so it must not be measured
        # either -- counting it would squeeze the rest of the line to make room
        # for something that is not there.
        assert text_length("ab\U00020000", 12) == pytest.approx(text_length("ab", 12))


class TestEncode:
    def test_every_code_point_becomes_its_own_two_byte_code(self):
        assert encode("Aé한") == b"\x00\x41\x00\xe9\xd5\x5c"

    def test_astral_code_points_are_left_out(self):
        assert encode("a\U00020000b") == b"\x00\x61\x00\x62"


class TestUnsupportedChars:
    @pytest.mark.parametrize("text", SAMPLES)
    def test_nothing_in_the_bmp_is_reported(self, text):
        assert unsupported_chars(text) == set()

    def test_astral_code_points_are_reported(self):
        # U+20000 is CJK Extension B; a two-byte code cannot address it.
        assert unsupported_chars("漢\U00020000字") == {"\U00020000"}


class TestTextSurvivesTheRoundTrip:
    """The only property that matters: what goes in comes back out."""

    @pytest.mark.parametrize("text", SAMPLES)
    def test_roundtrip_through_both_engines(self, text, tmp_path):
        """A text layer only one engine can read is not a searchable PDF.

        PDFium writes the layer here and MuPDF has no part in the shipped code,
        so agreement between them is what says the ToUnicode CMap is there and
        correct rather than that a reader still had the font to hand.
        """
        path = tmp_path / "roundtrip.pdf"
        reader = _document_with([text], path)
        assert text in reader[0].get_text()
        assert text in pdfium_text(path)
        reader.close()

    def test_a_line_running_down_the_page_survives_too(self, tmp_path):
        """Why the one glyph has an outline instead of being empty.

        PDFium sizes a character from its glyph's bounding box and drops any
        that comes out with no width. With an empty glyph that box is
        zero-height, which does not matter until the line is turned a quarter
        turn -- on a page scanned sideways, or one carrying ``/Rotate 90`` --
        and the zero becomes the width. The whole text layer then extracts as
        nothing in PDFium and as normal everywhere else, which is the kind of
        failure that gets shipped.
        """
        text = "한글 세로 줄"
        # Corners of a line running downwards: top-left, top-right, and so on
        # as the detector sees them once the line is turned.
        box = [[300, 100], [300, 500], [260, 500], [260, 100]]
        path = tmp_path / "turned.pdf"
        reader = _document_with([text], path, page_size=(612, 612), boxes=[box])

        assert text in reader[0].get_text()
        assert text in pdfium_text(path)
        reader.close()


class TestStretchingALinePastItsBox:
    """Where the width fitting stops being free, and for whom.

    Every glyph in this font advances the same half em, which is wrong for
    every script and deliberately so: :func:`text_overlay.overlay_text_on_page`
    scales the line horizontally until it fills the detected box, so the
    advance cancels out. What does not cancel out is what a reader makes of
    the gaps that scaling leaves between characters. MuPDF 1.28 reads a gap
    wide enough as a line break, and a combining mark -- which it expects to
    sit on top of its base rather than after it -- reaches that threshold
    first. The characters all survive; the word they spell does not, so a
    search for it fails in a MuPDF-based viewer.

    PDFium, which is what most viewers are built on, extracts the same file
    intact at any stretch. Nor does a real page reach these numbers: a
    detected box is the width of the line inside it, which is where the
    fixtures above put it. This pins the boundary rather than guarding it.
    """

    # Thai: three of these ten code points are vowel marks that sit above the
    # letter before them (category Mn) rather than beside it.
    TEXT = "สวัสดีครับ"

    def _stretched(self, factor, path):
        box = _fitted_box(self.TEXT, 40)
        box[1][0] = box[2][0] = box[0][0] + (box[1][0] - box[0][0]) * factor
        return _document_with([self.TEXT], path, boxes=[box])

    def test_a_line_fitted_to_its_box_reads_back_whole(self, tmp_path):
        reader = self._stretched(1.0, tmp_path / "fitted.pdf")
        assert self.TEXT in reader[0].get_text()
        reader.close()

    def test_pdfium_is_unaffected_however_far_it_is_stretched(self, tmp_path):
        path = tmp_path / "stretched.pdf"
        self._stretched(4.0, path).close()
        assert self.TEXT in pdfium_text(path)

    def test_mupdf_breaks_the_line_up_once_it_is_stretched_far_enough(self, tmp_path):
        reader = self._stretched(4.0, tmp_path / "stretched-mupdf.pdf")
        extracted = reader[0].get_text()
        assert self.TEXT not in extracted, (
            "MuPDF no longer splits stretched clusters; if this is a fixed "
            "upstream behaviour the surrounding class can go"
        )
        # Nothing is lost, only regrouped -- which is why this is a note about
        # searching rather than a data-loss bug.
        assert "".join(extracted.split()) == self.TEXT
        reader.close()


class TestTheFontCostsAlmostNothing:
    """What the text layer adds to a document, once, whatever its page count."""

    def test_the_embedded_font_stays_under_a_few_kilobytes(self, tmp_path):
        path = tmp_path / "sized.pdf"
        _document_with(SAMPLES, path).close()
        assert path.stat().st_size < 8_000

    def test_one_copy_serves_every_page(self, tmp_path):
        doc = opened(source_pdf(pages=5))
        result = OcrResult(
            text="한글 검색",
            confidence=0.9,
            bbox=[[50, 40], [550, 40], [550, 70], [50, 70]],
        )
        for page in doc:
            overlay_text_on_page(page, [result], 612, 792)

        path = tmp_path / "multipage.pdf"
        reader = written(doc, path)
        embedded = {font[0] for page in reader for font in page.get_fonts(full=True)}
        assert len(embedded) == 1
        reader.close()
