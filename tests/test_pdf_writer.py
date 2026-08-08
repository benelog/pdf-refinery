"""Tests for PDF writer module."""

import textwrap
from unittest.mock import patch

import fitz
import pytest

from pdf_refinery.fonts import FontResolver, span_metrics
from pdf_refinery.ocr_engine import OcrResult
from pdf_refinery.pdf_writer import (
    has_text,
    overlay_text_on_page,
    rasterize_page,
    remove_text_layer,
)


def _make_page():
    """Create a fresh fitz page (612x792 pts)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    return doc, page


def _count_ink(page, dpi=72):
    """Count dark pixels, i.e. how much is actually drawn on the page."""
    samples = page.get_pixmap(dpi=dpi, alpha=False).samples
    return sum(1 for b in samples if b < 128)


class TestOverlayTextOnPage:
    def test_empty_results(self):
        doc, page = _make_page()
        stats = overlay_text_on_page(page, [], image_width=1224, image_height=1584)
        assert stats.inserted == 0
        doc.close()

    def test_single_result_coordinates(self):
        doc, page = _make_page()
        # Horizontal text: top-left (100,200), top-right (300,200),
        # bottom-right (300,250), bottom-left (100,250)
        result = OcrResult(
            text="Hello",
            confidence=0.9,
            bbox=[[100, 200], [300, 200], [300, 250], [100, 250]],
        )

        with patch.object(page, "insert_text", wraps=page.insert_text) as spy:
            stats = overlay_text_on_page(page, [result], image_width=612, image_height=792)

            assert stats.inserted == 1
            spy.assert_called_once()
            call_kwargs = spy.call_args
            point = call_kwargs.kwargs.get("point") or call_kwargs[1].get("point", call_kwargs[0][0])
            # The baseline sits an ascent below the top of the box, so the
            # line's own ascent and descent fill it exactly.
            ascent, descent = span_metrics(FontResolver().resolve("Hello"))
            assert point.x == pytest.approx(100.0)
            assert point.y == pytest.approx(
                200 + (250 - 200) * ascent / (ascent + descent)
            )

        doc.close()

    def test_coordinate_scaling(self):
        doc, page = _make_page()
        # Image is 2x the page size → scale = 0.5
        result = OcrResult(
            text="Scaled",
            confidence=0.9,
            bbox=[[200, 400], [600, 400], [600, 500], [200, 500]],
        )

        with patch.object(page, "insert_text", wraps=page.insert_text) as spy:
            stats = overlay_text_on_page(page, [result], image_width=1224, image_height=1584)

            assert stats.inserted == 1
            call_kwargs = spy.call_args
            point = call_kwargs.kwargs.get("point") or call_kwargs[0][0]
            # scale = 0.5, so tl=(100,200) and bl=(100,250) on the page.
            ascent, descent = span_metrics(FontResolver().resolve("Scaled"))
            assert point.x == pytest.approx(100.0)
            assert point.y == pytest.approx(
                200 + (250 - 200) * ascent / (ascent + descent)
            )

        doc.close()

    def test_the_span_a_reader_gets_back_covers_the_detected_box(self):
        """The point of sizing by measured metrics rather than a constant.

        A viewer highlights, and ``get_text()`` reports, the span box -- not
        the glyphs. If that box does not match the box OCR detected, a search
        hit is drawn off the words it found, and at worst over the line above.
        """
        for text in ("Hello world", "한글 검색 테스트", "混用 漢字"):
            doc, page = _make_page()
            box = [[100, 200], [500, 200], [500, 250], [100, 250]]
            overlay_text_on_page(
                page,
                [OcrResult(text=text, confidence=0.9, bbox=box)],
                image_width=612, image_height=792,
            )
            spans = [
                span
                for block in page.get_text("dict")["blocks"] if block["type"] == 0
                for line in block["lines"] for span in line["spans"]
            ]
            assert len(spans) == 1, text
            x0, y0, x1, y1 = spans[0]["bbox"]
            assert y0 == pytest.approx(200, abs=0.5), text
            assert y1 == pytest.approx(250, abs=0.5), text
            doc.close()

    def test_tiny_font_skipped(self):
        doc, page = _make_page()
        # bbox height = 1px, scale_y * 0.85 < 1 → should be skipped
        result = OcrResult(
            text="Tiny",
            confidence=0.9,
            bbox=[[10, 10], [50, 10], [50, 11], [10, 11]],
        )

        stats = overlay_text_on_page(page, [result], image_width=612, image_height=792)
        assert stats.inserted == 0
        doc.close()

    def test_multiple_results(self, sample_ocr_results):
        doc, page = _make_page()
        stats = overlay_text_on_page(page, sample_ocr_results, image_width=612, image_height=792)
        assert stats.inserted == 2
        doc.close()


class TestLineIsFittedToItsBox:
    """The overlay must end where the scanned line ends.

    Left at its natural width, a line of Hangul set at the box's height is
    several times wider than the box. It then runs off the crop box, and
    ``get_text()`` returns only the part that stayed on the page -- so the tail
    of every line goes missing while the recogniser looks perfect. Measured on
    a real book scan, this cost 16.4% character error against 1.2% for the
    same text before it was written out.
    """

    LINE = "황재섭 정도면 대충 짐작할 수 있을 터였다 그가 나를 물끄러미 바라"

    def _overlay_full_width_line(self):
        doc, page = _make_page()
        # A line spanning the page, as a book scan produces.
        result = OcrResult(
            text=self.LINE,
            confidence=0.9,
            bbox=[[20, 200], [592, 200], [592, 240], [20, 240]],
        )
        overlay_text_on_page(page, [result], image_width=612, image_height=792)
        return doc, page

    def test_the_whole_line_is_still_extractable(self):
        doc, page = self._overlay_full_width_line()
        extracted = "".join(page.get_text().split())
        assert extracted == "".join(self.LINE.split())
        doc.close()

    def test_the_line_stays_inside_the_page(self):
        doc, page = self._overlay_full_width_line()
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block["lines"]
            for span in line["spans"]
        ]
        assert spans, "nothing was written"
        for span in spans:
            assert span["bbox"][2] <= page.rect.x1 + 1
        doc.close()

    def test_natural_width_would_have_overflowed(self):
        """Guards the guard: without fitting there is nothing to fix."""
        doc, page = _make_page()
        natural = fitz.get_text_length(self.LINE, fontname="korea", fontsize=34)
        assert natural > page.rect.width, (
            "the fixture no longer reproduces the overflow this test is about"
        )
        doc.close()


class TestRotatedLinesLandOnTheirBox:
    """A turned line must cover its box, not run off in some other direction.

    ``prerotate`` turns the opposite way from ``atan2``, so passing the angle
    through unchanged sent every line the wrong way -- barely visible at the
    fraction of a degree a straight scan produces, and completely wrong for a
    page scanned sideways, where the text left the page and stopped being
    extractable at all.
    """

    TEXT = "황재섭 정도면 대충 짐작할 수 있을 터였다 그가"

    # bbox corners are (top-left, top-right, bottom-right, bottom-left) of the
    # line as the detector sees it, so rotating the line rotates the corners.
    @pytest.mark.parametrize(
        "bbox,expected",
        [
            ([[100, 300], [500, 300], [500, 340], [100, 340]], (100, 300, 500, 340)),
            ([[300, 100], [300, 500], [260, 500], [260, 100]], (260, 100, 300, 500)),
            ([[500, 340], [100, 340], [100, 300], [500, 300]], (100, 300, 500, 340)),
            ([[300, 500], [300, 100], [340, 100], [340, 500]], (300, 100, 340, 500)),
            ([[100, 300], [498, 335], [494, 375], [96, 340]], (96, 300, 498, 376)),
        ],
        ids=["0deg", "90deg", "180deg", "270deg", "skewed"],
    )
    def test_the_line_covers_its_box(self, bbox, expected):
        doc = fitz.open()
        page = doc.new_page(width=612, height=612)
        result = OcrResult(text=self.TEXT, confidence=0.9, bbox=bbox)

        overlay_text_on_page(page, [result], image_width=612, image_height=612)

        assert "".join(page.get_text().split()) == "".join(self.TEXT.split())
        spans = [
            span["bbox"]
            for block in page.get_text("dict")["blocks"] if block["type"] == 0
            for line in block["lines"] for span in line["spans"]
        ]
        got = (
            min(s[0] for s in spans), min(s[1] for s in spans),
            max(s[2] for s in spans), max(s[3] for s in spans),
        )
        assert got == pytest.approx(expected, abs=2.0)
        doc.close()


class TestOverlayTextIsExtractable:
    """A text layer that cannot be extracted is worthless, whatever the OCR said."""

    @pytest.mark.parametrize(
        "text",
        ["한글 검색 테스트", "Hello World", "日本語のテスト", "中文测试", "café"],
    )
    def test_text_survives_the_overlay(self, text):
        doc, page = _make_page()
        result = OcrResult(
            text=text,
            confidence=0.9,
            bbox=[[100, 200], [400, 200], [400, 240], [100, 240]],
        )

        assert overlay_text_on_page(page, [result], 612, 792).inserted == 1
        assert text in page.get_text()
        doc.close()

    def test_mixed_scripts_across_lines(self):
        doc, page = _make_page()
        texts = ["한글 줄", "English line", "日本語の行"]
        results = [
            OcrResult(
                text=t,
                confidence=0.9,
                bbox=[[10, 20 + i * 60], [400, 20 + i * 60],
                      [400, 60 + i * 60], [10, 60 + i * 60]],
            )
            for i, t in enumerate(texts)
        ]

        assert overlay_text_on_page(page, results, 612, 792).inserted == 3
        extracted = page.get_text()
        for t in texts:
            assert t in extracted
        doc.close()


def _page_with_text(text, **kwargs):
    doc, page = _make_page()
    for i, line in enumerate(textwrap.wrap(text, 60) or [""]):
        page.insert_text(fitz.Point(50, 100 + i * 14), line, fontsize=12, **kwargs)
    return doc, page


class TestHasText:
    def test_blank_page(self):
        doc, page = _make_page()
        assert has_text(page) is False
        doc.close()

    def test_page_of_body_text(self):
        doc, page = _page_with_text("Something worth keeping. " * 10)
        assert has_text(page) is True
        doc.close()

    def test_invisible_text_counts(self):
        doc, page = _page_with_text("Hidden but extractable. " * 10, render_mode=3)
        assert has_text(page) is True
        doc.close()

    def test_stray_page_number_does_not_count(self):
        # A scanner-stamped page number is the classic reason a whole page of
        # body text ends up skipped and unsearchable.
        doc, page = _page_with_text("117")
        assert has_text(page) is False
        doc.close()

    def test_running_header_does_not_count(self):
        doc, page = _page_with_text("Chapter 5 - The Beginning of Everything")
        assert has_text(page) is False
        doc.close()

    def test_threshold_is_adjustable(self):
        doc, page = _page_with_text("117")
        assert has_text(page, min_chars=0) is True
        assert has_text(page, min_chars=3) is True
        assert has_text(page, min_chars=4) is False
        doc.close()

    def test_whitespace_does_not_pad_the_count(self):
        doc, page = _page_with_text(" ".join("x" * 1 for _ in range(60)))
        # 60 characters of content, however much space sits between them.
        assert has_text(page, min_chars=60) is True
        assert has_text(page, min_chars=61) is False
        doc.close()


class TestRasterizePage:
    def test_strips_text_but_keeps_appearance(self):
        doc, page = _make_page()
        page.insert_text(fitz.Point(100, 200), "Visible text", fontsize=24)
        ink_before = _count_ink(page)
        assert ink_before > 0

        rasterize_page(page, dpi=150)

        # The text layer is gone...
        assert page.get_text().strip() == ""
        # ...but the glyphs are still visible, unlike after a bare redaction.
        ink_after = _count_ink(page)
        assert ink_after == pytest.approx(ink_before, rel=0.2)
        doc.close()

    def test_redaction_alone_would_lose_the_content(self):
        """Contrast case: this is the data loss rasterize_page avoids."""
        doc, page = _make_page()
        page.insert_text(fitz.Point(100, 200), "Visible text", fontsize=24)

        remove_text_layer(page)

        assert _count_ink(page) == 0
        doc.close()


class TestRemoveTextLayer:
    def test_no_text_returns_false(self):
        doc, page = _make_page()
        assert remove_text_layer(page) is False
        doc.close()

    def test_removes_existing_text(self):
        doc, page = _make_page()
        page.insert_text(fitz.Point(100, 200), "Existing text", fontsize=12)
        assert page.get_text().strip() == "Existing text"

        assert remove_text_layer(page) is True
        assert page.get_text().strip() == ""
        doc.close()

    def test_removes_invisible_text(self):
        doc, page = _make_page()
        page.insert_text(fitz.Point(100, 200), "Hidden", fontsize=12, render_mode=3)
        assert page.get_text().strip() == "Hidden"

        assert remove_text_layer(page) is True
        assert page.get_text().strip() == ""
        doc.close()
