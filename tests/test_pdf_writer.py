"""Tests for PDF writer module."""

from unittest.mock import patch

import fitz
import pytest

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
        count = overlay_text_on_page(page, [], image_width=1224, image_height=1584)
        assert count == 0
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
            count = overlay_text_on_page(page, [result], image_width=612, image_height=792)

            assert count == 1
            spy.assert_called_once()
            call_kwargs = spy.call_args
            point = call_kwargs.kwargs.get("point") or call_kwargs[1].get("point", call_kwargs[0][0])
            # baseline = tl + (bl - tl) * 0.85
            assert point.x == pytest.approx(100.0)  # tl.x + 0 * 0.85
            assert point.y == pytest.approx(200 + (250 - 200) * 0.85)  # 242.5

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
            count = overlay_text_on_page(page, [result], image_width=1224, image_height=1584)

            assert count == 1
            call_kwargs = spy.call_args
            point = call_kwargs.kwargs.get("point") or call_kwargs[0][0]
            # scale = 0.5, tl=(100,200), bl=(100,250)
            # baseline = tl + (bl - tl) * 0.85
            assert point.x == pytest.approx(100.0)
            assert point.y == pytest.approx(200 + (250 - 200) * 0.85)  # 242.5

        doc.close()

    def test_tiny_font_skipped(self):
        doc, page = _make_page()
        # bbox height = 1px, scale_y * 0.85 < 1 → should be skipped
        result = OcrResult(
            text="Tiny",
            confidence=0.9,
            bbox=[[10, 10], [50, 10], [50, 11], [10, 11]],
        )

        count = overlay_text_on_page(page, [result], image_width=612, image_height=792)
        assert count == 0
        doc.close()

    def test_multiple_results(self, sample_ocr_results):
        doc, page = _make_page()
        count = overlay_text_on_page(page, sample_ocr_results, image_width=612, image_height=792)
        assert count == 2
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

        assert overlay_text_on_page(page, [result], 612, 792) == 1
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

        assert overlay_text_on_page(page, results, 612, 792) == 3
        extracted = page.get_text()
        for t in texts:
            assert t in extracted
        doc.close()


class TestHasText:
    def test_blank_page(self):
        doc, page = _make_page()
        assert has_text(page) is False
        doc.close()

    def test_page_with_text(self):
        doc, page = _make_page()
        page.insert_text(fitz.Point(100, 200), "Something", fontsize=12)
        assert has_text(page) is True
        doc.close()

    def test_invisible_text_counts(self):
        doc, page = _make_page()
        page.insert_text(fitz.Point(100, 200), "Hidden", fontsize=12, render_mode=3)
        assert has_text(page) is True
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
