"""Tests for PDF writer module."""

from unittest.mock import patch

import fitz
import pytest

from pdf_refinery.ocr_engine import OcrResult
from pdf_refinery.pdf_writer import overlay_text_on_page


def _make_page():
    """Create a fresh fitz page (612x792 pts)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    return doc, page


class TestOverlayTextOnPage:
    def test_empty_results(self):
        doc, page = _make_page()
        count = overlay_text_on_page(page, [], image_width=1224, image_height=1584)
        assert count == 0
        doc.close()

    def test_single_result_coordinates(self):
        doc, page = _make_page()
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
            assert point.x == pytest.approx(100.0)  # scale_x = 1.0
            assert point.y == pytest.approx(250.0)  # bottom_y * scale_y = 250 * 1.0

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
            # scale_x = 612/1224 = 0.5, scale_y = 792/1584 = 0.5
            assert point.x == pytest.approx(100.0)  # 200 * 0.5
            assert point.y == pytest.approx(250.0)  # 500 * 0.5

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
