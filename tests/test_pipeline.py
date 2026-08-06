"""Tests for pipeline module."""

from unittest.mock import MagicMock, patch

import click
import fitz
import numpy as np
import pytest

from pdf_refinery.ocr_engine import OcrResult
from pdf_refinery.pipeline import parse_page_range, run_ocr_pipeline


class TestParsePageRange:
    @pytest.mark.parametrize(
        "pages_str, total, expected",
        [
            ("3", 10, [2]),
            ("1-5", 10, [0, 1, 2, 3, 4]),
            ("1,3,5", 10, [0, 2, 4]),
            ("1-3,7,10-12", 15, [0, 1, 2, 6, 9, 10, 11]),
            ("8-15", 10, [7, 8, 9]),
            ("11", 10, []),
            ("1", 10, [0]),
            ("10", 10, [9]),
            (" 1 , 3 ", 10, [0, 2]),
        ],
    )
    def test_parse_page_range(self, pages_str, total, expected):
        assert parse_page_range(pages_str, total) == expected


class TestRunOcrPipeline:
    @patch("pdf_refinery.pipeline.overlay_text_on_page")
    @patch("pdf_refinery.pipeline.page_to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_basic_execution(self, mock_engine_cls, mock_page_to_image, mock_overlay, tmp_pdf, tmp_path):
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = [
            OcrResult(text="Hello", confidence=0.9, bbox=[[0, 0], [50, 0], [50, 20], [0, 20]])
        ]
        mock_engine_cls.return_value = mock_engine
        mock_page_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)
        mock_overlay.return_value = 1

        output = tmp_path / "out.pdf"
        run_ocr_pipeline(input_path=tmp_pdf, output_path=output)

        mock_engine.recognize.assert_called_once()
        mock_overlay.assert_called_once()

    @patch("pdf_refinery.pipeline.overlay_text_on_page")
    @patch("pdf_refinery.pipeline.page_to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_page_range(self, mock_engine_cls, mock_page_to_image, mock_overlay, multi_page_pdf, tmp_path):
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = []
        mock_engine_cls.return_value = mock_engine
        mock_page_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)
        mock_overlay.return_value = 0

        output = tmp_path / "out.pdf"
        run_ocr_pipeline(input_path=multi_page_pdf, output_path=output, pages="2,4")

        assert mock_overlay.call_count == 2

    @patch("pdf_refinery.pipeline.overlay_text_on_page")
    @patch("pdf_refinery.pipeline.page_to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_multiple_languages(self, mock_engine_cls, mock_page_to_image, mock_overlay, tmp_pdf, tmp_path):
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = []
        mock_engine_cls.return_value = mock_engine
        mock_page_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)
        mock_overlay.return_value = 0

        output = tmp_path / "out.pdf"
        run_ocr_pipeline(input_path=tmp_pdf, output_path=output, langs=["en", "korean"])

        assert mock_engine_cls.call_count == 2
        mock_engine_cls.assert_any_call(lang="en")
        mock_engine_cls.assert_any_call(lang="korean")

    def test_rejects_output_equal_to_input(self, tmp_pdf):
        with pytest.raises(click.ClickException):
            run_ocr_pipeline(input_path=tmp_pdf, output_path=tmp_pdf)

    def test_leaves_input_file_untouched(self, tmp_pdf, tmp_path):
        before = tmp_pdf.read_bytes()
        with patch("pdf_refinery.pipeline.OcrEngine") as mock_engine_cls:
            mock_engine_cls.return_value.recognize.return_value = []
            run_ocr_pipeline(input_path=tmp_pdf, output_path=tmp_path / "out.pdf")
        assert tmp_pdf.read_bytes() == before


class TestExistingTextPolicy:
    """Pages that already have text must not be silently destroyed."""

    @pytest.fixture
    def text_pdf(self, tmp_path):
        path = tmp_path / "with_text.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text(fitz.Point(72, 100), "Original content", fontsize=24)
        doc.save(path)
        doc.close()
        return path

    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_skips_pages_with_text_by_default(self, mock_engine_cls, text_pdf, tmp_path):
        mock_engine_cls.return_value.recognize.return_value = []

        output = tmp_path / "out.pdf"
        run_ocr_pipeline(input_path=text_pdf, output_path=output)

        mock_engine_cls.return_value.recognize.assert_not_called()
        doc = fitz.open(output)
        assert "Original content" in doc[0].get_text()
        doc.close()

    @patch("pdf_refinery.pipeline.page_to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_force_ocr_reprocesses_the_page(
        self, mock_engine_cls, mock_page_to_image, text_pdf, tmp_path
    ):
        mock_engine_cls.return_value.recognize.return_value = [
            OcrResult(text="Original content", confidence=0.9,
                      bbox=[[100, 100], [400, 100], [400, 140], [100, 140]])
        ]
        mock_page_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)

        output = tmp_path / "out.pdf"
        run_ocr_pipeline(input_path=text_pdf, output_path=output, force_ocr=True)

        mock_engine_cls.return_value.recognize.assert_called_once()
        doc = fitz.open(output)
        # The text now comes from OCR, and the page still shows its content.
        assert "Original content" in doc[0].get_text()
        doc.close()
