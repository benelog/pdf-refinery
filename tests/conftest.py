"""Shared test fixtures."""

import fitz
import pytest

from pdf_refinery.ocr_engine import OcrResult


@pytest.fixture
def sample_ocr_results():
    """Known OCR results for testing."""
    return [
        OcrResult(
            text="Hello",
            confidence=0.95,
            bbox=[[10, 20], [100, 20], [100, 50], [10, 50]],
        ),
        OcrResult(
            text="World",
            confidence=0.88,
            bbox=[[10, 60], [100, 60], [100, 90], [10, 90]],
        ),
    ]


@pytest.fixture
def tmp_pdf(tmp_path):
    """Create a minimal 1-page PDF."""
    path = tmp_path / "test.pdf"
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def multi_page_pdf(tmp_path):
    """Create a 5-page PDF."""
    path = tmp_path / "multi.pdf"
    doc = fitz.open()
    for _ in range(5):
        doc.new_page(width=612, height=792)
    doc.save(path)
    doc.close()
    return path
