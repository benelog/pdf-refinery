"""Shared test fixtures."""

import pytest

from pdf_refinery.ocr_engine import OcrResult
from tests.helpers import source_pdf


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
    """A minimal 1-page PDF on disk."""
    path = tmp_path / "test.pdf"
    path.write_bytes(source_pdf())
    return path


@pytest.fixture
def multi_page_pdf(tmp_path):
    """A 5-page PDF on disk."""
    path = tmp_path / "multi.pdf"
    path.write_bytes(source_pdf(pages=5))
    return path
