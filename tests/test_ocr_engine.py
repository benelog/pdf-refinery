"""Tests for OCR engine module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_refinery.ocr_engine import OcrEngine, OcrResult


class TestOcrResult:
    def test_field_access(self):
        r = OcrResult(text="hello", confidence=0.95, bbox=[[0, 0], [10, 0], [10, 10], [0, 10]])
        assert r.text == "hello"
        assert r.confidence == 0.95
        assert len(r.bbox) == 4


class TestOcrEngine:
    @patch("pdf_refinery.ocr_engine.PaddleOCR")
    def test_confidence_filtering(self, mock_paddle_cls):
        mock_ocr = MagicMock()
        mock_paddle_cls.return_value = mock_ocr

        mock_ocr.predict.return_value = [{
            "rec_texts": ["high", "low", "mid"],
            "rec_scores": [0.9, 0.4, 0.7],
            "dt_polys": [
                np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                np.array([[20, 0], [30, 0], [30, 10], [20, 10]]),
                np.array([[40, 0], [50, 0], [50, 10], [40, 10]]),
            ],
        }]

        engine = OcrEngine(lang="en")
        results = engine.recognize(np.zeros((100, 100, 3), dtype=np.uint8), confidence=0.5)

        assert len(results) == 2
        assert results[0].text == "high"
        assert results[1].text == "mid"

    @patch("pdf_refinery.ocr_engine.PaddleOCR")
    def test_empty_result(self, mock_paddle_cls):
        mock_ocr = MagicMock()
        mock_paddle_cls.return_value = mock_ocr
        mock_ocr.predict.return_value = []

        engine = OcrEngine(lang="en")
        results = engine.recognize(np.zeros((100, 100, 3), dtype=np.uint8))

        assert results == []

    @patch("pdf_refinery.ocr_engine.PaddleOCR")
    def test_bbox_conversion_from_numpy(self, mock_paddle_cls):
        mock_ocr = MagicMock()
        mock_paddle_cls.return_value = mock_ocr

        poly = np.array([[10.5, 20.3], [100.1, 20.3], [100.1, 50.7], [10.5, 50.7]])
        mock_ocr.predict.return_value = [{
            "rec_texts": ["test"],
            "rec_scores": [0.95],
            "dt_polys": [poly],
        }]

        engine = OcrEngine(lang="en")
        results = engine.recognize(np.zeros((100, 100, 3), dtype=np.uint8))

        assert len(results) == 1
        assert isinstance(results[0].bbox, list)
        assert isinstance(results[0].bbox[0], list)
        assert results[0].bbox[0] == pytest.approx([10.5, 20.3])
