"""PaddleOCR wrapper for text recognition."""

from dataclasses import dataclass

import numpy as np
from paddleocr import PaddleOCR


@dataclass
class OcrResult:
    """A single OCR detection result."""

    text: str
    confidence: float
    bbox: list[list[float]]  # Four corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


class OcrEngine:
    """Wrapper around PaddleOCR for page-level text recognition."""

    def __init__(self, lang: str = "en"):
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def recognize(self, image: np.ndarray, confidence: float = 0.5) -> list[OcrResult]:
        """Run OCR on a page image and return filtered results.

        Args:
            image: Page image as a numpy array (RGB).
            confidence: Minimum confidence threshold.

        Returns:
            List of OcrResult with confidence above the threshold.
        """
        raw = self._ocr.predict(image)
        if not raw:
            return []

        r = raw[0]
        texts = r["rec_texts"]
        scores = r["rec_scores"]
        polys = r["dt_polys"]

        results = []
        for text, conf, poly in zip(texts, scores, polys):
            if conf >= confidence:
                bbox = poly.tolist()
                results.append(OcrResult(text=text, confidence=conf, bbox=bbox))
        return results
