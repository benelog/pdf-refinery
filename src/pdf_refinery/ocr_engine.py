"""PaddleOCR wrapper for text recognition."""

from dataclasses import dataclass

import cv2
import numpy as np
from paddleocr import PaddleOCR


@dataclass
class OcrResult:
    """A single OCR detection result."""

    text: str
    confidence: float
    bbox: list[list[float]]  # Four corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR accuracy."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def _poly_to_rect(poly: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert polygon points to axis-aligned (x1, y1, x2, y2)."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _rect_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two axis-aligned rects (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def deduplicate_results(
    results: list[OcrResult], iou_threshold: float = 0.5,
) -> list[OcrResult]:
    """Remove duplicate detections by IoU, keeping higher confidence."""
    sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
    kept: list[tuple[OcrResult, tuple[float, float, float, float]]] = []
    for result in sorted_results:
        rect = _poly_to_rect(result.bbox)
        if all(_rect_iou(rect, k_rect) < iou_threshold for _, k_rect in kept):
            kept.append((result, rect))
    return [r for r, _ in kept]


class OcrEngine:
    """Wrapper around PaddleOCR for page-level text recognition."""

    def __init__(self, lang: str = "en"):
        self._ocr = PaddleOCR(
            use_textline_orientation=True,
            lang=lang,
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
            text_det_unclip_ratio=1.8,
            text_recognition_batch_size=16,
        )

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
