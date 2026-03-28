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


def _bbox_iou(a: list[list[float]], b: list[list[float]]) -> float:
    """Compute IoU between two axis-aligned bounding boxes from polygon points."""
    ax1, ay1 = min(p[0] for p in a), min(p[1] for p in a)
    ax2, ay2 = max(p[0] for p in a), max(p[1] for p in a)
    bx1, by1 = min(p[0] for p in b), min(p[1] for p in b)
    bx2, by2 = max(p[0] for p in b), max(p[1] for p in b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def deduplicate_results(
    results: list[OcrResult], iou_threshold: float = 0.5,
) -> list[OcrResult]:
    """Remove duplicate detections by IoU, keeping higher confidence."""
    results = sorted(results, key=lambda r: r.confidence, reverse=True)
    kept: list[OcrResult] = []
    for result in results:
        if all(_bbox_iou(result.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(result)
    return kept


class OcrEngine:
    """Wrapper around PaddleOCR for page-level text recognition."""

    def __init__(self, lang: str = "en"):
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.8,
            rec_batch_num=16,
        )

    def recognize(self, image: np.ndarray, confidence: float = 0.5) -> list[OcrResult]:
        """Run OCR on a page image and return filtered results.

        Args:
            image: Page image as a numpy array (RGB).
            confidence: Minimum confidence threshold.

        Returns:
            List of OcrResult with confidence above the threshold.
        """
        preprocessed = preprocess_image(image)
        raw = self._ocr.predict(preprocessed)
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
