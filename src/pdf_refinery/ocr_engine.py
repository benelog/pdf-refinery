"""Text recognition: the engine protocol, and the PaddleOCR implementation.

:class:`TextRecogniser` is the whole of what the pipeline depends on. Everything
else here -- the language tables, the model names, the orientation switch -- is
PaddleOCR's business, and a second engine would bring its own.

``paddleocr`` is imported where it is used rather than at module level. Its
import costs about three seconds and pings the model hosts, and most of what
this module offers -- the result type, the preprocessing, the deduplication --
does not need it. Keeping it out of the import graph is what lets ``--help``
answer instantly and offline.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class OcrResult:
    """A single OCR detection result."""

    text: str
    confidence: float
    bbox: list[list[float]]  # Four corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


# Codes worth naming in an error message. PaddleOCR supports many more, but its
# table is a local variable inside a private method and cannot be enumerated.
COMMON_LANGS = (
    "en", "korean", "japan", "ch", "chinese_cht",
    "fr", "german", "es", "it", "pt", "ru", "th", "ar", "hi", "vi",
)

# The one recognition model worth naming: PaddleOCR only picks it for Chinese
# and Japanese, so a Latin document gets the lighter mobile model by default.
SERVER_REC_MODEL = "PP-OCRv5_server_rec"

# Languages that model's dictionary covers -- it is the combined Chinese,
# English and Japanese model. Forcing it on Korean or Cyrillic text does not
# trade accuracy for speed, it recognises nothing.
SERVER_REC_LANGS = frozenset({"ch", "chinese_cht", "japan", "en"})

# Codes users reach for that PaddleOCR does not use.
LANG_ALIASES = {
    "ko": "korean", "kr": "korean", "kor": "korean",
    "ja": "japan", "jp": "japan", "jpn": "japan",
    "zh": "ch", "cn": "ch", "zh_cn": "ch",
    "zh_tw": "chinese_cht", "tw": "chinese_cht",
    "eng": "en", "english": "en",
}


def is_supported_lang(lang: str) -> bool:
    """Return whether PaddleOCR has a recognition model for ``lang``.

    The check maps the code to model names without constructing a pipeline, so
    it downloads nothing and can run before any work starts. It reads a private
    PaddleOCR method; if that ever changes shape the language is accepted, so a
    library upgrade cannot make this CLI reject codes that actually work.
    """
    from paddleocr import PaddleOCR

    try:
        _, rec_model = PaddleOCR._get_ocr_model_names(None, lang, None)
    except Exception:
        return True
    return rec_model is not None


PREPROCESS_MODES = ("none", "binarize")
DEFAULT_PREPROCESS = "binarize"

# Whether to run PaddleOCR's per-line orientation classifier, which judges each
# detected line upright or upside down and turns it before recognition.
#
# Off, because it is wrong far too often to be a default. On page 1 of the
# leaflet in ``bench/sample-2`` it reported 14 of 31 lines as upside down; none
# were. Every line it turns is then recognised upside down, and Hangul read that
# way comes back as other plausible Hangul rather than as nothing -- a line
# reading "D버튼을 눌러 맞추고 C버튼을 눌러 세팅을 완료합니다." was returned as
# "릉록릉글록릉글" with 0.455 confidence. So the damage is silent: no error, no
# empty result, just a page half filled with wrong text.
#
# Measured cost of leaving it on, CER-ns with everything else at defaults:
# sample-2 0.4571 against 0.0251, sample-1 0.0174 against 0.0145. It loses on
# both corpora and turns the leaflet from three times better than the text layer
# it replaces into eight times worse.
#
# It is kept as an option because a page really can be scanned with lines upside
# down, and nothing else here would fix that.
DEFAULT_TEXTLINE_ORIENTATION = False


def preprocess_image(img: np.ndarray, mode: str = DEFAULT_PREPROCESS) -> np.ndarray:
    """Prepare a rendered page for the recogniser.

    ``binarize`` denoises and thresholds to black and white, the habit of
    pre-neural OCR. The theory says not to: PP-OCRv5 was trained on
    anti-aliased greyscale and thresholding destroys exactly that. The theory
    lost. Measured over the three corpora in ``bench/`` at 300 DPI, against
    ``none``:

    ============  ===================  ==========================
    corpus        CER-ns               word errors
    ============  ===================  ==========================
    sample-1      0.0145 -> 0.0112     403 -> 117 of 806
    sample-2      0.0251 -> 0.0174     253 -> 99 of 302
    sample-3      0.0019 -> 0.0029       7 -> 9 of 474
    ============  ===================  ==========================

    The characters barely move. What moves is **word boundaries**: on the two
    Korean scans the recogniser runs whole phrases together when handed
    greyscale, and separates them when handed black and white. Since the
    overlay reproduces the recognised string exactly, that is the difference
    between word search working and not. The cost is two words on the clean
    Latin scan, and 10-20% more time -- nearly all of it in the denoise.

    This flipped twice before, on evidence that turned out to be measuring the
    textline-orientation bug rather than the preprocessing. These numbers are
    from after that fix, and are the first taken over three scanners and two
    scripts.

    Args:
        img: Page image in RGB order.
        mode: One of :data:`PREPROCESS_MODES`.

    Returns:
        An RGB image; the same array when ``mode`` is ``"none"``.
    """
    if mode == "none":
        return img
    if mode != "binarize":
        raise ValueError(
            f"Unknown preprocess mode '{mode}'; expected one of "
            f"{', '.join(PREPROCESS_MODES)}."
        )
    # Imported here rather than at module scope: this is the only code that
    # needs OpenCV, and the package does not declare it. See pyproject.toml.
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


PAGE_ROTATIONS = (0, 90, 180, 270)

_orientation_classifier = None


def detect_page_orientation(image: np.ndarray) -> int:
    """How far clockwise the page has been turned away from upright.

    Args:
        image: The rendered page in RGB, **before** any preprocessing. This
            matters: over three corpora at all four orientations the
            classifier answered all sixteen correctly on the rendered
            greyscale, at 0.92 confidence, and only thirteen on the same pages
            binarized, dropping as low as 0.54. Thresholding removes what it
            reads.

    Returns:
        One of :data:`PAGE_ROTATIONS`. Turning the page that many degrees
        anticlockwise stands it upright.

    Acting on this is the difference between a sideways scan being searchable
    and not. Fed the Latin corpus re-laid on its side, the pipeline extracted
    46 characters of noise from a 1050-character page and matched none of its
    139 words; with the page turned first, 1051 characters and 132 of the 139.
    """
    global _orientation_classifier
    if _orientation_classifier is None:
        from paddleocr import DocImgOrientationClassification

        _orientation_classifier = DocImgOrientationClassification()

    result = _orientation_classifier.predict(
        np.ascontiguousarray(image[:, :, ::-1])
    )[0]
    try:
        angle = int(result["label_names"][0])
    except (KeyError, IndexError, ValueError):
        return 0
    return angle if angle in PAGE_ROTATIONS else 0


def upright(image: np.ndarray, rotation: int) -> np.ndarray:
    """Turn ``image`` back upright, undoing ``rotation`` degrees clockwise."""
    if rotation == 0:
        return image
    # np.rot90 turns anticlockwise, which is the direction that undoes a
    # clockwise rotation, so the quarter-turn count is the rotation itself.
    return np.ascontiguousarray(np.rot90(image, k=rotation // 90))


def unrotate_results(
    results: list[OcrResult], rotation: int, upright_shape: tuple[int, ...],
) -> list[OcrResult]:
    """Map boxes found on an uprighted image back onto the original page.

    Without this the text layer is written in a frame the page does not have:
    the footer lands at the top and lines run off the edge, where nothing can
    extract them. That is what made PaddleOCR's own page-orientation option
    unusable -- it turns the image before detecting and hands the boxes back
    in the turned frame.

    Args:
        results: Detections in the coordinates of the uprighted image.
        rotation: The value :func:`detect_page_orientation` returned.
        upright_shape: ``shape`` of the uprighted image the boxes came from.
    """
    if rotation == 0:
        return results

    height, width = upright_shape[:2]

    def to_original(x: float, y: float) -> list[float]:
        # Inverse of np.rot90 by rotation//90 quarter turns, in (x, y).
        # Derived by rotating a marked image and comparing, not on paper: the
        # 90 and 270 cases are easy to write down the wrong way round, and
        # both still look plausible until a page is actually sideways.
        if rotation == 90:
            return [height - y, x]
        if rotation == 180:
            return [width - x, height - y]
        return [y, width - x]  # 270

    return [
        OcrResult(
            text=r.text,
            confidence=r.confidence,
            bbox=[to_original(px, py) for px, py in r.bbox],
        )
        for r in results
    ]


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
    """Remove duplicate detections by IoU, keeping higher confidence.

    Only reached when more than one language was asked for, which is rarely
    worth doing: the recognition models are multilingual, and a single
    ``-l korean`` reads the Latin in a Korean document. Measured on
    ``bench/sample-3``, an English scan, the Korean model made 5 character
    errors against the English model's 4.

    Two objections to this rule were raised and both were measured and
    dropped:

    Comparing confidence across two models has no theoretical basis, since
    nothing calibrates them against each other. In practice the separation is
    wide. On page 1 of ``bench/sample-2`` the Korean model averaged 0.862 and
    the English model, reading the same Korean lines, 0.555; confidence picked
    the right one for 30 of the 31 boxes. The obvious alternative -- prefer
    whichever returned more text -- picked right only 23 times, because a
    failing model sometimes answers at length ('A HE (2101E HE)' for
    'A버튼(라이트버튼)').

    A threshold of 0.5 might merge neighbouring lines in dense body text. The
    largest IoU between two genuinely different lines, over all three corpora,
    is 0.10 -- on the tightest-set of them, the letterpress prose in sample-3.
    The threshold has five times the margin it needs.
    """
    sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
    kept: list[tuple[OcrResult, tuple[float, float, float, float]]] = []
    for result in sorted_results:
        rect = _poly_to_rect(result.bbox)
        if all(_rect_iou(rect, k_rect) < iou_threshold for _, k_rect in kept):
            kept.append((result, rect))
    return [r for r, _ in kept]


def sort_reading_order(results: list[OcrResult]) -> list[OcrResult]:
    """Order detections the way the page is read: top to bottom, left to right.

    The detector returns boxes in its own order, which is close to reading
    order but not equal to it. Sorting purely by the top edge would still
    scramble a line, because boxes on one line rarely share a top edge exactly.
    Boxes are therefore grouped into lines by vertical overlap first, and only
    ordered by ``x`` within a line.
    """
    if not results:
        return []

    boxed = [(r, _poly_to_rect(r.bbox)) for r in results]
    boxed.sort(key=lambda item: item[1][1])

    lines: list[list[tuple[OcrResult, tuple[float, float, float, float]]]] = []
    for item in boxed:
        _, (_, y1, _, y2) = item
        centre = (y1 + y2) / 2
        for line in lines:
            ref_y1 = min(rect[1] for _, rect in line)
            ref_y2 = max(rect[3] for _, rect in line)
            # A box joins a line when its centre sits inside that line's band.
            # Centre-in-band rather than IoU: a short box and a full-width one
            # on the same line overlap very little by area.
            if ref_y1 <= centre <= ref_y2:
                line.append(item)
                break
        else:
            lines.append([item])

    ordered = []
    for line in lines:
        line.sort(key=lambda item: item[1][0])
        ordered.extend(result for result, _ in line)
    return ordered


class TextRecogniser(Protocol):
    """What the pipeline needs from a recognition engine.

    The pipeline hands over a page image and receives placed text; it knows
    nothing else about how the reading happened. Keeping that boundary explicit
    is what makes an engine swap a local change rather than one that reaches
    into the overlay and the benchmark.

    An implementation must return results in reading order and must already
    have applied ``confidence``, so the caller cannot forget to. Boxes are in
    the pixel coordinates of the image it was given.
    """

    def recognize(
        self, image: np.ndarray, confidence: float = 0.5
    ) -> list[OcrResult]:
        """Read ``image`` (RGB) and return the lines found, in reading order."""
        ...


class PaddleEngine:
    """Wrapper around PaddleOCR for page-level text recognition."""

    def __init__(
        self,
        lang: str = "en",
        rec_model: str | None = None,
        unwarp: bool = False,
        textline_orientation: bool = DEFAULT_TEXTLINE_ORIENTATION,
    ):
        """Build a recogniser.

        Args:
            lang: PaddleOCR language code, which selects the detection and
                recognition models.
            rec_model: Recognition model name overriding the one ``lang``
                selects, e.g. ``"PP-OCRv5_server_rec"``. Only the languages
                that model's dictionary covers benefit; see
                :func:`server_rec_covers`.
            unwarp: Correct page curvature before detection (UVDoc). Aimed at
                the bound edge of a photographed or flatbed-scanned book.
            textline_orientation: Let PaddleOCR decide per detected line
                whether it is upside down, and turn it before recognition.
                Off by default because it misfires on upright pages and
                corrupts them silently; see
                :data:`DEFAULT_TEXTLINE_ORIENTATION`.

        Note:
            PaddleOCR's own page-orientation option is deliberately left off.
            It turns the image before detecting and returns the boxes in the
            turned frame, so overlaying them on the page a reader has puts the
            footer at the top and pushes a third of the lines off the edge
            where nothing can extract them. Whole-page rotation is handled by
            :func:`detect_page_orientation` and :func:`unrotate_results`
            instead, which keep the two frames in step.
        """
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            use_textline_orientation=textline_orientation,
            lang=lang,
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
            text_det_unclip_ratio=1.8,
            text_recognition_batch_size=16,
            use_doc_unwarping=unwarp,
            use_doc_orientation_classify=False,
            **({"text_recognition_model_name": rec_model} if rec_model else {}),
        )

    def recognize(self, image: np.ndarray, confidence: float = 0.5) -> list[OcrResult]:
        """Run OCR on a page image and return filtered results in reading order.

        Args:
            image: Page image as a numpy array in RGB order.
            confidence: Minimum confidence threshold.

        Returns:
            List of OcrResult with confidence above the threshold.
        """
        # PaddleOCR passes numpy input straight through as BGR (see PaddleX
        # ReadImage), so RGB pages must be converted or the colours invert.
        # Reversing the channel axis is what cv2.COLOR_RGB2BGR does, checked
        # against it; the copy is needed because Paddle wants contiguous input.
        raw = self._ocr.predict(np.ascontiguousarray(image[:, :, ::-1]))
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
        return sort_reading_order(results)


# Retained because ``OcrEngine`` was this class's name before the protocol
# above existed and it is what the pipeline and tests refer to.
OcrEngine = PaddleEngine

# The engines available to build. One entry today; the point of the mapping is
# that adding a second is a line here plus a class, not a change to the
# pipeline. Deliberately not exposed as a CLI option while it holds one name --
# an option with a single choice tells a user nothing and implies a decision
# they do not have. Add `--engine` with a Choice over these keys at the same
# time as the second entry.
#
# Open question, and the reason this seam exists: whether another engine is
# worth having. The candidates and what each would be for:
#
#   RapidOCR    the same PP-OCR models under ONNXRuntime. No accuracy change to
#               be expected -- the point would be dropping the paddlepaddle
#               dependency, which is most of this project's install size.
#   Tesseract   mature and light, but likely behind PP-OCRv5 on Korean.
#   Surya       strong on line detection and reading order; slow without a GPU.
#
# None has been measured. Each means installing a large runtime, which is a
# decision about someone's machine rather than about this code. The benchmark
# in bench/ is what a comparison would run against, and it now covers two
# scripts and three scanners, so the comparison is finally worth running.
ENGINES: dict[str, type] = {"paddle": PaddleEngine}

DEFAULT_ENGINE = "paddle"


def create_engine(name: str = DEFAULT_ENGINE, **options) -> TextRecogniser:
    """Build the named engine.

    Args:
        name: A key of :data:`ENGINES`.
        **options: Passed to the engine's constructor. These are
            engine-specific; ``lang`` is the only one every engine is expected
            to accept.

    Raises:
        ValueError: If ``name`` is not a known engine.
    """
    try:
        engine_cls = ENGINES[name]
    except KeyError:
        raise ValueError(
            f"Unknown OCR engine '{name}'; expected one of "
            f"{', '.join(sorted(ENGINES))}."
        ) from None
    return engine_cls(**options)
