"""Tests for OCR engine module."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_refinery.ocr_engine import (
    DEFAULT_PREPROCESS,
    DEFAULT_TEXTLINE_ORIENTATION,
    OcrEngine,
    OcrResult,
    PaddleEngine,
    create_engine,
    preprocess_image,
    unrotate_results,
    upright,
)


class TestOcrResult:
    def test_field_access(self):
        r = OcrResult(text="hello", confidence=0.95, bbox=[[0, 0], [10, 0], [10, 10], [0, 10]])
        assert r.text == "hello"
        assert r.confidence == 0.95
        assert len(r.bbox) == 4


class TestOcrEngine:
    @patch("paddleocr.PaddleOCR")
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

    @patch("paddleocr.PaddleOCR")
    def test_empty_result(self, mock_paddle_cls):
        mock_ocr = MagicMock()
        mock_paddle_cls.return_value = mock_ocr
        mock_ocr.predict.return_value = []

        engine = OcrEngine(lang="en")
        results = engine.recognize(np.zeros((100, 100, 3), dtype=np.uint8))

        assert results == []

    @patch("paddleocr.PaddleOCR")
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


class TestWholePageRotation:
    """A sideways page must come back with its text on the right words.

    PaddleOCR's own page-orientation option reads such a page correctly and
    then reports the boxes in the turned frame, which puts the text layer
    somewhere the page does not have. Doing the turn here means the inverse is
    ours to apply -- and getting that inverse backwards is easy, which is what
    these tests are for.
    """

    @staticmethod
    def _marked_page():
        page = np.zeros((40, 30, 3), dtype=np.uint8)
        page[3:6, 2:9] = 255  # one "line of text" near the top left
        return page

    @staticmethod
    def _ink_box(image):
        ys, xs = np.where(image[..., 0] == 255)
        return xs.min(), ys.min(), xs.max(), ys.max()

    @pytest.mark.parametrize("rotation", [0, 90, 180, 270])
    def test_upright_undoes_the_scan_rotation(self, rotation):
        original = self._marked_page()
        scanned = np.ascontiguousarray(np.rot90(original, k=-(rotation // 90)))
        assert np.array_equal(upright(scanned, rotation), original)

    @pytest.mark.parametrize("rotation", [0, 90, 180, 270])
    def test_boxes_land_where_the_ink_is_on_the_page_as_scanned(self, rotation):
        original = self._marked_page()
        scanned = np.ascontiguousarray(np.rot90(original, k=-(rotation // 90)))
        straight = upright(scanned, rotation)

        x0, y0, x1, y1 = self._ink_box(straight)
        found = OcrResult(
            text="line", confidence=0.9,
            bbox=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        )
        mapped = unrotate_results([found], rotation, straight.shape)[0]

        xs = [p[0] for p in mapped.bbox]
        ys = [p[1] for p in mapped.bbox]
        want = self._ink_box(scanned)
        got = (min(xs), min(ys), max(xs), max(ys))
        assert all(abs(a - b) <= 1 for a, b in zip(got, want))

    def test_no_rotation_leaves_the_results_alone(self):
        r = [OcrResult("x", 0.9, [[1, 2], [3, 2], [3, 4], [1, 4]])]
        assert unrotate_results(r, 0, (40, 30)) is r


class TestPreprocessDefault:
    """Pages are thresholded to black and white before recognition.

    Counter-intuitive -- PP-OCRv5 is trained on anti-aliased greyscale -- and
    it flipped twice on evidence that was really measuring the
    textline-orientation bug. Measured again over three corpora after that fix,
    ``binarize`` cut word errors from 403 to 117 on sample-1 and 253 to 99 on
    sample-2, for two words lost on sample-3. Asserted here so the value is a
    decision with a reason rather than something that drifts back.
    """

    def test_binarize_is_the_default(self):
        assert DEFAULT_PREPROCESS == "binarize"

    def test_binarize_produces_two_tone_rgb(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        out = preprocess_image(img, mode="binarize")
        assert out.shape == img.shape
        assert set(np.unique(out)) <= {0, 255}

    def test_an_unknown_mode_is_refused(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="binarize"):
            preprocess_image(img, mode="sharpen")


class TestEngineSelection:
    """The pipeline depends on the protocol, not on PaddleOCR."""

    def test_paddle_engine_satisfies_the_protocol(self):
        assert isinstance(PaddleEngine, type)
        assert hasattr(PaddleEngine, "recognize")

    @patch("paddleocr.PaddleOCR")
    def test_create_engine_builds_the_default(self, mock_paddle_cls):
        engine = create_engine(lang="korean")
        assert isinstance(engine, PaddleEngine)

    @patch("paddleocr.PaddleOCR")
    def test_options_reach_the_engine(self, mock_paddle_cls):
        create_engine("paddle", lang="korean", unwarp=True)
        assert mock_paddle_cls.call_args.kwargs["use_doc_unwarping"] is True

    def test_an_unknown_engine_names_the_ones_that_exist(self):
        with pytest.raises(ValueError, match="paddle"):
            create_engine("tesseract", lang="en")


class TestOpenCvIsOptional:
    """Only ``--preprocess binarize`` may need OpenCV.

    The package cannot declare opencv-python: paddlex pins
    opencv-contrib-python to an exact version and both ship the same ``cv2``
    module, so naming the other one installs a second copy over the first.
    Keeping cv2 off the default path means an environment without it fails on
    that one option rather than on startup. See pyproject.toml.
    """

    def test_importing_the_module_does_not_pull_in_cv2(self):
        script = (
            "import sys\n"
            "import pdf_refinery.ocr_engine\n"
            "print('LOADED', 'cv2' in sys.modules)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.splitlines()[-1] == "LOADED False"

    def test_the_default_mode_returns_the_page_untouched(self):
        img = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
        assert preprocess_image(img, mode="none") is img

    @patch("paddleocr.PaddleOCR")
    def test_channels_are_reversed_for_paddle(self, mock_paddle_cls):
        # PaddleX reads a numpy page as BGR while the renderer produces RGB.
        # This used to be cv2.cvtColor; the replacement has to match it exactly
        # or every page is recognised through inverted colours.
        mock_ocr = MagicMock()
        mock_paddle_cls.return_value = mock_ocr
        mock_ocr.predict.return_value = []

        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 0] = 10  # R
        rgb[..., 2] = 30  # B
        OcrEngine(lang="en").recognize(rgb)

        passed = mock_ocr.predict.call_args.args[0]
        assert passed[0, 0, 0] == 30 and passed[0, 0, 2] == 10
        assert passed.flags["C_CONTIGUOUS"]


class TestTextlineOrientation:
    """The per-line orientation classifier stays off unless asked for.

    It judged 14 of the 31 lines on an upright page of ``bench/sample-2`` to be
    upside down and had them recognised that way, which does not fail loudly --
    upside-down Hangul comes back as different, plausible Hangul. Measured
    CER-ns on that page went from 0.0251 to 0.4571 with it on. The value is
    asserted here rather than left to the library because it is PaddleOCR's own
    default, so an upgrade could reinstate it silently.
    """

    def test_it_is_off_by_default(self):
        assert DEFAULT_TEXTLINE_ORIENTATION is False

    @patch("paddleocr.PaddleOCR")
    def test_default_engine_does_not_enable_it(self, mock_paddle_cls):
        OcrEngine(lang="korean")
        _, kwargs = mock_paddle_cls.call_args
        assert kwargs["use_textline_orientation"] is False

    @patch("paddleocr.PaddleOCR")
    def test_it_can_still_be_asked_for(self, mock_paddle_cls):
        OcrEngine(lang="korean", textline_orientation=True)
        _, kwargs = mock_paddle_cls.call_args
        assert kwargs["use_textline_orientation"] is True
