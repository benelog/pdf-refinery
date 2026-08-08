"""Integration tests - run the full OCR pipeline with real PaddleOCR."""

import fitz
import numpy as np
import pytest

from pdf_refinery.ocr_engine import OcrEngine, preprocess_image, deduplicate_results, OcrResult
from pdf_refinery.pdf_reader import page_to_image
from pdf_refinery.pdf_writer import overlay_text_on_page, remove_text_layer
from pdf_refinery.pipeline import run_ocr_pipeline


def _create_scanned_pdf(path, texts, font_size=36):
    """Create a PDF that simulates a scanned document.

    Renders text onto a page, rasterizes it to an image,
    then embeds that image as a new PDF (no selectable text layer).
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    y = 100
    for text in texts:
        page.insert_text(fitz.Point(72, y), text, fontsize=font_size)
        y += font_size * 2

    # Rasterize to image
    pixmap = page.get_pixmap(dpi=300)
    doc.close()

    # Create a new PDF with only the image (no text layer)
    img_doc = fitz.open()
    img_page = img_doc.new_page(width=612, height=792)
    img_page.insert_image(img_page.rect, pixmap=pixmap)
    img_doc.save(path)
    img_doc.close()


def _create_multipage_scanned_pdf(path, pages_texts, font_size=36):
    """Create a multi-page scanned PDF."""
    # First render all pages with text
    rendered = []
    for texts in pages_texts:
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        y = 100
        for text in texts:
            page.insert_text(fitz.Point(72, y), text, fontsize=font_size)
            y += font_size * 2
        rendered.append(page.get_pixmap(dpi=300))
        doc.close()

    # Create image-only PDF
    img_doc = fitz.open()
    for pixmap in rendered:
        img_page = img_doc.new_page(width=612, height=792)
        img_page.insert_image(img_page.rect, pixmap=pixmap)
    img_doc.save(path)
    img_doc.close()


@pytest.fixture
def scanned_pdf(tmp_path):
    path = tmp_path / "scanned.pdf"
    _create_scanned_pdf(path, ["Hello World", "Integration Test"])
    return path


@pytest.fixture
def multipage_scanned_pdf(tmp_path):
    path = tmp_path / "multipage.pdf"
    _create_multipage_scanned_pdf(path, [
        ["Page One", "First Content"],
        ["Page Two", "Second Content"],
        ["Page Three", "Third Content"],
    ])
    return path


class TestPreprocessImage:
    def test_output_shape_matches_input(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocess_image(img, mode="binarize")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_binarize_produces_binary_output(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_image(img, mode="binarize")
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_the_default_thresholds_the_page(self):
        """The default binarizes, against what the theory would predict.

        PP-OCRv5 is trained on anti-aliased greyscale, so handing it a
        threshold should lose information -- and the character error barely
        moves either way. What moves is word boundaries: on the two Korean
        corpora in bench/ the recogniser ran phrases together when given
        greyscale, and word errors fell from 403 to 117 and 253 to 99 when it
        was given black and white. Left untouched, the page still works;
        searching it for a word often does not.
        """
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert set(np.unique(preprocess_image(img))).issubset({0, 255})
        assert np.array_equal(preprocess_image(img, mode="none"), img)

    def test_an_unknown_mode_is_refused(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="binarize"):
            preprocess_image(img, mode="adaptive")


class TestDeduplicateResults:
    def test_removes_overlapping_lower_confidence(self):
        high = OcrResult(text="hello", confidence=0.9, bbox=[[0, 0], [100, 0], [100, 20], [0, 20]])
        low = OcrResult(text="hello", confidence=0.6, bbox=[[5, 2], [105, 2], [105, 22], [5, 22]])
        result = deduplicate_results([low, high])
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_keeps_non_overlapping(self):
        a = OcrResult(text="left", confidence=0.9, bbox=[[0, 0], [50, 0], [50, 20], [0, 20]])
        b = OcrResult(text="right", confidence=0.8, bbox=[[200, 0], [300, 0], [300, 20], [200, 20]])
        result = deduplicate_results([a, b])
        assert len(result) == 2

    def test_empty_input(self):
        assert deduplicate_results([]) == []

    def test_single_input(self):
        r = OcrResult(text="only", confidence=0.9, bbox=[[0, 0], [50, 0], [50, 20], [0, 20]])
        result = deduplicate_results([r])
        assert len(result) == 1


class TestOcrEngineIntegration:
    """Tests that exercise real PaddleOCR inference."""

    def test_recognizes_text_from_rendered_page(self, scanned_pdf):
        doc = fitz.open(scanned_pdf)
        page = doc[0]
        image = page_to_image(page, dpi=300)
        doc.close()

        preprocessed = preprocess_image(image)
        engine = OcrEngine(lang="en")
        results = engine.recognize(preprocessed, confidence=0.5)

        recognized_text = " ".join(r.text for r in results).lower()
        assert "hello" in recognized_text or "world" in recognized_text

    def test_returns_valid_bboxes(self, scanned_pdf):
        doc = fitz.open(scanned_pdf)
        page = doc[0]
        image = page_to_image(page, dpi=300)
        doc.close()

        preprocessed = preprocess_image(image)
        engine = OcrEngine(lang="en")
        results = engine.recognize(preprocessed, confidence=0.5)

        assert len(results) > 0
        for r in results:
            assert len(r.bbox) == 4
            for point in r.bbox:
                assert len(point) == 2
                assert all(isinstance(v, (int, float)) for v in point)


class TestOverlayIntegration:
    def test_overlay_makes_text_searchable(self, scanned_pdf):
        doc = fitz.open(scanned_pdf)
        page = doc[0]

        # Confirm no text layer initially
        assert page.get_text().strip() == ""

        image = page_to_image(page, dpi=300)
        img_h, img_w = image.shape[:2]
        preprocessed = preprocess_image(image)

        engine = OcrEngine(lang="en")
        results = engine.recognize(preprocessed, confidence=0.5)
        overlay_text_on_page(page, results, img_w, img_h)

        # Now the page should have searchable text
        text = page.get_text().strip().lower()
        assert len(text) > 0
        doc.close()


class TestRemoveTextLayerIntegration:
    def test_removes_text_preserves_image(self, scanned_pdf):
        doc = fitz.open(scanned_pdf)
        page = doc[0]

        # Add text, then remove it
        page.insert_text(fitz.Point(100, 400), "Temporary text", fontsize=14)
        assert "Temporary text" in page.get_text()

        remove_text_layer(page)
        assert page.get_text().strip() == ""

        # Image should still be present
        images = page.get_images()
        assert len(images) > 0
        doc.close()


class TestFullPipeline:
    def test_single_page(self, scanned_pdf, tmp_path):
        output = tmp_path / "output.pdf"
        run_ocr_pipeline(input_path=scanned_pdf, output_path=output)

        assert output.exists()
        doc = fitz.open(output)
        assert len(doc) == 1
        text = doc[0].get_text().strip().lower()
        assert len(text) > 0
        doc.close()

    def test_output_contains_recognized_words(self, scanned_pdf, tmp_path):
        output = tmp_path / "output.pdf"
        run_ocr_pipeline(input_path=scanned_pdf, output_path=output)

        doc = fitz.open(output)
        text = doc[0].get_text().strip().lower()
        # At least one of the original words should be recognized
        assert any(w in text for w in ["hello", "world", "integration", "test"])
        doc.close()

    def test_multipage_all_pages_processed(self, multipage_scanned_pdf, tmp_path):
        output = tmp_path / "output.pdf"
        run_ocr_pipeline(input_path=multipage_scanned_pdf, output_path=output)

        doc = fitz.open(output)
        assert len(doc) == 3
        for i in range(3):
            text = doc[i].get_text().strip()
            assert len(text) > 0, f"Page {i+1} has no text"
        doc.close()

    def test_page_range(self, multipage_scanned_pdf, tmp_path):
        output = tmp_path / "output.pdf"
        run_ocr_pipeline(
            input_path=multipage_scanned_pdf, output_path=output, pages="1,3",
        )

        doc = fitz.open(output)
        assert len(doc) == 3
        # Pages 1 and 3 should have text, page 2 should not
        assert len(doc[0].get_text().strip()) > 0
        assert doc[1].get_text().strip() == ""
        assert len(doc[2].get_text().strip()) > 0
        doc.close()

    def test_confidence_threshold(self, scanned_pdf, tmp_path):
        output_low = tmp_path / "low.pdf"
        output_high = tmp_path / "high.pdf"

        run_ocr_pipeline(input_path=scanned_pdf, output_path=output_low, confidence=0.3)
        run_ocr_pipeline(input_path=scanned_pdf, output_path=output_high, confidence=0.99)

        doc_low = fitz.open(output_low)
        doc_high = fitz.open(output_high)
        text_low = doc_low[0].get_text().strip()
        text_high = doc_high[0].get_text().strip()
        # Higher threshold should yield equal or fewer text blocks
        assert len(text_low) >= len(text_high)
        doc_low.close()
        doc_high.close()

    def test_verbose_output(self, scanned_pdf, tmp_path, capsys):
        output = tmp_path / "output.pdf"
        run_ocr_pipeline(input_path=scanned_pdf, output_path=output, verbose=True)

        captured = capsys.readouterr()
        assert "text blocks detected" in captured.out

    def test_preserves_original_file(self, scanned_pdf, tmp_path):
        original_bytes = scanned_pdf.read_bytes()
        output = tmp_path / "output.pdf"
        run_ocr_pipeline(input_path=scanned_pdf, output_path=output)

        assert scanned_pdf.read_bytes() == original_bytes
