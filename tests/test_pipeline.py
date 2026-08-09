"""Tests for pipeline module."""

import json
from unittest.mock import MagicMock, patch

import click
import fitz
import numpy as np
import pytest

from pdf_refinery.ocr_engine import OcrResult
from pdf_refinery.pdf_writer import OverlayStats
from pdf_refinery.pipeline import (
    PAGE_SEPARATOR,
    parse_page_range,
    progress_path_for,
    run_ocr_pipeline,
)
from tests.helpers import source_pdf, source_pdf_with_text

RECOGNISED = [
    OcrResult(
        text="Recognised line",
        confidence=0.9,
        bbox=[[10, 10], [300, 10], [300, 40], [10, 40]],
    )
]


@pytest.fixture
def fake_ocr():
    """Stand in for rendering and recognition so the pipeline logic runs fast.

    The text overlay itself stays real: the resume tests care about what
    actually ends up in the output PDF.
    """
    with patch("pdf_refinery.pipeline.OcrEngine") as engine_cls, \
         patch("pdf_refinery.pdf_document.Page.to_image") as to_image, \
         patch("pdf_refinery.pipeline.preprocess_image",
               side_effect=lambda img, mode=None: img):
        to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)
        engine_cls.return_value.recognize.return_value = RECOGNISED
        yield engine_cls.return_value


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
    @patch("pdf_refinery.pdf_document.Page.to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_basic_execution(self, mock_engine_cls, mock_to_image, mock_overlay, tmp_pdf, tmp_path):
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = [
            OcrResult(text="Hello", confidence=0.9, bbox=[[0, 0], [50, 0], [50, 20], [0, 20]])
        ]
        mock_engine_cls.return_value = mock_engine
        mock_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)
        mock_overlay.return_value = OverlayStats(1, 0)

        output = tmp_path / "out.pdf"
        run_ocr_pipeline(input_path=tmp_pdf, output_path=output)

        mock_engine.recognize.assert_called_once()
        mock_overlay.assert_called_once()

    @patch("pdf_refinery.pipeline.overlay_text_on_page")
    @patch("pdf_refinery.pdf_document.Page.to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_page_range(self, mock_engine_cls, mock_to_image, mock_overlay, multi_page_pdf, tmp_path):
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = []
        mock_engine_cls.return_value = mock_engine
        mock_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)
        mock_overlay.return_value = OverlayStats(0, 0)

        output = tmp_path / "out.pdf"
        run_ocr_pipeline(input_path=multi_page_pdf, output_path=output, pages="2,4")

        assert mock_overlay.call_count == 2

    @patch("pdf_refinery.pipeline.overlay_text_on_page")
    @patch("pdf_refinery.pdf_document.Page.to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_multiple_languages(self, mock_engine_cls, mock_to_image, mock_overlay, tmp_pdf, tmp_path):
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = []
        mock_engine_cls.return_value = mock_engine
        mock_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)
        mock_overlay.return_value = OverlayStats(0, 0)

        output = tmp_path / "out.pdf"
        run_ocr_pipeline(input_path=tmp_pdf, output_path=output, langs=["en", "korean"])

        assert mock_engine_cls.call_count == 2
        mock_engine_cls.assert_any_call(
            lang="en", rec_model=None, unwarp=False, textline_orientation=False)
        mock_engine_cls.assert_any_call(
            lang="korean", rec_model=None, unwarp=False, textline_orientation=False)

    @patch("pdf_refinery.pipeline.overlay_text_on_page")
    @patch("pdf_refinery.pdf_document.Page.to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_textline_orientation_reaches_the_engine(
        self, mock_engine_cls, mock_to_image, mock_overlay, tmp_pdf, tmp_path
    ):
        # Asked for explicitly it must arrive; it is off by default because it
        # corrupts upright pages without reporting anything (see
        # ocr_engine.DEFAULT_TEXTLINE_ORIENTATION), so a flag that silently
        # failed to reach the engine would be indistinguishable from working.
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = []
        mock_engine_cls.return_value = mock_engine
        mock_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)
        mock_overlay.return_value = OverlayStats(0, 0)

        run_ocr_pipeline(
            input_path=tmp_pdf,
            output_path=tmp_path / "out.pdf",
            textline_orientation=True,
        )

        assert mock_engine_cls.call_args.kwargs["textline_orientation"] is True

    def test_rejects_output_equal_to_input(self, tmp_pdf):
        with pytest.raises(click.ClickException):
            run_ocr_pipeline(input_path=tmp_pdf, output_path=tmp_pdf)

    def test_rejects_page_range_matching_no_page(self, multi_page_pdf, tmp_path):
        # "10-20" on a 5-page file used to process nothing and still report
        # success, leaving an output file that looked complete.
        output = tmp_path / "out.pdf"
        with pytest.raises(click.ClickException, match="selects no pages"):
            run_ocr_pipeline(
                input_path=multi_page_pdf, output_path=output, pages="10-20"
            )
        assert not output.exists()

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
        path.write_bytes(source_pdf_with_text(["Original content"] * 10, fontsize=24))
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

    @patch("pdf_refinery.pdf_document.Page.to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_force_ocr_reprocesses_the_page(
        self, mock_engine_cls, mock_to_image, text_pdf, tmp_path
    ):
        mock_engine_cls.return_value.recognize.return_value = [
            OcrResult(text="Original content", confidence=0.9,
                      bbox=[[100, 100], [400, 100], [400, 140], [100, 140]])
        ]
        mock_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)

        output = tmp_path / "out.pdf"
        run_ocr_pipeline(input_path=text_pdf, output_path=output, force_ocr=True)

        mock_engine_cls.return_value.recognize.assert_called_once()
        doc = fitz.open(output)
        # The text now comes from OCR, and the page still shows its content.
        assert "Original content" in doc[0].get_text()
        doc.close()

    @patch("pdf_refinery.pdf_document.Page.to_image")
    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_a_stray_header_does_not_suppress_the_page(
        self, mock_engine_cls, mock_to_image, tmp_path
    ):
        # A scanned page whose only text layer is a page number must still be
        # read; skipping it would leave the body unsearchable.
        path = tmp_path / "stray.pdf"
        path.write_bytes(source_pdf_with_text(["117"], fontsize=10))

        mock_engine_cls.return_value.recognize.return_value = []
        mock_to_image.return_value = np.zeros((792, 612, 3), dtype=np.uint8)

        run_ocr_pipeline(input_path=path, output_path=tmp_path / "out.pdf")

        mock_engine_cls.return_value.recognize.assert_called_once()

    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_threshold_of_zero_restores_skip_on_any_text(
        self, mock_engine_cls, tmp_path
    ):
        path = tmp_path / "stray.pdf"
        path.write_bytes(source_pdf_with_text(["117"], fontsize=10))

        mock_engine_cls.return_value.recognize.return_value = []

        run_ocr_pipeline(
            input_path=path, output_path=tmp_path / "out.pdf", skip_text_threshold=0
        )

        mock_engine_cls.return_value.recognize.assert_not_called()


class TestVerboseOutput:
    """Verbose logging and the progress bar cannot share a line."""

    def test_progress_bar_is_dropped_when_verbose(self, fake_ocr, multi_page_pdf, tmp_path):
        with patch("pdf_refinery.pipeline.click.progressbar") as bar:
            run_ocr_pipeline(
                input_path=multi_page_pdf,
                output_path=tmp_path / "out.pdf",
                verbose=True,
            )
        bar.assert_not_called()

    def test_progress_bar_is_used_otherwise(self, fake_ocr, multi_page_pdf, tmp_path):
        with patch("pdf_refinery.pipeline.click.progressbar") as bar:
            bar.return_value.__enter__.return_value = []
            run_ocr_pipeline(
                input_path=multi_page_pdf, output_path=tmp_path / "out.pdf"
            )
        bar.assert_called_once()


class TestSidecar:
    def test_writes_one_entry_per_page(self, fake_ocr, multi_page_pdf, tmp_path):
        sidecar = tmp_path / "out.txt"
        run_ocr_pipeline(
            input_path=multi_page_pdf,
            output_path=tmp_path / "out.pdf",
            sidecar=sidecar,
        )

        pages = sidecar.read_text(encoding="utf-8").split(PAGE_SEPARATOR)
        assert len(pages) == 6  # five pages, then a trailing empty tail
        assert pages[0].strip() == "Recognised line"

    @patch("pdf_refinery.pipeline.OcrEngine")
    def test_skipped_pages_are_transcribed_too(self, mock_engine_cls, tmp_path):
        # The sidecar is a transcript of the document, not a log of OCR calls.
        path = tmp_path / "with_text.pdf"
        path.write_bytes(source_pdf_with_text(["Original content"] * 10, fontsize=24))

        mock_engine_cls.return_value.recognize.return_value = []
        sidecar = tmp_path / "out.txt"
        run_ocr_pipeline(
            input_path=path, output_path=tmp_path / "out.pdf", sidecar=sidecar
        )

        assert "Original content" in sidecar.read_text(encoding="utf-8")


class TestCheckpointAndResume:
    """An interrupted book-length run must not lose everything."""

    def test_progress_file_is_gone_after_a_clean_run(
        self, fake_ocr, multi_page_pdf, tmp_path
    ):
        output = tmp_path / "out.pdf"
        run_ocr_pipeline(
            input_path=multi_page_pdf, output_path=output, checkpoint_every=2
        )
        assert not progress_path_for(output).exists()

    def test_interrupted_run_keeps_the_pages_it_finished(
        self, fake_ocr, multi_page_pdf, tmp_path
    ):
        output = tmp_path / "out.pdf"
        fake_ocr.recognize.side_effect = [
            RECOGNISED, RECOGNISED, RuntimeError("killed mid-run")
        ]

        with pytest.raises(RuntimeError):
            run_ocr_pipeline(
                input_path=multi_page_pdf, output_path=output, checkpoint_every=2
            )

        assert output.exists()
        state = json.loads(progress_path_for(output).read_text(encoding="utf-8"))
        assert state["done"] == [0, 1]
        assert state["input"] == str(multi_page_pdf.resolve())

        doc = fitz.open(output)
        assert "Recognised line" in doc[0].get_text()
        doc.close()

    def test_resume_only_processes_what_is_left(
        self, fake_ocr, multi_page_pdf, tmp_path
    ):
        output = tmp_path / "out.pdf"
        fake_ocr.recognize.side_effect = [
            RECOGNISED, RECOGNISED, RuntimeError("killed mid-run")
        ]
        with pytest.raises(RuntimeError):
            run_ocr_pipeline(
                input_path=multi_page_pdf, output_path=output, checkpoint_every=2
            )

        fake_ocr.recognize.side_effect = None
        fake_ocr.recognize.return_value = RECOGNISED
        fake_ocr.recognize.reset_mock()

        run_ocr_pipeline(
            input_path=multi_page_pdf,
            output_path=output,
            checkpoint_every=2,
            resume=True,
        )

        assert fake_ocr.recognize.call_count == 3  # pages 3, 4 and 5 only
        assert not progress_path_for(output).exists()

        doc = fitz.open(output)
        assert all("Recognised line" in page.get_text() for page in doc)
        doc.close()

    def test_resume_appends_to_the_sidecar_without_repeating(
        self, fake_ocr, multi_page_pdf, tmp_path
    ):
        output = tmp_path / "out.pdf"
        sidecar = tmp_path / "out.txt"
        fake_ocr.recognize.side_effect = [
            RECOGNISED, RECOGNISED, RuntimeError("killed mid-run")
        ]
        with pytest.raises(RuntimeError):
            run_ocr_pipeline(
                input_path=multi_page_pdf, output_path=output,
                checkpoint_every=2, sidecar=sidecar,
            )
        assert sidecar.read_text(encoding="utf-8").count(PAGE_SEPARATOR) == 2

        fake_ocr.recognize.side_effect = None
        run_ocr_pipeline(
            input_path=multi_page_pdf, output_path=output,
            checkpoint_every=2, sidecar=sidecar, resume=True,
        )
        assert sidecar.read_text(encoding="utf-8").count(PAGE_SEPARATOR) == 5

    def test_resume_without_a_checkpoint_is_refused(self, multi_page_pdf, tmp_path):
        with pytest.raises(click.ClickException, match="Nothing to resume"):
            run_ocr_pipeline(
                input_path=multi_page_pdf,
                output_path=tmp_path / "out.pdf",
                resume=True,
            )

    def test_resume_refuses_a_checkpoint_from_another_document(
        self, fake_ocr, multi_page_pdf, tmp_path
    ):
        output = tmp_path / "out.pdf"
        fake_ocr.recognize.side_effect = [
            RECOGNISED, RECOGNISED, RuntimeError("killed mid-run")
        ]
        with pytest.raises(RuntimeError):
            run_ocr_pipeline(
                input_path=multi_page_pdf, output_path=output, checkpoint_every=2
            )

        other = tmp_path / "other.pdf"
        other.write_bytes(source_pdf())

        fake_ocr.recognize.side_effect = None
        with pytest.raises(click.ClickException, match="belongs to a run over"):
            run_ocr_pipeline(input_path=other, output_path=output, resume=True)

    def test_checkpointing_off_leaves_no_progress_file(
        self, fake_ocr, multi_page_pdf, tmp_path
    ):
        output = tmp_path / "out.pdf"
        fake_ocr.recognize.side_effect = [
            RECOGNISED, RECOGNISED, RuntimeError("killed mid-run")
        ]
        with pytest.raises(RuntimeError):
            run_ocr_pipeline(
                input_path=multi_page_pdf, output_path=output, checkpoint_every=0
            )
        assert not progress_path_for(output).exists()
        assert not output.exists()

    def test_resume_finishes_a_document_that_already_has_a_text_layer(
        self, fake_ocr, multi_page_pdf, tmp_path
    ):
        # A resumed run reopens an output the earlier run already wrote a font
        # and a text layer into, and keeps writing into it. It does not go
        # looking for that font -- the pages it adds embed their own copy -- so
        # the size check below is what keeps that from mattering: a duplicate
        # is a couple of kilobytes and does not grow with the page count.
        output = tmp_path / "out.pdf"
        fake_ocr.recognize.side_effect = [
            RECOGNISED, RECOGNISED, RuntimeError("killed mid-run")
        ]
        with pytest.raises(RuntimeError):
            run_ocr_pipeline(
                input_path=multi_page_pdf, output_path=output, checkpoint_every=2,
            )

        fake_ocr.recognize.side_effect = None
        run_ocr_pipeline(
            input_path=multi_page_pdf, output_path=output,
            checkpoint_every=2, resume=True,
        )

        doc = fitz.open(output)
        assert all("Recognised line" in page.get_text() for page in doc)
        doc.close()
        assert output.stat().st_size < 16_000
