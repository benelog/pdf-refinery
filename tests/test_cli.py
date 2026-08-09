"""Tests for CLI module."""

import subprocess
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pdf_refinery.cli import main
from pdf_refinery.pipeline import progress_path_for


@patch("pdf_refinery.pipeline.run_ocr_pipeline")
class TestCli:
    def test_default_output_name(self, mock_pipeline, tmp_pdf):
        runner = CliRunner()
        result = runner.invoke(main, ["ocr", str(tmp_pdf), "-l", "en"])

        assert result.exit_code == 0
        call_kwargs = mock_pipeline.call_args.kwargs
        expected_output = tmp_pdf.with_stem(f"{tmp_pdf.stem}_ocr")
        assert call_kwargs["output_path"] == expected_output

    def test_custom_output(self, mock_pipeline, tmp_pdf, tmp_path):
        out = tmp_path / "custom.pdf"
        runner = CliRunner()
        result = runner.invoke(main, ["ocr", str(tmp_pdf), "-o", str(out), "-l", "en"])

        assert result.exit_code == 0
        assert mock_pipeline.call_args.kwargs["output_path"] == out

    def test_multiple_languages(self, mock_pipeline, tmp_pdf):
        runner = CliRunner()
        result = runner.invoke(main, ["ocr", str(tmp_pdf), "-l", "korean", "-l", "en"])

        assert result.exit_code == 0
        assert mock_pipeline.call_args.kwargs["langs"] == ["korean", "en"]

    def test_all_options_passed(self, mock_pipeline, tmp_pdf, tmp_path):
        out = tmp_path / "out.pdf"
        runner = CliRunner()
        result = runner.invoke(main, [
            "ocr", str(tmp_pdf), "-l", "en",
            "-o", str(out),
            "-l", "en",
            "--dpi", "150",
            "--pages", "1-3",
            "--confidence", "0.8",
            "-v",
        ])

        assert result.exit_code == 0
        kw = mock_pipeline.call_args.kwargs
        assert kw["dpi"] == 150
        assert kw["pages"] == "1-3"
        assert kw["confidence"] == 0.8
        assert kw["verbose"] is True

    def test_missing_input_file(self, mock_pipeline):
        runner = CliRunner()
        result = runner.invoke(main, ["ocr"])

        assert result.exit_code != 0
        mock_pipeline.assert_not_called()

    def test_nonexistent_file(self, mock_pipeline):
        runner = CliRunner()
        result = runner.invoke(main, ["ocr", "/nonexistent/file.pdf"])

        assert result.exit_code != 0
        mock_pipeline.assert_not_called()


@patch("pdf_refinery.pipeline.run_ocr_pipeline")
class TestOptionValidation:
    """Bad input must fail before the pipeline starts, not after an hour."""

    def _fails(self, runner, mock_pipeline, args, expected):
        result = runner.invoke(main, args)
        assert result.exit_code != 0
        assert expected in result.output
        mock_pipeline.assert_not_called()

    def test_a_missing_language_is_refused_rather_than_guessed(
        self, mock_pipeline, tmp_pdf
    ):
        """No default language, because the wrong one is a total loss.

        A Korean scan recognised as English measured 97% character error --
        not a worse result but an empty one -- and the run still exited zero
        and wrote an output PDF of a believable size. Any default merely
        chooses whose documents fail that way.
        """
        result = CliRunner().invoke(main, ["ocr", str(tmp_pdf)])

        assert result.exit_code != 0
        assert "No language given" in result.output
        assert "-l korean" in result.output
        mock_pipeline.assert_not_called()

    def test_unknown_language_suggests_the_real_code(self, mock_pipeline, tmp_pdf):
        result = CliRunner().invoke(main, ["ocr", str(tmp_pdf), "-l", "ko"])

        assert result.exit_code != 0
        assert "Did you mean 'korean'?" in result.output
        mock_pipeline.assert_not_called()

    @pytest.mark.parametrize("code", ["zzz", "jp", "cn"])
    def test_unknown_languages_are_rejected(self, mock_pipeline, tmp_pdf, code):
        self._fails(CliRunner(), mock_pipeline, ["ocr", str(tmp_pdf), "-l", code],
                    "is not a PaddleOCR language code")

    @pytest.mark.parametrize("code", ["en", "korean", "japan", "ch"])
    def test_known_languages_pass(self, mock_pipeline, tmp_pdf, code):
        result = CliRunner().invoke(main, ["ocr", str(tmp_pdf), "-l", code])
        assert result.exit_code == 0

    @pytest.mark.parametrize("pages", ["abc", "1-", "0", "10-1", "", "1,,3"])
    def test_malformed_page_ranges_are_rejected(self, mock_pipeline, tmp_pdf, pages):
        self._fails(CliRunner(), mock_pipeline,
                    ["ocr", str(tmp_pdf), "-l", "en", "--pages", pages], "--pages")

    @pytest.mark.parametrize("pages", ["1-10", "1,3,5", "1-3,7,10-12", " 1 , 3 "])
    def test_valid_page_ranges_pass(self, mock_pipeline, tmp_pdf, pages):
        result = CliRunner().invoke(main, ["ocr", str(tmp_pdf), "-l", "en", "--pages", pages])
        assert result.exit_code == 0

    def test_confidence_given_as_a_percentage_is_rejected(self, mock_pipeline, tmp_pdf):
        # A run with --confidence 50 would filter out every result and finish
        # with an empty text layer.
        self._fails(CliRunner(), mock_pipeline,
                    ["ocr", str(tmp_pdf), "-l", "en", "--confidence", "50"], "--confidence")

    def test_absurd_dpi_is_rejected(self, mock_pipeline, tmp_pdf):
        self._fails(CliRunner(), mock_pipeline,
                    ["ocr", str(tmp_pdf), "-l", "en", "--dpi", "5000"], "--dpi")


@patch("pdf_refinery.pipeline.run_ocr_pipeline")
class TestOutputOverwrite:
    def test_existing_output_is_not_replaced_silently(self, mock_pipeline, tmp_pdf, tmp_path):
        out = tmp_path / "out.pdf"
        out.write_bytes(b"previous result")

        result = CliRunner().invoke(main, ["ocr", str(tmp_pdf), "-l", "en", "-o", str(out)])

        assert result.exit_code != 0
        assert "--overwrite" in result.output
        mock_pipeline.assert_not_called()
        assert out.read_bytes() == b"previous result"

    def test_overwrite_flag_allows_it(self, mock_pipeline, tmp_pdf, tmp_path):
        out = tmp_path / "out.pdf"
        out.write_bytes(b"previous result")

        result = CliRunner().invoke(
            main, ["ocr", str(tmp_pdf), "-l", "en", "-o", str(out), "--overwrite"]
        )

        assert result.exit_code == 0
        mock_pipeline.assert_called_once()

    def test_default_output_name_is_guarded_too(self, mock_pipeline, tmp_pdf):
        default = tmp_pdf.with_stem(f"{tmp_pdf.stem}_ocr")
        default.write_bytes(b"previous result")

        result = CliRunner().invoke(main, ["ocr", str(tmp_pdf), "-l", "en"])

        assert result.exit_code != 0
        mock_pipeline.assert_not_called()


@patch("pdf_refinery.pipeline.run_ocr_pipeline")
class TestResumeOptions:
    def test_resume_and_overwrite_conflict(self, mock_pipeline, tmp_pdf):
        result = CliRunner().invoke(
            main, ["ocr", str(tmp_pdf), "-l", "en", "--resume", "--overwrite"]
        )
        assert result.exit_code != 0
        assert "Pick one" in result.output
        mock_pipeline.assert_not_called()

    def test_resume_accepts_the_existing_output(self, mock_pipeline, tmp_pdf, tmp_path):
        out = tmp_path / "out.pdf"
        out.write_bytes(b"half-finished run")

        result = CliRunner().invoke(
            main, ["ocr", str(tmp_pdf), "-l", "en", "-o", str(out), "--resume"]
        )

        assert result.exit_code == 0
        assert mock_pipeline.call_args.kwargs["resume"] is True

    def test_an_interrupted_run_is_pointed_at_resume(self, mock_pipeline, tmp_pdf, tmp_path):
        out = tmp_path / "out.pdf"
        out.write_bytes(b"half-finished run")
        progress_path_for(out).write_text("{}")

        result = CliRunner().invoke(main, ["ocr", str(tmp_pdf), "-l", "en", "-o", str(out)])

        assert result.exit_code != 0
        assert "--resume" in result.output
        mock_pipeline.assert_not_called()

    def test_sidecar_is_not_overwritten_silently(self, mock_pipeline, tmp_pdf, tmp_path):
        sidecar = tmp_path / "out.txt"
        sidecar.write_text("previous transcript")

        result = CliRunner().invoke(
            main, ["ocr", str(tmp_pdf), "-l", "en", "--sidecar", str(sidecar)]
        )

        assert result.exit_code != 0
        assert "--overwrite" in result.output
        mock_pipeline.assert_not_called()
        assert sidecar.read_text() == "previous transcript"

    def test_new_options_reach_the_pipeline(self, mock_pipeline, tmp_pdf, tmp_path):
        sidecar = tmp_path / "out.txt"
        result = CliRunner().invoke(main, [
            "ocr", str(tmp_pdf), "-l", "en",
            "--sidecar", str(sidecar),
            "--checkpoint-every", "5",
            "--skip-text-threshold", "40",
        ])

        assert result.exit_code == 0
        kw = mock_pipeline.call_args.kwargs
        assert kw["sidecar"] == sidecar
        assert kw["checkpoint_every"] == 5
        assert kw["skip_text_threshold"] == 40


class TestStartupCost:
    """--help must not pay for PaddleOCR's import or its network check."""

    def test_help_does_not_import_paddleocr(self):
        script = (
            "import sys\n"
            "from click.testing import CliRunner\n"
            "from pdf_refinery.cli import main\n"
            "result = CliRunner().invoke(main, ['ocr', '--help'])\n"
            "assert result.exit_code == 0, result.output\n"
            "print('LOADED', 'paddleocr' in sys.modules)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.splitlines()[-1] == "LOADED False"
