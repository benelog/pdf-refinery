"""Tests for CLI module."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from pdf_refinery.cli import main


@patch("pdf_refinery.cli.run_ocr_pipeline")
class TestCli:
    def test_default_output_name(self, mock_pipeline, tmp_pdf):
        runner = CliRunner()
        result = runner.invoke(main, ["ocr", str(tmp_pdf)])

        assert result.exit_code == 0
        call_kwargs = mock_pipeline.call_args.kwargs
        expected_output = tmp_pdf.with_stem(f"{tmp_pdf.stem}_ocr")
        assert call_kwargs["output_path"] == expected_output

    def test_custom_output(self, mock_pipeline, tmp_pdf, tmp_path):
        out = tmp_path / "custom.pdf"
        runner = CliRunner()
        result = runner.invoke(main, ["ocr", str(tmp_pdf), "-o", str(out)])

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
            "ocr", str(tmp_pdf),
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
