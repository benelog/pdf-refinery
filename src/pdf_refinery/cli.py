"""Command-line interface for PDF Refinery."""

from pathlib import Path

import click

from pdf_refinery.pipeline import run_ocr_pipeline


@click.group()
@click.version_option(package_name="pdf-refinery")
def main():
    """PDF Refinery - Transform scanned PDFs into searchable documents."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Output file path. Defaults to <input>_ocr.pdf.")
@click.option("-l", "--lang", multiple=True, default=("en",),
              help="OCR language code (repeatable, e.g., -l korean -l en).")
@click.option("--dpi", default=300, type=int,
              help="DPI for page rendering.")
@click.option("--pages", default=None, type=str,
              help='Page range to process (e.g., "1-10", "1,3,5").')
@click.option("--confidence", default=0.5, type=float,
              help="Minimum confidence threshold for OCR results.")
@click.option("--force-ocr", is_flag=True,
              help="Re-OCR pages that already contain text, replacing it. "
                   "Such pages are skipped by default.")
@click.option("--font-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Font for the text layer. Use this when the built-in fonts "
                   "cannot encode the document's characters.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
def ocr(input_file: Path, output: Path | None, lang: tuple[str, ...], dpi: int,
        pages: str | None, confidence: float, force_ocr: bool,
        font_file: str | None, verbose: bool):
    """Apply OCR to a scanned PDF to make it searchable."""
    if output is None:
        output = input_file.with_stem(f"{input_file.stem}_ocr")

    run_ocr_pipeline(
        input_path=input_file,
        output_path=output,
        langs=list(lang),
        dpi=dpi,
        pages=pages,
        confidence=confidence,
        verbose=verbose,
        force_ocr=force_ocr,
        font_file=font_file,
    )
