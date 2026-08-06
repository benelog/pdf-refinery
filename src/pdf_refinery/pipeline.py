"""OCR pipeline orchestration."""

from pathlib import Path

import click

from pdf_refinery.fonts import FontResolver
from pdf_refinery.ocr_engine import OcrEngine, deduplicate_results, preprocess_image
from pdf_refinery.pdf_reader import open_pdf, page_to_image
from pdf_refinery.pdf_writer import (
    has_text,
    overlay_text_on_page,
    rasterize_page,
)


def parse_page_range(pages_str: str, total_pages: int) -> list[int]:
    """Parse a page range string into a list of 0-based page indices.

    Supports formats like "1-10", "1,3,5", "1-3,7,10-12".
    Input is 1-based, output is 0-based.
    """
    indices = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start))
            end = min(total_pages, int(end))
            indices.update(range(start - 1, end))
        else:
            idx = int(part) - 1
            if 0 <= idx < total_pages:
                indices.add(idx)
    return sorted(indices)


def run_ocr_pipeline(
    input_path: Path,
    output_path: Path,
    langs: list[str] | None = None,
    dpi: int = 300,
    pages: str | None = None,
    confidence: float = 0.5,
    verbose: bool = False,
    force_ocr: bool = False,
    font_file: str | None = None,
) -> None:
    """Run the full OCR pipeline on a scanned PDF.

    Pages that already contain text are left untouched unless ``force_ocr`` is
    set, in which case they are flattened to an image before being re-read.

    Args:
        input_path: Path to the input scanned PDF.
        output_path: Path for the output searchable PDF.
        langs: List of PaddleOCR language codes.
        dpi: DPI for page rendering.
        pages: Optional page range string (1-based).
        confidence: Minimum OCR confidence threshold.
        verbose: Enable verbose output.
        force_ocr: Re-OCR pages that already contain text, replacing it.
        font_file: Font used for the text layer, overriding the built-in fonts.
    """
    if langs is None:
        langs = ["en"]
    if input_path.resolve() == output_path.resolve():
        raise click.ClickException("Output path must differ from the input path.")

    doc = open_pdf(input_path)
    total_pages = len(doc)

    if pages:
        page_indices = parse_page_range(pages, total_pages)
    else:
        page_indices = list(range(total_pages))

    click.echo(f"Processing {len(page_indices)} page(s) from '{input_path.name}'...")
    click.echo(f"Languages: {', '.join(langs)}")

    engines = [OcrEngine(lang=lang) for lang in langs]
    font_resolver = FontResolver(font_file=font_file)
    total_blocks = 0
    skipped = 0

    with click.progressbar(page_indices, label="OCR progress") as bar:
        for page_idx in bar:
            page = doc[page_idx]

            if has_text(page):
                if not force_ocr:
                    skipped += 1
                    if verbose:
                        click.echo(
                            f"\n  Page {page_idx + 1}: already has text, skipped"
                        )
                    continue
                rasterize_page(page, dpi=dpi)

            image = page_to_image(page, dpi=dpi)
            img_h, img_w = image.shape[:2]
            preprocessed = preprocess_image(image)

            results = []
            for engine in engines:
                results.extend(engine.recognize(preprocessed, confidence=confidence))
            if len(engines) > 1:
                results = deduplicate_results(results)

            if verbose:
                click.echo(f"\n  Page {page_idx + 1}: {len(results)} text blocks detected")

            total_blocks += overlay_text_on_page(
                page, results, img_w, img_h, font_resolver=font_resolver,
            )

    if font_resolver.uses_embedded_font:
        doc.subset_fonts()
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    if skipped:
        click.echo(
            f"Skipped {skipped} page(s) that already had text "
            f"(use --force-ocr to replace it)."
        )
    if font_resolver.dropped_chars:
        sample = "".join(sorted(font_resolver.dropped_chars)[:20])
        click.echo(
            f"Warning: {len(font_resolver.dropped_chars)} character(s) could not be "
            f"encoded and are not searchable: {sample}\n"
            f"Supply a font covering them with --font-file."
        )
    click.echo(f"Done. {total_blocks} text blocks added to '{output_path.name}'.")
