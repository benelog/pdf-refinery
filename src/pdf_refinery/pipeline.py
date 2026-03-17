"""OCR pipeline orchestration."""

import shutil
from pathlib import Path

import click

from pdf_refinery.ocr_engine import OcrEngine
from pdf_refinery.pdf_reader import open_pdf, page_to_image
from pdf_refinery.pdf_writer import overlay_text_on_page


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
) -> None:
    """Run the full OCR pipeline on a scanned PDF.

    Args:
        input_path: Path to the input scanned PDF.
        output_path: Path for the output searchable PDF.
        langs: List of PaddleOCR language codes.
        dpi: DPI for page rendering.
        pages: Optional page range string (1-based).
        confidence: Minimum OCR confidence threshold.
        verbose: Enable verbose output.
    """
    if langs is None:
        langs = ["en"]

    # Copy input to output first, then modify in place
    shutil.copy2(input_path, output_path)

    doc = open_pdf(output_path)
    total_pages = len(doc)

    if pages:
        page_indices = parse_page_range(pages, total_pages)
    else:
        page_indices = list(range(total_pages))

    click.echo(f"Processing {len(page_indices)} page(s) from '{input_path.name}'...")
    click.echo(f"Languages: {', '.join(langs)}")

    engines = [OcrEngine(lang=lang) for lang in langs]
    total_blocks = 0

    with click.progressbar(page_indices, label="OCR progress") as bar:
        for page_idx in bar:
            page = doc[page_idx]

            # Render page to image
            image = page_to_image(page, dpi=dpi)
            img_h, img_w = image.shape[:2]

            # Run OCR with each language engine and merge results
            results = []
            for engine in engines:
                results.extend(engine.recognize(image, confidence=confidence))

            if verbose:
                click.echo(f"\n  Page {page_idx + 1}: {len(results)} text blocks detected")

            # Overlay invisible text
            count = overlay_text_on_page(page, results, img_w, img_h)
            total_blocks += count

    doc.save(output_path, incremental=True, encryption=0)
    doc.close()

    click.echo(f"Done. {total_blocks} text blocks added to '{output_path.name}'.")
