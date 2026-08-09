"""Command-line interface for PDF Refinery.

Every option is validated here, before the pipeline opens the PDF or PaddleOCR
downloads a model. A run can take an hour, so a typo must fail in the first
second rather than at the end -- or, worse, produce an output that looks fine.

Nothing that pulls in ``paddleocr`` is imported at module level. That import
costs about three seconds and contacts the model hosts, which would make even
``--help`` slow and dependent on the network. Click resolves ``--help`` and
``--version`` eagerly, before any other callback runs, so deferring these
imports into the callbacks and the command body keeps them out of that path.
The two option defaults imported below deliberately live in modules that carry
no such dependency.
"""

import difflib
from pathlib import Path

import click

from pdf_refinery.ocr_engine import (
    COMMON_LANGS,
    DEFAULT_PREPROCESS,
    DEFAULT_TEXTLINE_ORIENTATION,
    PREPROCESS_MODES,
    SERVER_REC_LANGS,
    SERVER_REC_MODEL,
)
from pdf_refinery.pdf_writer import MIN_TEXT_CHARS
from pdf_refinery.pipeline import DEFAULT_CHECKPOINT_EVERY


def _validate_langs(ctx, param, value: tuple[str, ...]) -> tuple[str, ...]:
    """Reject language codes PaddleOCR has no model for, and suggest a fix."""
    from pdf_refinery.ocr_engine import LANG_ALIASES, is_supported_lang

    for lang in value:
        if is_supported_lang(lang):
            continue
        hint = LANG_ALIASES.get(lang.lower())
        if hint is None:
            close = difflib.get_close_matches(lang, COMMON_LANGS, n=1, cutoff=0.6)
            hint = close[0] if close else None
        suggestion = f" Did you mean '{hint}'?" if hint else ""
        raise click.BadParameter(
            f"'{lang}' is not a PaddleOCR language code.{suggestion}\n"
            f"Common codes: {', '.join(COMMON_LANGS)}."
        )
    return value


def _validate_pages(ctx, param, value: str | None) -> str | None:
    """Check page-range syntax so a typo fails now, not as an empty result.

    Ranges are still clamped to the document later; what is rejected here is
    input that cannot mean anything, such as ``"abc"``, ``"0"`` or ``"10-1"``.
    """
    if value is None:
        return None

    def bad(reason: str) -> click.BadParameter:
        return click.BadParameter(
            f"{reason} Expected 1-based page numbers like \"1-10\", \"1,3,5\" "
            "or \"1-3,7\"."
        )

    if not value.strip():
        raise bad("Page range is empty.")

    for part in value.split(","):
        part = part.strip()
        if not part:
            raise bad("Page range has an empty entry.")
        bounds = part.split("-", 1) if "-" in part else [part]
        try:
            numbers = [int(b.strip()) for b in bounds]
        except ValueError:
            raise bad(f"'{part}' is not a page number.")
        if any(n < 1 for n in numbers):
            raise bad(f"'{part}' is out of range: page numbers start at 1.")
        if len(numbers) == 2 and numbers[0] > numbers[1]:
            raise bad(f"'{part}' runs backwards.")
    return value


@click.group()
@click.version_option(package_name="pdf-refinery")
def main():
    """PDF Refinery - Transform scanned PDFs into searchable documents."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Output file path. Defaults to <input>_ocr.pdf.")
@click.option("-l", "--lang", multiple=True, default=(), callback=_validate_langs,
              help="OCR language code, e.g. korean or en. Required: there is "
                   "no safe default. Repeatable, but each language costs a "
                   "full extra pass over every page.")
@click.option("--dpi", default=300, type=click.IntRange(72, 1200),
              help="DPI for page rendering.")
@click.option("--pages", default=None, type=str, callback=_validate_pages,
              help='Page range to process (e.g., "1-10", "1,3,5").')
@click.option("--confidence", default=0.5, type=click.FloatRange(0.0, 1.0),
              help="Minimum confidence threshold for OCR results (0.0-1.0).")
@click.option("--force-ocr", is_flag=True,
              help="Re-OCR pages that already contain text, replacing it. "
                   "Such pages are skipped by default.")
@click.option("--skip-text-threshold", default=MIN_TEXT_CHARS, type=click.IntRange(0),
              help="Characters a page needs before it counts as already "
                   "searchable and is skipped. Keeps a stray page number or "
                   "header from suppressing OCR of the whole page.")
@click.option("--sidecar", type=click.Path(dir_okay=False, path_type=Path),
              default=None,
              help="Also write the recognised text to this plain-text file, "
                   "one page per form feed.")
@click.option("--checkpoint-every", default=DEFAULT_CHECKPOINT_EVERY,
              type=click.IntRange(0), metavar="PAGES",
              help="Save the output every N pages so an interrupted run can be "
                   "continued with --resume. 0 saves only at the end.")
@click.option("--resume", is_flag=True,
              help="Continue an interrupted run from its last checkpoint, "
                   "reusing the existing output file.")
@click.option("--overwrite", is_flag=True,
              help="Replace the output file if it already exists.")
@click.option("--preprocess", type=click.Choice(PREPROCESS_MODES),
              default=DEFAULT_PREPROCESS, show_default=True,
              help="Image preparation before recognition. 'binarize' denoises "
                   "and thresholds to black and white, which is what makes "
                   "the recogniser separate words instead of running them "
                   "together; 'none' hands it the rendered page as-is, which "
                   "is faster and about as accurate per character.")
@click.option("--rec-model", default=None, metavar="NAME",
              help="PaddleOCR recognition model overriding the one the "
                   f"language selects, e.g. {SERVER_REC_MODEL}. That heavier "
                   "model is not the obvious win it sounds like: on the Latin "
                   "benchmark it fixed one character, broke five words, and "
                   "took twice as long.")
@click.option("--unwarp", is_flag=True,
              help="Flatten page curvature before detection. Aimed at the "
                   "bound edge of a book photographed or pressed against the "
                   "glass; on a flat scan it costs time and accuracy both.")
@click.option("--textline-orientation", is_flag=True,
              default=DEFAULT_TEXTLINE_ORIENTATION,
              help="Let the recogniser turn individual lines it judges to be "
                   "upside down. Use only for a scan that really has some; on "
                   "an upright page it mislabels lines often, and each one it "
                   "turns is read as convincing nonsense rather than failing.")
@click.option("--auto-rotate", is_flag=True,
              help="Detect pages scanned sideways or upside down and read "
                   "them the right way up. The page itself is left as it is; "
                   "only the text layer is placed correctly.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
def ocr(input_file: Path, output: Path | None, lang: tuple[str, ...], dpi: int,
        pages: str | None, confidence: float, force_ocr: bool, overwrite: bool,
        skip_text_threshold: int, sidecar: Path | None,
        checkpoint_every: int, resume: bool, preprocess: str,
        rec_model: str | None, unwarp: bool, textline_orientation: bool,
        auto_rotate: bool, verbose: bool):
    """Apply OCR to a scanned PDF to make it searchable."""
    from pdf_refinery.pipeline import progress_path_for, run_ocr_pipeline

    if not lang:
        # There used to be a default of 'en'. A Korean scan read as English
        # measured 97% character error -- not a degraded result but an empty
        # one -- and the run still reported success and wrote a plausible
        # output file. Any default just chooses whose documents fail that way,
        # so the option is asked for instead.
        raise click.UsageError(
            "No language given. Pass -l with the document's language, for "
            "example '-l korean' or '-l en'.\n"
            f"Common codes: {', '.join(COMMON_LANGS)}."
        )

    if rec_model == SERVER_REC_MODEL:
        # Its dictionary is Chinese plus English and kana. Pointed at Korean or
        # Cyrillic it does not run slower-but-better, it returns nothing
        # usable -- and that would only show up at the end of a long run.
        uncovered = [lg for lg in lang if lg not in SERVER_REC_LANGS]
        if uncovered:
            raise click.BadParameter(
                f"{SERVER_REC_MODEL} only recognises "
                f"{', '.join(sorted(SERVER_REC_LANGS))}; it has no characters "
                f"for {', '.join(uncovered)}. Drop --rec-model to use the "
                f"model built for that language."
            )

    if output is None:
        output = input_file.with_stem(f"{input_file.stem}_ocr")

    if resume and overwrite:
        raise click.UsageError(
            "--resume continues the existing output while --overwrite discards "
            "it. Pick one."
        )

    # --resume is the one case where an existing output is the point, so the
    # guard below only applies to a fresh run.
    if output.exists() and not overwrite and not resume:
        resumable = progress_path_for(output).exists()
        hint = (
            "Pass --resume to continue that run, --overwrite to start over, "
            if resumable else
            "Pass --overwrite to replace it, "
        )
        note = (
            "An interrupted run left this behind. " if resumable else ""
        )
        raise click.ClickException(
            f"'{output}' already exists. {note}{hint}or choose another path "
            "with -o."
        )

    if sidecar is not None and sidecar.exists() and not overwrite and not resume:
        raise click.ClickException(
            f"'{sidecar}' already exists. Pass --overwrite to replace it, or "
            "choose another path with --sidecar."
        )

    run_ocr_pipeline(
        input_path=input_file,
        output_path=output,
        langs=list(lang),
        dpi=dpi,
        pages=pages,
        confidence=confidence,
        verbose=verbose,
        force_ocr=force_ocr,
        sidecar=sidecar,
        skip_text_threshold=skip_text_threshold,
        checkpoint_every=checkpoint_every,
        resume=resume,
        preprocess=preprocess,
        rec_model=rec_model,
        unwarp=unwarp,
        textline_orientation=textline_orientation,
        auto_rotate=auto_rotate,
    )
