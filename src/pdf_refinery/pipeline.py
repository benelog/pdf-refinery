"""OCR pipeline orchestration.

A book-length run takes tens of minutes to hours, which changes what "correct"
means: work has to survive an interrupt. Pages are therefore saved in batches
and the indices already finished are recorded next to the output, so ``--resume``
can pick the run back up instead of starting from page one.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import click

from pdf_refinery.ocr_engine import (
    DEFAULT_PREPROCESS,
    DEFAULT_TEXTLINE_ORIENTATION,
    create_engine,
    deduplicate_results,
    detect_page_orientation,
    preprocess_image,
    unrotate_results,
    upright,
)
from pdf_refinery.pdf_document import open_pdf
from pdf_refinery.text_overlay import (
    MIN_TEXT_CHARS,
    has_text,
    overlay_text_on_page,
)

# Pages between intermediate saves. Each save rewrites the whole PDF, so this
# trades a little throughput for a bounded amount of lost work on an interrupt.
DEFAULT_CHECKPOINT_EVERY = 20

PROGRESS_SUFFIX = ".progress"
PARTIAL_SUFFIX = ".part"
PROGRESS_VERSION = 1

# Form feed between pages, matching what ocrmypdf's sidecar emits.
PAGE_SEPARATOR = "\f"


def parse_page_selections(pages_str: str) -> list[tuple[int, int]]:
    """Parse a page range string into 1-based ``(start, end)`` pairs.

    Supports formats like "1-10", "1,3,5", "1-3,7,10-12"; a single number is
    the pair ``(n, n)``. Input that cannot mean anything -- ``"abc"``, ``"0"``,
    ``"10-1"`` -- raises ValueError with a reason fit to show the user. This is
    the one place that knows the syntax, so the CLI's early check and the
    parse the pipeline runs on cannot drift apart.
    """
    if not pages_str.strip():
        raise ValueError("Page range is empty.")
    selections = []
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Page range has an empty entry.")
        bounds = part.split("-", 1) if "-" in part else [part]
        try:
            numbers = [int(b.strip()) for b in bounds]
        except ValueError:
            raise ValueError(f"'{part}' is not a page number.") from None
        if any(n < 1 for n in numbers):
            raise ValueError(f"'{part}' is out of range: page numbers start at 1.")
        if len(numbers) == 2 and numbers[0] > numbers[1]:
            raise ValueError(f"'{part}' runs backwards.")
        selections.append((numbers[0], numbers[-1]))
    return selections


def parse_page_range(pages_str: str, total_pages: int) -> list[int]:
    """Parse a page range string into a list of 0-based page indices.

    Input is 1-based, output is 0-based; pages beyond ``total_pages`` are
    dropped. Raises ValueError as :func:`parse_page_selections` does.
    """
    indices = set()
    for start, end in parse_page_selections(pages_str):
        indices.update(range(start - 1, min(end, total_pages)))
    return sorted(indices)


def progress_path_for(output_path: Path) -> Path:
    """Path of the file recording which pages of ``output_path`` are finished."""
    return output_path.with_name(output_path.name + PROGRESS_SUFFIX)


def _write_progress(progress_path: Path, input_path: Path, done: set[int]) -> None:
    progress_path.write_text(
        json.dumps(
            {
                "version": PROGRESS_VERSION,
                "input": str(input_path.resolve()),
                "done": sorted(done),
            }
        ),
        encoding="utf-8",
    )


def _load_progress(progress_path: Path, input_path: Path, output_path: Path) -> set[int]:
    """Read the finished-page set left behind by an interrupted run."""
    if not output_path.exists() or not progress_path.exists():
        raise click.ClickException(
            f"Nothing to resume: no '{progress_path.name}' next to the output. "
            "A run only leaves one behind once it reaches a checkpoint; run "
            "again without --resume."
        )
    try:
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        source = state["input"]
        done = {int(i) for i in state["done"]}
    except (ValueError, TypeError, KeyError) as exc:
        raise click.ClickException(
            f"'{progress_path.name}' is not readable as progress state ({exc}). "
            "Delete it and the output, then run again without --resume."
        )
    if source != str(input_path.resolve()):
        raise click.ClickException(
            f"'{progress_path.name}' belongs to a run over '{source}', not "
            f"'{input_path}'. Resuming would merge two documents."
        )
    return done


def _save(doc, output_path: Path, final: bool = False) -> None:
    """Write the document without ever leaving a half-written output behind.

    On a resumed run the document was opened *from* ``output_path``, so an
    interrupt during a direct save would destroy the only copy of the work so
    far. Writing a sibling file and renaming keeps the previous checkpoint
    readable until the new one is complete.
    """
    partial = output_path.with_name(output_path.name + PARTIAL_SUFFIX)
    doc.save(partial, final=final)
    os.replace(partial, output_path)


class _Transcript:
    """Plain-text transcript of the pages, written in step with checkpoints.

    Text is buffered until a checkpoint rather than written per page, so the
    transcript never runs ahead of the recorded progress: a page written here
    but not yet marked done would be transcribed twice after ``--resume``.
    """

    def __init__(self, path: Path, append: bool):
        self._path = path
        self._mode = "a" if append else "w"
        self._buffer: list[str] = []

    def add_page(self, texts: list[str]) -> None:
        self._buffer.append("\n".join(texts))

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self._path, self._mode, encoding="utf-8") as handle:
            for page_text in self._buffer:
                handle.write(page_text + "\n" + PAGE_SEPARATOR)
        self._mode = "a"
        self._buffer.clear()


@contextmanager
def _tracked(page_indices: list[int], verbose: bool):
    """Iterate pages, showing a progress bar only where it stays readable.

    Verbose mode prints a line per page, and a click progress bar redraws on
    that same line; together they produce shredded output.
    """
    if verbose:
        yield page_indices
    else:
        with click.progressbar(page_indices, label="OCR progress") as bar:
            yield bar


def run_ocr_pipeline(
    input_path: Path,
    output_path: Path,
    langs: list[str] | None = None,
    dpi: int = 300,
    pages: str | None = None,
    confidence: float = 0.5,
    verbose: bool = False,
    force_ocr: bool = False,
    sidecar: Path | None = None,
    skip_text_threshold: int = MIN_TEXT_CHARS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    resume: bool = False,
    preprocess: str = DEFAULT_PREPROCESS,
    rec_model: str | None = None,
    unwarp: bool = False,
    textline_orientation: bool = DEFAULT_TEXTLINE_ORIENTATION,
    auto_rotate: bool = False,
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
        sidecar: Optional path for a plain-text transcript of every page.
        skip_text_threshold: Characters a page needs before it counts as
            already searchable. See :data:`text_overlay.MIN_TEXT_CHARS`.
        checkpoint_every: Pages between intermediate saves; 0 disables them,
            which also makes the run unresumable.
        resume: Continue an interrupted run from its last checkpoint.
        preprocess: Image preparation mode; see
            :func:`ocr_engine.preprocess_image`.
        rec_model: Recognition model overriding the language's default.
        unwarp: Correct page curvature before detection.
        textline_orientation: Turn lines the classifier judges upside down
            before recognising them. See :class:`ocr_engine.PaddleEngine`.
        auto_rotate: Detect a page scanned sideways or upside down, read it
            the right way up, and place the text back on the page as it is.
    """
    if langs is None:
        langs = ["en"]
    if input_path.resolve() == output_path.resolve():
        raise click.ClickException("Output path must differ from the input path.")

    progress_path = progress_path_for(output_path)
    if resume:
        done = _load_progress(progress_path, input_path, output_path)
        doc = open_pdf(output_path)
    else:
        done = set()
        doc = open_pdf(input_path)

    total_pages = len(doc)

    if pages:
        try:
            page_indices = parse_page_range(pages, total_pages)
        except ValueError as exc:
            doc.close()
            raise click.ClickException(str(exc))
        if not page_indices:
            # Without this the run "succeeds" with zero pages and writes an
            # output that looks like a finished job.
            doc.close()
            raise click.ClickException(
                f"Page range '{pages}' selects no pages; "
                f"'{input_path.name}' has {total_pages}."
            )
    else:
        page_indices = list(range(total_pages))

    pending = [idx for idx in page_indices if idx not in done]

    if resume:
        click.echo(
            f"Resuming '{output_path.name}': {len(done)} page(s) already done, "
            f"{len(pending)} to go."
        )
    else:
        click.echo(f"Processing {len(pending)} page(s) from '{input_path.name}'...")
    click.echo(f"Languages: {', '.join(langs)}")

    engines = [
        create_engine(
            lang=lang,
            rec_model=rec_model,
            unwarp=unwarp,
            textline_orientation=textline_orientation,
        )
        for lang in langs
    ]
    transcript = _Transcript(sidecar, append=bool(done)) if sidecar else None
    total_blocks = 0
    total_too_small = 0
    dropped_chars: set[str] = set()
    skipped = 0
    since_checkpoint = 0

    def checkpoint() -> None:
        # The output is written before the progress note, so an interrupt
        # between the two costs at worst a page that gets read twice. The other
        # order would instead mark a page done that never reached the file,
        # leaving it silently unsearchable.
        nonlocal since_checkpoint
        _save(doc, output_path)
        _write_progress(progress_path, input_path, done)
        if transcript:
            transcript.flush()
        since_checkpoint = 0

    with _tracked(pending, verbose) as tracked:
        for page_idx in tracked:
            page = doc[page_idx]

            already_readable = has_text(page, min_chars=skip_text_threshold)
            if already_readable and not force_ocr:
                skipped += 1
                done.add(page_idx)
                if transcript:
                    transcript.add_page([page.text().strip()])
                if verbose:
                    click.echo(f"  Page {page_idx + 1}: already has text, skipped")
                continue

            # Rendered before anything is written to the page, which is what
            # lets the same rendering both feed the recogniser and, on a page
            # being re-read, replace the page it came from.
            image = page.to_image(dpi=dpi)
            image_height, image_width = image.shape[:2]
            if already_readable:
                page.replace_with_image(image)

            # Orientation is judged on the rendered page, before preprocessing,
            # and the turn is done here rather than inside PaddleOCR. Both
            # matter: the classifier reads the greyscale far better than the
            # binarized page, and doing the turn here is what lets the boxes be
            # mapped back onto the page the reader actually has.
            rotation = detect_page_orientation(image) if auto_rotate else 0
            straight = upright(image, rotation)
            if rotation and verbose:
                click.echo(f"  Page {page_idx + 1}: turned {rotation}deg to read")

            preprocessed = preprocess_image(straight, mode=preprocess)

            results = []
            for engine in engines:
                results.extend(engine.recognize(preprocessed, confidence=confidence))
            if len(engines) > 1:
                results = deduplicate_results(results)
            results = unrotate_results(results, rotation, straight.shape)

            if verbose:
                click.echo(f"  Page {page_idx + 1}: {len(results)} text blocks detected")

            stats = overlay_text_on_page(page, results, image_width, image_height)
            total_blocks += stats.inserted
            total_too_small += stats.too_small
            dropped_chars |= stats.dropped_chars
            if transcript:
                transcript.add_page([r.text for r in results])

            done.add(page_idx)
            since_checkpoint += 1
            if checkpoint_every and since_checkpoint >= checkpoint_every:
                checkpoint()
                if verbose:
                    click.echo(f"  Checkpoint saved ({len(done)} page(s) complete)")

    _save(doc, output_path, final=True)
    doc.close()
    if transcript:
        transcript.flush()
    progress_path.unlink(missing_ok=True)

    if skipped:
        click.echo(
            f"Skipped {skipped} page(s) that already had text "
            f"(use --force-ocr to replace it)."
        )
    if total_too_small:
        click.echo(
            f"Note: {total_too_small} detected line(s) were too small to place "
            f"and are not searchable. A higher --dpi usually recovers them."
        )
    if dropped_chars:
        sample = "".join(sorted(dropped_chars)[:20])
        click.echo(
            f"Warning: {len(dropped_chars)} character(s) lie above U+FFFF, which "
            f"the text layer cannot encode, and are not searchable: {sample}"
        )
    if sidecar:
        click.echo(f"Text transcript written to '{sidecar}'.")
    click.echo(f"Done. {total_blocks} text blocks added to '{output_path.name}'.")
