"""Recognition by a vision language model, placed with PaddleOCR's boxes.

A language model reads a page better than PaddleOCR does -- measured over
``bench/``, GPT-6 Sol made 6 character errors on the Korean book where
PaddleOCR made 22, and 2 word errors on the leaflet where it made 71 -- but it
returns a transcription and no coordinates. The invisible text layer needs
both: each line has to be drawn where that line is printed, or selecting and
highlighting land on the wrong words.

So the two are combined. PaddleOCR finds the lines and reads them as it always
does; the model transcribes the whole page; and the transcription is aligned
character by character against PaddleOCR's lines, so each box gets the
model's reading of the text inside it. The two readings agree on nearly every
character, which is what makes the alignment reliable -- and where they do not
agree enough for a line, that line keeps PaddleOCR's text rather than trust an
alignment that may have gone wrong.

The model is reached through the Codex CLI, so it runs on whatever account
Codex is logged in with and needs no API key here. Every page image is sent to
OpenAI; that is the whole cost of the accuracy, and the reason this is an
option rather than the default.
"""

import difflib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import click
import numpy as np

from pdf_refinery.ocr_engine import OcrResult, PaddleEngine

DEFAULT_CODEX_MODEL = "gpt-6-sol"

# Measured on bench/ with GPT-6 Sol, "high" matched "low" to within one
# character on every corpus while taking two to three times as long and
# three times the output tokens.
CODEX_REASONING_EFFORT = "low"

# A page takes 15-40 seconds. Ten minutes is long enough never to cut off a
# slow answer, and short enough that a hung call does not stall a book.
CODEX_TIMEOUT_SECONDS = 600

TRANSCRIPTION_PROMPT = """\
The attached image is one page of a scanned document. Transcribe every piece \
of text on it exactly as printed, in reading order, including the running \
header, page number and footer.

Rules:
- Reproduce the characters faithfully: keep the original spelling, typos, \
punctuation marks (e.g. 「」《》“”), Hanja and spacing. Do not correct, \
translate, summarise or add anything.
- Put each printed line on its own line.
- Output only the transcription, with no commentary, no Markdown and no code \
fences.
- Do not run any commands or read any files; the image is all you need.
"""

# How close a line's aligned text must stay to PaddleOCR's reading of it, as a
# difflib ratio over the non-space characters, before it is used. The two
# readings of a correctly aligned line differ by a character or two, so they
# sit near 1.0; a line the alignment has attached the wrong words to falls far
# below. Half is the point where the aligned text no longer resembles what the
# box contains.
MIN_LINE_AGREEMENT = 0.5


def codex_available() -> bool:
    """Whether the ``codex`` command can be run at all."""
    return shutil.which("codex") is not None


def _owners(line_chars: list[int], line_text: str, page_chars: list[int], page_text: str):
    """Assign each non-space character of the page transcription to a line.

    ``line_chars`` and ``page_chars`` are the positions of the non-space
    characters in ``line_text`` (PaddleOCR's lines joined) and ``page_text``
    (the transcription). Returns, for each page character, the index into
    ``line_chars`` it was aligned with, or None where it has no counterpart.
    """
    a = [line_text[i] for i in line_chars]
    b = [page_text[i] for i in page_chars]
    aligned: list[int | None] = [None] * len(b)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == "equal":
            for j in range(b2 - b1):
                aligned[b1 + j] = a1 + j
        elif op == "replace":
            # A misread: spread the model's characters over PaddleOCR's in
            # proportion, so a replacement straddling two lines splits
            # between them rather than landing wholly on one.
            for j in range(b1, b2):
                aligned[j] = a1 + (j - b1) * (a2 - a1) // (b2 - b1)
    return aligned


def align_transcription(lines: list[str], transcription: str) -> list[str | None]:
    """Distribute a page transcription over the lines PaddleOCR found.

    Args:
        lines: PaddleOCR's text for each detected line, in reading order.
        transcription: The model's reading of the whole page.

    Returns:
        For each line, the model's text for it; None where the line should
        keep PaddleOCR's text, because what the model had for it does not
        resemble what PaddleOCR read in that box, or because the model had
        nothing for it; or ``""`` where the line should be dropped.

    A line is dropped only when both readings say it holds no text: the model
    read nothing there, and PaddleOCR read no letter or digit. Those are specks
    and rules the detector boxed -- on the leaflet in ``bench/sample-2``,
    boxes read as ``#``, ``*``, ``-`` and ``:`` beside the photograph. Neither
    reading alone is enough. PaddleOCR's alone is not: a comma or period that
    wrapped onto a box of its own reads just the same, and dropping every
    such line lost more real punctuation on ``sample-1`` than it removed
    specks. The model's alone is not either, because a model can skip a line
    of real text.

    Text the model read that PaddleOCR did not detect at all has no box to go
    in and is dropped; there is nowhere on the page to place it.
    """
    joined = ""
    char_line: list[int] = []
    line_chars: list[int] = []
    for index, line in enumerate(lines):
        for ch in line:
            if not ch.isspace():
                line_chars.append(len(joined))
                char_line.append(index)
            joined += ch
        joined += "\n"

    page_chars = [i for i, ch in enumerate(transcription) if not ch.isspace()]
    aligned = _owners(line_chars, joined, page_chars, transcription)
    owner: list[int | None] = [
        None if a is None else char_line[a] for a in aligned
    ]

    # Characters only the model saw, such as a corner bracket PaddleOCR's
    # dictionary lacks. They belong to a neighbouring line; the model's own
    # line breaks say which. After a break they open the next line, otherwise
    # they continue the previous one.
    for k in range(len(owner)):
        if owner[k] is not None:
            continue
        prev = next((j for j in range(k - 1, -1, -1) if owner[j] is not None), None)
        nxt = next((j for j in range(k + 1, len(owner)) if owner[j] is not None), None)
        if prev is None and nxt is None:
            break
        if prev is None:
            owner[k] = owner[nxt]
        elif nxt is None:
            owner[k] = owner[prev]
        else:
            gap = transcription[page_chars[prev]:page_chars[k]]
            owner[k] = owner[nxt] if "\n" in gap else owner[prev]

    spans: dict[int, list[int]] = {}
    for k, line in enumerate(owner):
        if line is not None:
            spans.setdefault(line, [k, k])[1] = k

    out: list[str | None] = []
    for index, original in enumerate(lines):
        span = spans.get(index)
        if span is None:
            out.append(None if any(ch.isalnum() for ch in original) else "")
            continue
        start, end = page_chars[span[0]], page_chars[span[1]]
        text = " ".join(transcription[start:end + 1].split())
        agreement = difflib.SequenceMatcher(
            None, "".join(original.split()), "".join(text.split()), autojunk=False
        ).ratio()
        out.append(text if agreement >= MIN_LINE_AGREEMENT else None)
    return out


def transcribe_with_codex(image: np.ndarray, model: str = DEFAULT_CODEX_MODEL) -> str:
    """Have the model transcribe ``image`` (RGB) through the Codex CLI.

    Raises:
        RuntimeError: If Codex fails, times out or answers with nothing.
    """
    import cv2

    with tempfile.TemporaryDirectory(prefix="pdf-refinery-") as workdir:
        page = Path(workdir) / "page.png"
        answer = Path(workdir) / "answer.txt"
        cv2.imwrite(str(page), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        command = [
            "codex", "exec",
            "-m", model,
            "-c", f'model_reasoning_effort="{CODEX_REASONING_EFFORT}"',
            # Read-only and ephemeral: the model is asked to read one image,
            # and nothing it does should touch the disk or linger as a session.
            "-s", "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-rules",
            "-i", str(page),
            "-o", str(answer),
            "-",
        ]
        try:
            proc = subprocess.run(
                command, input=TRANSCRIPTION_PROMPT, capture_output=True,
                text=True, cwd=workdir, timeout=CODEX_TIMEOUT_SECONDS,
                env={**os.environ, "NO_COLOR": "1"},
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"codex did not answer within {CODEX_TIMEOUT_SECONDS}s") from None
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            raise RuntimeError(
                f"codex exited {proc.returncode}: {detail[-1] if detail else 'no output'}"
            )
        text = answer.read_text(encoding="utf-8") if answer.exists() else ""
        if not text.strip():
            raise RuntimeError("codex returned an empty transcription")
        return text


class CodexEngine:
    """PaddleOCR's boxes, with the text read by a model through Codex."""

    def __init__(self, lang: str = "en", codex_model: str = DEFAULT_CODEX_MODEL,
                 **paddle_options):
        """Build the engine.

        Args:
            lang: PaddleOCR language code, for detection and the fallback
                reading. The model itself is not told the language.
            codex_model: The model Codex runs.
            **paddle_options: Passed to :class:`PaddleEngine`.
        """
        self._paddle = PaddleEngine(lang=lang, **paddle_options)
        self._model = codex_model
        self.pages_kept_as_paddle = 0

    def recognize(
        self, image: np.ndarray, confidence: float = 0.5,
        rendered: np.ndarray | None = None,
    ) -> list[OcrResult]:
        results = self._paddle.recognize(image, confidence=confidence)
        if not results:
            # No boxes, so nothing the transcription could be placed on.
            return results
        try:
            # The model reads the page as rendered, not as thresholded for
            # PaddleOCR: on bench/sample-1, the binarized page cost it 12 and
            # 13 character errors over two runs against 7 for the greyscale,
            # nearly all of them dense Hangul syllables losing a stroke.
            seen = image if rendered is None else rendered
            transcription = transcribe_with_codex(seen, model=self._model)
        except RuntimeError as exc:
            # One failed call must not end a run over a whole book: the page
            # still gets PaddleOCR's reading, and the user is told.
            self.pages_kept_as_paddle += 1
            click.echo(f"Warning: {exc}; this page keeps PaddleOCR's text.", err=True)
            return results
        texts = align_transcription([r.text for r in results], transcription)
        return [
            OcrResult(text=r.text if t is None else t, confidence=r.confidence, bbox=r.bbox)
            for r, t in zip(results, texts)
            if t != ""
        ]
