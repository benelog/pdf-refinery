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

The model is reached through a coding agent's command-line tool -- the Codex
CLI for OpenAI's models, Claude Code for Anthropic's -- so it runs on whatever
account that tool is logged in with and needs no API key here. Every page
image is sent to that provider; that is the whole cost of the accuracy, and the
reason this is an option rather than the default.

The consensus engine asks both and votes line by line, character by character,
with PaddleOCR's own reading as the arbiter where the two models disagree. See
:func:`merge_readings`.
"""

import base64
import difflib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
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

# An alias, not a dated model name: Claude Code resolves it to the newest Opus
# the account can use, so the default does not go stale with each release.
DEFAULT_CLAUDE_MODEL = "opus"

# Pinned rather than inherited. Claude Code otherwise takes the effort level
# from its settings or from CLAUDE_EFFORT -- which a parent Claude Code session
# exports to everything it runs, at "high" -- so the same command would think
# for a different length depending on where it was started. Low matches the
# Codex setting above, where high bought nothing measurable for transcription.
CLAUDE_EFFORT = "low"

# The same budget as Codex, for the same reason.
CLAUDE_TIMEOUT_SECONDS = 600

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


def claude_available() -> bool:
    """Whether the ``claude`` command (Claude Code) can be run at all."""
    return shutil.which("claude") is not None


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


def _tight(text: str) -> str:
    return "".join(text.split())


def merge_readings(paddle: str, a: str | None, b: str | None) -> str | None:
    """Combine two models' readings of one line, with PaddleOCR as arbiter.

    Args:
        paddle: What PaddleOCR read in the box.
        a: The preferred model's text for the line, as
            :func:`align_transcription` returns it: a string, None for "keep
            PaddleOCR's text", or ``""`` for "drop the line". Ties go to it.
        b: The other model's text for the line, in the same form.

    Returns:
        The line's text, None to keep PaddleOCR's, or ``""`` to drop the line.

    Where both models read the line the same way, that is the answer. Where
    only one has a usable reading -- the other found nothing there, or text
    that did not resemble the box -- that one is used, as its engine alone
    would. Where they disagree, the two readings are aligned character by
    character; the stretches they agree on are kept, and each stretch they
    disagree on is settled by which version leaves the whole line closer to
    what PaddleOCR read. Two models disagreeing about a character is exactly
    when a third, independent reading is worth asking.

    Closeness is measured with the spaces removed, so a disagreement that is
    only about spacing always ties and goes to ``a``. That is deliberate:
    spacing is the one thing PaddleOCR is worst at -- it runs the words of
    large type together -- and letting it vote would pull a correctly spaced
    reading back towards ``A버튼(라이트버튼)``, which a word search for
    ``버튼`` does not find. It is also why two readings differing only in
    spacing are not treated as agreeing: the space is the part that decides
    whether the word can be searched for, so it is voted on like any other
    character, and settled by the model that spaces better.

    The arbiter has known biases of its own. PaddleOCR's Korean dictionary
    has no corner brackets and reads ``톰`` as ``통`` more often than not, so
    where one model has the right character and the other agrees with
    PaddleOCR's habit, the vote goes the wrong way. It only gets the chance
    when the two models already disagree.

    A line is dropped only when both models say to drop it.
    """
    if not a and not b:
        return "" if a == "" and b == "" else None
    if not b:
        return a
    if not a:
        return b
    if a == b:
        return a

    target = _tight(paddle)

    def closeness(text: str) -> float:
        return difflib.SequenceMatcher(None, target, _tight(text), autojunk=False).ratio()

    merged: list[str] = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == "equal":
            merged.append(a[a1:a2])
            continue
        # Each disagreement is judged in the context of the line as decided so
        # far, with the preferred reading standing in for what is still ahead.
        head, tail = "".join(merged), a[a2:]
        with_a = closeness(head + a[a1:a2] + tail)
        with_b = closeness(head + b[b1:b2] + tail)
        merged.append(b[b1:b2] if with_b > with_a else a[a1:a2])
    return " ".join("".join(merged).split())


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


def _png_base64(image: np.ndarray) -> str:
    import cv2

    ok, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("the page could not be encoded as PNG")
    return base64.b64encode(png.tobytes()).decode("ascii")


def _claude_result(stdout: str) -> dict | None:
    """The final ``result`` event of a stream-json run, or None if absent."""
    result = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    return result


def transcribe_with_claude(image: np.ndarray, model: str = DEFAULT_CLAUDE_MODEL) -> str:
    """Have the model transcribe ``image`` (RGB) through Claude Code.

    The image goes in on stdin as a stream-json user message, inline and
    base64-encoded, so nothing is written to disk and the model needs no tool
    to read a file -- which is why it can be given none at all.

    Raises:
        RuntimeError: If Claude Code fails, times out, reports an error or
            answers with nothing.
    """
    message = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": _png_base64(image),
                }},
                {"type": "text", "text": TRANSCRIPTION_PROMPT},
            ],
        },
    }
    command = [
        "claude", "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        # stream-json output requires it; the events are parsed, not shown.
        "--verbose",
        "--model", model,
        "--effort", CLAUDE_EFFORT,
        # No tools, no saved session, and none of the user's settings, hooks
        # or MCP servers: the call reads one image and answers, and nothing
        # it does should touch the disk, linger in the session list, or
        # depend on how this machine's Claude Code happens to be configured.
        "--tools", "",
        "--no-session-persistence",
        "--setting-sources", "",
        "--strict-mcp-config",
    ]
    # An empty working directory, so the project the user ran this from --
    # its CLAUDE.md, its git status -- is not read into the model's context.
    with tempfile.TemporaryDirectory(prefix="pdf-refinery-") as workdir:
        try:
            proc = subprocess.run(
                command, input=json.dumps(message) + "\n", capture_output=True,
                text=True, cwd=workdir, timeout=CLAUDE_TIMEOUT_SECONDS,
                env={**os.environ, "NO_COLOR": "1"},
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"claude did not answer within {CLAUDE_TIMEOUT_SECONDS}s") from None
    result = _claude_result(proc.stdout)
    if proc.returncode != 0:
        # Claude Code reports most failures -- not logged in, a usage limit,
        # an unknown model -- as the result text, not on stderr.
        detail = (
            [str(result.get("result") or result.get("subtype"))] if result
            else (proc.stderr or proc.stdout).strip().splitlines()
        )
        raise RuntimeError(
            f"claude exited {proc.returncode}: {detail[-1] if detail else 'no output'}"
        )
    if result is None:
        raise RuntimeError("claude finished without a result")
    if result.get("is_error") or result.get("subtype") != "success":
        raise RuntimeError(
            f"claude reported an error ({result.get('subtype')}): "
            f"{str(result.get('result') or '').strip() or 'no detail'}"
        )
    text = result.get("result")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("claude returned an empty transcription")
    return text


def _with_lines(results: list[OcrResult], texts: list[str | None]) -> list[OcrResult]:
    """PaddleOCR's boxes with the given text; None keeps its own, ``""`` drops."""
    return [
        OcrResult(text=r.text if t is None else t, confidence=r.confidence, bbox=r.bbox)
        for r, t in zip(results, texts)
        if t != ""
    ]


class _ModelEngine:
    """PaddleOCR's boxes, with the text read by one model."""

    def __init__(self, lang: str, model: str, **paddle_options):
        self._paddle = PaddleEngine(lang=lang, **paddle_options)
        self._model = model
        self.pages_kept_as_paddle = 0

    def _transcribe(self, image: np.ndarray) -> str:
        raise NotImplementedError

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
            # PaddleOCR: on bench/sample-1, the binarized page cost GPT-6 Sol
            # 12 and 13 character errors over two runs against 7 for the
            # greyscale, nearly all of them dense Hangul syllables losing a
            # stroke.
            seen = image if rendered is None else rendered
            transcription = self._transcribe(seen)
        except RuntimeError as exc:
            # One failed call must not end a run over a whole book: the page
            # still gets PaddleOCR's reading, and the user is told.
            self.pages_kept_as_paddle += 1
            click.echo(f"Warning: {exc}; this page keeps PaddleOCR's text.", err=True)
            return results
        return _with_lines(
            results, align_transcription([r.text for r in results], transcription)
        )


class CodexEngine(_ModelEngine):
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
        super().__init__(lang, codex_model, **paddle_options)

    def _transcribe(self, image: np.ndarray) -> str:
        return transcribe_with_codex(image, model=self._model)


class ClaudeEngine(_ModelEngine):
    """PaddleOCR's boxes, with the text read by a model through Claude Code."""

    def __init__(self, lang: str = "en", claude_model: str = DEFAULT_CLAUDE_MODEL,
                 **paddle_options):
        """Build the engine.

        Args:
            lang: PaddleOCR language code, for detection and the fallback
                reading. The model itself is not told the language.
            claude_model: The model Claude Code runs, by alias or full name.
            **paddle_options: Passed to :class:`PaddleEngine`.
        """
        super().__init__(lang, claude_model, **paddle_options)

    def _transcribe(self, image: np.ndarray) -> str:
        return transcribe_with_claude(image, model=self._model)


# Which model's reading wins a vote PaddleOCR cannot settle -- above all, every
# disagreement about spacing. Claude, because it was the better reader alone
# on every corpus in bench/: Opus through Claude Code made 0, 0 and 0
# character errors and 2, 0 and 4 word errors on sample-1, -2 and -3, where
# GPT-6 Sol through Codex made 7, 1 and 0 and 16, 1 and 4.
CONSENSUS_PREFERRED = "claude"


class ConsensusEngine:
    """PaddleOCR's boxes, with the text voted on by two models.

    Both models transcribe every page, at the same time, so a page takes about
    as long as the slower of the two calls rather than both in turn. Each
    transcription is aligned to PaddleOCR's lines on its own, and the two are
    then merged line by line with :func:`merge_readings`.

    A page on which one call fails is read by the other model alone, exactly
    as its own engine would read it; only when both fail does the page keep
    PaddleOCR's text.
    """

    def __init__(self, lang: str = "en", codex_model: str = DEFAULT_CODEX_MODEL,
                 claude_model: str = DEFAULT_CLAUDE_MODEL, **paddle_options):
        """Build the engine.

        Args:
            lang: PaddleOCR language code, for detection and the fallback
                reading. Neither model is told the language.
            codex_model: The model Codex runs.
            claude_model: The model Claude Code runs.
            **paddle_options: Passed to :class:`PaddleEngine`.
        """
        self._paddle = PaddleEngine(lang=lang, **paddle_options)
        self._models = {"codex": codex_model, "claude": claude_model}
        self.pages_kept_as_paddle = 0
        self.pages_read_by_one_model = 0

    def _transcribe_both(self, image: np.ndarray) -> tuple[dict[str, str], dict[str, str]]:
        """Both models' transcriptions, and the error of each call that failed."""
        calls = {"codex": transcribe_with_codex, "claude": transcribe_with_claude}
        readings: dict[str, str] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = {
                name: pool.submit(call, image, model=self._models[name])
                for name, call in calls.items()
            }
            for name, future in futures.items():
                try:
                    readings[name] = future.result()
                except RuntimeError as exc:
                    errors[name] = str(exc)
        return readings, errors

    def recognize(
        self, image: np.ndarray, confidence: float = 0.5,
        rendered: np.ndarray | None = None,
    ) -> list[OcrResult]:
        results = self._paddle.recognize(image, confidence=confidence)
        if not results:
            return results
        # The page as rendered, for the reason given in _ModelEngine.
        seen = image if rendered is None else rendered
        readings, errors = self._transcribe_both(seen)
        lines = [r.text for r in results]

        if not readings:
            self.pages_kept_as_paddle += 1
            click.echo(
                f"Warning: {'; '.join(errors.values())}; this page keeps "
                "PaddleOCR's text.",
                err=True,
            )
            return results
        if errors:
            (alone, transcription), = readings.items()
            self.pages_read_by_one_model += 1
            click.echo(
                f"Warning: {'; '.join(errors.values())}; this page is read by "
                f"{alone} alone.",
                err=True,
            )
            return _with_lines(results, align_transcription(lines, transcription))

        other = "claude" if CONSENSUS_PREFERRED == "codex" else "codex"
        a = align_transcription(lines, readings[CONSENSUS_PREFERRED])
        b = align_transcription(lines, readings[other])
        return _with_lines(
            results, [merge_readings(*line) for line in zip(lines, a, b)]
        )
