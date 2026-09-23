---
name: pdf-ocr
description: Make a scanned PDF searchable with the pdf-refinery CLI, installing the tool first if it is missing. Use when the user wants to OCR a scanned PDF or book, add an invisible text layer so the PDF can be searched or copied from, or get the text out of a scan (including Korean, Japanese, Chinese and Latin-script documents). Not for PDFs that already have selectable text unless the user wants them re-read.
---

# OCR a scanned PDF with pdf-refinery

`pdf-refinery` renders each page, reads it with PaddleOCR, and writes an
invisible text layer over the original image, so the PDF looks the same but
becomes searchable and selectable. It can optionally have a vision model read
the text through the Codex CLI instead (`--engine codex`), which is much more
accurate on Korean but sends every page image to OpenAI.

Work through the steps in order. A book takes one to three hours, so every
choice that can be checked on two pages is checked there first.

## 1. Make sure the tool is installed

```bash
command -v pdf-refinery && pdf-refinery --version
```

If that fails, install it from PyPI with `uv` (preferred) or `pipx`, pinning
Python 3.12. PaddlePaddle only ships wheels for a few Python versions, and the
tool holds PaddlePaddle to 3.1–3.2 because other versions fail to run the
models.

```bash
# uv (install uv first if needed: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv tool install --python 3.12 pdf-refinery

# or pipx
pipx install --python python3.12 pdf-refinery
```

Then run `pdf-refinery --version` again. If the shell cannot find it, the
tool's bin directory (usually `~/.local/bin`) is not on `PATH`; run
`uv tool update-shell` or call it by full path.

To upgrade later: `uv tool upgrade pdf-refinery` (or
`pipx upgrade pdf-refinery`).

Things that need the network and may need sandbox approval:

- the install itself (about 1.2 GB of dependencies, mostly PaddlePaddle);
- the first OCR run for each language, which downloads 100–150 MB of
  PaddleOCR models to `~/.paddlex`.

If an existing PaddlePaddle outside 3.1–3.2 is visible to the environment, a
run fails immediately. The isolated `uv tool` / `pipx` install avoids that.

## 2. Establish the language

`-l` is required and has no default, on purpose. The wrong language does not
degrade the result, it destroys it: a Korean scan read with `-l en` came back
96% wrong while the run still exited 0 and wrote a normal-looking PDF.

Codes are PaddleOCR's, not ISO 639:

| Script | Code |
|---|---|
| Korean (Latin mixed in is fine) | `korean` |
| English / Latin-only | `en` |
| Japanese | `japan` |
| Simplified Chinese | `ch` |
| Traditional Chinese | `chinese_cht` |
| French, German, Spanish, Italian, Portuguese | `fr`, `german`, `es`, `it`, `pt` |
| Russian, Thai, Arabic, Hindi, Vietnamese | `ru`, `th`, `ar`, `hi`, `vi` |

If the user did not say, look at a page rather than guess. With a `uv`
install, render the first pages to images and view them:

```bash
"$(uv tool dir)/pdf-refinery/bin/pypdfium2" render input.pdf -o /tmp/peek --pages 1-2 --scale 1
```

Use one code for the document's main script. `-l korean` already reads the
Latin inside Korean text. Every extra `-l` runs the whole pass again, which
doubles the time.

## 3. Choose the engine

- `--engine paddle` (default): everything stays on this machine and costs
  nothing. Use this unless the user asks for higher accuracy.
- `--engine codex`: PaddleOCR still finds the lines, but GPT-6 Sol reads the
  text through the Codex CLI. On the project's Korean benchmark, character
  errors fell from 22 to 7 and word errors from 71 to 1 on a leaflet.
  Before using it:
  - **ask the user**, because every page image is sent to OpenAI;
  - check `codex login status` reports a login;
  - give the cost: about $3 per 300 pages at API list prices, or usage
    against the ChatGPT plan when Codex is logged in that way;
  - allow for time: add 20–35 s per page;
  - keep a single `-l`; the CLI refuses more than one with this engine.

  Do not pass `--codex-model gpt-6-luna` to save money. It rewrote Korean
  text into different plausible words and did worse than PaddleOCR alone.

## 4. Trial run on two pages

```bash
pdf-refinery ocr -l korean --pages 1-2 --force-ocr \
  -o /tmp/trial.pdf --sidecar /tmp/trial.txt --overwrite input.pdf
```

Read `/tmp/trial.txt` and compare it with the page image:

- Mostly wrong characters, or the wrong script → wrong `-l`. Fix it before
  going on.
- Text rotated or garbled on a sideways or upside-down scan → add
  `--auto-rotate`.
- `Note: N detected line(s) were too small to place` → small type; try
  `--dpi 400`.

Pick trial pages that carry body text; a cover or blank page proves nothing.

## 5. Full run

```bash
pdf-refinery ocr -l korean --sidecar output.txt -o output.pdf input.pdf
```

- Allow roughly 10–30 s per page on a CPU: a 300-page book takes 1–2.5 hours.
  Start it in the background and check on it, rather than blocking on one
  command that may hit a tool timeout.
- Progress is saved every 20 pages. If the run stops, continue with the same
  command plus `--resume`. Do not delete `output.pdf.progress`.
- The output is never replaced silently. If it already exists, the run
  stops. Pass `--overwrite` only when the user wants the old result gone.
- Pages that already have a text layer (100+ characters) are skipped. Pass
  `--force-ocr` to re-read them, for example when the existing layer is
  poor. The default output name is `<input>_ocr.pdf`.

## 6. Check and report

- Confirm the run ended with `Done. N text blocks added`, and that the
  sidecar has text for every page (pages are separated by form feeds, `\f`).
- Read a few pages of the sidecar for sanity.
- Pass on any warnings the run printed:
  - lines too small to place;
  - characters above U+FFFF that could not be encoded;
  - `this page keeps PaddleOCR's text` from a failed Codex call.
- Tell the user where the PDF and the transcript are, which language and
  engine were used, and how many pages were skipped as already searchable.

## Options not to reach for

- `--textline-orientation`: judges individual lines upside down and gets it
  wrong on upright pages, turning correct text into different, plausible
  text. Use it only for a scan that really has inverted lines.
- `--unwarp`: only for photographed pages curving at the binding. It hurts
  flat scans.
- `--preprocess none`: slightly better on clean Latin scans but runs Korean
  words together. Leave the default for Korean.
- `--rec-model PP-OCRv5_server_rec`: slower and no better on the benchmark.
  It has no Hangul, and the CLI refuses it with `-l korean`.

`pdf-refinery ocr --help` lists everything else.
