# PDF Refinery

[](https://github.com/benelog/pdf-refinery/actions/workflows/test.yml)
[](https://github.com/benelog/pdf-refinery)
[](https://opensource.org/licenses/MIT)

A command-line tool that transforms your scanned book PDFs into fully searchable documents using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

## Features

- **PaddleOCR Engine:** Industry-leading OCR with 111 language support
- **Invisible Text Layer:** Original appearance preserved with searchable/selectable text overlay
- **Simple Execution:** A single command is all it takes to make your entire PDF searchable

-----

## Requirements

- Python 3.10+
- A CPU build of PaddlePaddle, installed for you and held to 3.1 or 3.2.
  Neither neighbour works: 3.0 cannot load the detection model current
  PaddleOCR chooses, and 3.3 faults during inference on every PP-OCR model. If
  you already have PaddlePaddle, check its version before installing this — a
  copy outside that range is the likeliest reason a run fails immediately.

## Installation

```bash
pip install pdf-refinery
```

-----

## Usage

### Basic Syntax

```bash
pdf-refinery ocr [options] <input_file.pdf>
```

### Examples

```bash
# Basic usage - output saved as scanned_book_ocr.pdf
pdf-refinery ocr -l korean scanned_book.pdf

# Specify output file
pdf-refinery ocr -l korean --output searchable.pdf scanned_book.pdf

# Process specific pages with verbose output
pdf-refinery ocr -l korean --pages "1-10" --verbose scanned_book.pdf

# Also write out what OCR read, as plain text
pdf-refinery ocr -l korean --sidecar book.txt scanned_book.pdf

# Continue a run that was interrupted
pdf-refinery ocr -l korean --resume scanned_book.pdf
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `<input>_ocr.pdf` | Output file path |
| `-l, --lang` | **required** | OCR language code, repeatable |
| `--dpi` | `300` | DPI for page rendering (72–1200) |
| `--pages` | all | Page range (e.g., `"1-10"`, `"1,3,5"`) |
| `--confidence` | `0.5` | Minimum confidence threshold (0.0–1.0) |
| `--preprocess` | `binarize` | `binarize` or `none` (see [Tuning](#tuning)) |
| `--rec-model` | per language | Override PaddleOCR's recognition model |
| `--auto-rotate` | off | Read pages scanned sideways or upside down |
| `--engine` | `paddle` | `codex` has a vision model read the text (see [Reading with a model](#reading-with-a-model-through-codex)) |
| `--codex-model` | `gpt-6-sol` | Model the `codex` engine runs |
| `--textline-orientation` | off | Let the recogniser turn individual lines |
| `--unwarp` | off | Flatten page curvature before detection |
| `--force-ocr` | off | Re-OCR pages that already contain text |
| `--skip-text-threshold` | `100` | Characters a page needs before it counts as already searchable |
| `--sidecar` | none | Also write the recognised text to a plain-text file |
| `--checkpoint-every` | `20` | Save the output every N pages (`0` = only at the end) |
| `--resume` | off | Continue an interrupted run from its last checkpoint |
| `--overwrite` | off | Replace the output file if it already exists |
| `-v, --verbose` | off | Enable verbose output |
| `--version` | | Show version and exit |

Language codes are PaddleOCR's, not ISO 639: Korean is `korean` (not `ko`),
Japanese is `japan`, simplified Chinese is `ch`. An unknown code is rejected
before the run starts, with a suggestion where one is obvious.

**`-l` has no default and must be given.** Naming the wrong language can be not
a degraded result but an empty one: reading a Korean book scan with `-l en`
measures **96% character error** while the run exits successfully and writes an
output PDF of an entirely believable size.

The damage is one-directional, which is exactly why there is no safe default.
The Korean model's dictionary contains Latin, so an English scan read with
`-l korean` costs 5 character errors against the English model's 4 — nothing.
The English model has no Hangul, so the reverse destroys the document. Any
default just picks whose documents fail that way.

`-l` is repeatable, but each extra language runs the whole detection and
recognition pass again over every page — a two-language run takes roughly twice
as long. The recognition models are already multilingual, so try the single code
that matches the document's main script first (`-l korean` handles Latin
characters inside Korean text) and only add a second language if that leaves
text unread.

Every option is validated up front, and an existing output file is never
replaced without `--overwrite`. A run over a long book takes a while, so a typo
should not cost you an hour or a previous result.

### Pages that already contain text

Pages with extractable text are **skipped by default**, so a document that is
already searchable is left untouched. Pass `--force-ocr` to flatten such pages
to an image and read them again.

"Has text" means at least `--skip-text-threshold` characters (100 by default),
not merely one. Scans routinely carry a stray text layer — a stamped page
number, a running header, a watermark — and treating that as a finished page
would leave the entire body unsearchable. Lower the threshold to `0` to restore
the stricter "any text at all" rule.

### Long runs

A scanned book takes tens of minutes to hours, so the output is saved every
`--checkpoint-every` pages (20 by default) rather than only at the end. If a run
is interrupted, the finished pages are already in the output file and a
`<output>.progress` file records how far it got:

```bash
pdf-refinery ocr -l korean --resume scanned_book.pdf
```

picks up from there. The progress file is deleted once the run completes, and
starting a fresh run over an output that still has one will tell you so instead
of quietly discarding the work. Each save writes a temporary sibling and renames
it, so an interrupt during a save cannot leave a truncated PDF behind.

### Sidecar text

`--sidecar out.txt` writes what OCR read as plain text, one page per form feed
(`\f`), alongside the PDF. It is the quickest way to judge recognition quality
without opening the output, and it includes the text of skipped pages so the
file is a transcript of the whole document rather than a log of OCR calls.

### Fonts and the text layer

The text layer is invisible, so glyph shapes never matter — only whether the
font can encode the characters. There is therefore nothing to choose and no
option to set: every document is written with a **glyphless font**, the same
approach Tesseract and ocrmypdf use. It is a generated TrueType font holding
one glyph that draws nothing, which every code point in the Basic Multilingual
Plane maps to, so Hangul, Kana, Han, Thai, Devanagari, Cyrillic and Greek all
encode alike — there is no script it covers only partly.

Nothing is drawn from it, so it costs almost nothing: the font program is 664
bytes, and with its character map and the table that gives the text back the
whole text layer adds about 2 KB to a document, whatever its page count. There
is no subsetting step and no font file to supply.

The one thing outside its reach is a code point above U+FFFF, which the
two-byte encoding cannot address. Those are counted and reported at the end of
a run rather than dropped in silence; no recognition model shipped with
PaddleOCR produces them.

### Tuning

The defaults are what measured best over the three corpora in `bench/`: two
Korean scans and one Latin, from three different scanners, each with a
hand-transcribed ground truth. Character error rates are whitespace-insensitive.

| Setting | sample-1 | sample-2 | sample-3 |
|---|---|---|---|
| **defaults** (300 DPI, `binarize`) | **0.009** | **0.010** | 0.001 |
| `--preprocess none` | 0.015 | 0.025 | 0.002 |
| `--dpi 240` | 0.011 | 0.016 | 0.002 |
| `--dpi 150` | 0.015 | 0.007 | 0.002 |
| `--dpi 600` | 0.016 | 0.034 | 0.011 |
| `--unwarp` | 0.017 | 0.022 | 0.002 |
| `--textline-orientation` | 0.012 | 0.232 | 0.003 |
| the text layer the PDF already had | 0.017 | 0.146 | — |
| the wrong `-l` | 0.969 | 0.940 | 0.003 |

The rows below the first were measured before the threshold and detector
changes described next, so compare each with the 0.011 / 0.017 / 0.003 the
defaults scored then rather than with the first row.

Three changes since then were measured the same way and are now built in:

- **The binarization window is wider** — 51 pixels with an offset of 15, from
  31 and 10 — and **the detector's box threshold is PaddleOCR's own 0.6**,
  from 0.5. Together they took character errors from 25 to 22 on `sample-1`
  and 15 to 10 on `sample-2`, and left `sample-3` alone. Every window from 41
  to 71 with an offset of 12 to 20 beat the old setting on both Korean scans,
  so this is a plateau, not a lucky point. The box threshold is mostly what
  stops specks around the leaflet's photograph being read as `.*-1·*`.
- **The space after a period or comma is put back** where Hangul sits on both
  sides. The recogniser returns `거두었다.단지` for `거두었다. 단지`: the
  characters are right, but the page extracts as one run-on word and a word
  search for `단지` misses it. Word errors on `sample-1` fell from 126 to 56 of
  806, from 15.6% to 7.0%; on `sample-2` from 83 to 71. A comma after a single
  syllable is left alone, because lists like `(시,분,초)` are often printed
  unspaced.
- Measured and rejected: `unclip_ratio` 1.5 or 2.0 (worse on both Korean
  scans), no denoise before thresholding (leaflet errors more than doubled),
  Otsu's global threshold, and PP-OCRv6's detector in place of PP-OCRv5's for
  Korean (faster, but almost twice the word errors on the leaflet).

Still not fixed: the Korean recognition model's dictionary has no `「」《》`,
so corner brackets come back as `[ ]` and `< )`, and it reads `톰` as `통`
more often than not. Both are the model's limits, not the pipeline's.

Things worth knowing before you turn a knob:

- **`--preprocess binarize`** thresholds the page to black and white, which the
  theory says should lose information — PP-OCRv5 is trained on anti-aliased
  greyscale. Per character it barely matters. What it changes is **word
  boundaries**: on the Korean scans the recogniser runs phrases together when
  given greyscale, and word errors fall from 403 to 117 and from 253 to 99 when
  given black and white. `--preprocess none` is 20–35% faster and slightly
  better on the clean Latin scan, so it is worth trying on Latin documents.
- **`--dpi`** is flat between 200 and 300 — the differences above are one
  character each — and clearly worse at 400 and beyond. `--dpi 240` is about
  35% faster for no measured loss. The default stays at 300 because every
  corpus here is ordinary body text; a page of footnotes or dictionary entries
  is exactly where a lower resolution would start dropping small type, and
  nothing here would catch that.
- **`--unwarp`** is for a book photographed or pressed against the glass, where
  the text curves near the binding. On a flat scan it costs time and accuracy.
- **`--rec-model PP-OCRv5_server_rec`** sounds like the accurate setting. On the
  Latin corpus it fixed one character, broke five words, and took twice as long.

The last row is the one to take seriously. Reading a Korean scan as English is
not a degraded result, it is an empty one — and the run still exits 0 and writes
a believable-looking PDF. The reverse is harmless, because the Korean model's
dictionary contains Latin. That asymmetry is why `-l` has no default.

### Reading with a model through Codex

`--engine codex` keeps PaddleOCR's line positions but has a vision model read
the text, through the [Codex CLI](https://github.com/openai/codex). Codex must
be installed and logged in; the run uses whatever account it is logged in with.

```bash
pdf-refinery ocr -l korean --engine codex scanned_book.pdf
```

**Every page image is sent to OpenAI.** That is why it is not the default.

PaddleOCR still finds every line and reads it. The model transcribes the whole
page, and that transcription is aligned character by character against
PaddleOCR's lines, so each box gets the model's reading of the text inside it.
The invisible layer needs both halves: a model returns no coordinates, and
PaddleOCR reads less well. A line whose aligned text no longer resembles what
PaddleOCR read in that box keeps PaddleOCR's text. So does a whole page whose
Codex call fails, with a warning, so one bad call does not end a long run.
Specks that both readings agree hold no text are dropped.

Measured over `bench/`, through the whole pipeline and read back out of the
output PDF:

| | sample-1 chars / words | sample-2 chars / words | sample-3 chars / words | s/page |
|---|---|---|---|---|
| `paddle` | 22 / 56 (6.9%) | 10 / 71 (23.5%) | 2 / 5 | 16–31 |
| `codex`, `gpt-6-sol` | **7 / 16 (2.0%)** | **1 / 1 (0.3%)** | **0 / 4** | 23–37 |

The model reads corner brackets `「」《》` that PaddleOCR's Korean dictionary
does not have, and gets the word spacing of large type right where PaddleOCR
runs it together. It is handed the page as rendered, not as binarized for
PaddleOCR: thresholding cost it 12–13 character errors on `sample-1` against 7.

At API list prices, measured tokens put a 300-page book at about **$3** with
`gpt-6-sol` ($2 / $10 per million input / output tokens, about 2,800 image
tokens and 500 output tokens a page). Codex adds its own ~18k-token prompt to
every call. Run through a ChatGPT login, it counts against that plan's usage
instead.

`--codex-model gpt-6-luna` is about twenty times cheaper and not worth it. Read
on its own it made 394 character errors on `sample-1` against PaddleOCR's 22,
and not by failing to read. It rewrote text into other plausible words:
`캐번디시` became `케임브리지`, and a whole clause became `를 만나`.

The model follows the prompt's instruction to transcribe as printed, but not
perfectly: it corrected the leaflet's own typo `누루면서` to `누르면서`.

### Pages that are not the right way up

`--auto-rotate` handles a page fed through the scanner sideways or upside down.
Without it such a page is not "a bit worse" — it is lost. Measured on a Latin
scan re-laid on its side, a 1050-character page came back as **46 characters**
of noise with none of its 139 words matching; with the flag, 1051 characters and
132 of the 139.

The page itself is left exactly as it was. Only the text layer is placed
correctly, so the document still looks the way it was scanned and is searchable
anyway.

It is off by default because it costs a classifier pass per page, and because
the classifier must see the rendered greyscale rather than the binarized page —
on the three corpora at all four orientations it answered 16 of 16 correctly on
greyscale and 13 of 16 after thresholding.

`--textline-orientation` is a different thing and is best left alone. It judges
each *line* upside down or not, and it is wrong often enough to ruin a page: on
one upright leaflet page it called 14 of 31 lines inverted. A line read upside
down does not fail, it comes back as different, plausible text — `릉록릉글록릉글`
where the page says `D버튼을 눌러 맞추고 C버튼을 눌러 세팅을 완료합니다.` It was
the default until it was measured, and it cost `sample-2` 0.025 → 0.457 CER.

### Measuring changes yourself

`bench/` holds a corpus with per-page ground truth and `scripts/bench.py`
scores against it:

```bash
scripts/bench.py run --all     # measure every known variant
scripts/bench.py table         # compare what has been measured
```

It reports recognition accuracy and extracted-PDF accuracy separately, because
they fail independently — a text layer can be written in a font that cannot
encode the script, leaving recognition perfect and the output empty. See
`bench/README.md`.

-----

## Contributing

Contributions are welcome\! Please feel free to submit a pull request or open an issue to discuss proposed changes or report bugs.

### Development Setup

```bash
git clone git@github.com:benelog/pdf-refinery.git
cd pdf-refinery
pip install -e '.[dev]'
pytest -v
```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

PDF reading and writing go through [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
(Apache-2.0, over BSD-licensed PDFium) and [pikepdf](https://github.com/pikepdf/pikepdf)
(MPL-2.0, over QPDF), both of which the MIT licence above can sit next to.
PyMuPDF, which used to do this work, is AGPL v3 and is now a test-only
dependency — the suite reads every output back with an engine that had no hand
in writing it.
