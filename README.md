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
| `--textline-orientation` | off | Let the recogniser turn individual lines |
| `--unwarp` | off | Flatten page curvature before detection |
| `--force-ocr` | off | Re-OCR pages that already contain text |
| `--font-file` | built-in | Font for the text layer (single `.ttf`/`.otf`) |
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
font can encode the characters. Built-in fonts cover Latin, Hangul, Kana and
common Han, which is enough for most scanned books. For scripts they cannot
encode (Thai, Arabic, Devanagari, or heavy Hanja use), supply a font yourself:

```bash
pdf-refinery ocr -l korean --font-file /path/to/NotoSansKR-Regular.ttf book.pdf
```

The file must be a **single-face** `.ttf` or `.otf`. Font collections (`.ttc`,
`.otc` — including the widely installed `NotoSansCJK-Regular.ttc`) are rejected:
PyMuPDF cannot subset them, so the whole 15–20 MB font would end up embedded in
your output.

A supplied font is preferred but not forced. Lines it cannot encode fall back to
a built-in font rather than losing their text, and the count is reported at the
end. Any character no available font can encode is reported too, rather than
silently dropped.

### Tuning

The defaults are what measured best over the three corpora in `bench/`: two
Korean scans and one Latin, from three different scanners, each with a
hand-transcribed ground truth. Character error rates are whitespace-insensitive.

| Setting | sample-1 | sample-2 | sample-3 |
|---|---|---|---|
| **defaults** (300 DPI, `binarize`) | **0.011** | **0.017** | 0.003 |
| `--preprocess none` | 0.015 | 0.025 | 0.002 |
| `--dpi 240` | 0.011 | 0.016 | 0.002 |
| `--dpi 150` | 0.015 | 0.007 | 0.002 |
| `--dpi 600` | 0.016 | 0.034 | 0.011 |
| `--unwarp` | 0.017 | 0.022 | 0.002 |
| `--textline-orientation` | 0.012 | 0.232 | 0.003 |
| the text layer the PDF already had | 0.017 | 0.146 | — |
| the wrong `-l` | 0.969 | 0.940 | 0.003 |

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
