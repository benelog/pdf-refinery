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
pdf-refinery ocr scanned_book.pdf

# Specify output file and language
pdf-refinery ocr -l korean --output searchable.pdf scanned_book.pdf

# Multiple languages (e.g., Korean + English mixed document)
pdf-refinery ocr -l korean -l en scanned_book.pdf

# Process specific pages with verbose output
pdf-refinery ocr --pages "1-10" --verbose scanned_book.pdf
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `<input>_ocr.pdf` | Output file path |
| `-l, --lang` | `en` | OCR language code, repeatable (`-l korean -l en`) |
| `--dpi` | `300` | DPI for page rendering |
| `--pages` | all | Page range (e.g., `"1-10"`, `"1,3,5"`) |
| `--confidence` | `0.5` | Minimum confidence threshold |
| `--force-ocr` | off | Re-OCR pages that already contain text |
| `--font-file` | built-in | Font for the text layer (single `.ttf`/`.otf`) |
| `-v, --verbose` | off | Enable verbose output |
| `--version` | | Show version and exit |

### Pages that already contain text

Pages with extractable text are **skipped by default**, so a document that is
already searchable is left untouched. Pass `--force-ocr` to flatten such pages
to an image and read them again.

### Fonts and the text layer

The text layer is invisible, so glyph shapes never matter — only whether the
font can encode the characters. Built-in fonts cover Latin, Hangul, Kana and
common Han, which is enough for most scanned books. For scripts they cannot
encode (Thai, Arabic, Devanagari, or heavy Hanja use), supply a font yourself:

```bash
pdf-refinery ocr -l korean --font-file /path/to/NotoSansKR-Regular.ttf book.pdf
```

Any character that cannot be encoded is reported at the end of the run rather
than silently dropped.

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
