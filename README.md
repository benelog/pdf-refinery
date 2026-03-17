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
| `-v, --verbose` | off | Enable verbose output |
| `--version` | | Show version and exit |

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
