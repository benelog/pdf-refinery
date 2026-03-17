# PDF Refinery

> **Project in Planning Stage**
>
> This project is currently in the planning and design phase. The features described in this README are the intended goals, but the code is not yet implemented. Stay tuned for updates!
>
[](https://github.com/benelog/pdf-refinery)
[](https://opensource.org/licenses/MIT)

A command-line tool that transforms your scanned book PDFs into fully searchable documents using AI-powered OCR.

## Purpose

This tool is for anyone who scans their physical books to PDF. PDF Refinery applies AI-powered OCR to turn those image-based scans into searchable e-books with a single command.

-----

## Features

### AI-Powered OCR

Transform your image-based PDFs into fully searchable documents. PDF Refinery integrates with leading AI models to provide high-accuracy text recognition.

  * **Multiple Provider Options:** Choose the OCR engine that best fits your needs.
      * OpenAI API (GPT-4o)
      * Google Gemini AI
      * Upstage OCR API
  * **Simple Execution:** A single command is all it takes to make your entire PDF searchable.

-----

## Installation

*(This is a placeholder section. You will need to provide the actual installation command based on your packaging choice, e.g., PyPI, Homebrew, etc.)*

```bash
# Example for pip installation
pip install pdf-refinery
```

-----

## Usage

The tool is operated through a straightforward command-line interface.

### Basic Syntax

```bash
pdf-refinery ocr [options] <input_file.pdf>
```

### Examples

Make your scanned PDF searchable using the Gemini AI API. The output will be saved as `searchable_book.pdf`.

```bash
pdf-refinery ocr --api gemini --output searchable_book.pdf "My Scanned Book.pdf"
```

  * Supported APIs for the `--api` flag: `openai`, `gemini`, `upstage`.

-----

## Configuration

To use the OCR features, you need to configure your API keys. Create a `.env` file in your home directory (`~/.env`) or the project directory and add your keys:

```
# .env file
OPENAI_API_KEY="your-openai-api-key"
GEMINI_API_KEY="your-gemini-api-key"
UPSTAGE_API_KEY="your-upstage-api-key"
```

The tool will automatically load these keys when you run an OCR command.

-----

## Contributing

Contributions are welcome\! Please feel free to submit a pull request or open an issue to discuss proposed changes or report bugs.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
