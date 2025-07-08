# PDF Refinery

> **Project in Planning Stage**
>
> This project is currently in the planning and design phase. The features described in this README are the intended goals, but the code is not yet implemented. Stay tuned for updates!
> 
[](https://github.com/benelog/pdf-refinery)
[](https://opensource.org/licenses/MIT)

A command-line tool designed for individuals who scan their physical books and manage them as PDFs. PDF Refinery helps you create searchable, size-optimized digital book collections with ease.

## Purpose

This tool is for anyone who wants to digitize their library. If you scan your books to PDF, PDF Refinery provides the essential post-processing features to turn those scans into a clean, searchable, and storage-friendly e-book library.

-----

## Features

### AI-Powered OCR

Transform your image-based PDFs into fully searchable documents. PDF Refinery integrates with leading AI models to provide high-accuracy text recognition.

  * **Multiple Provider Options:** Choose the OCR engine that best fits your needs.
      * OpenAI API (GPT-4o)
      * Google Gemini AI
      * Upstage OCR API
  * **Simple Execution:** A single command is all it takes to make your entire PDF searchable.

### Advanced Size Optimization

Take full control over your PDF file sizes without sacrificing quality.

  * **Precise Compression Control:** Specify a target file size (e.g., 50MB, 100MB) and let the tool handle the complex compression settings to meet your goal.
  * **Smart Color Conversion:** Automatically convert monochrome books scanned in color mode to a 256-level grayscale palette. This significantly reduces file size while preserving the visual integrity of the original text and images.

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
pdf-refinery <command> [options] <input_file.pdf>
```

### Examples

#### **1. Running OCR**

Make your scanned PDF searchable using the Gemini AI API. The output will be saved as `searchable_book.pdf`.

```bash
pdf-refinery ocr --api gemini --output searchable_book.pdf "My Scanned Book.pdf"
```

  * Supported APIs for the `--api` flag: `openai`, `gemini`, `upstage`.

#### **2. Optimizing File Size**

Compress a PDF to a target size of approximately 50 MB.

```bash
pdf-refinery optimize --target-size 50mb --output optimized_book.pdf "My Scanned Book.pdf"
```

#### **3. Converting to Grayscale**

Convert a color-scanned book to grayscale to reduce its size. This is ideal for books that are primarily black and white.

```bash
pdf-refinery optimize --grayscale --output grayscale_book.pdf "My Color-Scanned Book.pdf"
```

#### **4. Combining Operations**

You can combine optimization and grayscale conversion in a single command.

```bash
pdf-refinery optimize --target-size 100mb --grayscale --output final_book.pdf "My Scanned Book.pdf"
```

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
