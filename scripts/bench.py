#!/usr/bin/env python3
"""Measure OCR accuracy against the ground truth in ``bench/``.

Every tuning decision in this tool was a hypothesis until this existed. The
script exists to settle them one at a time: name a variant, run it, compare.

Two accuracy numbers are always reported, because they fail independently:

``ocr``
    What the recogniser returned, read back from ``--sidecar``.
``pdf``
    What ``page.get_text()`` extracts from the output PDF.

A text layer written in a font that cannot encode Hangul leaves ``ocr`` perfect
and ``pdf`` empty. That was a real defect once, and averaging the two into one
score would have hidden it.

Each run happens in a child process. Peak memory then belongs to that run
alone, and PaddleOCR's module-level state cannot leak from one recognition
model to the next.

Usage::

    scripts/bench.py list
    scripts/bench.py run baseline
    scripts/bench.py run --all
    scripts/bench.py run baseline --set dpi=400 --name dpi-400
    scripts/bench.py table
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "bench"
RESULTS_DIR = BENCH_DIR / "results"

sys.path.insert(0, str(ROOT / "src"))

# Named here rather than duplicated: which languages the heavier recognition
# model can actually read is the package's fact, not the benchmark's.
from pdf_refinery.ocr_engine import SERVER_REC_LANGS, SERVER_REC_MODEL  # noqa: E402


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

def _wrong_language(corpus: "Corpus") -> dict:
    """Name a language the corpus is definitely not written in.

    Has to be derived rather than fixed: ``en`` is the wrong answer for the
    Korean corpora and the right one for the Latin corpus, so a constant here
    would quietly turn this variant into a second baseline.
    """
    return {"langs": ["korean" if _default_lang(corpus) == "en" else "en"]}


def _server_rec(corpus: "Corpus") -> dict | None:
    """Swap in the heavier recognition model, where its dictionary reaches.

    Returns None -- meaning "not applicable here" -- for languages the model
    has no characters for. Pointing it at Korean does not measure a slower,
    more accurate model; it measures an empty page.
    """
    if _default_lang(corpus) not in SERVER_REC_LANGS:
        return None
    return {"rec_model": SERVER_REC_MODEL}


# Values are either ``run_ocr_pipeline`` arguments (plus ``langs``), or a
# function of the corpus returning those -- or None to skip a corpus the
# variant cannot mean anything for. Anything absent takes the pipeline default,
# so "baseline" really is the shipped behaviour.
VARIANTS: dict[str, dict | Callable] = {
    "baseline": {},
    "binarize": {"preprocess": "binarize"},
    "unwarp": {"unwarp": True},
    "conf-0.3": {"confidence": 0.3},
    "dpi-150": {"dpi": 150},
    "dpi-200": {"dpi": 200},
    "dpi-240": {"dpi": 240},
    "dpi-400": {"dpi": 400},
    "dpi-600": {"dpi": 600},
    # What the binarize default buys, and what it costs.
    "no-preprocess": {"preprocess": "none"},
    "no-preprocess-200": {"preprocess": "none", "dpi": 200},
    # The one option that had no corpus to be measured on until sample-3.
    "server-rec": _server_rec,
    # Neither of the two below is a tuning knob. Both are mistakes kept in the
    # table so the size of each stays visible.
    "wrong-language": _wrong_language,
    # Was the default until it was measured: the per-line orientation
    # classifier turning upright lines over and recognising them upside down.
    "textline-ori": {"textline_orientation": True},
}


def resolve_variant(name: str, corpus: "Corpus") -> dict | None:
    """Return the overrides for ``name`` on ``corpus``, or None to skip it."""
    spec = VARIANTS[name]
    return dict(spec) if isinstance(spec, dict) else spec(corpus)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

# Ground truth is transcribed as the page prints it, with typographic quotes
# and dashes. No recogniser is graded on telling U+201C from U+0022, so both
# sides are folded to one representative before comparison.
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "´": "'", "ʼ": "'", "`": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", "　": " ",
})


def normalize(text: str) -> str:
    """Fold away differences no OCR engine should be graded on.

    NFC matters more than it looks: Hangul written as decomposed jamo scores as
    three characters per syllable against a composed ground truth, which would
    make a correct result look 200% wrong.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_PUNCT_FOLD)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b) -> int:
    """Levenshtein distance over any two sequences.

    Written out rather than pulled in: the corpus is a few thousand characters
    per run, and a benchmark that needs an install to run does not get run.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def score(truth: str, hypothesis: str) -> dict:
    """Edit distances between one page of truth and one page of output."""
    truth_n, hyp_n = normalize(truth), normalize(hypothesis)
    truth_tight = truth_n.replace(" ", "")
    hyp_tight = hyp_n.replace(" ", "")
    truth_words, hyp_words = truth_n.split(), hyp_n.split()
    return {
        "char_errors": edit_distance(truth_n, hyp_n),
        "chars": len(truth_n),
        # Korean word spacing is genuinely ambiguous and the recogniser is not
        # really being asked about it, so the no-space figure is the one to
        # trust when the two disagree.
        "char_errors_nospace": edit_distance(truth_tight, hyp_tight),
        "chars_nospace": len(truth_tight),
        "word_errors": edit_distance(truth_words, hyp_words),
        "words": len(truth_words),
    }


def aggregate(page_scores: list[dict]) -> dict:
    """Micro-average: total errors over total truth, not a mean of ratios.

    A mean of per-page rates lets a nearly blank page count as much as a full
    one.
    """
    total = {k: sum(p[k] for p in page_scores) for k in page_scores[0]} if page_scores else {}

    def ratio(errors: str, size: str) -> float:
        return total[errors] / total[size] if total.get(size) else 0.0

    return {
        "cer": ratio("char_errors", "chars"),
        "cer_nospace": ratio("char_errors_nospace", "chars_nospace"),
        "wer": ratio("word_errors", "words"),
        **total,
    }


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

class Corpus:
    def __init__(self, manifest_path: Path):
        self.dir = manifest_path.parent
        self.meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.name = self.meta.get("name", self.dir.name)
        self.pdf = (self.dir / self.meta["pdf"]).resolve()
        self.truth = {
            int(p.name[5:8]): p.read_text(encoding="utf-8")
            for p in sorted(self.dir.glob("page-*.gt.txt"))
        }

    @property
    def pages(self) -> list[int]:
        """1-based page numbers that have ground truth."""
        return sorted(self.truth)


def load_corpora() -> dict[str, Corpus]:
    return {
        c.name: c
        for c in (Corpus(m) for m in sorted(BENCH_DIR.glob("*/manifest.json")))
    }


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def _child_run(config: dict) -> None:
    """Run one pipeline configuration and report cost on stdout.

    This is the body of the child process spawned by :func:`run_variant`.
    """
    import resource

    from pdf_refinery.pipeline import run_ocr_pipeline

    started = time.perf_counter()
    run_ocr_pipeline(**{
        **config,
        "input_path": Path(config["input_path"]),
        "output_path": Path(config["output_path"]),
        "sidecar": Path(config["sidecar"]),
    })
    elapsed = time.perf_counter() - started
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("BENCH_RESULT " + json.dumps({
        "seconds": elapsed, "peak_rss_mb": peak_kb / 1024,
    }))


def read_pages(sidecar: Path) -> list[str]:
    """Split a sidecar transcript back into pages."""
    text = sidecar.read_text(encoding="utf-8")
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def extract_pdf_pages(pdf_path: Path, page_numbers: list[int]) -> list[str]:
    # MuPDF, which the pipeline does not use: the output is written with
    # PDFium and QPDF, so reading it back with an engine that had no hand in
    # making it is what makes this number a check rather than a tautology.
    import fitz

    with fitz.open(pdf_path) as doc:
        return [doc[n - 1].get_text() for n in page_numbers]


def run_variant(name: str, corpus: Corpus, overrides: dict, workdir: Path) -> dict:
    """Measure one configuration end to end and return its result record."""
    pages = corpus.pages
    output = workdir / f"{corpus.name}.{name}.pdf"
    sidecar = workdir / f"{corpus.name}.{name}.txt"
    for stale in (output, sidecar):
        stale.unlink(missing_ok=True)

    langs = overrides.pop("langs", None) or [_default_lang(corpus)]
    config = {
        "input_path": str(corpus.pdf),
        "output_path": str(output),
        "sidecar": str(sidecar),
        "langs": langs,
        "pages": ",".join(str(n) for n in pages),
        # The corpus PDFs carry a text layer from whatever OCR made them, so
        # without this every page is skipped and there is nothing to measure.
        "force_ocr": corpus.meta.get("force_ocr", False),
        "checkpoint_every": 0,
        **overrides,
    }

    env = {
        **os.environ,
        # Otherwise every child spends seconds pinging the model hosts.
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "PYTHONPATH": str(ROOT / "src"),
    }
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "_child", json.dumps(config)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"variant '{name}' failed (exit {proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
    cost = next(
        json.loads(line[len("BENCH_RESULT "):])
        for line in proc.stdout.splitlines() if line.startswith("BENCH_RESULT ")
    )

    ocr_pages = read_pages(sidecar)
    pdf_pages = extract_pdf_pages(output, pages)
    truth = [corpus.truth[n] for n in pages]
    if len(ocr_pages) != len(truth):
        raise RuntimeError(
            f"variant '{name}' transcribed {len(ocr_pages)} page(s) but the "
            f"corpus has {len(truth)}; the sidecar and the ground truth are "
            "not aligned and any score would be meaningless."
        )

    ocr_scores = [score(t, h) for t, h in zip(truth, ocr_pages)]
    pdf_scores = [score(t, h) for t, h in zip(truth, pdf_pages)]
    return {
        "variant": name,
        "corpus": corpus.name,
        "config": {k: v for k, v in config.items()
                   if k not in ("input_path", "output_path", "sidecar")},
        "seconds": cost["seconds"],
        "seconds_per_page": cost["seconds"] / len(pages),
        "peak_rss_mb": cost["peak_rss_mb"],
        "output_kb": output.stat().st_size / 1024,
        "ocr": aggregate(ocr_scores),
        "pdf": aggregate(pdf_scores),
        "per_page": [
            {"page": n, "ocr": o, "pdf": p}
            for n, o, p in zip(pages, ocr_scores, pdf_scores)
        ],
    }


def _default_lang(corpus: Corpus) -> str:
    return {"korean": "korean", "latin": "en"}.get(corpus.meta.get("script"), "en")


def score_incumbent(corpus: Corpus) -> dict | None:
    """Score the text layer the corpus PDF already carries, if it has one.

    Not ground truth and not this project's output -- it is whatever OCR
    produced the file. It is the number to beat, and without it a CER has no
    scale: 0.08 could be excellent or a regression.

    Returns None for a PDF that carries no text at all. Scoring one anyway
    reports a flat 1.0, which reads in the table as a catastrophic result
    rather than as the absence of a comparison.
    """
    hypotheses = extract_pdf_pages(corpus.pdf, corpus.pages)
    if not any(h.strip() for h in hypotheses):
        return None
    scores = [score(corpus.truth[n], h) for n, h in zip(corpus.pages, hypotheses)]
    return {
        "variant": "incumbent-text-layer",
        "corpus": corpus.name,
        "config": {"note": "text layer already present in the input PDF"},
        "seconds": 0.0, "seconds_per_page": 0.0, "peak_rss_mb": 0.0,
        "output_kb": corpus.pdf.stat().st_size / 1024,
        "ocr": aggregate(scores),
        "pdf": aggregate(scores),
        "per_page": [{"page": n, "ocr": s, "pdf": s}
                     for n, s in zip(corpus.pages, scores)],
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_COLUMNS = [
    ("variant", "{:<24}", "{:<24}"),
    ("ocr CER", "{:>9}", "{:>9.4f}"),
    ("ocr CER-ns", "{:>10}", "{:>10.4f}"),
    ("ocr WER", "{:>9}", "{:>9.4f}"),
    ("pdf CER", "{:>9}", "{:>9.4f}"),
    ("pdf CER-ns", "{:>10}", "{:>10.4f}"),
    ("s/page", "{:>8}", "{:>8.1f}"),
    ("RSS MB", "{:>8}", "{:>8.0f}"),
    ("out KB", "{:>8}", "{:>8.0f}"),
]


def _row_values(record: dict) -> list:
    return [
        record["variant"],
        record["ocr"]["cer"], record["ocr"]["cer_nospace"], record["ocr"]["wer"],
        record["pdf"]["cer"], record["pdf"]["cer_nospace"],
        record["seconds_per_page"], record["peak_rss_mb"], record["output_kb"],
    ]


def print_table(records: list[dict]) -> None:
    header = "  ".join(fmt.format(name) for name, fmt, _ in _COLUMNS)
    print(header)
    print("-" * len(header))
    for record in sorted(records, key=lambda r: r["ocr"]["cer_nospace"]):
        print("  ".join(
            fmt.format(value)
            for (_, _, fmt), value in zip(_COLUMNS, _row_values(record))
        ))
    print("\nCER-ns ignores whitespace. Lower is better throughout.")


def save(record: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{record['corpus']}.{record['variant']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_results(corpus_name: str | None = None) -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(RESULTS_DIR.glob("*.json"))]
    if corpus_name:
        records = [r for r in records if r["corpus"] == corpus_name]
    return records


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _parse_set(assignments: list[str]) -> dict:
    """Turn ``--set dpi=400 --set unwarp=true`` into pipeline arguments."""
    out: dict = {}
    for item in assignments:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        key = key.strip().replace("-", "_")
        value: object
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        elif key == "langs":
            value = raw.split("+")
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show corpora and known variants")

    run = sub.add_parser("run", help="measure one or more variants")
    run.add_argument("variant", nargs="?", help="variant name from `list`")
    run.add_argument("--all", action="store_true", help="run every known variant")
    run.add_argument("--corpus", default=None)
    run.add_argument("--set", dest="overrides", action="append", default=[],
                     metavar="KEY=VALUE", help="override a pipeline argument")
    run.add_argument("--name", default=None, help="store the result under this name")

    table = sub.add_parser("table", help="compare saved results")
    table.add_argument("--corpus", default=None)

    child = sub.add_parser("_child")
    child.add_argument("config")

    args = parser.parse_args()

    if args.command == "_child":
        _child_run(json.loads(args.config))
        return 0

    corpora = load_corpora()
    if not corpora:
        raise SystemExit(f"No corpora found under {BENCH_DIR}.")

    if args.command == "list":
        for corpus in corpora.values():
            print(f"{corpus.name}: {len(corpus.pages)} page(s) with ground truth "
                  f"-- {corpus.meta.get('description', '')}")
        print("\nVariants: " + ", ".join(VARIANTS))
        saved = load_results()
        print(f"Saved results: {len(saved)}"
              + (" (" + ", ".join(sorted(r['variant'] for r in saved)) + ")" if saved else ""))
        return 0

    if args.command == "table":
        records = load_results(args.corpus)
        if not records:
            raise SystemExit("Nothing measured yet; run a variant first.")
        print_table(records)
        return 0

    selected = list(corpora.values()) if args.corpus is None else [corpora[args.corpus]]
    if args.all:
        names = list(VARIANTS)
    elif args.variant:
        names = [args.variant]
    else:
        raise SystemExit("Give a variant name or --all.")

    workdir = BENCH_DIR / "_work"
    workdir.mkdir(parents=True, exist_ok=True)

    for corpus in selected:
        incumbent = score_incumbent(corpus)
        if incumbent is not None:
            save(incumbent)
        for name in names:
            if name not in VARIANTS:
                raise SystemExit(f"Unknown variant '{name}'. Known: {', '.join(VARIANTS)}")
            base = resolve_variant(name, corpus)
            if base is None:
                print(f"[{corpus.name}] {name} -- not applicable, skipped", flush=True)
                continue
            overrides = {**base, **_parse_set(args.overrides)}
            label = args.name or name
            print(f"[{corpus.name}] {label} ...", flush=True)
            record = run_variant(label, corpus, overrides, workdir)
            path = save(record)
            print(f"  ocr CER {record['ocr']['cer_nospace']:.4f}  "
                  f"pdf CER {record['pdf']['cer_nospace']:.4f}  "
                  f"{record['seconds_per_page']:.1f}s/page  -> {path.name}", flush=True)

    print()
    print_table(load_results(selected[0].name if len(selected) == 1 else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
