# Benchmark corpus

Every accuracy claim about this project has to come from here. Before this
directory existed, "the OCR got better" was an assertion no commit could check.

## Layout

```
bench/
  <corpus>/
    manifest.json      # which PDF, how to run it, what it exercises
    page-001.gt.txt    # ground truth, one file per 1-based page
    ...
  results/             # JSON written by scripts/bench.py (git-ignored)
```

`manifest.json` fields:

| field | meaning |
|---|---|
| `pdf` | path to the PDF, relative to the manifest |
| `script` | dominant script, for picking a default language |
| `force_ocr` | run with `--force-ocr`; required when the PDF already has a text layer |
| `description` / `notes` | what this corpus exercises, and any traps |

Ground truth is a faithful transcription of what a human reads on the page,
including typographic quotes, the running header/footer, and the source's own
typos. The scorer folds punctuation variants and whitespace, so those choices
do not decide the score — see `scripts/bench.py`.

One caveat the scorer cannot fold: the fullwidth comma `U+FF0C`. Neither corpus
prints one, so a hypothesis that emits it is charged for it. The incumbent
layer on `sample-1` does, which is a real part of why it scores 0.017 and not
lower.

## Running

```bash
scripts/bench.py list                     # corpora and saved runs
scripts/bench.py run baseline             # measure one named variant
scripts/bench.py run --all                # measure every named variant
scripts/bench.py run baseline --set dpi=400 --name baseline-400dpi
scripts/bench.py table                    # compare everything measured so far
```

Each run reports two separate numbers, because they fail independently:

- **ocr** — the strings the recogniser returned (from `--sidecar`).
- **pdf** — the strings `page.get_text()` gets back out of the output PDF.

A font that cannot encode Hangul leaves `ocr` untouched and destroys `pdf`.
That was a real defect (plan.md §1-1) and only the second number showed it.

## The three corpora here

| corpus | PDF | what it is | incumbent CER-ns |
|---|---|---|---|
| `sample-1` | `tests/sample-1.pdf` | 4 pages of a scanned book: justified Korean prose, serif, running header, ~310 DPI grayscale | 0.017 |
| `sample-2` | `tests/sample-2.pdf` | 2 pages of a scanned product leaflet: sans-serif at mixed sizes, hanging label column around a photograph, visible skew, ~300 DPI grayscale | 0.146 |
| `sample-3` | `tests/sample-3.pdf` | 2 pages of a 1914 letterpress novel: justified Latin prose, serif, 600 DPI grayscale Google scan, public domain | none |

They fail differently, which is the point of having more than one. `sample-1`
is a clean single column that the incumbent layer already reads at 1.7% CER-ns
— it measures whether a change *regresses* the easy case. `sample-2` is where
the incumbent collapses (page 1 alone is 29% CER, most of it rendered as
unrelated Hanja) — it measures whether a change *helps* the hard case.
`sample-3` is the only corpus that is not Korean, and the only one clean enough
that a change moving it at all deserves suspicion. A knob that moves one corpus
and not the others has not been shown to generalise — `--preprocess` flipped
twice on exactly that mistake.

`sample-2` also holds the layout traps: text is not one column, the button
labels sit around a large solid-black image, and the leaflet contains its own
typos, which the ground truth reproduces rather than corrects.

`sample-3` carries no text layer at all, so it has no `incumbent-text-layer`
row — the bench omits that variant rather than scoring an empty string as a
catastrophic 1.0. It exists because `--rec-model` could not be measured without
a Latin page: PaddleOCR picks a lighter recognition model for `en` than for
Chinese or Japanese, and whether the heavier one is worth forcing stayed an
open question as long as there was nothing to ask it of.

Still absent, in rough order of how much they would change a decision: a
low-contrast or fax-quality scan, a genuine two-column body, vertical writing,
and a page with tables or footnotes.

## Variants that are not tuning knobs

Two entries are mistakes, kept in the table so the size of each stays visible:

- `wrong-language` names a language the corpus is not written in. It is derived
  per corpus rather than fixed, because `en` is the wrong answer for the Korean
  corpora and the right one for the Latin one. It also shows the mistake is
  one-directional: `-l en` on Korean is ruinous, `-l korean` on English is not.
- `textline-ori` restores the per-line orientation classifier that used to be
  on by default, which costs `sample-2` 0.0251 → 0.4571 CER-ns.

A variant may be a function of the corpus, and may return `None` for "not
applicable here". `server-rec` does that for the Korean corpora: that model's
dictionary has no Hangul, so running it there would measure a blank page rather
than a slower, more accurate model.

## Adding a corpus

Anything larger than a megabyte or two should live behind a download script
with a checksum rather than in git, and should be public-domain material.

`sample-3` is the model to copy. It is two pages lifted out of a 384-page
archive.org scan, which keeps it to 113 KB — small enough to commit without the
download-script machinery — and it is public domain (published 1914; the source
record says `NOT_IN_COPYRIGHT`). Its `manifest.json` records the source URL and
which pages of the scan it holds, so the extract can be reproduced or extended.

**On what is committed here**: `sample-1` and `sample-2` are scans of published
material with no clear licence, kept in git for local measurement. That is fine
for a private repository and a question worth settling before this one is
published. Anything added from here on should follow `sample-3` instead.
