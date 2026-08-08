"""Font selection for the invisible text layer.

The overlay text is drawn with render mode 3 (invisible), so glyph shapes never
matter -- only whether the PDF font can *encode* a code point, because that is
what determines the text a reader extracts or searches.

PyMuPDF's Base14 ``helv`` is limited to Latin-1, and the built-in CJK fonts
cover Hangul syllables well but only a fraction of Han ideographs. Neither
``Font.has_glyph`` nor the font's cmap predicts what ``insert_text`` actually
encodes, so encodability is probed empirically and cached.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz

# PyMuPDF built-in fonts, keyed by the script they cover best.
BUILTIN_LATIN = "helv"
BUILTIN_KOREAN = "korea"
BUILTIN_JAPANESE = "japan"
BUILTIN_CHINESE = "china-s"

# Code point ranges used to pick a font for a line of text.
_HANGUL_RANGES = ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F),
                  (0xAC00, 0xD7A3), (0xD7B0, 0xD7FF))
_KANA_RANGES = ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D))
_HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))


def _in_ranges(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= code <= hi for lo, hi in ranges)


def detect_script(text: str) -> str:
    """Classify a line of text by the script that decides its font.

    Returns one of ``"hangul"``, ``"kana"``, ``"han"`` or ``"latin"``. Mixed
    lines resolve by priority: Hangul and Kana are unambiguous language
    signals, while Han alone is shared between Chinese and Japanese.
    """
    has_kana = has_hangul = has_han = False
    for ch in text:
        code = ord(ch)
        if _in_ranges(code, _HANGUL_RANGES):
            has_hangul = True
        elif _in_ranges(code, _KANA_RANGES):
            has_kana = True
        elif _in_ranges(code, _HAN_RANGES):
            has_han = True

    if has_hangul:
        return "hangul"
    if has_kana:
        return "kana"
    if has_han:
        return "han"
    return "latin"


_SCRIPT_TO_BUILTIN = {
    "hangul": BUILTIN_KOREAN,
    "kana": BUILTIN_JAPANESE,
    "han": BUILTIN_CHINESE,
    "latin": BUILTIN_LATIN,
}


@dataclass(frozen=True)
class FontSpec:
    """A font usable with :meth:`fitz.Page.insert_text`.

    Attributes:
        name: Resource name for the font inside the PDF.
        file: Path to a font file, or None to use a PyMuPDF built-in font.
    """

    name: str
    file: str | None = None

    @property
    def is_embedded(self) -> bool:
        return self.file is not None


# Probing means writing characters into a scratch page and reading them back,
# which is the only reliable way to know what insert_text encodes. A whole
# batch goes onto one page because creating the page dominates the cost, and
# results are cached since a document reuses the same small set of characters.
_PROBE_FONTSIZE = 5
_PROBE_LINE_HEIGHT = 8
_PROBE_COLS = 100
_PROBE_ROWS = 120
_PROBE_MARGIN = 10
_PROBE_PAGE = (
    _PROBE_MARGIN * 2 + _PROBE_COLS * _PROBE_FONTSIZE * 2,
    _PROBE_MARGIN * 2 + _PROBE_ROWS * _PROBE_LINE_HEIGHT,
)

_encodable_cache: dict[tuple[str, str | None, str], bool] = {}


def _probe_batch(spec: FontSpec, chars: list[str]) -> set[str]:
    """Return which of ``chars`` survive an insert/extract round trip.

    Characters are laid out in a grid that stays inside the page, because text
    drawn outside the crop box would not be extracted and would look like a
    font failure.
    """
    encodable: set[str] = set()
    page_w, page_h = _PROBE_PAGE
    per_page = _PROBE_COLS * _PROBE_ROWS

    for start in range(0, len(chars), per_page):
        batch = chars[start:start + per_page]
        doc = fitz.open()
        page = doc.new_page(width=page_w, height=page_h)
        try:
            for row in range(0, len(batch), _PROBE_COLS):
                line = "".join(batch[row:row + _PROBE_COLS])
                page.insert_text(
                    fitz.Point(
                        _PROBE_MARGIN,
                        _PROBE_MARGIN + (row // _PROBE_COLS + 1) * _PROBE_LINE_HEIGHT,
                    ),
                    line,
                    fontsize=_PROBE_FONTSIZE,
                    fontname=spec.name,
                    fontfile=spec.file,
                )
            extracted = page.get_text()
            encodable |= {ch for ch in batch if ch in extracted}
        except Exception:
            pass  # nothing in this batch is usable
        finally:
            doc.close()

    return encodable


def unsupported_chars(text: str, spec: FontSpec) -> set[str]:
    """Return the characters of ``text`` that ``spec`` cannot encode.

    Whitespace is ignored: it always round-trips, sometimes as a different
    space character, which would otherwise register as a false drop.

    Note:
        This function exists because the built-in fonts have real gaps --
        measured by round trip, the CJK ones encode all Hangul syllables but
        only 31% of Han and no Thai, Arabic or Devanagari, which is why
        ``--font-file`` exists. The way out of the whole problem is the
        glyphless font Tesseract uses: one empty glyph plus a ToUnicode CMap
        covers every code point in about 4 KB, and nothing here would need to
        choose a font or report a dropped character again. PyMuPDF's high-level
        API cannot build one; it needs a Type0/Identity-H font assembled
        directly, via fontTools or pikepdf. That is a new dependency for a
        problem ``--font-file`` already answers, so it has not been done.
    """
    candidates = {ch for ch in text if not ch.isspace()}
    unknown = [
        ch for ch in candidates
        if (spec.name, spec.file, ch) not in _encodable_cache
    ]
    if unknown:
        encodable = _probe_batch(spec, unknown)
        for ch in unknown:
            _encodable_cache[(spec.name, spec.file, ch)] = ch in encodable

    return {
        ch for ch in candidates
        if not _encodable_cache[(spec.name, spec.file, ch)]
    }


_measure_cache: dict[str, fitz.Font] = {}


def text_length(text: str, spec: FontSpec, fontsize: float) -> float:
    """Width ``text`` occupies when drawn with ``spec`` at ``fontsize``.

    Two different calls are needed because the two font kinds are written into
    the PDF differently, and each reports the other's metrics wrongly. A
    built-in CJK name goes in as a CID font whose advances only
    ``get_text_length`` knows; both were checked against the span box the
    viewer actually produces.
    """
    if not text:
        return 0.0
    if spec.file is None:
        return fitz.get_text_length(text, fontname=spec.name, fontsize=fontsize)
    font = _measure_cache.get(spec.file)
    if font is None:
        font = _measure_cache[spec.file] = fitz.Font(fontfile=spec.file)
    return font.text_length(text, fontsize)


_metrics_cache: dict[tuple[str, str | None], tuple[float, float]] = {}

# Text for the metrics probe. Only the font's own ascent and descent decide the
# span box, not which characters are in it, so this needs to be encodable
# rather than representative -- ASCII is the one thing every candidate font
# here can write.
_METRICS_PROBE_TEXT = "Hxq"
_METRICS_PROBE_SIZE = 100.0


def span_metrics(spec: FontSpec) -> tuple[float, float]:
    """Ascent and descent of a line drawn with ``spec``, in em, both positive.

    These decide the box a viewer highlights and ``get_text()`` reports, so
    they are what a line must be sized and placed by if the text layer is to
    line up with the scan underneath it.

    Measured by drawing and reading the span back, not taken from
    ``fitz.Font``. For a font loaded from a file the two agree. For the
    built-in CJK names they do not: ``fitz.Font("korea")`` declares
    1.043/-0.266, while the CID font PyMuPDF actually writes into the PDF
    behaves as 1.000/0.200. Trusting the declared numbers puts every Korean
    line about 8% of its height out of position.
    """
    key = (spec.name, spec.file)
    cached = _metrics_cache.get(key)
    if cached is not None:
        return cached

    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    baseline_y = 200.0
    page.insert_text(
        (50.0, baseline_y), _METRICS_PROBE_TEXT, fontsize=_METRICS_PROBE_SIZE,
        fontname=spec.name, fontfile=spec.file, render_mode=3,
    )
    spans = [
        span
        for block in page.get_text("dict")["blocks"] if block["type"] == 0
        for line in block["lines"] for span in line["spans"]
    ]
    if spans:
        _, y0, _, y1 = spans[-1]["bbox"]
        metrics = ((baseline_y - y0) / _METRICS_PROBE_SIZE,
                   (y1 - baseline_y) / _METRICS_PROBE_SIZE)
    else:
        # A font that cannot write even ASCII is already reported elsewhere;
        # fall back to the declared numbers rather than failing the overlay.
        font = fitz.Font(spec.name) if spec.file is None else fitz.Font(fontfile=spec.file)
        metrics = (font.ascender, -font.descender)
    doc.close()

    if metrics[0] + metrics[1] <= 0:
        metrics = (0.8, 0.2)
    _metrics_cache[key] = metrics
    return metrics


# Font collections hold several faces in one file. PyMuPDF's subset_fonts()
# cannot rewrite them -- it reports "Index bounds" and leaves the font whole --
# so a 18 MB NotoSansCJK-Regular.ttc lands in the output untouched.
COLLECTION_SUFFIXES = (".ttc", ".otc")
SINGLE_FACE_SUFFIXES = (".ttf", ".otf")


def check_font_file(path: str) -> None:
    """Raise ``ValueError`` if ``path`` is unusable for the text layer.

    Catches the two failures that otherwise surface only after a full OCR run:
    a font collection, which cannot be subset and bloats the output, and a file
    PyMuPDF cannot parse at all.
    """
    suffix = Path(path).suffix.lower()
    if suffix in COLLECTION_SUFFIXES:
        raise ValueError(
            f"'{Path(path).name}' is a font collection ({suffix}). PyMuPDF cannot "
            "subset collections, so the entire font -- often 15-20 MB -- would be "
            "embedded in the output. Use a single-face .ttf or .otf instead "
            "(e.g. NotoSansKR-Regular.ttf), or split the collection with "
            "'fonttools ttLib.ttCollection'."
        )
    if suffix not in SINGLE_FACE_SUFFIXES:
        raise ValueError(
            f"expected a {' or '.join(SINGLE_FACE_SUFFIXES)} file, got "
            f"'{Path(path).name}'."
        )
    try:
        fitz.Font(fontfile=path)
    except Exception as exc:
        raise ValueError(f"'{Path(path).name}' could not be read as a font: {exc}")


class FontResolver:
    """Chooses a font per line of OCR text.

    A font file supplied by the user is preferred, but it does not win
    unconditionally: a font that cannot encode a line would make that line
    unsearchable, while a built-in font might encode it fine. So each line goes
    to whichever candidate drops the fewest characters, and the number of lines
    that fell back is recorded for the final report.
    """

    def __init__(self, font_file: str | None = None):
        self._font_file = font_file
        self._embedded = (
            FontSpec(name="ocr-embedded", file=font_file) if font_file else None
        )
        self.dropped_chars: set[str] = set()
        self.fallback_lines = 0
        self._embedded_used = False

    def _candidates(self, text: str) -> list[FontSpec]:
        """Fonts to consider for ``text``, best guess first."""
        builtin = FontSpec(name=_SCRIPT_TO_BUILTIN[detect_script(text)])
        specs = [builtin] if self._embedded is None else [self._embedded, builtin]
        if builtin.name != BUILTIN_LATIN:
            # Hangul and Latin mix constantly; helv covers Latin-1 far better
            # than the built-in CJK fonts do.
            specs.append(FontSpec(name=BUILTIN_LATIN))
        return specs

    def resolve(self, text: str) -> FontSpec:
        """Pick a font for ``text`` and record any characters it cannot encode."""
        candidates = self._candidates(text)
        best = candidates[0]
        missing = unsupported_chars(text, best)
        for alternative in candidates[1:]:
            if not missing:
                break
            alt_missing = unsupported_chars(text, alternative)
            if len(alt_missing) < len(missing):
                best, missing = alternative, alt_missing

        if self._embedded is not None and best is not self._embedded:
            self.fallback_lines += 1
        if best.is_embedded:
            self._embedded_used = True
        self.dropped_chars |= missing
        return best

    @property
    def uses_embedded_font(self) -> bool:
        """Whether an embedded font actually ended up in the document.

        Subsetting is pointless -- and the flag misleading -- when every line
        fell back to a built-in font.
        """
        return self._embedded_used
