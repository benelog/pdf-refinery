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


class FontResolver:
    """Chooses a font per line of OCR text.

    An explicit font file wins for every line, since a font supplied by the
    user is assumed to cover the document. Otherwise the script of the line
    selects a PyMuPDF built-in font.
    """

    def __init__(self, font_file: str | None = None):
        self._font_file = font_file
        self._embedded = (
            FontSpec(name="ocr-embedded", file=font_file) if font_file else None
        )
        self.dropped_chars: set[str] = set()

    def resolve(self, text: str) -> FontSpec:
        """Pick a font for ``text`` and record any characters it cannot encode."""
        spec = self._embedded or FontSpec(
            name=_SCRIPT_TO_BUILTIN[detect_script(text)]
        )
        self.dropped_chars |= unsupported_chars(text, spec)
        return spec

    @property
    def uses_embedded_font(self) -> bool:
        return self._embedded is not None
