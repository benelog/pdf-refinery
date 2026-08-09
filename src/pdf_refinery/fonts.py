"""The font used for the invisible text layer.

The overlay is drawn with render mode 3 (invisible), so glyph shapes never
matter -- only whether the PDF font can *encode* a code point, because that is
what determines the text a reader extracts or searches. That makes the usual
font problem the wrong problem to solve. Instead of choosing a face that
happens to cover the document's script, this builds the one font that covers
every script at once: a *glyphless* font, the same trick Tesseract and ocrmypdf
use for their text layers.

It is a two-glyph TrueType font -- ``.notdef`` and one outline that encloses no
area -- reached through a ``CIDToGIDMap`` that sends every one of the 65536
Identity-H codes to that single glyph. :func:`embed_font` wraps it as a
Type0/CIDFontType2 and pairs it with an identity ToUnicode CMap, which is all a
reader needs to give the text back. Extraction is verified against two
independent engines (PDFium, which renders pages here, and MuPDF, which has no
part in the shipped code at all).

The whole thing costs about two kilobytes in the output -- 366 bytes of font
program, 150 for the map, and 1.7 KB of ToUnicode CMap -- once per document
however many pages draw with it. That is why there is no subsetting step and no
``--font-file`` option any more: what those existed for was the coverage gap
this font does not have.

The one thing outside its reach is a code point above U+FFFF, which a two-byte
Identity-H code cannot address; see :func:`unsupported_chars`.
"""

import struct
import zlib

import pikepdf
from pikepdf import Array, Dictionary, Name, Stream, String

# Resource name the font is registered under inside the PDF.
FONTNAME = "glyphless"

UNITS_PER_EM = 1000
_ADVANCE = 500
_ASCENT = 800
_DESCENT = -200

# Glyph 0 is ``.notdef`` and glyph 1 is what every code point resolves to.
_NUM_GLYPHS = 2
_GLYPH = 1

# An Identity-H code is two bytes wide, so this is the last code point the
# text layer can carry.
_MAX_CODE_POINT = 0xFFFF

# What a caller needs to place a line, in em.
ADVANCE_EM = _ADVANCE / UNITS_PER_EM
LINE_ASCENT = _ASCENT / UNITS_PER_EM
LINE_DESCENT = -_DESCENT / UNITS_PER_EM


def text_length(text: str, fontsize: float) -> float:
    """Width ``text`` occupies when drawn at ``fontsize``.

    Every glyph has the same advance, so this is arithmetic rather than a
    metrics lookup. What it counts is what :func:`encode` will write, not what
    was passed in, so a line carrying a character the font cannot encode is
    still fitted to the width of the part that gets drawn. The number is
    deliberately not the width the line had on the page --
    :func:`pdf_writer.overlay_text_on_page` divides the detected box by it and
    scales the line horizontally to fit, so what matters is only that it is
    proportional to the character count.
    """
    encodable = sum(1 for ch in text if ord(ch) <= _MAX_CODE_POINT)
    return encodable * ADVANCE_EM * fontsize


def unsupported_chars(text: str) -> set[str]:
    """Return the characters of ``text`` the font cannot encode.

    Only code points above U+FFFF qualify: an Identity-H code is two bytes
    wide, and reaching the astral planes would mean a second encoding for
    characters no recognition model in use here emits. They are reported
    rather than dropped in silence, because a lost character is a word that
    cannot be searched for and nothing else in the run would mention it.
    """
    return {ch for ch in text if ord(ch) > _MAX_CODE_POINT}


def encode(text: str) -> bytes:
    """Encode ``text`` as the Identity-H code string the layer is written with.

    Each code point becomes its own two-byte code, which the CMap turns back
    into that same code point when a reader extracts it. Anything
    :func:`unsupported_chars` reports is skipped -- there is no code for it --
    which is why :func:`text_length` counts the same way.
    """
    return b"".join(
        struct.pack(">H", ord(ch)) for ch in text if ord(ch) <= _MAX_CODE_POINT
    )


# ---------------------------------------------------------------------------
# TrueType assembly
#
# Written out with struct rather than pulled in from fontTools: the font is ten
# fixed tables holding two glyphs, the layout never varies, and a build-time
# dependency for a hundred lines of packing would be the larger thing to
# maintain.
# ---------------------------------------------------------------------------

def _glyf() -> bytes:
    """The one glyph, as an outline that draws nothing.

    It would be simpler to leave it empty -- Tesseract's glyphless font does,
    and nothing here is ever painted, since the layer is drawn invisible. It is
    not empty because PDFium measures a character by its glyph's bounding box
    and discards any character whose box comes out with no width. An empty
    glyph is zero-height, which costs nothing while the text runs across the
    page and everything the moment it runs down it: on a page turned by
    ``/Rotate``, or scanned sideways, that zero height becomes the width and
    PDFium extracts none of the line, while MuPDF reads it perfectly -- which
    is the kind of defect that ships. A two-point contour encloses no area and
    still has a bounding box, which is the whole reason this is not ``b""``.
    """
    contour = (
        struct.pack(">hhhhh", 1, 0, _DESCENT, _ADVANCE, _ASCENT)  # bbox
        + struct.pack(">HH", 1, 0)      # one contour ending at point 1, no hinting
        + bytes([0x01, 0x01])           # both points on-curve, coordinates as int16
        + struct.pack(">hh", 0, _ADVANCE)          # x, as deltas from the origin
        + struct.pack(">hh", _DESCENT, _ASCENT - _DESCENT)  # y
    )
    return contour + bytes(-len(contour) % 4)


def _loca() -> bytes:
    """Glyph 0 runs from 0 to 0 -- ``.notdef`` is empty -- and glyph 1 follows."""
    return struct.pack(">III", 0, 0, len(_glyf()))


def _cmap() -> bytes:
    """The smallest legal format-4 subtable, present only for well-formedness.

    Nothing consults it: under Identity-H the code is the CID and the
    ``CIDToGIDMap`` turns that into the glyph, so the font's own character
    mapping never comes into it. A TrueType font without a cmap is still
    invalid, hence the terminator segment the format requires and nothing else.
    """
    segments = 1
    subtable = struct.pack(
        ">HHHHHHH",
        4,                  # format
        16 + 8 * segments,  # length
        0,                  # language
        segments * 2,
        2,                  # searchRange
        0,                  # entrySelector
        0,                  # rangeShift
    )
    subtable += struct.pack(">H", 0xFFFF)   # endCode
    subtable += struct.pack(">H", 0)        # reservedPad
    subtable += struct.pack(">H", 0xFFFF)   # startCode
    subtable += struct.pack(">h", 1)        # idDelta: sends U+FFFF to .notdef
    subtable += struct.pack(">H", 0)        # idRangeOffset
    # One encoding record: platform 3 (Windows), encoding 1 (BMP).
    return struct.pack(">HHHHI", 0, 1, 3, 1, 12) + subtable


def _head() -> bytes:
    return struct.pack(
        ">IIIIHHqqhhhhHHhhh",
        0x00010000,   # version
        0x00010000,   # fontRevision
        0,            # checkSumAdjustment, patched in by build_font
        0x5F0F3CF5,   # magicNumber
        0x0003,       # flags
        UNITS_PER_EM,
        0, 0,         # created, modified
        0, _DESCENT, _ADVANCE, _ASCENT,   # bounding box
        0,            # macStyle
        8,            # lowestRecPPEM
        2,            # fontDirectionHint
        1,            # indexToLocFormat: 32-bit offsets
        0,            # glyphDataFormat
    )


def _hhea() -> bytes:
    return struct.pack(
        ">IhhhHhhhhhhhhhhhh",
        0x00010000, _ASCENT, _DESCENT, 0,
        _ADVANCE, 0, 0, 0,   # advanceWidthMax and the side bearings
        1, 0, 0,             # caret slope and offset
        0, 0, 0, 0,          # reserved
        0,                   # metricDataFormat
        1,                   # numberOfHMetrics
    )


def _maxp() -> bytes:
    return struct.pack(">IH" + "H" * 13, 0x00010000, _NUM_GLYPHS, *([0] * 13))


def _hmtx() -> bytes:
    """One metric record, then a left side bearing for the remaining glyph.

    ``numberOfHMetrics`` is 1, which the format defines as "every glyph after
    the first has the last stated advance".
    """
    return struct.pack(">Hh", _ADVANCE, 0) + bytes(2 * (_NUM_GLYPHS - 1))


def _post() -> bytes:
    # Version 3.0: no glyph names, which is the point.
    return struct.pack(">IIhhIIIII", 0x00030000, 0, 0, 0, 1, 0, 0, 0, 0)


def _os2() -> bytes:
    return struct.pack(
        ">HhHHH" + "h" * 8 + "hhh" + "10s" + "IIII" + "4s" + "HHH"
        + "hhh" + "HH" + "II" + "hhHHH",
        4,                    # version
        _ADVANCE,             # xAvgCharWidth
        400, 5, 0,            # weight, width, fsType (installable)
        650, 700, 0, 140,     # subscript
        650, 700, 0, 480,     # superscript
        50, 250,              # strikeout
        0,                    # sFamilyClass
        bytes(10),            # panose
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,  # ulUnicodeRange
        b"NONE",
        0x0040,               # fsSelection: regular
        0x0000, 0xFFFF,       # usFirstCharIndex, usLastCharIndex
        _ASCENT, _DESCENT, 0,     # sTypo{Ascender,Descender,LineGap}
        _ASCENT, -_DESCENT,       # usWin{Ascent,Descent}, both unsigned
        0xFFFFFFFF, 0xFFFFFFFF,  # ulCodePageRange
        500, 700,             # sxHeight, sCapHeight
        0x0020, 0x0020, 1,
    )


def _name() -> bytes:
    records = b""
    strings = b""
    for name_id, text in ((1, "GlyphLess"), (2, "Regular"), (4, "GlyphLess"),
                          (5, "Version 1.000"), (6, "GlyphLess")):
        encoded = text.encode("utf-16-be")
        records += struct.pack(">HHHHHH", 3, 1, 0x0409, name_id,
                               len(encoded), len(strings))
        strings += encoded
    return struct.pack(">HHH", 0, 5, 6 + 12 * 5) + records + strings


def _checksum(data: bytes) -> int:
    data += bytes(-len(data) % 4)
    return sum(struct.unpack(f">{len(data) // 4}I", data)) & 0xFFFFFFFF


def build_font() -> bytes:
    """Assemble the glyphless font.

    Exposed for the tests; callers that need the bytes want :func:`font_file`,
    which builds them once.
    """
    tables = {
        b"OS/2": _os2(),
        b"cmap": _cmap(),
        b"glyf": _glyf(),
        b"head": _head(),
        b"hhea": _hhea(),
        b"hmtx": _hmtx(),
        b"loca": _loca(),
        b"maxp": _maxp(),
        b"name": _name(),
        b"post": _post(),
    }

    count = len(tables)
    entry_selector = count.bit_length() - 1
    search_range = 16 * (1 << entry_selector)
    directory = b""
    body = b""
    offset = 12 + 16 * count
    head_offset = 0
    for tag in sorted(tables):
        data = tables[tag]
        if tag == b"head":
            head_offset = offset
        directory += struct.pack(">4sIII", tag, _checksum(data), offset, len(data))
        padded = data + bytes(-len(data) % 4)
        body += padded
        offset += len(padded)

    font = struct.pack(">IHHHH", 0x00010000, count, search_range, entry_selector,
                       16 * count - search_range) + directory + body

    # checkSumAdjustment is the eight bytes into head that make the whole file
    # sum to the constant below. Nothing in this project's read path checks it;
    # a stricter validator elsewhere would.
    adjustment = (0xB1B0AFBA - _checksum(font)) & 0xFFFFFFFF
    cursor = head_offset + 8
    return font[:cursor] + struct.pack(">I", adjustment) + font[cursor + 4:]


_font_file: bytes | None = None


def font_file() -> bytes:
    """The assembled font, built once per process and shared by every document."""
    global _font_file
    if _font_file is None:
        _font_file = build_font()
    return _font_file


# ---------------------------------------------------------------------------
# PDF font object
# ---------------------------------------------------------------------------

def _cid_to_gid() -> bytes:
    """Every CID resolves to the one glyph.

    Two bytes per CID for all 65536 of them: 128 KB of the same two bytes,
    which deflates to about 150. This is what makes the font itself two glyphs
    rather than 65535. The alternative, ``/Identity``, would need a distinct
    glyph -- and so a ``loca`` entry -- per code point, and a ``loca`` of
    ascending offsets is the one part of such a font that does not compress
    away: 131 KB even deflated, against 150 bytes for this.
    """
    return struct.pack(">H", _GLYPH) * (_MAX_CODE_POINT + 1)


def _to_unicode() -> bytes:
    """The CMap that turns the codes back into the text they stand for.

    Without it a reader has a document full of Identity-H codes and no way to
    know they are Unicode, so the text extracts as nothing -- which is the
    whole product. The mapping is the identity, but it cannot be stated as one
    range: ``bfrange`` increments the last byte of its destination only, so a
    range wider than 256 codes is undefined. It takes 256 rows, in the batches
    of 100 the specification allows, and deflates to under two kilobytes.
    """
    rows = [
        b"<%04X> <%04X> <%04X>\n" % (hi << 8, (hi << 8) | 0xFF, hi << 8)
        for hi in range(256)
    ]
    batches = b""
    for start in range(0, len(rows), 100):
        batch = rows[start:start + 100]
        batches += b"%d beginbfrange\n" % len(batch) + b"".join(batch) + b"endbfrange\n"
    return (
        b"/CIDInit /ProcSet findresource begin\n"
        b"12 dict begin\nbegincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        b"/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        + batches
        + b"endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    )


def _deflated(pdf: pikepdf.Pdf, data: bytes) -> pikepdf.Object:
    """A stream compressed on the way in rather than at save time."""
    stream = Stream(pdf, b"")
    stream.write(zlib.compress(data, 9), filter=Name.FlateDecode)
    return stream


def embed_font(pdf: pikepdf.Pdf) -> pikepdf.Object:
    """Add the glyphless font to ``pdf`` and return the font object.

    Call this once per document: the return value is an indirect object, so
    every page that references it shares the one embedded copy.
    """
    program = _deflated(pdf, font_file())
    program.Length1 = len(font_file())

    descriptor = pdf.make_indirect(Dictionary(
        Type=Name.FontDescriptor,
        FontName=Name("/GlyphLess"),
        Flags=4,  # symbolic: no standard encoding describes what this covers
        FontBBox=Array([0, _DESCENT, _ADVANCE, _ASCENT]),
        ItalicAngle=0,
        Ascent=_ASCENT,
        Descent=_DESCENT,
        CapHeight=_ASCENT,
        StemV=80,
        FontFile2=pdf.make_indirect(program),
    ))
    descendant = pdf.make_indirect(Dictionary(
        Type=Name.Font,
        Subtype=Name.CIDFontType2,
        BaseFont=Name("/GlyphLess"),
        CIDSystemInfo=Dictionary(
            Registry=String("Adobe"), Ordering=String("Identity"), Supplement=0
        ),
        FontDescriptor=descriptor,
        DW=_ADVANCE,
        CIDToGIDMap=pdf.make_indirect(_deflated(pdf, _cid_to_gid())),
    ))
    return pdf.make_indirect(Dictionary(
        Type=Name.Font,
        Subtype=Name.Type0,
        BaseFont=Name("/GlyphLess"),
        Encoding=Name("/Identity-H"),
        DescendantFonts=Array([descendant]),
        ToUnicode=pdf.make_indirect(Stream(pdf, _to_unicode())),
    ))
