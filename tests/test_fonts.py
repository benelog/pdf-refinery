"""Tests for font selection of the invisible text layer."""

import fitz
import pytest

from pdf_refinery.fonts import (
    BUILTIN_CHINESE,
    BUILTIN_JAPANESE,
    BUILTIN_KOREAN,
    BUILTIN_LATIN,
    FontResolver,
    FontSpec,
    check_font_file,
    detect_script,
    unsupported_chars,
)


class TestDetectScript:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("한글 테스트", "hangul"),
            ("ㄱㄴㄷ", "hangul"),
            ("日本語です", "kana"),
            ("カタカナ", "kana"),
            ("中文测试", "han"),
            ("漢字", "han"),
            ("Hello World", "latin"),
            ("café naïve", "latin"),
            ("12345", "latin"),
            ("", "latin"),
        ],
    )
    def test_detect_script(self, text, expected):
        assert detect_script(text) == expected

    def test_hangul_wins_over_han(self):
        # Korean texts mix Hanja; the Hangul is the reliable language signal.
        assert detect_script("한글과 漢字") == "hangul"

    def test_kana_wins_over_han(self):
        # Han alone is ambiguous, kana pins the text to Japanese.
        assert detect_script("漢字と仮名") == "kana"


class TestUnsupportedChars:
    def test_latin_font_rejects_hangul(self):
        assert unsupported_chars("한글", FontSpec(BUILTIN_LATIN)) == {"한", "글"}

    def test_korean_font_accepts_hangul(self):
        assert unsupported_chars("한글 검색", FontSpec(BUILTIN_KOREAN)) == set()

    def test_whitespace_is_never_reported(self):
        assert unsupported_chars("  \t\n", FontSpec(BUILTIN_LATIN)) == set()

    def test_ascii_supported_everywhere(self):
        for name in (BUILTIN_LATIN, BUILTIN_KOREAN, BUILTIN_JAPANESE, BUILTIN_CHINESE):
            assert unsupported_chars("Hello 123", FontSpec(name)) == set()


class TestFontResolver:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("한글", BUILTIN_KOREAN),
            ("カタカナ", BUILTIN_JAPANESE),
            ("中文", BUILTIN_CHINESE),
            ("Hello", BUILTIN_LATIN),
        ],
    )
    def test_resolves_by_script(self, text, expected):
        assert FontResolver().resolve(text).name == expected

    def test_explicit_font_file_wins_when_it_encodes(self, latin_font_file):
        resolver = FontResolver(font_file=latin_font_file)
        assert resolver.resolve("Hello").file == latin_font_file
        assert resolver.uses_embedded_font
        assert resolver.fallback_lines == 0

    def test_builtin_is_not_embedded(self):
        resolver = FontResolver()
        assert resolver.resolve("Hello").is_embedded is False
        assert resolver.uses_embedded_font is False

    def test_records_dropped_characters(self):
        resolver = FontResolver()
        # Thai has no coverage in any built-in font.
        resolver.resolve("สวัสดี")
        assert resolver.dropped_chars

    def test_no_drops_for_supported_scripts(self):
        resolver = FontResolver()
        for text in ("한글 검색 테스트", "Hello World", "日本語のテスト", "中文测试"):
            resolver.resolve(text)
        assert resolver.dropped_chars == set()


class TestFallbackFromUnsuitableFontFile:
    """A supplied font must not silently cost the user their text layer."""

    def test_falls_back_to_builtin_for_hangul(self, latin_font_file):
        resolver = FontResolver(font_file=latin_font_file)
        spec = resolver.resolve("한글 검색")

        assert spec.name == BUILTIN_KOREAN
        assert spec.file is None
        assert resolver.dropped_chars == set()
        assert resolver.fallback_lines == 1

    def test_unreadable_font_file_never_blocks_the_text_layer(self, tmp_path):
        broken = tmp_path / "broken.ttf"
        broken.write_bytes(b"not a font")
        resolver = FontResolver(font_file=str(broken))

        assert resolver.resolve("Hello").name == BUILTIN_LATIN
        assert resolver.uses_embedded_font is False

    def test_mixed_line_picks_the_font_dropping_least(self, latin_font_file):
        # Neither candidate covers the whole line; korea drops only 'é'.
        resolver = FontResolver(font_file=latin_font_file)
        assert resolver.resolve("한글 café").name == BUILTIN_KOREAN

    def test_builtin_only_still_falls_back_to_latin(self):
        # Latin-1 coverage is 32% in the built-in CJK fonts, 97% in helv.
        resolver = FontResolver()
        assert resolver.resolve("한글").name == BUILTIN_KOREAN
        assert resolver.dropped_chars == set()


class TestCheckFontFile:
    def test_rejects_font_collections(self, tmp_path):
        collection = tmp_path / "NotoSansCJK-Regular.ttc"
        collection.write_bytes(b"")
        with pytest.raises(ValueError, match="collection"):
            check_font_file(str(collection))

    def test_rejects_unknown_suffix(self, tmp_path):
        other = tmp_path / "font.woff2"
        other.write_bytes(b"")
        with pytest.raises(ValueError, match=r"\.ttf or \.otf"):
            check_font_file(str(other))

    def test_rejects_unreadable_file(self, tmp_path):
        broken = tmp_path / "broken.ttf"
        broken.write_bytes(b"not a font")
        with pytest.raises(ValueError, match="could not be read"):
            check_font_file(str(broken))

    def test_accepts_a_real_single_face_font(self, latin_font_file):
        check_font_file(latin_font_file)


class TestSelectedFontsActuallyEncode:
    """The whole point of font selection: text must survive a round trip."""

    @pytest.mark.parametrize(
        "text",
        ["한글 검색 테스트", "Hello World", "日本語のテスト", "中文测试", "café"],
    )
    def test_roundtrip_through_pdf(self, text):
        spec = FontResolver().resolve(text)
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            fitz.Point(50, 100), text, fontsize=12,
            fontname=spec.name, fontfile=spec.file, render_mode=3,
        )
        assert text in page.get_text()
        doc.close()
