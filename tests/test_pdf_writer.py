"""Tests for PDF writer module."""

import numpy as np
import pytest
from pikepdf import Name

from pdf_refinery.fonts import LINE_ASCENT, LINE_DESCENT, text_length
from pdf_refinery.ocr_engine import OcrResult
from pdf_refinery.pdf_document import Matrix
from pdf_refinery.pdf_writer import has_text, overlay_text_on_page
from tests.helpers import (
    ink,
    opened,
    pdfium_text,
    source_pdf,
    source_pdf_with_text,
    span_bounds,
    spans,
    written,
)

# Where the baseline of a line sits inside its detected box.
BASELINE_RATIO = LINE_ASCENT / (LINE_ASCENT + LINE_DESCENT)


def _overlaid(tmp_path, results, image_size, page_size=(612, 792), name="out.pdf"):
    """Overlay ``results``, write the document out, and reopen it as a reader.

    Everything is checked through a saved file rather than in memory: the
    text layer is a content stream and a font object, and whether those are
    right is only observable once something else parses them.
    """
    doc = opened(source_pdf(width=page_size[0], height=page_size[1]))
    stats = overlay_text_on_page(doc[0], results, *image_size)
    path = tmp_path / name
    return stats, written(doc, path), path


class TestOverlayTextOnPage:
    def test_empty_results(self, tmp_path):
        stats, reader, _ = _overlaid(tmp_path, [], (1224, 1584))
        assert stats.inserted == 0
        assert reader[0].get_text().strip() == ""
        reader.close()

    def test_single_result_coordinates(self, tmp_path):
        # Horizontal text: top-left (100,200), top-right (300,200),
        # bottom-right (300,250), bottom-left (100,250)
        result = OcrResult(
            text="Hello",
            confidence=0.9,
            bbox=[[100, 200], [300, 200], [300, 250], [100, 250]],
        )
        stats, reader, _ = _overlaid(tmp_path, [result], (612, 792))

        assert stats.inserted == 1
        assert span_bounds(reader[0]) == pytest.approx((100, 200, 300, 250), abs=0.5)
        reader.close()

    def test_coordinate_scaling(self, tmp_path):
        # Image is 2x the page size, so the box halves onto the page.
        result = OcrResult(
            text="Scaled",
            confidence=0.9,
            bbox=[[200, 400], [600, 400], [600, 500], [200, 500]],
        )
        stats, reader, _ = _overlaid(tmp_path, [result], (1224, 1584))

        assert stats.inserted == 1
        assert span_bounds(reader[0]) == pytest.approx((100, 200, 300, 250), abs=0.5)
        reader.close()

    def test_the_span_a_reader_gets_back_covers_the_detected_box(self, tmp_path):
        """The point of sizing by declared metrics rather than a constant.

        A viewer highlights, and text extraction reports, the span box -- not
        the glyphs. If that box does not match the box OCR detected, a search
        hit is drawn off the words it found, and at worst over the line above.
        """
        for index, text in enumerate(("Hello world", "한글 검색 테스트", "混用 漢字")):
            box = [[100, 200], [500, 200], [500, 250], [100, 250]]
            _, reader, _ = _overlaid(
                tmp_path,
                [OcrResult(text=text, confidence=0.9, bbox=box)],
                (612, 792),
                name=f"span-{index}.pdf",
            )
            assert len(spans(reader[0])) == 1, text
            _, y0, _, y1 = spans(reader[0])[0]["bbox"]
            assert y0 == pytest.approx(200, abs=0.5), text
            assert y1 == pytest.approx(250, abs=0.5), text
            reader.close()

    def test_the_baseline_sits_an_ascent_below_the_top_of_the_box(self, tmp_path):
        result = OcrResult(
            text="Hello",
            confidence=0.9,
            bbox=[[100, 200], [300, 200], [300, 250], [100, 250]],
        )
        _, reader, _ = _overlaid(tmp_path, [result], (612, 792))

        span = spans(reader[0])[0]
        assert span["origin"][1] == pytest.approx(200 + 50 * BASELINE_RATIO, abs=0.5)
        reader.close()

    def test_tiny_font_skipped(self, tmp_path):
        # A 1px box on an image twice the page size is half a point tall, so
        # the line would be set below 1pt and is dropped instead.
        result = OcrResult(
            text="Tiny",
            confidence=0.9,
            bbox=[[10, 10], [50, 10], [50, 11], [10, 11]],
        )

        stats, reader, _ = _overlaid(tmp_path, [result], (1224, 1584))
        assert stats.inserted == 0
        assert stats.too_small == 1
        assert reader[0].get_text().strip() == ""
        reader.close()

    def test_multiple_results(self, tmp_path, sample_ocr_results):
        stats, reader, _ = _overlaid(tmp_path, sample_ocr_results, (612, 792))
        assert stats.inserted == 2
        reader.close()

    def test_nothing_is_added_to_a_page_with_no_detections(self, tmp_path):
        """A page OCR found nothing on must not gain a font or a stream."""
        doc = opened(source_pdf())
        overlay_text_on_page(doc[0], [], 612, 792)
        reader = written(doc, tmp_path / "empty.pdf")
        assert reader[0].get_fonts() == []
        reader.close()


class TestLineIsFittedToItsBox:
    """The overlay must end where the scanned line ends.

    Left at its natural width, a line of Hangul set at the box's height is
    several times wider than the box. It then runs off the crop box, and text
    extraction returns only the part that stayed on the page -- so the tail of
    every line goes missing while the recogniser looks perfect. Measured on a
    real book scan, this cost 16.4% character error against 1.2% for the same
    text before it was written out.
    """

    LINE = "황재섭 정도면 대충 짐작할 수 있을 터였다 그가 나를 물끄러미 바라"

    def _overlay_full_width_line(self, tmp_path):
        # A line spanning the page, as a book scan produces.
        result = OcrResult(
            text=self.LINE,
            confidence=0.9,
            bbox=[[20, 200], [592, 200], [592, 240], [20, 240]],
        )
        return _overlaid(tmp_path, [result], (612, 792))

    def test_the_whole_line_is_still_extractable(self, tmp_path):
        _, reader, path = self._overlay_full_width_line(tmp_path)
        assert "".join(reader[0].get_text().split()) == "".join(self.LINE.split())
        assert "".join(pdfium_text(path).split()) == "".join(self.LINE.split())
        reader.close()

    def test_the_line_stays_inside_the_page(self, tmp_path):
        _, reader, _ = self._overlay_full_width_line(tmp_path)
        for span in spans(reader[0]):
            assert span["bbox"][2] <= reader[0].rect.x1 + 1
        reader.close()

    def test_natural_width_would_have_overflowed(self):
        """Guards the guard: without fitting there is nothing to fix."""
        # The overlay sets the line at the height of its box, 240 - 200.
        assert text_length(self.LINE, fontsize=40) > 612, (
            "the fixture no longer reproduces the overflow this test is about"
        )


class TestRotatedLinesLandOnTheirBox:
    """A turned line must cover its box, not run off in some other direction.

    The line is written flat and turned by the same matrix that scales it, so
    the horizontal squeeze acts along the line. Turning it afterwards instead
    would make that squeeze act across the line -- barely visible at the
    fraction of a degree a straight scan produces, and completely wrong for a
    page scanned sideways, where the text leaves the page and stops being
    extractable at all.
    """

    TEXT = "황재섭 정도면 대충 짐작할 수 있을 터였다 그가"

    # bbox corners are (top-left, top-right, bottom-right, bottom-left) of the
    # line as the detector sees it, so rotating the line rotates the corners.
    @pytest.mark.parametrize(
        "bbox,expected",
        [
            ([[100, 300], [500, 300], [500, 340], [100, 340]], (100, 300, 500, 340)),
            ([[300, 100], [300, 500], [260, 500], [260, 100]], (260, 100, 300, 500)),
            ([[500, 340], [100, 340], [100, 300], [500, 300]], (100, 300, 500, 340)),
            ([[300, 500], [300, 100], [340, 100], [340, 500]], (300, 100, 340, 500)),
            ([[100, 300], [498, 335], [494, 375], [96, 340]], (96, 300, 498, 376)),
        ],
        ids=["0deg", "90deg", "180deg", "270deg", "skewed"],
    )
    def test_the_line_covers_its_box(self, tmp_path, bbox, expected):
        result = OcrResult(text=self.TEXT, confidence=0.9, bbox=bbox)
        _, reader, _ = _overlaid(tmp_path, [result], (612, 612), page_size=(612, 612))

        assert "".join(reader[0].get_text().split()) == "".join(self.TEXT.split())
        assert span_bounds(reader[0]) == pytest.approx(expected, abs=2.0)
        reader.close()


class TestOverlayTextIsExtractable:
    """A text layer that cannot be extracted is worthless, whatever the OCR said."""

    @pytest.mark.parametrize(
        "text",
        ["한글 검색 테스트", "Hello World", "日本語のテスト", "中文测试", "café"],
    )
    def test_text_survives_the_overlay(self, tmp_path, text):
        result = OcrResult(
            text=text,
            confidence=0.9,
            bbox=[[100, 200], [400, 200], [400, 240], [100, 240]],
        )

        stats, reader, path = _overlaid(tmp_path, [result], (612, 792))
        assert stats.inserted == 1
        assert text in reader[0].get_text()
        assert text in pdfium_text(path)
        reader.close()

    def test_mixed_scripts_across_lines(self, tmp_path):
        texts = ["한글 줄", "English line", "日本語の行"]
        results = [
            OcrResult(
                text=text,
                confidence=0.9,
                bbox=[[10, 20 + i * 60], [400, 20 + i * 60],
                      [400, 60 + i * 60], [10, 60 + i * 60]],
            )
            for i, text in enumerate(texts)
        ]

        stats, reader, path = _overlaid(tmp_path, results, (612, 792))
        assert stats.inserted == 3
        extracted = reader[0].get_text()
        by_pdfium = pdfium_text(path)
        for text in texts:
            assert text in extracted
            assert text in by_pdfium
        reader.close()

    def test_the_font_is_embedded_once_however_many_pages_use_it(self, tmp_path):
        doc = opened(source_pdf(pages=5))
        result = OcrResult(
            text="한글 검색",
            confidence=0.9,
            bbox=[[10, 20], [400, 20], [400, 60], [10, 60]],
        )
        for page in doc:
            overlay_text_on_page(page, [result], 612, 792)

        reader = written(doc, tmp_path / "multipage.pdf")
        embedded = {font[0] for page in reader for font in page.get_fonts(full=True)}
        assert len(embedded) == 1
        reader.close()


class TestOverlayRespectsHowThePageIsShown:
    """A page is not always the rectangle its media box describes.

    ``/Rotate`` turns the page for display and a crop box moves its origin,
    and neither is rare in a scan. The overlay is given boxes in the
    coordinates of the *rendered* page, so a page space that ignored either
    would put every line somewhere else. This checks the whole way round: the
    line is placed by page-space coordinates, the file is written, reopened,
    and rendered, and the mark is looked for where it was asked for.
    """

    @pytest.mark.parametrize("rotation", [0, 90, 180, 270])
    def test_a_turned_page_places_lines_by_what_is_rendered(self, tmp_path, rotation):
        doc = opened(source_pdf(width=400, height=600, rotation=rotation))
        page = doc[0]
        page_width, page_height = page.size
        assert (page_width, page_height) == (
            (600, 400) if rotation in (90, 270) else (400, 600)
        )

        overlay_text_on_page(
            page,
            [OcrResult(text="Hello", confidence=0.9,
                       bbox=[[50, 100], [250, 100], [250, 140], [50, 140]])],
            image_width=int(page_width), image_height=int(page_height),
        )
        path = tmp_path / f"rot{rotation}.pdf"
        written(doc, path).close()

        assert "Hello" in pdfium_text(path)

    @pytest.mark.parametrize("rotation", [0, 90, 180, 270])
    def test_page_space_is_the_rendered_image(self, tmp_path, rotation):
        """The claim :attr:`Page.placement` makes, checked by rendering it.

        A visible rectangle is drawn through ``placement`` at a known place in
        page space; it has to come back at exactly those pixels.
        """
        doc = opened(source_pdf(width=400, height=600, rotation=rotation))
        page = doc[0]
        # The unit square, scaled and moved to page-space (50,100)-(250,140).
        placed = Matrix(200, 0, 0, 40, 50, 100).then(page.placement)
        page.add_content(
            ("q %s cm 0 0 0 rg 0 0 1 1 re f Q\n"
             % " ".join(f"{value:g}" for value in placed)).encode("ascii")
        )
        path = tmp_path / f"placed{rotation}.pdf"
        written(doc, path).close()

        image = opened(path.read_bytes())[0].to_image(dpi=72)
        rows, columns = np.where(image[:, :, 0] < 128)
        assert (columns.min(), columns.max()) == (50, 249)
        assert (rows.min(), rows.max()) == (100, 139)

    def test_a_cropped_page_is_measured_by_its_crop_box(self, tmp_path):
        doc = opened(source_pdf(width=400, height=600, cropbox=(20, 30, 380, 500)))
        page = doc[0]
        assert page.size == pytest.approx((360, 470))
        assert page.to_image(dpi=72).shape[:2] == (470, 360)


class TestHasText:
    def _page(self, lines, **insert):
        return opened(source_pdf_with_text(lines, **insert))[0]

    def test_blank_page(self):
        assert has_text(opened(source_pdf())[0]) is False

    def test_page_of_body_text(self):
        assert has_text(self._page(["Something worth keeping."] * 10)) is True

    def test_invisible_text_counts(self):
        page = self._page(["Hidden but extractable."] * 10, render_mode=3)
        assert has_text(page) is True

    def test_stray_page_number_does_not_count(self):
        # A scanner-stamped page number is the classic reason a whole page of
        # body text ends up skipped and unsearchable.
        assert has_text(self._page(["117"])) is False

    def test_running_header_does_not_count(self):
        assert has_text(self._page(["Chapter 5 - The Beginning of Everything"])) is False

    def test_threshold_is_adjustable(self):
        page = self._page(["117"])
        assert has_text(page, min_chars=0) is True
        assert has_text(page, min_chars=3) is True
        assert has_text(page, min_chars=4) is False

    def test_whitespace_does_not_pad_the_count(self):
        page = self._page([" ".join("x" * 1 for _ in range(60))])
        # 60 characters of content, however much space sits between them.
        assert has_text(page, min_chars=60) is True
        assert has_text(page, min_chars=61) is False


class TestReplacingAPageWithItsRendering:
    """How a page that already has text is made re-readable.

    Erasing the text alone would take the visible glyphs with it, because on
    such a page the text *is* the appearance. Keeping only a rendering of the
    page loses the text layer and nothing else.
    """

    def _page_of_text(self):
        return opened(source_pdf_with_text(["Visible text"], fontsize=24))[0]

    def test_strips_text_but_keeps_appearance(self, tmp_path):
        page = self._page_of_text()
        before = page.to_image(dpi=150)
        assert ink(before) > 0

        page.replace_with_image(before)
        reader = written(page.document, tmp_path / "flat.pdf")

        # The text layer is gone...
        assert reader[0].get_text().strip() == ""
        # ...but the glyphs are still visible, unlike after erasing the text.
        after = opened((tmp_path / "flat.pdf").read_bytes())[0].to_image(dpi=150)
        assert ink(after) == pytest.approx(ink(before), rel=0.2)
        reader.close()

    def test_the_page_keeps_its_size(self, tmp_path):
        page = self._page_of_text()
        before = page.size
        page.replace_with_image(page.to_image(dpi=150))
        assert page.size == before

    @pytest.mark.parametrize("inherited", [False, True], ids=["on-page", "inherited"])
    def test_a_turned_page_is_not_turned_twice(self, tmp_path, inherited):
        """The rendering is already the right way up, so ``/Rotate`` has to go.

        It can be inherited from the page tree rather than set on the page,
        in which case deleting the page's own key would uncover the parent's
        and turn the flattened page anyway.
        """
        doc = opened(source_pdf(width=400, height=600, rotation=90))
        page = doc[0]
        if inherited:
            doc.pdf.Root.Pages[Name.Rotate] = 90
            del page.obj.obj[Name.Rotate]
        assert page.rotation == 90

        before = page.size
        page.replace_with_image(page.to_image(dpi=72))
        assert page.rotation == 0
        assert page.size == before

        path = tmp_path / f"flat-{inherited}.pdf"
        written(doc, path).close()
        assert opened(path.read_bytes())[0].size == before
