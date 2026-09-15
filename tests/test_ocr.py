"""Tests for gst_einvoice.ocr: binary resolution, preprocessing, word offsets and span confidence.

The offset invariant (``page.text[w.start:w.end] == w.text`` for every word) is the
point of the module and is asserted over a real OCR run, not a synthetic one.
Expectations that pin text are transcribed from the fixture by hand.
"""

import io
import os
import shutil

import pymupdf
import pytest
from PIL import Image, ImageDraw

from gst_einvoice import ocr
from gst_einvoice.ingest import ingest
from gst_einvoice.ocr import (
    OcrPage,
    OcrWord,
    confidence_for_span,
    deskew,
    normalize_contrast,
    ocr_image,
    ocr_pdf_page,
    upscale_low_dpi,
)


def _tesseract_present() -> bool:
    return shutil.which("tesseract") is not None or os.path.exists(ocr.TESSERACT_FALLBACK)


requires_tesseract = pytest.mark.skipif(
    not _tesseract_present(),
    reason=(
        "the tesseract executable was not found on PATH or at "
        f"{ocr.TESSERACT_FALLBACK}; install it with "
        '"winget install UB-Mannheim.TesseractOCR"'
    ),
)

# Rendered into every page fixture. Short, plain and large enough that Tesseract
# reads it exactly, so the expected text below can be transcribed by hand.
INVOICE_TEXT = (
    "TAX INVOICE\n"
    "Acme Industries Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Invoice No INV-2026-0042\n"
    "HSN 8471 Laptop Stand\n"
)

# Hand-transcribed from INVOICE_TEXT: the same five lines, trailing newline dropped
# because OCR text is assembled from words and ends at the last word.
EXPECTED_OCR_TEXT = (
    "TAX INVOICE\n"
    "Acme Industries Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Invoice No INV-2026-0042\n"
    "HSN 8471 Laptop Stand"
)

SECOND_PAGE_TEXT = "CONTINUATION SHEET\nHSN 9954 Installation Service\n"


def _render(text: str = INVOICE_TEXT, *, dpi: int = 300, fontsize: int = 13) -> Image.Image:
    """Rasterise one text page to a PIL image, the way a scan of it would look."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=fontsize)
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    doc.close()
    image = Image.open(io.BytesIO(png))
    image.load()
    return image


def _png_bytes(text: str, *, dpi: int = 150) -> bytes:
    """Render a page of text to PNG using a throwaway document (build 1's helper pattern)."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=13)
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    doc.close()
    return png


def _build_scanned_pdf(path, *texts) -> str:
    """A PDF whose pages are full-page images of rendered text, so they carry no text layer."""
    doc = pymupdf.open()
    for text in texts:
        page = doc.new_page()
        page.insert_image(page.rect, stream=_png_bytes(text))
    doc.save(str(path))
    doc.close()
    return str(path)


def _bars(width: int = 600, height: int = 400, mode: str = "RGB") -> Image.Image:
    """A stand-in for a page of text: eight horizontal black bars on white."""
    img = Image.new(mode, (width, height), "white")
    draw = ImageDraw.Draw(img)
    for i in range(8):
        top = 40 + i * 40
        draw.rectangle([80, top, width - 80, top + 8], fill="black")
    return img


@pytest.fixture(scope="module")
def clean_page() -> OcrPage:
    """One real OCR run at 300 dpi, shared by the invariant tests."""
    return ocr_image(_render(), page=1)


# --- fixture sanity -----------------------------------------------------------------


def test_scanned_fixture_pages_are_the_ones_build_one_routes_to_ocr(tmp_path):
    """The image-only fixture really has no text layer, so OCR is the only way in."""
    path = _build_scanned_pdf(tmp_path / "scan.pdf", INVOICE_TEXT, SECOND_PAGE_TEXT)
    result = ingest(path)
    assert [page.method for page in result.pages] == ["ocr_required", "ocr_required"]
    assert [page.text for page in result.pages] == ["", ""]


# --- _resolve_tesseract -------------------------------------------------------------


def test_resolve_tesseract_prefers_path_over_the_windows_default(monkeypatch):
    monkeypatch.setattr(ocr, "_TESSERACT_CMD", None)
    monkeypatch.setattr(ocr.shutil, "which", lambda name: "/usr/bin/tesseract")
    assert ocr._resolve_tesseract() == "/usr/bin/tesseract"


def test_resolve_tesseract_falls_back_to_the_default_install(monkeypatch, tmp_path):
    fallback = tmp_path / "tesseract.exe"
    fallback.write_bytes(b"")
    monkeypatch.setattr(ocr, "_TESSERACT_CMD", None)
    monkeypatch.setattr(ocr, "TESSERACT_FALLBACK", str(fallback))
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    assert ocr._resolve_tesseract() == str(fallback)


def test_resolve_tesseract_missing_names_both_places_and_how_to_install(monkeypatch, tmp_path):
    missing = tmp_path / "nowhere" / "tesseract.exe"
    monkeypatch.setattr(ocr, "_TESSERACT_CMD", None)
    monkeypatch.setattr(ocr, "TESSERACT_FALLBACK", str(missing))
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        ocr._resolve_tesseract()
    message = str(excinfo.value)
    assert "PATH" in message
    assert str(missing) in message
    assert "winget install UB-Mannheim.TesseractOCR" in message


def test_resolve_tesseract_looks_only_once(monkeypatch):
    calls: list[str] = []

    def fake_which(name):
        calls.append(name)
        return "/usr/bin/tesseract"

    monkeypatch.setattr(ocr, "_TESSERACT_CMD", None)
    monkeypatch.setattr(ocr.shutil, "which", fake_which)
    assert ocr._resolve_tesseract() == "/usr/bin/tesseract"
    assert ocr._resolve_tesseract() == "/usr/bin/tesseract"
    assert calls == ["tesseract"]


# --- upscale_low_dpi ----------------------------------------------------------------


def test_upscale_leaves_an_image_at_min_width_untouched():
    img = Image.new("RGB", (1700, 900), "white")
    out, note = upscale_low_dpi(img)
    assert out is img
    assert note is None


def test_upscale_never_downscales_a_high_resolution_scan():
    img = Image.new("RGB", (2480, 3508), "white")
    out, note = upscale_low_dpi(img)
    assert out.size == (2480, 3508)
    assert note is None


def test_upscale_reaches_min_width_and_reports_the_factor():
    img = Image.new("RGB", (1000, 700), "white")
    out, note = upscale_low_dpi(img)
    assert out.size == (1700, 1190)  # 1.7x, aspect ratio preserved
    assert note == "upscaled 1.70x"


def test_upscale_factor_is_capped_at_four():
    img = Image.new("RGB", (300, 200), "white")
    out, note = upscale_low_dpi(img)
    assert out.size == (1200, 800)  # 4.0x, not the 5.67x that would reach 1700
    assert out.width < 1700
    assert note == "upscaled 4.00x"


def test_upscale_honours_a_custom_min_width():
    img = Image.new("RGB", (400, 400), "white")
    assert upscale_low_dpi(img, min_width=800)[0].size == (800, 800)
    assert upscale_low_dpi(img, min_width=400)[1] is None


def test_upscale_note_does_not_round_a_real_resize_away():
    """A 200-dpi A4 scan is 1653px and really is enlarged; the note must not read as 1x.

    1700 / 1653 = 1.0284, which one decimal renders as "upscaled 1.0x" -- a record
    that contradicts what happened to the image the confidences were measured on.
    """
    img = Image.new("RGB", (1653, 2339), "white")
    out, note = upscale_low_dpi(img)
    assert out.width == 1700
    assert note == "upscaled 1.03x"


def test_upscale_skips_a_resize_it_could_only_record_as_one_times():
    """A hair under min_width, the record would read "upscaled 1.00x" -- so nothing is done.

    1700 / 1699 = 1.0006. Resampling and then recording it as 1.00x would say "no
    resize" about an image that was resampled, which is the same dishonest record
    the two-decimal note exists to prevent, one decimal place down.
    """
    img = Image.new("RGB", (1699, 2200), "white")
    out, note = upscale_low_dpi(img)
    assert out is img
    assert out.width == 1699  # left below min_width rather than misreported
    assert note is None


def test_upscale_still_resizes_just_under_that_boundary():
    """The skip is only the 1.00x band: 1684px is 1.01x and is really enlarged."""
    img = Image.new("RGB", (1684, 2200), "white")
    out, note = upscale_low_dpi(img)
    assert out is not img
    assert out.width == 1700
    assert note == "upscaled 1.01x"


# --- deskew -------------------------------------------------------------------------


def test_deskew_leaves_a_straight_page_unchanged():
    img = _bars()
    out, note = deskew(img)
    assert out is img
    assert note is None


@pytest.mark.parametrize("dpi", [200, 240, 250, 260, 300])
def test_deskew_leaves_a_straight_rendered_page_unchanged_at_every_width(dpi):
    """A page that was never skewed must record no skew, whatever its resolution.

    The synthetic bars above are one size; a real render at 250 dpi (2066 px) is the
    case that caught a skew search run on a downscaled proxy, which aliased the text
    rows and reported -0.25deg on a page that is perfectly straight.
    """
    img = _render(dpi=dpi)
    out, note = deskew(img)
    assert note is None
    assert out is img


def test_deskew_leaves_a_blank_page_unchanged():
    img = Image.new("RGB", (600, 400), "white")
    out, note = deskew(img)
    assert out is img
    assert note is None


@pytest.mark.parametrize(
    ("applied", "expected_note"),
    [(2.0, "deskewed -2.00deg"), (-3.0, "deskewed 3.00deg"), (4.5, "deskewed -4.50deg")],
)
def test_deskew_finds_the_rotation_that_undoes_a_known_skew(applied, expected_note):
    skewed = _bars().rotate(applied, expand=True, fillcolor="white")
    _, note = deskew(skewed)
    assert note == expected_note


def test_deskew_ignores_a_skew_smaller_than_one_step():
    skewed = _bars().rotate(0.1, expand=True, fillcolor="white")
    out, note = deskew(skewed)
    assert out is skewed
    assert note is None


def test_deskew_never_rotates_beyond_max_angle():
    skewed = _bars().rotate(4.0, expand=True, fillcolor="white")
    assert deskew(skewed, max_angle=2.0)[1] == "deskewed -2.00deg"


def test_deskew_expands_the_canvas_and_fills_the_corners_white():
    skewed = _bars(mode="L").rotate(3.0, expand=True, fillcolor=255)
    out, note = deskew(skewed)
    assert note == "deskewed -3.00deg"
    assert out.size > skewed.size
    assert out.getpixel((0, 0)) == 255
    assert out.getpixel((out.width - 1, 0)) == 255


# --- normalize_contrast -------------------------------------------------------------


def test_normalize_contrast_converts_colour_to_grayscale():
    img = Image.new("RGB", (10, 10), (255, 255, 255))
    img.putpixel((0, 0), (0, 0, 0))
    out, note = normalize_contrast(img)
    assert out.mode == "L"
    assert note == "contrast normalised"


def test_normalize_contrast_stretches_a_flat_histogram():
    img = Image.new("L", (10, 10), 200)
    img.putpixel((0, 0), 100)
    out, note = normalize_contrast(img)
    assert note == "contrast normalised"
    # The 100..200 band is stretched over the full range (PIL's LUT truncates the top).
    assert out.getpixel((0, 0)) == 0
    assert out.getpixel((9, 9)) >= 250


def test_normalize_contrast_is_not_recorded_when_it_changes_nothing():
    img = Image.new("L", (10, 10), 255)
    img.putpixel((0, 0), 0)
    out, note = normalize_contrast(img)
    assert out is img
    assert note is None


# --- ocr_image: the offset invariant ------------------------------------------------


@requires_tesseract
def test_every_word_offset_indexes_exactly_into_the_page_text(clean_page):
    assert clean_page.words
    for word in clean_page.words:
        assert clean_page.text[word.start : word.end] == word.text


@requires_tesseract
def test_page_text_is_words_joined_by_single_spaces_and_lines_by_newlines(clean_page):
    lines: dict[int, list[str]] = {}
    for word in clean_page.words:
        lines.setdefault(word.line, []).append(word.text)
    rebuilt = "\n".join(" ".join(lines[index]) for index in sorted(lines))
    assert rebuilt == clean_page.text
    assert clean_page.text.count("\n") == len(lines) - 1


@requires_tesseract
def test_page_text_carries_no_padding_tabs_or_double_spaces(clean_page):
    assert "  " not in clean_page.text
    assert "\t" not in clean_page.text
    assert clean_page.text == clean_page.text.strip()
    assert " \n" not in clean_page.text and "\n " not in clean_page.text


@requires_tesseract
def test_line_indices_are_zero_based_contiguous_and_in_reading_order(clean_page):
    indices = [word.line for word in clean_page.words]
    assert indices == sorted(indices)
    assert sorted(set(indices)) == list(range(len(set(indices))))
    assert indices[0] == 0


@requires_tesseract
def test_clean_render_is_read_exactly(clean_page):
    assert clean_page.text == EXPECTED_OCR_TEXT
    assert clean_page.page == 1


@requires_tesseract
def test_confidences_are_fractions_and_the_mean_is_their_mean(clean_page):
    assert all(0.0 <= word.confidence <= 1.0 for word in clean_page.words)
    expected = sum(word.confidence for word in clean_page.words) / len(clean_page.words)
    assert clean_page.mean_confidence == pytest.approx(expected)
    assert clean_page.mean_confidence > 0.5


# --- ocr_image: the confidences themselves ------------------------------------------

# A hand-written stand-in for pytesseract's TSV dict. The conf values are transcribed
# here and asserted below as fractions, so a regression that reads the wrong column,
# applies a floor, or constants the confidence cannot pass. Two rows exist only to
# pin the drop rule one guard at a time: real Tesseract layout rows are blank AND
# conf -1 at once, so each guard would otherwise mask the other.
#   - "LEGACY" carries conf -1 with non-blank text  -> only the conf guard drops it.
#   - the whitespace row carries conf 99            -> only the blank guard drops it.
# The final row pins the third component of the line key: "PANEL" sits in block 2 with
# the same par_num and line_num as the row before it, because Tesseract numbers lines
# within a block and a second column on the page restarts that numbering. Every
# rendered fixture here is one block, so without this row nothing would notice
# block_num dropping out of the key.
FAKE_TSV = {
    "level": [1, 2, 3, 5, 5, 4, 5, 5, 5, 5],
    "text": ["", "", "", "TAX", "INVOICE", "LEGACY", "GSTIN", "27AAPFU0939F1ZV", "   ", "PANEL"],
    "conf": [-1, -1, -1, 96, 92, -1, 88, 61, 99, 77],
    "block_num": [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    "par_num": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "line_num": [0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    "word_num": [0, 0, 0, 1, 2, 0, 1, 2, 3, 1],
}


@pytest.fixture
def fake_tsv_page(monkeypatch) -> OcrPage:
    """One OCR run against FAKE_TSV -- no Tesseract involved, so the numbers are known."""
    monkeypatch.setattr(ocr, "_TESSERACT_CMD", "tesseract")  # skip the binary lookup
    monkeypatch.setattr(ocr.pytesseract, "image_to_data", lambda *args, **kwargs: FAKE_TSV)
    return ocr_image(Image.new("L", (1700, 2200), "white"), page=3)


def test_word_confidences_are_the_tesseract_values_divided_by_one_hundred(fake_tsv_page):
    """conf 96/92/88/61/77 -> 0.96/0.92/0.88/0.61/0.77, in TSV order, each its own value."""
    assert [word.text for word in fake_tsv_page.words] == [
        "TAX",
        "INVOICE",
        "GSTIN",
        "27AAPFU0939F1ZV",
        "PANEL",
    ]
    assert [word.confidence for word in fake_tsv_page.words] == [0.96, 0.92, 0.88, 0.61, 0.77]
    # 4.14 / 5, worked out by hand from the five conf values above.
    assert fake_tsv_page.mean_confidence == pytest.approx(0.828)
    assert fake_tsv_page.page == 3


def test_a_layout_row_is_dropped_even_when_it_carries_text(fake_tsv_page):
    """conf == -1 means "not a word", whatever the text column says."""
    assert "LEGACY" not in fake_tsv_page.text
    assert all(word.text != "LEGACY" for word in fake_tsv_page.words)


def test_a_blank_row_is_dropped_even_when_its_confidence_is_high(fake_tsv_page):
    """A whitespace-only row at conf 99 must leave no empty word and no stray space."""
    assert all(word.text.strip() for word in fake_tsv_page.words)
    # Transcribed by hand from FAKE_TSV: three kept lines, the blank row leaving no trace.
    assert fake_tsv_page.text == "TAX INVOICE\nGSTIN 27AAPFU0939F1ZV\nPANEL"
    assert "  " not in fake_tsv_page.text
    assert not fake_tsv_page.text.endswith(" ")


def test_dropped_rows_do_not_disturb_the_offsets(fake_tsv_page):
    for word in fake_tsv_page.words:
        assert fake_tsv_page.text[word.start : word.end] == word.text
    # Counted by hand over the text above: "TAX INVOICE\n" is 12 chars, "GSTIN " 6 more,
    # the 15-char GSTIN ends at 33 where the second newline sits.
    assert fake_tsv_page.words[3].start == 18
    assert fake_tsv_page.words[-1].start == 34


def test_a_new_block_starts_a_new_line_even_when_the_line_number_repeats(fake_tsv_page):
    """Tesseract numbers lines within a block, so block_num belongs in the line key.

    "PANEL" carries the same par_num and line_num as the GSTIN before it and differs
    only in block_num -- the shape of a two-panel invoice, where the buyer's column is
    a second block that restarts the numbering. Keying on (par, line) alone would
    splice the buyer's panel onto the seller's line and the spans would follow.
    """
    panel = fake_tsv_page.words[-1]
    assert panel.text == "PANEL"
    assert panel.line == 2
    assert [word.line for word in fake_tsv_page.words] == [0, 0, 1, 1, 2]
    assert fake_tsv_page.text.endswith("27AAPFU0939F1ZV\nPANEL")


@requires_tesseract
def test_a_badly_read_word_stays_low_while_the_page_mean_hides_it(clean_page):
    """The case the whole part exists for: one bad field on an otherwise clean page.

    At 50 dpi the GSTIN is misread; it is not silently dropped, it is reported at a
    confidence that says so, while the page mean stays high enough to look fine.
    """
    degraded = ocr_image(_render(dpi=50))
    assert degraded.text != EXPECTED_OCR_TEXT
    assert "27AAPFU0939F1ZV" not in degraded.text  # misread, and honestly so

    worst_degraded = min(word.confidence for word in degraded.words)
    worst_clean = min(word.confidence for word in clean_page.words)
    assert worst_degraded < 0.6 < worst_clean
    assert degraded.mean_confidence > 0.8  # a page-level number would have hidden it


@requires_tesseract
def test_a_clean_high_resolution_page_records_no_upscale_or_deskew(clean_page):
    assert not any(note.startswith("upscaled") for note in clean_page.preprocessing)
    assert not any(note.startswith("deskewed") for note in clean_page.preprocessing)
    assert "contrast normalised" in clean_page.preprocessing


@requires_tesseract
def test_a_blank_page_yields_no_words_and_no_mean_confidence():
    page = ocr_image(Image.new("RGB", (2480, 3508), "white"))
    assert page.text == ""
    assert page.words == ()
    assert page.mean_confidence is None


@requires_tesseract
def test_page_number_is_carried_through():
    page = ocr_image(_render(dpi=150), page=7)
    assert page.page == 7


@requires_tesseract
def test_a_low_dpi_capture_is_upscaled_and_still_read():
    page = ocr_image(_render(dpi=72))
    assert page.preprocessing[0] == "upscaled 2.86x"  # 1700 / 595 px, two decimals
    assert "27AAPFU0939F1ZV" in page.text


@requires_tesseract
def test_a_skewed_scan_is_deskewed_before_reading():
    skewed = _render(dpi=200).rotate(1.5, expand=True, fillcolor="white")
    page = ocr_image(skewed)
    assert "deskewed -1.50deg" in page.preprocessing
    assert page.text == EXPECTED_OCR_TEXT


@requires_tesseract
def test_preprocessing_records_the_steps_in_order():
    page = ocr_image(_render(dpi=72).rotate(2.0, expand=True, fillcolor="white"))
    assert page.preprocessing[0].startswith("upscaled")
    assert page.preprocessing[1] == "deskewed -2.00deg"
    assert page.preprocessing[2] == "contrast normalised"


# --- ocr_pdf_page -------------------------------------------------------------------


@requires_tesseract
def test_ocr_pdf_page_reads_the_requested_one_based_page(tmp_path):
    path = _build_scanned_pdf(tmp_path / "scan.pdf", INVOICE_TEXT, SECOND_PAGE_TEXT)
    first = ocr_pdf_page(path, 1, dpi=200)
    second = ocr_pdf_page(path, 2, dpi=200)
    assert first.page == 1
    assert "27AAPFU0939F1ZV" in first.text
    assert second.page == 2
    assert "CONTINUATION SHEET" in second.text
    assert "9954" in second.text


@requires_tesseract
def test_ocr_pdf_page_offsets_index_into_its_own_text(tmp_path):
    path = _build_scanned_pdf(tmp_path / "scan.pdf", INVOICE_TEXT)
    page = ocr_pdf_page(path, 1, dpi=200)
    for word in page.words:
        assert page.text[word.start : word.end] == word.text


@pytest.mark.parametrize("page_number", [0, -1, 2, 99])
def test_ocr_pdf_page_rejects_a_page_outside_the_document(tmp_path, page_number):
    path = _build_scanned_pdf(tmp_path / "scan.pdf", INVOICE_TEXT)
    with pytest.raises(IndexError) as excinfo:
        ocr_pdf_page(path, page_number)
    assert "1-based" in str(excinfo.value)


@requires_tesseract
def test_ocr_pdf_page_reads_a_plain_image_file(tmp_path):
    png_path = tmp_path / "scan.png"
    _render(dpi=200).save(png_path)
    page = ocr_pdf_page(png_path, 1, dpi=200)
    assert "27AAPFU0939F1ZV" in page.text


@requires_tesseract
def test_ocr_pdf_page_dpi_controls_the_raster_and_therefore_the_upscale(tmp_path):
    path = _build_scanned_pdf(tmp_path / "scan.pdf", INVOICE_TEXT)
    low = ocr_pdf_page(path, 1, dpi=72)
    high = ocr_pdf_page(path, 1, dpi=300)
    assert any(note.startswith("upscaled") for note in low.preprocessing)
    assert not any(note.startswith("upscaled") for note in high.preprocessing)


# --- confidence_for_span ------------------------------------------------------------

# Built by hand so the offsets and the expected minimum are transcribed, not computed.
SPAN_TEXT = "GSTIN 27AAPFU0939F1ZV\nInvoice INV-42"
SPAN_PAGE = OcrPage(
    page=1,
    text=SPAN_TEXT,
    words=(
        OcrWord(text="GSTIN", confidence=0.95, start=0, end=5, line=0),
        OcrWord(text="27AAPFU0939F1ZV", confidence=0.62, start=6, end=21, line=0),
        OcrWord(text="Invoice", confidence=0.88, start=22, end=29, line=1),
        OcrWord(text="INV-42", confidence=0.91, start=30, end=36, line=1),
    ),
    mean_confidence=0.84,
    preprocessing=(),
)


def test_span_fixture_offsets_are_transcribed_correctly():
    for word in SPAN_PAGE.words:
        assert SPAN_PAGE.text[word.start : word.end] == word.text
    assert len(SPAN_TEXT) == 36


def test_confidence_for_span_returns_the_word_covering_the_span():
    assert confidence_for_span(SPAN_PAGE, 6, 21) == 0.62
    assert confidence_for_span(SPAN_PAGE, 0, 5) == 0.95


def test_confidence_for_span_is_the_minimum_not_the_mean():
    # Mean of 0.95 and 0.62 is 0.785; the worst read is what must survive.
    assert confidence_for_span(SPAN_PAGE, 0, 21) == 0.62
    assert confidence_for_span(SPAN_PAGE, 0, 36) == 0.62


def test_confidence_for_span_spans_lines():
    assert confidence_for_span(SPAN_PAGE, 22, 36) == 0.88


def test_confidence_for_span_counts_a_partial_overlap():
    assert confidence_for_span(SPAN_PAGE, 3, 9) == 0.62  # tail of "GSTIN" plus head of the GSTIN
    assert confidence_for_span(SPAN_PAGE, 20, 24) == 0.62


def test_confidence_for_span_returns_none_when_nothing_overlaps():
    assert confidence_for_span(SPAN_PAGE, 36, 40) is None
    assert confidence_for_span(SPAN_PAGE, 5, 6) is None  # the separating space only
    assert confidence_for_span(SPAN_PAGE, 21, 22) is None  # the line break only


def test_confidence_for_span_returns_none_for_an_empty_span():
    assert confidence_for_span(SPAN_PAGE, 10, 10) is None
    assert confidence_for_span(SPAN_PAGE, 10, 4) is None


def test_confidence_for_span_returns_none_on_a_page_with_no_words():
    empty = OcrPage(page=1, text="", words=(), mean_confidence=None, preprocessing=())
    assert confidence_for_span(empty, 0, 10) is None
