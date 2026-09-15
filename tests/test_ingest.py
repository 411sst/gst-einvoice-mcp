"""Tests for gst_einvoice.ingest: per-page routing, detectors, and end-to-end ingestion."""

import pymupdf
import pytest

from gst_einvoice.ingest import (
    ACCEPTED_EXTENSIONS,
    IngestResult,
    PageResult,
    Refusal,
    detect_export_or_sez,
    detect_multi_currency,
    detect_reverse_charge,
    ingest,
    route_page,
)

CLEAN_TEXT = (
    "TAX INVOICE\n"
    "Acme Industries Pvt Ltd\n"
    "GSTIN: 27AAPFU0939F1ZV\n"
    "Invoice No: INV-2026-0042   Date: 01-04-2026\n"
    "Buyer: Beta Traders, GSTIN 29AABCT1332L1ZU\n"
    "HSN 8471 Laptop 2 nos Rs. 1,00,000.00\n"
    "CGST 9% Rs. 9,000.00   SGST 9% Rs. 9,000.00\n"
    "Total Rs. 1,18,000.00\n"
    "Whether tax is payable on reverse charge basis: No\n"
)

# Passes the length gate (>= 100 characters) but carries none of the keywords.
PARTIAL_TEXT = (
    "Page 2 of 3 - continuation sheet - Acme Industries Pvt Ltd,\n"
    "Plot 14, MIDC Andheri East, Mumbai 400093, Tel 022-12345678\n"
)


def _png_bytes() -> bytes:
    """Render a page of text to PNG using a throwaway document."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), CLEAN_TEXT, fontsize=11)
    return page.get_pixmap(dpi=100).tobytes("png")


def _add_text_page(doc: pymupdf.Document, text: str) -> None:
    doc.new_page().insert_text((50, 72), text, fontsize=11)


def _add_image_page(doc: pymupdf.Document, png: bytes, header: str | None = None) -> None:
    page = doc.new_page()
    page.insert_image(page.rect, stream=png)
    if header is not None:
        page.insert_text((50, 30), header, fontsize=8)


def _build_pdf(path, *specs) -> str:
    """Each spec is ("text", str), ("image", None) or ("partial", header_text)."""
    png = _png_bytes()
    doc = pymupdf.open()
    for kind, payload in specs:
        if kind == "text":
            _add_text_page(doc, payload)
        elif kind == "image":
            _add_image_page(doc, png)
        else:
            _add_image_page(doc, png, header=payload)
    doc.save(str(path))
    doc.close()
    return str(path)


# --- fixture sanity ---------------------------------------------------------------


def test_fixture_texts_satisfy_their_intended_gates():
    assert len(CLEAN_TEXT.strip()) >= 100
    assert len(PARTIAL_TEXT.strip()) >= 100
    assert route_page(CLEAN_TEXT) == "native"
    assert route_page(PARTIAL_TEXT) == "ocr_required"


# --- route_page ---------------------------------------------------------------------


def test_route_page_short_text_with_keywords_is_ocr_required():
    text = "TAX INVOICE GSTIN 27AAPFU0939F1ZV HSN 8471 CGST SGST IGST"
    assert len(text.strip()) < 100
    assert route_page(text) == "ocr_required"


def test_route_page_empty_is_ocr_required():
    assert route_page("") == "ocr_required"
    assert route_page("   \n\n  ") == "ocr_required"


def test_route_page_whitespace_padding_does_not_count_toward_length():
    text = "GSTIN " + " " * 200
    assert route_page(text) == "ocr_required"


def test_route_page_long_text_without_keywords_is_ocr_required():
    assert route_page(PARTIAL_TEXT) == "ocr_required"


@pytest.mark.parametrize("keyword", ["GSTIN", "gstin", "HSN", "SGST", "CGST", "IGST", "Tax Invoice"])
def test_route_page_each_keyword_passes_gate_two(keyword):
    text = PARTIAL_TEXT + f"\n{keyword}: value"
    assert route_page(text) == "native"


def test_route_page_tax_invoice_split_across_newline_passes():
    assert route_page("TAX\nINVOICE\n" + PARTIAL_TEXT) == "native"
    assert route_page("TAX \t \nINVOICE\n" + PARTIAL_TEXT) == "native"


def test_route_page_keywords_must_be_whole_words():
    text = PARTIAL_TEXT + "\nHSNCODE BIGSTINK CGSTX taxinvoice TAX-INVOICING"
    assert route_page(text) == "ocr_required"


def test_route_page_keyword_adjacent_to_punctuation_still_counts():
    assert route_page(PARTIAL_TEXT + "\nGSTIN/UIN:27AAPFU0939F1ZV") == "native"
    assert route_page(PARTIAL_TEXT + "\nCGST@9%") == "native"


# --- export / SEZ detector ---------------------------------------------------------


@pytest.mark.parametrize(
    "snippet",
    [
        "SUPPLY MEANT FOR EXPORT ON PAYMENT OF INTEGRATED TAX",
        "Supply meant for export",
        "Export Invoice No. EXP/2026/17",
        "Bill of Export No. 4471 dated 03-04-2026",
        "with payment of IGST",
        "With Payment of Integrated Tax",
        "without payment of IGST",
        "Without payment of integrated tax",
        "Under LUT No. AD270321000123X",
        "Letter of Undertaking ARN AD2703...",
        "Supplied under bond",
        "Supply to SEZ unit for authorised operations",
        "Buyer is an SEZ Developer",
        "Supply to SEZ",
        "Supplies to SEZ without payment of tax",
        "Supply meant for SEZ",
        "Shipping Bill No. 7788990",
        "Supply meant\nfor export",
    ],
)
def test_export_sez_positive(snippet):
    refusal = detect_export_or_sez(f"Invoice details\n{snippet}\nTotal 100")
    assert refusal is not None
    assert refusal.kind == "export_sez"
    assert refusal.field == "TranDtls.SupTyp"


@pytest.mark.parametrize(
    "snippet",
    [
        "Sharma Exports Pvt Ltd",
        "Export quality basmati rice 25 kg",
        "Import Export Code (IEC): 0512345678",
        "SEZ",
        "Deemed export supply under section 147",
        "Plot 5, Cochin Special Economic Zone, Kakkanad, Kochi 682037",
        "Exporters and Importers of fine tea",
        "Bonded warehouse charges",
        "Mode of Shipping\nBill To: Beta Traders",
        "Shipping Bill To: Beta Traders",
        "",
    ],
)
def test_export_sez_negative(snippet):
    assert detect_export_or_sez(f"TAX INVOICE\n{snippet}\nGSTIN 27AAPFU0939F1ZV") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Supply meant for export with payment of IGST", "EXPWP"),
        ("Supply meant for export under LUT without payment of integrated tax", "EXPWOP"),
        ("Supply to SEZ unit with payment of integrated tax", "SEZWP"),
        ("Supply to SEZ developer under bond", "SEZWOP"),
        ("Export Invoice", "EXPWP or EXPWOP"),
        ("SEZ unit", "SEZWP or SEZWOP"),
        ("with payment of IGST", "EXPWP or SEZWP"),
        ("under LUT", "EXPWOP or SEZWOP"),
        ("Letter of Undertaking", "EXPWOP or SEZWOP"),
    ],
)
def test_export_sez_message_names_likely_suptyp_values(text, expected):
    refusal = detect_export_or_sez(text)
    assert refusal is not None
    assert f"most likely {expected}." in refusal.message
    assert f"SupTyp set to {expected}." in refusal.message


def test_export_sez_message_is_actionable():
    refusal = detect_export_or_sez("Declaration: supply meant\nfor export under LUT", page=3)
    assert refusal is not None
    assert refusal.evidence == '"supply meant for export" (page 3)'
    msg = refusal.message
    assert '"supply meant for export"' in msg and "on page 3" in msg  # (1) what, where
    assert "TranDtls.SupTyp" in msg and "EXPWOP" in msg  # (2) which field, likely value
    assert "no GSTIN" in msg and "BuyerDtls.Gstin" in msg and "tax-split" in msg  # (3) why
    assert "e-invoice portal" in msg and "offline utility" in msg  # (4) what to do
    assert "boilerplate" in msg


def test_export_sez_message_why_clause_matches_family():
    # An SEZ unit/developer is GST-registered: the message must not claim it has no GSTIN.
    sez = detect_export_or_sez("Supply to SEZ unit with payment of IGST")
    assert sez is not None
    assert "no GSTIN" not in sez.message
    assert "zero-rated" in sez.message and "inter-state" in sez.message and "tax-split" in sez.message
    exp = detect_export_or_sez("Export Invoice")
    assert exp is not None
    assert "no GSTIN by design" in exp.message and "zero-rated" not in exp.message
    # Payment-only wording cannot tell the families apart, so both reasons are stated.
    both = detect_export_or_sez("with payment of IGST")
    assert both is not None
    assert "no GSTIN by design" in both.message and "zero-rated" in both.message


def test_export_sez_evidence_without_page():
    refusal = detect_export_or_sez("Shipping Bill No. 1")
    assert refusal is not None
    assert refusal.evidence == '"Shipping Bill"'
    assert "in the document text" in refusal.message


def test_export_sez_evidence_is_earliest_phrase_in_text_order():
    # "Supply to SEZ unit" matches two phrases; the one that starts first is quoted.
    refusal = detect_export_or_sez("Supply to SEZ unit with payment of IGST")
    assert refusal is not None
    assert refusal.evidence == '"Supply to SEZ"'
    refusal = detect_export_or_sez("Buyer: SEZ unit. Supply to SEZ.")
    assert refusal is not None
    assert refusal.evidence == '"SEZ unit"'


# --- multi-currency detector -------------------------------------------------------


@pytest.mark.parametrize(
    "snippet",
    [
        "TotInvValFc: 1200.00",
        "Amount in foreign currency",
        "Exchange rate: 83.20",
        "Conversion Rate 83.20",
        "Total $ 1,200.00",
        "US$1200",
        "Price €500",
        "£ 100",
        "¥1000",
        "Total: USD 1,200.00",
        "Total 1,200.00 USD",
        "USD1200",
        "EUR 500",
        "Currency: USD",
        "Currency - AED",
        "Currency\nSGD",
        "currency: GBP",
        "Currency of invoice: USD\nTotal 1,200.00",
        "Currency (USD)",
        "Total 1,200.00 USD Only",
        "Grand total 1,200.00 USD\nAmount in words: one thousand two hundred",
    ],
)
def test_multi_currency_positive(snippet):
    refusal = detect_multi_currency(f"TAX INVOICE\n{snippet}\nGSTIN 27AAPFU0939F1ZV")
    assert refusal is not None
    assert refusal.kind == "multi_currency"
    assert refusal.field == "ValDtls.TotInvValFc"


@pytest.mark.parametrize(
    "snippet",
    [
        "Total INR 1,18,000.00",
        "Rs. 1,18,000.00",
        "Rs 118000",
        "₹1,18,000.00",
        "Rupees One Lakh Eighteen Thousand Only",
        "Currency: INR",
        "2 CAD drawings 998314 5,000.00",
        "1\nCAD drawings",
        "SAR filing for FY 2025-26",
        "Amount payable in USD",
        "Weight 1200 kg NOK",
        "CAD 3D model",
        "Sharma Exports Pvt Ltd, Rs. 500.00",
        "",
    ],
)
def test_multi_currency_negative(snippet):
    assert detect_multi_currency(f"TAX INVOICE\n{snippet}\nGSTIN 27AAPFU0939F1ZV") is None


def test_multi_currency_message_is_actionable():
    refusal = detect_multi_currency("Total: USD 1,200.00", page=2)
    assert refusal is not None
    assert refusal.evidence == '"USD 1,200.00" (page 2)'
    msg = refusal.message
    assert '"USD 1,200.00"' in msg and "on page 2" in msg  # (1)
    assert "ValDtls.TotInvValFc" in msg  # (2)
    assert "rupee" in msg and "misparse" in msg  # (3)
    assert "INR amounts" in msg and "exchange rate" in msg and "re-run" in msg  # (4)
    assert "export/SEZ refusal" in msg


# --- reverse charge detector -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Whether tax is payable on reverse charge basis: Yes",
        "Whether tax is payable on reverse charge basis : Y",
        "Whether tax is payable under reverse charge: Yes",
        "Reverse Charge: Yes",
        "Reverse Charge - Applicable",
        "Reverse charge mechanism: True",
        "Reverse Charge (RCM): Yes",
        "Reverse Charge (Yes/No): Yes",
        "Reverse Charge:\nYes",
        "Reverse charge applicable: Yes",
        "This supply is subject to reverse charge under section 9(3)",
        "RCM: Yes",
        "RCM applicable",
        "Reverse\ncharge: Yes",
        "Reverse charge applicable as per section 9(3)",
        "Reverse Charge Applicable\nYes",
        "Reverse Charge:\n\nYes",
        "Tax payable under reverse charge: Yes",
        # Embedded in a sentence: words sit beside the phrase on its own line, so it
        # reads as a declaration rather than as a column heading.
        "This supply attracts reverse charge.",
        "Note: tax payable under reverse charge per Notification 13/2017",
    ],
)
def test_reverse_charge_positive(text):
    assert detect_reverse_charge(f"TAX INVOICE\n{text}\nGSTIN 27AAPFU0939F1ZV") is True


@pytest.mark.parametrize(
    "text",
    [
        "Reverse charge applicable",
        "Reverse charge is applicable",
        "Tax payable under reverse charge",
        "Supply attracts reverse charge",
        "Reverse Charge Applicable",
    ],
)
def test_a_phrase_filling_its_whole_line_needs_an_explicit_answer(text):
    """A phrase alone on its line is a label or grid header, not a declaration.

    Reading one as an affirmation sets RegRev and stands validate_tax_split down
    for the whole invoice, silently, on a document that is not under reverse
    charge. The opposite error runs the check and raises a dismissible warning.
    That asymmetry is why these six read as unresolved.
    """
    assert detect_reverse_charge(text) is False
    assert detect_reverse_charge(f"TAX INVOICE\n{text}\nGSTIN 27AAPFU0939F1ZV") is False


@pytest.mark.parametrize(
    "text",
    [
        # Row-major grid extraction: the label row is emitted first, the value row after.
        "Reverse Charge Applicable\nPlace of Supply\nNo\nMaharashtra (27)",
        "Reverse Charge Applicable\nE-Way Bill No.\nNo\n1234567890",
        "Reverse Charge Applicable\nDated\nNo\n1-Apr-2026",
        "Tax payable on\nreverse charge\nbasis\nNo",
    ],
)
def test_a_row_major_grid_label_is_not_an_affirmation(text):
    """Many ERP generators emit every header cell, then every value cell."""
    assert detect_reverse_charge(f"TAX INVOICE\n{text}\nGSTIN 27AAPFU0939F1ZV") is False


@pytest.mark.parametrize(
    "text",
    [
        "Whether tax is payable on reverse charge basis: No",
        "Whether tax is payable on reverse charge basis: N",
        "Whether tax is payable under reverse charge: No",
        "Whether tax is payable on reverse charge basis",
        "Reverse Charge: No",
        "Reverse Charge: N",
        "Reverse charge: Not applicable",
        "Reverse charge: N/A",
        "Reverse charge: NA",
        "Reverse Charge (Yes/No): No",
        "Reverse Charge (RCM): No",
        "Reverse Charge:\nNo",
        "Reverse charge applicable: No",
        "Reverse charge applicability: N",
        "Reverse charge",
        "Reverse charge mechanism",
        "RCM: No",
        "RCM",
        "Tax is payable on reverse charge: No",
        "Reverse Charge: Yes/No",
        # Negated assertive phrases.
        "Not subject to reverse charge",
        "This invoice is not subject to reverse charge",
        "Reverse charge is not applicable",
        "No tax is payable under reverse charge",
        "Subject to reverse charge: No",
        # Label-only layouts: the assertive phrase is a form label without a "Yes".
        "Reverse Charge Applicable\n\nNo",
        "Reverse Charge Applicable: -",
        "Reverse Charge Applicable: Non-applicable",
        "Reverse Charge Applicable    E-Way Bill No.\nNo    1234",
        "Reverse Charge Applicable Yes/No",
        "Tax payable under reverse charge:",
        # Label whose value is an amount, dash or NIL, on the same line or the next one
        # ("Amount of Tax subject to Reverse Charge" is a standard invoice cell).
        "Amount of Tax subject to Reverse Charge\n0.00",
        "Amount of Tax subject to Reverse Charge 0.00",
        "Amount of Tax subject to Reverse Charge: 0.00",
        "Tax Amount subject to Reverse Charge NIL",
        "Reverse Charge Applicable\n\n0.00",
        "Reverse Charge Applicable\n-",
        "Tax payable under reverse charge\n0.00",
        "Tax payable under reverse charge\nRs. 0.00",
        "Tax is payable on reverse charge: No\nAmount of Tax subject to Reverse Charge\n0.00",
        "",
    ],
)
def test_reverse_charge_negative(text):
    assert detect_reverse_charge(f"TAX INVOICE\n{text}\nGSTIN 27AAPFU0939F1ZV") is False


@pytest.mark.parametrize(
    "text",
    [
        "Reverse Charge Applicable",
        "Reverse charge applicable\n\n",
        "Amount of Tax subject to Reverse Charge",
        "Tax payable under reverse charge",
    ],
)
def test_reverse_charge_bare_label_with_nothing_after_it_is_not_affirmative(text):
    # A grid header whose answer is out of reach is unresolved, not "Yes": the safe
    # failure is a tax-split warning the accountant can dismiss, not a silent skip.
    assert detect_reverse_charge(text) is False


def test_reverse_charge_answer_does_not_bleed_across_lines():
    assert detect_reverse_charge("Reverse charge: No\nE-way bill required: Yes") is False


def test_reverse_charge_examines_every_mention():
    # The second mention starts inside the first mention's tail and must still be seen.
    assert detect_reverse_charge("Whether tax is payable on reverse charge basis: No\nRCM: Yes") is True
    assert detect_reverse_charge("Reverse Charge: No\nRCM: Yes") is True


# --- ingest: routing fixtures ------------------------------------------------------


def test_ingest_clean_text_pdf_is_native(tmp_path):
    result = ingest(_build_pdf(tmp_path / "clean.pdf", ("text", CLEAN_TEXT)))
    assert isinstance(result, IngestResult)
    assert [p.method for p in result.pages] == ["native"]
    assert result.pages[0].page == 1
    assert "GSTIN: 27AAPFU0939F1ZV" in result.pages[0].text
    assert result.refusals == []
    assert result.reverse_charge is False


def test_ingest_image_only_pdf_is_ocr_required(tmp_path):
    result = ingest(_build_pdf(tmp_path / "scan.pdf", ("image", None)))
    assert result.pages == [PageResult(page=1, method="ocr_required", text="")]


def test_ingest_partial_text_layer_fails_keyword_gate(tmp_path):
    path = _build_pdf(tmp_path / "partial.pdf", ("partial", PARTIAL_TEXT))
    raw = pymupdf.open(path)[0].get_text("text")
    assert len(raw.strip()) >= 100  # the length gate passes on this page
    result = ingest(path)
    assert result.pages == [PageResult(page=1, method="ocr_required", text="")]


def test_ingest_multi_page_routes_per_page(tmp_path):
    path = _build_pdf(
        tmp_path / "multi.pdf",
        ("text", CLEAN_TEXT),
        ("image", None),
        ("partial", PARTIAL_TEXT),
    )
    result = ingest(path)
    assert [p.page for p in result.pages] == [1, 2, 3]
    assert [p.method for p in result.pages] == ["native", "ocr_required", "ocr_required"]
    assert result.pages[0].text.strip() != ""
    assert result.pages[1].text == "" and result.pages[2].text == ""


def test_ingest_png_input_is_one_ocr_required_page(tmp_path):
    png_path = tmp_path / "scan.png"
    png_path.write_bytes(_png_bytes())
    result = ingest(png_path)
    assert result.pages == [PageResult(page=1, method="ocr_required", text="")]
    assert result.refusals == [] and result.reverse_charge is False


def test_ingest_accepts_path_like_and_upper_case_extension(tmp_path):
    path = tmp_path / "CLEAN.PDF"
    _build_pdf(path, ("text", CLEAN_TEXT))
    assert ingest(path).pages[0].method == "native"


def test_ingest_short_page_with_keywords_is_ocr_required(tmp_path):
    result = ingest(_build_pdf(tmp_path / "short.pdf", ("text", "TAX INVOICE\nGSTIN 27AAPFU0939F1ZV")))
    assert result.pages == [PageResult(page=1, method="ocr_required", text="")]


def test_ingest_tax_invoice_split_across_lines_is_native(tmp_path):
    text = "TAX\nINVOICE\n" + PARTIAL_TEXT
    assert "TAX INVOICE" not in text
    result = ingest(_build_pdf(tmp_path / "split.pdf", ("text", text)))
    assert result.pages[0].method == "native"


@pytest.mark.parametrize("ext", [".docx", ".txt", ".xlsx", ""])
def test_ingest_rejects_unsupported_extension(tmp_path, ext):
    path = tmp_path / f"invoice{ext}"
    path.write_bytes(b"not a document")
    with pytest.raises(ValueError) as excinfo:
        ingest(path)
    assert repr(ext) in str(excinfo.value)
    for accepted in ACCEPTED_EXTENSIONS:
        assert accepted in str(excinfo.value)


def test_ingest_password_protected_pdf_raises_value_error(tmp_path):
    path = str(tmp_path / "locked.pdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 72), CLEAN_TEXT, fontsize=11)
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
    doc.close()
    with pytest.raises(ValueError, match="password-protected"):
        ingest(path)


def test_ingest_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "does-not-exist.pdf")


# --- ingest: detectors end-to-end --------------------------------------------------


def test_ingest_export_phrase_yields_export_refusal_with_page(tmp_path):
    export_page = CLEAN_TEXT.replace("Rs. ", "") + "Supply meant for export under LUT without payment of IGST\n"
    path = _build_pdf(tmp_path / "export.pdf", ("text", CLEAN_TEXT), ("text", export_page))
    result = ingest(path)
    assert [r.kind for r in result.refusals] == ["export_sez"]
    refusal = result.refusals[0]
    assert isinstance(refusal, Refusal)
    assert refusal.field == "TranDtls.SupTyp"
    assert refusal.evidence == '"Supply meant for export" (page 2)'
    assert "TranDtls.SupTyp" in refusal.message
    assert '"Supply meant for export"' in refusal.message
    assert "on page 2" in refusal.message
    assert "EXPWOP" in refusal.message


def test_ingest_export_invoice_in_usd_yields_both_refusals(tmp_path):
    text = (
        "EXPORT INVOICE\n"
        "Sharma Exports Pvt Ltd, GSTIN: 27AAPFU0939F1ZV\n"
        "Buyer: Globex Corp, 1 Main St, Springfield, USA\n"
        "Supply meant for export on payment of integrated tax\n"
        "HSN 0902 Tea 1000 kg   Total: USD 1,200.00\n"
        "Exchange rate: 83.20   IGST 18%\n"
        "Shipping Bill No. 7788990\n"
    )
    result = ingest(_build_pdf(tmp_path / "export-usd.pdf", ("text", text)))
    assert {r.kind for r in result.refusals} == {"export_sez", "multi_currency"}
    by_kind = {r.kind: r for r in result.refusals}
    assert "EXPWP" in by_kind["export_sez"].message
    assert "(page 1)" in by_kind["multi_currency"].evidence


def test_ingest_domestic_exporter_named_company_is_not_refused(tmp_path):
    text = (
        "TAX INVOICE\n"
        "Sharma Exports Pvt Ltd\n"
        "GSTIN: 27AAPFU0939F1ZV   Import Export Code: 0512345678\n"
        "Buyer: Beta Traders, GSTIN 29AABCT1332L1ZU\n"
        "HSN 0902 Tea 100 kg Rs. 50,000.00\n"
        "CGST 2.5% Rs. 1,250.00   SGST 2.5% Rs. 1,250.00\n"
        "Total Rs. 52,500.00   Currency: INR\n"
        "Whether tax is payable on reverse charge basis: No\n"
    )
    result = ingest(_build_pdf(tmp_path / "domestic.pdf", ("text", text)))
    assert result.pages[0].method == "native"
    assert result.refusals == []
    assert result.reverse_charge is False


def test_ingest_reverse_charge_yes_sets_flag_without_refusal(tmp_path):
    text = CLEAN_TEXT.replace("reverse charge basis: No", "reverse charge basis: Yes")
    result = ingest(_build_pdf(tmp_path / "rcm.pdf", ("text", text)))
    assert result.reverse_charge is True
    assert result.refusals == []


def test_ingest_detectors_skip_pages_routed_to_ocr(tmp_path):
    # The revealing page is under the length gate, so its text is dropped and the
    # refusal is missed in build 1 (documented limitation; build 2 re-runs on OCR text).
    path = _build_pdf(
        tmp_path / "hidden.pdf",
        ("text", CLEAN_TEXT),
        ("text", "Supply meant for export. Total USD 1,200.00. RCM: Yes"),
    )
    result = ingest(path)
    assert [p.method for p in result.pages] == ["native", "ocr_required"]
    assert result.refusals == []
    assert result.reverse_charge is False


def test_ingest_reports_first_page_where_each_refusal_kind_appears(tmp_path):
    page2 = CLEAN_TEXT + "Currency: USD\n"
    page3 = CLEAN_TEXT + "Supply to SEZ unit with payment of IGST\n"
    result = ingest(_build_pdf(tmp_path / "pages.pdf", ("text", CLEAN_TEXT), ("text", page2), ("text", page3)))
    by_kind = {r.kind: r for r in result.refusals}
    assert by_kind["multi_currency"].evidence == '"Currency: USD" (page 2)'
    assert by_kind["export_sez"].evidence == '"Supply to SEZ" (page 3)'
    assert "SEZWP" in by_kind["export_sez"].message


def test_result_dataclasses_are_frozen():
    page = PageResult(page=1, method="native", text="x")
    with pytest.raises(AttributeError):
        page.text = "y"  # type: ignore[misc]
