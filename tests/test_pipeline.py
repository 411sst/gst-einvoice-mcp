"""Tests for gst_einvoice.pipeline: end-to-end assembly of the INV-01 payload.

What this file pins is what Part D owns: which fields reached the payload, that
every emitted field carries a provenance entry, that nothing was fabricated,
``missing_fields``, page methods, refusal handling, and the order of the
warnings block. Warnings produced by stage 1 and stage 2 are asserted to be
*present* -- by check name or field path, never by count or wording -- because
those stages own their own messages and are free to change them.

Every pinned expectation (payload values, provenance paths, arithmetic) is
transcribed by hand from the fixture text above it, not computed with the code
under test. ``27AABCB5507N1ZJ`` is a checksum-valid Maharashtra GSTIN, needed so
an intra-state fixture can have two parties in the same state; it was confirmed
through build 1's ``validate_gstin`` before being written here as a literal.

Stage 2 is driven by a fake client throughout: there is no API key in this
environment and nothing here may touch the network.
"""

import json
import os
import shutil

import pymupdf
import pytest

from gst_einvoice import ocr
from gst_einvoice.extract_llm import DEFAULT_MODEL
from gst_einvoice.extract_rules import extract_rules
from gst_einvoice.ingest import detect_multi_currency, ingest
from gst_einvoice.ocr import ocr_pdf_page
from gst_einvoice.pipeline import PipelineResult, extract_invoice
from gst_einvoice.schema import ExtractionResult, PageMeta
from gst_einvoice.validators import validate_invoice


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

VALIDATOR_CHECKS = frozenset(
    {
        "validate_item_total",
        "validate_invoice_total",
        "validate_val_dtls_sums",
        "validate_tax_split",
    }
)


# --- fake Groq client ---------------------------------------------------------------


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        self._client.calls.append(kwargs)
        return _Response(self._client.content)


class _Chat:
    def __init__(self, client):
        self.completions = _Completions(client)


class FakeClient:
    """Exposes ``chat.completions.create`` and hands back one canned response."""

    def __init__(self, payload):
        self.content = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls = []
        self.chat = _Chat(self)

    @property
    def prompt(self) -> str:
        return self.calls[0]["messages"][0]["content"]


# --- fixture builders ---------------------------------------------------------------


def _text_pdf(path, *texts) -> str:
    """A PDF with one text-layer page per argument."""
    doc = pymupdf.open()
    for text in texts:
        doc.new_page().insert_text((50, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


def _page_png(text: str) -> bytes:
    """Render text to PNG, the way build 1's tests make a scanned page."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=12)
    png = page.get_pixmap(dpi=150).tobytes("png")
    doc.close()
    return png


def _mixed_pdf(path, *specs) -> str:
    """Each spec is ("text", str) for a text layer or ("image", str) for a scan."""
    doc = pymupdf.open()
    for kind, payload in specs:
        page = doc.new_page()
        if kind == "text":
            page.insert_text((50, 72), payload, fontsize=11)
        else:
            page.insert_image(page.rect, stream=_page_png(payload))
    doc.save(str(path))
    doc.close()
    return str(path)


def _image_file(path, text: str) -> str:
    """A bare page image on disk, with no PDF around it -- a photograph of an invoice."""
    path.write_bytes(_page_png(text))
    return str(path)


def leaf_paths(value, prefix: str = "") -> list[str]:
    """Every leaf path of a dumped payload (``ItemList[0].SlNo``, ``ValDtls.AssVal``)."""
    if isinstance(value, dict):
        paths: list[str] = []
        for key, sub in value.items():
            paths.extend(leaf_paths(sub, f"{prefix}.{key}" if prefix else key))
        return paths
    if isinstance(value, list):
        paths = []
        for index, sub in enumerate(value):
            paths.extend(leaf_paths(sub, f"{prefix}[{index}]"))
        return paths
    return [prefix]


# --- fixture 1: clean text-layer PDF, intra-state (CGST + SGST), one item ------------

CLEAN_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0042\n"
    "Invoice Date: 17/04/2026\n"
    "Buyer: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Sl No 01 Laptop Stand\n"
    "Qty 4 NOS Rate 1500.00\n"
    "Taxable 6000.00 GST 18%\n"
    "CGST 540.00 SGST 540.00 IGST 0.00\n"
    "HSN/SAC: 8471\n"
    "Goods once sold will not be taken back or exchanged\n"
    "Total Invoice Value 7080.00\n"
)

CLEAN_RESPONSE = {
    "items": [
        {
            "SlNo": "01",
            "PrdDesc": "Laptop Stand",
            "HsnCd": "8471",
            "Qty": 4,
            "Unit": "NOS",
            "UnitPrice": 1500.00,
            "TotAmt": 6000.00,
            "Discount": None,
            "AssAmt": 6000.00,
            "GstRt": 18,
            "CgstAmt": 540.00,
            "SgstAmt": 540.00,
            "IgstAmt": 0.00,
            "CesAmt": None,
            "TotItemVal": 7080.00,
        }
    ],
    "seller": {
        "LglNm": "Nimbus Components Pvt Ltd",
        "Addr1": "Plot 14 MIDC Andheri East",
        "Loc": "Mumbai",
        "Pin": "400093",
    },
    "buyer": {
        "LglNm": "Kanchan Electricals LLP",
        "Addr1": "221 Laxmi Road Shivajinagar",
        "Loc": "Pune",
        "Pin": "411005",
    },
    "totals": {
        "AssVal": 6000.00,
        "CgstVal": 540.00,
        "SgstVal": 540.00,
        "IgstVal": 0.00,
        "CesVal": None,
        "TotInvVal": 7080.00,
        "RndOffAmt": None,
    },
}

# Transcribed by hand from CLEAN_TEXT and CLEAN_RESPONSE. Stage 1 supplies the
# GSTINs (matched), the state codes (27 = Maharashtra, from the GSTIN prefix), the
# invoice number and the date; stage 2 supplies the names, addresses, line item and
# totals; the pipeline supplies Version/TaxSch/SupTyp/Typ/Pos and the zero defaults.
# Arithmetic checked by hand: 4 x 1500.00 = 6000.00 taxable, 18% intra-state splits
# 540.00 + 540.00, so TotItemVal = 6000 + 540 + 540 = 7080.00 = TotInvVal.
EXPECTED_CLEAN_PAYLOAD = {
    "Version": "1.1",
    "TranDtls": {"TaxSch": "GST", "SupTyp": "B2B"},
    "DocDtls": {"Typ": "INV", "No": "INV-2026-0042", "Dt": "17/04/2026"},
    "SellerDtls": {
        "Gstin": "27AAPFU0939F1ZV",
        "LglNm": "Nimbus Components Pvt Ltd",
        "Addr1": "Plot 14 MIDC Andheri East",
        "Loc": "Mumbai",
        "Pin": 400093,
        "Stcd": "27",
    },
    "BuyerDtls": {
        "Gstin": "27AABCB5507N1ZJ",
        "LglNm": "Kanchan Electricals LLP",
        "Addr1": "221 Laxmi Road Shivajinagar",
        "Loc": "Pune",
        "Pin": 411005,
        "Stcd": "27",
        "Pos": "27",
    },
    "ItemList": [
        {
            "SlNo": "01",
            "PrdDesc": "Laptop Stand",
            "IsServc": "N",
            "HsnCd": "8471",
            "Qty": 4,
            "Unit": "NOS",
            "UnitPrice": 1500.00,
            "TotAmt": 6000.00,
            "Discount": 0,
            "AssAmt": 6000.00,
            "GstRt": 18,
            "CgstAmt": 540.00,
            "SgstAmt": 540.00,
            "IgstAmt": 0.00,
            "CesAmt": 0,
            "StateCesAmt": 0,
            "OthChrg": 0,
            "TotItemVal": 7080.00,
        }
    ],
    "ValDtls": {
        "AssVal": 6000.00,
        "CgstVal": 540.00,
        "SgstVal": 540.00,
        "IgstVal": 0.00,
        "CesVal": 0,
        "StCesVal": 0,
        "RndOffAmt": 0,
        "TotInvVal": 7080.00,
    },
}

# The same payload's field paths, written out by hand rather than walked out of it.
EXPECTED_CLEAN_PATHS = frozenset(
    {
        "Version",
        "TranDtls.TaxSch",
        "TranDtls.SupTyp",
        "DocDtls.Typ",
        "DocDtls.No",
        "DocDtls.Dt",
        "SellerDtls.Gstin",
        "SellerDtls.LglNm",
        "SellerDtls.Addr1",
        "SellerDtls.Loc",
        "SellerDtls.Pin",
        "SellerDtls.Stcd",
        "BuyerDtls.Gstin",
        "BuyerDtls.LglNm",
        "BuyerDtls.Addr1",
        "BuyerDtls.Loc",
        "BuyerDtls.Pin",
        "BuyerDtls.Stcd",
        "BuyerDtls.Pos",
        "ItemList[0].SlNo",
        "ItemList[0].PrdDesc",
        "ItemList[0].IsServc",
        "ItemList[0].HsnCd",
        "ItemList[0].Qty",
        "ItemList[0].Unit",
        "ItemList[0].UnitPrice",
        "ItemList[0].TotAmt",
        "ItemList[0].Discount",
        "ItemList[0].AssAmt",
        "ItemList[0].GstRt",
        "ItemList[0].CgstAmt",
        "ItemList[0].SgstAmt",
        "ItemList[0].IgstAmt",
        "ItemList[0].CesAmt",
        "ItemList[0].StateCesAmt",
        "ItemList[0].OthChrg",
        "ItemList[0].TotItemVal",
        "ValDtls.AssVal",
        "ValDtls.CgstVal",
        "ValDtls.SgstVal",
        "ValDtls.IgstVal",
        "ValDtls.CesVal",
        "ValDtls.StCesVal",
        "ValDtls.RndOffAmt",
        "ValDtls.TotInvVal",
    }
)


@pytest.fixture
def clean_result(tmp_path):
    client = FakeClient(CLEAN_RESPONSE)
    path = _text_pdf(tmp_path / "clean.pdf", CLEAN_TEXT)
    return extract_invoice(path, client=client), client


# --- clean text-layer PDF -----------------------------------------------------------


def test_clean_pdf_produces_the_hand_transcribed_inv01_payload(clean_result):
    result, _ = clean_result
    assert isinstance(result, PipelineResult)
    assert result.missing_fields == ()
    assert result.refusals == ()
    assert isinstance(result.extraction, ExtractionResult)
    assert result.extraction.invoice.model_dump(exclude_none=True) == EXPECTED_CLEAN_PAYLOAD


def test_clean_pdf_records_one_page_meta_with_the_ingest_method(clean_result):
    result, _ = clean_result
    assert result.meta.pages == [PageMeta(page=1, method="native")]


def test_extraction_meta_is_the_same_meta_the_result_carries(clean_result):
    result, _ = clean_result
    assert result.extraction.extraction_meta == result.meta


def test_clean_pdf_emits_no_validator_warnings_because_its_arithmetic_is_consistent(
    clean_result,
):
    result, _ = clean_result
    assert [w for w in result.meta.warnings if w.check in VALIDATOR_CHECKS] == []


def test_every_emitted_field_has_a_provenance_entry(clean_result):
    result, _ = clean_result
    emitted = leaf_paths(result.extraction.invoice.model_dump(exclude_none=True))
    assert set(emitted) == EXPECTED_CLEAN_PATHS
    missing_provenance = [p for p in emitted if p not in result.meta.field_provenance]
    assert missing_provenance == []


@pytest.mark.parametrize(
    ("path", "source"),
    [
        ("SellerDtls.Gstin", "regex"),
        ("BuyerDtls.Gstin", "regex"),
        ("SellerDtls.Stcd", "derived"),
        ("BuyerDtls.Stcd", "derived"),
        ("DocDtls.No", "regex"),
        ("DocDtls.Dt", "regex"),
        ("SellerDtls.LglNm", "llm"),
        ("BuyerDtls.Addr1", "llm"),
        ("ItemList[0].PrdDesc", "llm"),
        ("ItemList[0].TotItemVal", "llm"),
        ("ValDtls.TotInvVal", "llm"),
        ("ItemList[0].IsServc", "derived"),
        ("BuyerDtls.Pos", "assumed"),
        ("TranDtls.SupTyp", "assumed"),
        ("TranDtls.TaxSch", "assumed"),
        ("DocDtls.Typ", "assumed"),
        ("Version", "assumed"),
        ("ItemList[0].StateCesAmt", "assumed"),
        ("ValDtls.StCesVal", "assumed"),
    ],
)
def test_provenance_names_the_stage_each_field_came_from(clean_result, path, source):
    result, _ = clean_result
    assert result.meta.field_provenance[path]["source"] == source


def test_a_state_code_is_recorded_as_derived_because_it_was_never_matched(clean_result):
    # CLEAN_TEXT prints no state code anywhere: "27" reaches the payload only because
    # it is the first two characters of 27AAPFU0939F1ZV and 27AABCB5507N1ZJ, put
    # through build 1's lookup_state_code. Calling that "regex" would claim the state
    # was read off the page, on the two fields that decide CGST/SGST versus IGST.
    # Stage 1's own label for these two, read from stage 1 rather than assumed here:
    assert extract_rules(CLEAN_TEXT).seller_stcd.method == "derived"
    assert extract_rules(CLEAN_TEXT).buyer_stcd.method == "derived"
    provenance = clean_result[0].meta.field_provenance
    assert provenance["SellerDtls.Stcd"]["source"] == "derived"
    assert provenance["BuyerDtls.Stcd"]["source"] == "derived"
    # And the GSTINs they were derived from are still matched text, so they stay "regex".
    assert provenance["SellerDtls.Gstin"]["source"] == "regex"
    assert provenance["BuyerDtls.Gstin"]["source"] == "regex"


def test_provenance_on_a_native_page_records_no_ocr_confidence(clean_result):
    result, _ = clean_result
    for entry in result.meta.field_provenance.values():
        assert entry["ocr_confidence"] is None


def test_stage_warnings_reach_meta_warnings(clean_result):
    result, _ = clean_result
    assert any(w.check == "extract_llm" for w in result.meta.warnings)


def test_place_of_supply_is_flagged_as_assumed_not_read(clean_result):
    result, _ = clean_result
    notes = [w for w in result.meta.warnings if w.check == "pipeline" and w.field == "BuyerDtls.Pos"]
    assert len(notes) == 1
    assert notes[0].severity == "info"
    assert "assumed" in notes[0].message


def test_the_confirmed_stage_one_fields_are_passed_to_the_model(clean_result):
    result, client = clean_result
    assert len(client.calls) == 1
    for confirmed in ("27AAPFU0939F1ZV", "27AABCB5507N1ZJ", "INV-2026-0042", "17/04/2026", "8471"):
        assert confirmed in client.prompt


def test_the_default_model_is_used_and_an_explicit_one_is_passed_through(tmp_path):
    path = _text_pdf(tmp_path / "clean.pdf", CLEAN_TEXT)
    default = FakeClient(CLEAN_RESPONSE)
    extract_invoice(path, client=default)
    assert default.calls[0]["model"] == DEFAULT_MODEL

    chosen = FakeClient(CLEAN_RESPONSE)
    extract_invoice(path, client=chosen, model="llama-3.1-8b-instant")
    assert chosen.calls[0]["model"] == "llama-3.1-8b-instant"


def test_client_is_a_required_keyword_argument(tmp_path):
    path = _text_pdf(tmp_path / "clean.pdf", CLEAN_TEXT)
    with pytest.raises(TypeError):
        extract_invoice(path)


# --- inter-state invoice (IGST) -----------------------------------------------------

INTERSTATE_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0044\n"
    "Invoice Date: 19/04/2026\n"
    "Buyer: Rohtas Distributors Pvt Ltd\n"
    "GSTIN 07AAGFF2194N1Z1\n"
    "Khasra 12 Naraina Industrial Area\n"
    "New Delhi 110028\n"
    "Sl No 01 Cable Reel\n"
    "Qty 2 NOS Rate 2500.00 Taxable 5000.00 GST 12% CGST 0.00 SGST 0.00 IGST 600.00\n"
    "Line Total 5600.00\n"
    "HSN/SAC: 8544\n"
    "Goods once sold will not be taken back or exchanged\n"
    "Total Invoice Value 5600.00\n"
)

INTERSTATE_RESPONSE = {
    "items": [
        {
            "SlNo": "01",
            "PrdDesc": "Cable Reel",
            "HsnCd": "8544",
            "Qty": 2,
            "Unit": "NOS",
            "UnitPrice": 2500.00,
            "TotAmt": 5000.00,
            "Discount": None,
            "AssAmt": 5000.00,
            "GstRt": 12,
            "CgstAmt": 0.00,
            "SgstAmt": 0.00,
            "IgstAmt": 600.00,
            "CesAmt": None,
            "TotItemVal": 5600.00,
        }
    ],
    "seller": {
        "LglNm": "Nimbus Components Pvt Ltd",
        "Addr1": "Plot 14 MIDC Andheri East",
        "Loc": "Mumbai",
        "Pin": "400093",
    },
    "buyer": {
        "LglNm": "Rohtas Distributors Pvt Ltd",
        "Addr1": "Khasra 12 Naraina Industrial Area",
        "Loc": "New Delhi",
        "Pin": "110028",
    },
    "totals": {
        "AssVal": 5000.00,
        "CgstVal": 0.00,
        "SgstVal": 0.00,
        "IgstVal": 600.00,
        "CesVal": None,
        "TotInvVal": 5600.00,
        "RndOffAmt": None,
    },
}


def test_inter_state_invoice_keeps_igst_and_passes_the_tax_split_check(tmp_path):
    # Maharashtra (27) seller to a Delhi (07) buyer: 5000.00 x 12% = 600.00 IGST,
    # no CGST or SGST, so TotInvVal = 5000.00 + 600.00 = 5600.00.
    path = _text_pdf(tmp_path / "igst.pdf", INTERSTATE_TEXT)
    result = extract_invoice(path, client=FakeClient(INTERSTATE_RESPONSE))
    assert result.missing_fields == ()
    invoice = result.extraction.invoice
    assert invoice.SellerDtls.Stcd == "27"
    assert invoice.BuyerDtls.Stcd == "07"
    assert invoice.BuyerDtls.Pos == "07"
    assert (invoice.ItemList[0].CgstAmt, invoice.ItemList[0].SgstAmt) == (0.0, 0.0)
    assert invoice.ItemList[0].IgstAmt == 600.00
    assert invoice.ValDtls.IgstVal == 600.00
    assert invoice.ValDtls.TotInvVal == 5600.00
    assert [w for w in result.meta.warnings if w.check in VALIDATOR_CHECKS] == []


# --- mixed GST rates across line items ----------------------------------------------

MIXED_RATES_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0043\n"
    "Invoice Date: 18/04/2026\n"
    "Buyer: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Sl No 01 Laptop Stand\n"
    "Qty 2 NOS Rate 1500.00 Taxable 3000.00 GST 18% CGST 270.00 SGST 270.00\n"
    "Line Total 3540.00\n"
    "HSN/SAC: 8471\n"
    "Sl No 02 Printer Paper\n"
    "Qty 10 PKT Rate 200.00 Taxable 2000.00 GST 12% CGST 120.00 SGST 120.00\n"
    "Line Total 2240.00\n"
    "HSN/SAC: 4802\n"
    "Goods once sold will not be taken back or exchanged\n"
    "Taxable Value Total 5000.00\n"
    "CGST Total 390.00 SGST Total 390.00 IGST Total 0.00\n"
    "Total Invoice Value 5780.00\n"
)

MIXED_RATES_RESPONSE = {
    "items": [
        {
            "SlNo": "01",
            "PrdDesc": "Laptop Stand",
            "HsnCd": "8471",
            "Qty": 2,
            "Unit": "NOS",
            "UnitPrice": 1500.00,
            "TotAmt": 3000.00,
            "Discount": None,
            "AssAmt": 3000.00,
            "GstRt": 18,
            "CgstAmt": 270.00,
            "SgstAmt": 270.00,
            "IgstAmt": 0.00,
            "CesAmt": None,
            "TotItemVal": 3540.00,
        },
        {
            "SlNo": "02",
            "PrdDesc": "Printer Paper",
            "HsnCd": "4802",
            "Qty": 10,
            "Unit": "PKT",
            "UnitPrice": 200.00,
            "TotAmt": 2000.00,
            "Discount": None,
            "AssAmt": 2000.00,
            "GstRt": 12,
            "CgstAmt": 120.00,
            "SgstAmt": 120.00,
            "IgstAmt": 0.00,
            "CesAmt": None,
            "TotItemVal": 2240.00,
        },
    ],
    "seller": {
        "LglNm": "Nimbus Components Pvt Ltd",
        "Addr1": "Plot 14 MIDC Andheri East",
        "Loc": "Mumbai",
        "Pin": "400093",
    },
    "buyer": {
        "LglNm": "Kanchan Electricals LLP",
        "Addr1": "221 Laxmi Road Shivajinagar",
        "Loc": "Pune",
        "Pin": "411005",
    },
    "totals": {
        "AssVal": 5000.00,
        "CgstVal": 390.00,
        "SgstVal": 390.00,
        "IgstVal": 0.00,
        "CesVal": None,
        "TotInvVal": 5780.00,
        "RndOffAmt": None,
    },
}


def test_mixed_rates_keep_each_items_own_rate_and_still_reconcile(tmp_path):
    # Row 1: 3000.00 at 18% -> 270.00 + 270.00. Row 2: 2000.00 at 12% -> 120.00 + 120.00.
    # AssVal 5000.00, CgstVal 390.00, SgstVal 390.00, TotInvVal 5780.00.
    path = _text_pdf(tmp_path / "mixed.pdf", MIXED_RATES_TEXT)
    result = extract_invoice(path, client=FakeClient(MIXED_RATES_RESPONSE))
    assert result.missing_fields == ()
    items = result.extraction.invoice.ItemList
    assert [item.GstRt for item in items] == [18.0, 12.0]
    assert [item.HsnCd for item in items] == ["8471", "4802"]
    assert [item.TotItemVal for item in items] == [3540.00, 2240.00]
    assert result.extraction.invoice.ValDtls.TotInvVal == 5780.00
    assert [w for w in result.meta.warnings if w.check in VALIDATOR_CHECKS] == []


def test_every_emitted_field_of_a_two_item_invoice_has_provenance(tmp_path):
    path = _text_pdf(tmp_path / "mixed.pdf", MIXED_RATES_TEXT)
    result = extract_invoice(path, client=FakeClient(MIXED_RATES_RESPONSE))
    emitted = leaf_paths(result.extraction.invoice.model_dump(exclude_none=True))
    assert [p for p in emitted if p not in result.meta.field_provenance] == []
    assert result.meta.field_provenance["ItemList[1].PrdDesc"]["source"] == "llm"
    assert result.meta.field_provenance["ItemList[1].IsServc"]["source"] == "derived"


# --- single-digit row numbers, the ordinary Indian invoice ----------------------

# Every other fixture here numbers its rows "01"/"02". The great majority of real
# invoices number them 1..9, which is a one-character SlNo -- and SlNo is mandatory
# in INV-01, so a stage 2 that cannot ground one character emits no Item, and Part D
# correctly refuses to invent one: no payload at all for the commonest shape there
# is. This fixture is the end-to-end check that the ordinary case survives the whole
# pipeline.

SINGLE_DIGIT_ROWS_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0046\n"
    "Invoice Date: 21/04/2026\n"
    "Buyer: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Sl No 1 Laptop Stand\n"
    "Qty 2 NOS Rate 1500.00 Taxable 3000.00 GST 18% CGST 270.00 SGST 270.00\n"
    "Line Total 3540.00\n"
    "HSN/SAC: 8471\n"
    "Sl No 2 Printer Paper\n"
    "Qty 10 PKT Rate 200.00 Taxable 2000.00 GST 12% CGST 120.00 SGST 120.00\n"
    "Line Total 2240.00\n"
    "HSN/SAC: 4802\n"
    "Goods once sold will not be taken back or exchanged\n"
    "Taxable Value Total 5000.00\n"
    "CGST Total 390.00 SGST Total 390.00 IGST Total 0.00\n"
    "Total Invoice Value 5780.00\n"
)

SINGLE_DIGIT_ROWS_RESPONSE = {
    "items": [
        {
            "SlNo": "1",
            "PrdDesc": "Laptop Stand",
            "HsnCd": "8471",
            "Qty": 2,
            "Unit": "NOS",
            "UnitPrice": 1500.00,
            "TotAmt": 3000.00,
            "Discount": None,
            "AssAmt": 3000.00,
            "GstRt": 18,
            "CgstAmt": 270.00,
            "SgstAmt": 270.00,
            "IgstAmt": 0.00,
            "CesAmt": None,
            "TotItemVal": 3540.00,
        },
        {
            "SlNo": "2",
            "PrdDesc": "Printer Paper",
            "HsnCd": "4802",
            "Qty": 10,
            "Unit": "PKT",
            "UnitPrice": 200.00,
            "TotAmt": 2000.00,
            "Discount": None,
            "AssAmt": 2000.00,
            "GstRt": 12,
            "CgstAmt": 120.00,
            "SgstAmt": 120.00,
            "IgstAmt": 0.00,
            "CesAmt": None,
            "TotItemVal": 2240.00,
        },
    ],
    "seller": {
        "LglNm": "Nimbus Components Pvt Ltd",
        "Addr1": "Plot 14 MIDC Andheri East",
        "Loc": "Mumbai",
        "Pin": "400093",
    },
    "buyer": {
        "LglNm": "Kanchan Electricals LLP",
        "Addr1": "221 Laxmi Road Shivajinagar",
        "Loc": "Pune",
        "Pin": "411005",
    },
    "totals": {
        "AssVal": 5000.00,
        "CgstVal": 390.00,
        "SgstVal": 390.00,
        "IgstVal": 0.00,
        "CesVal": None,
        "TotInvVal": 5780.00,
        "RndOffAmt": None,
    },
}


def test_rows_numbered_one_and_two_produce_a_complete_payload(tmp_path):
    # Transcribed by hand from SINGLE_DIGIT_ROWS_TEXT: row 1 is 2 x 1500.00 = 3000.00
    # at 18% -> 270.00 + 270.00, line total 3540.00; row 2 is 10 x 200.00 = 2000.00 at
    # 12% -> 120.00 + 120.00, line total 2240.00. AssVal 5000.00, CgstVal 390.00,
    # SgstVal 390.00, IgstVal 0.00, TotInvVal 5780.00. Both parties are in state 27,
    # so the tax is CGST + SGST.
    path = _text_pdf(tmp_path / "single_digit_rows.pdf", SINGLE_DIGIT_ROWS_TEXT)
    result = extract_invoice(path, client=FakeClient(SINGLE_DIGIT_ROWS_RESPONSE))

    assert result.refusals == ()
    assert result.missing_fields == ()
    assert isinstance(result.extraction, ExtractionResult)
    invoice = result.extraction.invoice
    assert [item.SlNo for item in invoice.ItemList] == ["1", "2"]
    assert [item.PrdDesc for item in invoice.ItemList] == ["Laptop Stand", "Printer Paper"]
    assert [item.TotItemVal for item in invoice.ItemList] == [3540.00, 2240.00]
    assert invoice.ValDtls.AssVal == 5000.00
    assert invoice.ValDtls.CgstVal == 390.00
    assert invoice.ValDtls.SgstVal == 390.00
    assert invoice.ValDtls.TotInvVal == 5780.00
    # A complete payload, and the arithmetic build 1's validators check reconciles.
    assert [w for w in result.meta.warnings if w.check in VALIDATOR_CHECKS] == []
    emitted = leaf_paths(invoice.model_dump(exclude_none=True))
    assert [p for p in emitted if p not in result.meta.field_provenance] == []
    # The row numbers were kept, not laundered: stage 2 reports that one character is
    # weak corroboration. Asserted by check and field path only -- stage 2 owns the
    # wording, and how many notes it emits is its business, not this file's.
    assert any(
        w.check == "extract_llm" and w.field == "ItemList[0].SlNo" for w in result.meta.warnings
    )


# --- cess -----------------------------------------------------------------------

CESS_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0045\n"
    "Invoice Date: 20/04/2026\n"
    "Buyer: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Sl No 01 Aerated Drink Crate\n"
    "Qty 5 CTN Rate 1000.00 Taxable 5000.00 GST 28% CGST 700.00 SGST 700.00 IGST 0.00\n"
    "Compensation Cess 600.00\n"
    "Line Total 7000.00\n"
    "HSN/SAC: 2202\n"
    "Goods once sold will not be taken back or exchanged\n"
    "Total Invoice Value 7000.00\n"
)

CESS_RESPONSE = {
    "items": [
        {
            "SlNo": "01",
            "PrdDesc": "Aerated Drink Crate",
            "HsnCd": "2202",
            "Qty": 5,
            "Unit": "CTN",
            "UnitPrice": 1000.00,
            "TotAmt": 5000.00,
            "Discount": None,
            "AssAmt": 5000.00,
            "GstRt": 28,
            "CgstAmt": 700.00,
            "SgstAmt": 700.00,
            "IgstAmt": 0.00,
            "CesAmt": 600.00,
            "TotItemVal": 7000.00,
        }
    ],
    "seller": {
        "LglNm": "Nimbus Components Pvt Ltd",
        "Addr1": "Plot 14 MIDC Andheri East",
        "Loc": "Mumbai",
        "Pin": "400093",
    },
    "buyer": {
        "LglNm": "Kanchan Electricals LLP",
        "Addr1": "221 Laxmi Road Shivajinagar",
        "Loc": "Pune",
        "Pin": "411005",
    },
    "totals": {
        "AssVal": 5000.00,
        "CgstVal": 700.00,
        "SgstVal": 700.00,
        "IgstVal": 0.00,
        "CesVal": 600.00,
        "TotInvVal": 7000.00,
        "RndOffAmt": None,
    },
}


def test_cess_is_carried_into_the_item_and_the_totals(tmp_path):
    # 5000.00 at 28% -> 700.00 + 700.00, plus 600.00 cess:
    # TotItemVal = 5000 + 700 + 700 + 600 = 7000.00 = TotInvVal.
    path = _text_pdf(tmp_path / "cess.pdf", CESS_TEXT)
    result = extract_invoice(path, client=FakeClient(CESS_RESPONSE))
    assert result.missing_fields == ()
    invoice = result.extraction.invoice
    assert invoice.ItemList[0].CesAmt == 600.00
    assert invoice.ValDtls.CesVal == 600.00
    assert invoice.ValDtls.TotInvVal == 7000.00
    assert result.meta.field_provenance["ItemList[0].CesAmt"]["source"] == "llm"
    assert result.meta.field_provenance["ValDtls.CesVal"]["source"] == "llm"
    assert [w for w in result.meta.warnings if w.check in VALIDATOR_CHECKS] == []


# --- IsServc derivation ---------------------------------------------------------

SERVICE_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0046\n"
    "Invoice Date: 21/04/2026\n"
    "Buyer: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Sl No 01 Software Support Service\n"
    "Qty 1 JOB Rate 20000.00 Taxable 20000.00 GST 18% CGST 1800.00 SGST 1800.00 IGST 0.00\n"
    "Line Total 23600.00\n"
    "HSN/SAC: 998313\n"
    "Goods once sold will not be taken back or exchanged\n"
    "Total Invoice Value 23600.00\n"
)

SERVICE_RESPONSE = {
    "items": [
        {
            "SlNo": "01",
            "PrdDesc": "Software Support Service",
            "HsnCd": "998313",
            "Qty": 1,
            "Unit": "JOB",
            "UnitPrice": 20000.00,
            "TotAmt": 20000.00,
            "Discount": None,
            "AssAmt": 20000.00,
            "GstRt": 18,
            "CgstAmt": 1800.00,
            "SgstAmt": 1800.00,
            "IgstAmt": 0.00,
            "CesAmt": None,
            "TotItemVal": 23600.00,
        }
    ],
    "seller": {
        "LglNm": "Nimbus Components Pvt Ltd",
        "Addr1": "Plot 14 MIDC Andheri East",
        "Loc": "Mumbai",
        "Pin": "400093",
    },
    "buyer": {
        "LglNm": "Kanchan Electricals LLP",
        "Addr1": "221 Laxmi Road Shivajinagar",
        "Loc": "Pune",
        "Pin": "411005",
    },
    "totals": {
        "AssVal": 20000.00,
        "CgstVal": 1800.00,
        "SgstVal": 1800.00,
        "IgstVal": 0.00,
        "CesVal": None,
        "TotInvVal": 23600.00,
        "RndOffAmt": None,
    },
}


def test_a_six_digit_sac_beginning_99_is_marked_as_a_service(tmp_path):
    path = _text_pdf(tmp_path / "service.pdf", SERVICE_TEXT)
    result = extract_invoice(path, client=FakeClient(SERVICE_RESPONSE))
    assert result.missing_fields == ()
    assert result.extraction.invoice.ItemList[0].HsnCd == "998313"
    assert result.extraction.invoice.ItemList[0].IsServc == "Y"
    assert result.meta.field_provenance["ItemList[0].IsServc"]["source"] == "derived"


def test_a_goods_hsn_is_not_marked_as_a_service(clean_result):
    result, _ = clean_result
    assert result.extraction.invoice.ItemList[0].HsnCd == "8471"
    assert result.extraction.invoice.ItemList[0].IsServc == "N"


def test_a_row_that_never_reaches_the_payload_leaves_no_isservc_provenance(tmp_path):
    # IsServc is derived from the HSN before the row's other fields are known to be
    # readable. This row is missing its Unit, so it never becomes an Item and no
    # payload is produced at all; a provenance entry for its IsServc would describe a
    # field the result does not contain, which is the same mismatch the missing paths
    # have their provenance taken back for.
    response = json.loads(json.dumps(CLEAN_RESPONSE))
    response["items"][0]["Unit"] = None
    path = _text_pdf(tmp_path / "no_unit.pdf", CLEAN_TEXT)
    result = extract_invoice(path, client=FakeClient(response))
    assert result.extraction is None
    assert result.missing_fields == ("ItemList[0].Unit",)
    assert "ItemList[0].IsServc" not in result.meta.field_provenance
    # The rest of the row was read, so what stage 2 did supply is still recorded.
    assert result.meta.field_provenance["ItemList[0].PrdDesc"]["source"] == "llm"


# --- reverse charge -------------------------------------------------------------


def test_reverse_charge_sets_regrev_and_is_recorded_as_assumed(tmp_path):
    text = CLEAN_TEXT + "Reverse Charge Applicable: Yes\n"
    path = _text_pdf(tmp_path / "rcm.pdf", text)
    result = extract_invoice(path, client=FakeClient(CLEAN_RESPONSE))
    assert result.missing_fields == ()
    assert result.extraction.invoice.TranDtls.RegRev == "Y"
    assert result.meta.field_provenance["TranDtls.RegRev"]["source"] == "assumed"


def test_without_a_reverse_charge_declaration_regrev_is_left_out(clean_result):
    result, _ = clean_result
    assert result.extraction.invoice.TranDtls.RegRev is None
    assert "TranDtls.RegRev" not in result.extraction.invoice.model_dump(exclude_none=True)["TranDtls"]


# --- the central guarantee: missing, never fabricated ---------------------------

# The same seller, but the buyer's address, town and PIN are genuinely absent.
NO_BUYER_ADDRESS_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0042\n"
    "Invoice Date: 17/04/2026\n"
    "Buyer: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "Sl No 01 Laptop Stand\n"
    "Qty 4 NOS Rate 1500.00\n"
    "Taxable 6000.00 GST 18%\n"
    "CGST 540.00 SGST 540.00 IGST 0.00\n"
    "HSN/SAC: 8471\n"
    "Goods once sold will not be taken back or exchanged\n"
    "Total Invoice Value 7080.00\n"
)


def _response_with_buyer(buyer: dict) -> dict:
    response = json.loads(json.dumps(CLEAN_RESPONSE))
    response["buyer"] = buyer
    return response


def test_a_field_the_model_could_not_find_is_reported_missing_not_filled(tmp_path):
    # The model answers null for the three fields the document does not print.
    client = FakeClient(
        _response_with_buyer(
            {"LglNm": "Kanchan Electricals LLP", "Addr1": None, "Loc": None, "Pin": None}
        )
    )
    path = _text_pdf(tmp_path / "no_addr.pdf", NO_BUYER_ADDRESS_TEXT)
    result = extract_invoice(path, client=client)

    assert result.extraction is None
    assert result.missing_fields == ("BuyerDtls.Addr1", "BuyerDtls.Loc", "BuyerDtls.Pin")
    for path_name in result.missing_fields:
        assert path_name not in result.meta.field_provenance
        assert any(
            w.check == "pipeline" and w.field == path_name for w in result.meta.warnings
        ), f"no pipeline note explains why {path_name} is missing"
        assert any(
            w.check == "extract_llm" and w.field == path_name for w in result.meta.warnings
        ), f"stage 2's warning for {path_name} did not reach meta.warnings"


def test_an_ungrounded_value_is_dropped_and_reported_missing_not_kept(tmp_path):
    # "Oceanic Freight Systems Limited" is nowhere in the document. It is plausible,
    # which is exactly why it must not survive into the payload.
    fabricated = "Oceanic Freight Systems Limited"
    assert fabricated not in NO_BUYER_ADDRESS_TEXT
    client = FakeClient(
        _response_with_buyer({"LglNm": fabricated, "Addr1": None, "Loc": None, "Pin": None})
    )
    path = _text_pdf(tmp_path / "fabricated.pdf", NO_BUYER_ADDRESS_TEXT)
    result = extract_invoice(path, client=client)

    assert result.extraction is None
    assert "BuyerDtls.LglNm" in result.missing_fields
    assert "BuyerDtls.LglNm" not in result.meta.field_provenance
    assert any(
        w.check == "extract_llm" and w.field == "BuyerDtls.LglNm" for w in result.meta.warnings
    )


def test_a_missing_mandatory_field_still_leaves_the_readable_fields_in_provenance(tmp_path):
    client = FakeClient(
        _response_with_buyer(
            {"LglNm": "Kanchan Electricals LLP", "Addr1": None, "Loc": None, "Pin": None}
        )
    )
    path = _text_pdf(tmp_path / "no_addr.pdf", NO_BUYER_ADDRESS_TEXT)
    result = extract_invoice(path, client=client)
    assert result.meta.field_provenance["SellerDtls.Gstin"]["source"] == "regex"
    assert result.meta.field_provenance["BuyerDtls.LglNm"]["source"] == "llm"
    assert result.meta.pages == [PageMeta(page=1, method="native")]


def test_no_line_items_leaves_itemlist_missing_rather_than_inventing_a_row(tmp_path):
    response = json.loads(json.dumps(CLEAN_RESPONSE))
    response["items"] = []
    path = _text_pdf(tmp_path / "no_items.pdf", CLEAN_TEXT)
    result = extract_invoice(path, client=FakeClient(response))
    assert result.extraction is None
    assert result.missing_fields == ("ItemList",)
    assert any(w.check == "pipeline" and w.field == "ItemList" for w in result.meta.warnings)


def test_an_unusable_response_reports_every_stage_two_field_as_missing(tmp_path):
    path = _text_pdf(tmp_path / "garbage.pdf", CLEAN_TEXT)
    result = extract_invoice(path, client=FakeClient("this is not JSON at all"))
    assert result.extraction is None
    # Stage 1's fields survive; everything stage 2 was responsible for is reported.
    assert result.missing_fields == (
        "SellerDtls.LglNm",
        "SellerDtls.Addr1",
        "SellerDtls.Loc",
        "SellerDtls.Pin",
        "BuyerDtls.LglNm",
        "BuyerDtls.Addr1",
        "BuyerDtls.Loc",
        "BuyerDtls.Pin",
        "ItemList",
        "ValDtls.AssVal",
        "ValDtls.CgstVal",
        "ValDtls.SgstVal",
        "ValDtls.IgstVal",
        "ValDtls.TotInvVal",
    )
    assert [e for e in result.meta.field_provenance.values() if e["source"] == "llm"] == []
    assert any(w.check == "extract_llm" and w.field == "ItemList" for w in result.meta.warnings)


def test_a_pin_that_is_not_a_number_is_missing_rather_than_coerced(tmp_path):
    # The model returns the whole printed line. It is grounded, but it is not a PIN.
    response = json.loads(json.dumps(CLEAN_RESPONSE))
    response["buyer"]["Pin"] = "Pune 411005"
    path = _text_pdf(tmp_path / "bad_pin.pdf", CLEAN_TEXT)
    result = extract_invoice(path, client=FakeClient(response))
    assert result.extraction is None
    assert result.missing_fields == ("BuyerDtls.Pin",)
    note = [w for w in result.meta.warnings if w.check == "pipeline" and w.field == "BuyerDtls.Pin"]
    assert len(note) == 1
    assert "Pune 411005" in note[0].message
    # A path reported missing must not also carry a provenance entry saying it was
    # read: stage 2 did record one for the grounded "Pune 411005" before the pipeline
    # rejected it as a PIN, and the metadata is the only output there is here. This is
    # the same invariant the null-field case above asserts.
    assert "BuyerDtls.Pin" not in result.meta.field_provenance
    assert result.meta.field_provenance["BuyerDtls.Loc"]["source"] == "llm"


def test_a_pin_written_in_non_ascii_digits_is_missing_rather_than_coerced(tmp_path):
    # "²³" is printed on the page, so stage 2 grounds it and hands it over as
    # the PIN. str.isdigit() says it is digits and int() refuses it, so a guard that
    # trusts isdigit() lets a ValueError out of extract_invoice instead of reporting the
    # field missing; Arabic-Indic digits pass both and would be converted into a PIN
    # nobody printed. Neither is an INV-01 PIN code, so neither is kept.
    text = CLEAN_TEXT.replace("Pune 411005", "Pune ²³ 411005")
    response = json.loads(json.dumps(CLEAN_RESPONSE))
    response["buyer"]["Pin"] = "²³"
    path = _text_pdf(tmp_path / "superscript_pin.pdf", text)
    result = extract_invoice(path, client=FakeClient(response))
    assert result.extraction is None
    assert result.missing_fields == ("BuyerDtls.Pin",)
    note = [w for w in result.meta.warnings if w.check == "pipeline" and w.field == "BuyerDtls.Pin"]
    assert len(note) == 1
    assert "²³" in note[0].message
    assert "BuyerDtls.Pin" not in result.meta.field_provenance


# --- party disambiguation: the positional fallback always warns -----------------

# No "Seller"/"Buyer"/"Bill To" label anywhere, so stage 1 must fall back to
# document position, and must warn even though the fallback gets it right.
UNLABELLED_TEXT = (
    "TAX INVOICE\n"
    "Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0042\n"
    "Invoice Date: 17/04/2026\n"
    "Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Sl No 01 Laptop Stand\n"
    "Qty 4 NOS Rate 1500.00\n"
    "Taxable 6000.00 GST 18%\n"
    "CGST 540.00 SGST 540.00 IGST 0.00\n"
    "HSN/SAC: 8471\n"
    "Goods once sold will not be taken back or exchanged\n"
    "Total Invoice Value 7080.00\n"
)


def test_positional_party_fallback_warns_even_when_it_succeeds(tmp_path):
    path = _text_pdf(tmp_path / "unlabelled.pdf", UNLABELLED_TEXT)
    result = extract_invoice(path, client=FakeClient(CLEAN_RESPONSE))

    # The fallback succeeds: the topmost GSTIN is the seller, as printed.
    assert result.extraction is not None
    assert result.extraction.invoice.SellerDtls.Gstin == "27AAPFU0939F1ZV"
    assert result.extraction.invoice.BuyerDtls.Gstin == "27AABCB5507N1ZJ"
    # And it still warns, because getting it backwards inverts who owes the tax.
    assert any(
        w.check == "extract_rules" and w.field in {"SellerDtls.Gstin", "BuyerDtls.Gstin"}
        for w in result.meta.warnings
    )


def test_labelled_parties_do_not_raise_the_positional_warning(clean_result):
    result, _ = clean_result
    assert not any(
        w.check == "extract_rules" and w.field in {"SellerDtls.Gstin", "BuyerDtls.Gstin"}
        for w in result.meta.warnings
    )


# --- validators -----------------------------------------------------------------

# Same invoice, but the printed grand total is 7800.00 where the parts add to 7080.00.
# The invoice-level total is transposed to 7800.00 while the item row still prints its
# own 7080.00. Both numbers must stay on the page: stage 2 grounds every value against
# the document, so replacing the only printed 7080.00 would (correctly) drop the item's
# TotItemVal as invented and the invoice would never reach the validators at all.
TRANSPOSED_TOTAL_TEXT = CLEAN_TEXT.replace(
    "Total Invoice Value 7080.00",
    "Item Total 7080.00\nTotal Invoice Value 7800.00",
)


def _transposed_response() -> dict:
    response = json.loads(json.dumps(CLEAN_RESPONSE))
    response["totals"]["TotInvVal"] = 7800.00
    return response


def test_an_inconsistent_total_is_reported_by_the_validators_but_still_extracted(tmp_path):
    # 6000.00 + 540.00 + 540.00 = 7080.00, not the 7800.00 the document prints.
    path = _text_pdf(tmp_path / "bad_total.pdf", TRANSPOSED_TOTAL_TEXT)
    result = extract_invoice(path, client=FakeClient(_transposed_response()))
    assert result.extraction is not None
    assert result.extraction.invoice.ValDtls.TotInvVal == 7800.00
    assert any(w.check == "validate_invoice_total" for w in result.meta.warnings)


# Three arithmetic disagreements at once, so the forwarded block is a list and not a
# single warning that any ordering would satisfy. Transcribed by hand from the text:
#   row:     4 x 1500.00 = 6000.00 taxable, 18% intra-state -> 540.00 + 540.00, so the
#            row adds to 7080.00 and the page prints 7000.00 -- item total disagrees.
#   totals:  the CGST total prints 500.00 where the only row's CGST is 540.00 --
#            ValDtls sums disagree.
#   invoice: 6000.00 + 500.00 + 540.00 + 0.00 = 7040.00 and the page prints 7800.00 --
#            invoice total disagrees.
# The tax split itself is right (540.00 each on 6000.00 at 18%, intra-state 27 to 27),
# so that check stays quiet and the block is not simply "everything failed".
MANY_DISAGREEMENTS_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0047\n"
    "Invoice Date: 22/04/2026\n"
    "Buyer: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Sl No 01 Laptop Stand\n"
    "Qty 4 NOS Rate 1500.00\n"
    "Taxable 6000.00 GST 18%\n"
    "CGST 540.00 SGST 540.00 IGST 0.00\n"
    "Line Total 7000.00\n"
    "HSN/SAC: 8471\n"
    "Taxable Value Total 6000.00\n"
    "CGST Total 500.00 SGST Total 540.00 IGST Total 0.00\n"
    "Total Invoice Value 7800.00\n"
)

MANY_DISAGREEMENTS_RESPONSE = {
    "items": [
        {
            "SlNo": "01",
            "PrdDesc": "Laptop Stand",
            "HsnCd": "8471",
            "Qty": 4,
            "Unit": "NOS",
            "UnitPrice": 1500.00,
            "TotAmt": 6000.00,
            "Discount": None,
            "AssAmt": 6000.00,
            "GstRt": 18,
            "CgstAmt": 540.00,
            "SgstAmt": 540.00,
            "IgstAmt": 0.00,
            "CesAmt": None,
            "TotItemVal": 7000.00,
        }
    ],
    "seller": {
        "LglNm": "Nimbus Components Pvt Ltd",
        "Addr1": "Plot 14 MIDC Andheri East",
        "Loc": "Mumbai",
        "Pin": "400093",
    },
    "buyer": {
        "LglNm": "Kanchan Electricals LLP",
        "Addr1": "221 Laxmi Road Shivajinagar",
        "Loc": "Pune",
        "Pin": "411005",
    },
    "totals": {
        "AssVal": 6000.00,
        "CgstVal": 500.00,
        "SgstVal": 540.00,
        "IgstVal": 0.00,
        "CesVal": None,
        "TotInvVal": 7800.00,
        "RndOffAmt": None,
    },
}


def test_the_forwarded_validator_block_is_what_build_one_returns_for_this_invoice(tmp_path):
    # "At least one validator warning arrived" would pass on a pipeline that forwarded
    # the first and dropped the rest, which on a document disagreeing with itself in
    # several places is most of the finding gone. So the block is compared against
    # build 1's own validate_invoice, run here on the invoice the pipeline emitted:
    # the pipeline owns the forwarding, not the wording or the count, and this pins
    # the forwarding without restating a single message of build 1's.
    path = _text_pdf(tmp_path / "many_disagreements.pdf", MANY_DISAGREEMENTS_TEXT)
    result = extract_invoice(path, client=FakeClient(MANY_DISAGREEMENTS_RESPONSE))
    assert result.missing_fields == ()
    assert result.extraction is not None

    expected = validate_invoice(result.extraction.invoice, 0.05)
    # The fixture really does disagree in more than one place, so the comparison has
    # something to be wrong about.
    assert len({w.check for w in expected}) > 1
    assert [w for w in result.meta.warnings if w.check in VALIDATOR_CHECKS] == expected
    # And they arrive last, as one unbroken block in build 1's own order.
    assert result.meta.warnings[-len(expected) :] == expected


def test_the_tolerance_argument_reaches_the_validators(tmp_path):
    path = _text_pdf(tmp_path / "bad_total.pdf", TRANSPOSED_TOTAL_TEXT)
    result = extract_invoice(path, client=FakeClient(_transposed_response()), tolerance=1000.0)
    assert result.extraction is not None
    assert [w for w in result.meta.warnings if w.check in VALIDATOR_CHECKS] == []


def test_warnings_run_stage_one_then_stage_two_then_pipeline_then_validators(tmp_path):
    path = _text_pdf(tmp_path / "ordered.pdf", TRANSPOSED_TOTAL_TEXT)
    result = extract_invoice(path, client=FakeClient(_transposed_response()))
    checks = [w.check for w in result.meta.warnings]
    last_stage = max(i for i, c in enumerate(checks) if c in {"extract_rules", "extract_llm"})
    first_pipeline = min(i for i, c in enumerate(checks) if c == "pipeline")
    first_validator = min(i for i, c in enumerate(checks) if c in VALIDATOR_CHECKS)
    assert last_stage < first_pipeline < first_validator


def test_stage_one_warnings_come_before_stage_two_warnings(tmp_path):
    # The test above collapses both stages into one group, so the one boundary of the
    # specified order it cannot see is Part B's warnings before Part C's. This fixture
    # raises both: stage 1 warns because it disambiguated the parties by position, and
    # stage 2 warns about the fields the canned response leaves null.
    path = _text_pdf(tmp_path / "both_stages.pdf", UNLABELLED_TEXT)
    result = extract_invoice(path, client=FakeClient(CLEAN_RESPONSE))
    checks = [w.check for w in result.meta.warnings]
    assert "extract_rules" in checks
    assert "extract_llm" in checks
    last_rules = max(i for i, c in enumerate(checks) if c == "extract_rules")
    first_llm = min(i for i, c in enumerate(checks) if c == "extract_llm")
    assert last_rules < first_llm


# --- refusals -------------------------------------------------------------------

EXPORT_TEXT = CLEAN_TEXT.replace("TAX INVOICE", "EXPORT INVOICE")

# The other refusal kind, on a text layer: an ISO currency code beside a number is
# what build 1 refuses, and 2950.00 is not a number the rupee fixture prints, so the
# refusal cannot be read as a rupee figure that merely moved.
FOREIGN_CURRENCY_TEXT = CLEAN_TEXT.replace(
    "Total Invoice Value 7080.00", "Grand Total USD 2950.00"
)


@pytest.mark.parametrize(
    ("text", "kind", "field"),
    [
        (EXPORT_TEXT, "export_sez", "TranDtls.SupTyp"),
        (FOREIGN_CURRENCY_TEXT, "multi_currency", "ValDtls.TotInvValFc"),
    ],
    ids=["export_sez", "multi_currency"],
)
def test_a_refused_document_stops_before_extraction(tmp_path, text, kind, field):
    # Both of build 1's refusal kinds take the same path out of the pipeline, and both
    # have to: the contract's licence to hard-code SupTyp rests on the first, and the
    # assumption that every amount is in rupees rests on the second.
    client = FakeClient(CLEAN_RESPONSE)
    path = _text_pdf(tmp_path / "refused.pdf", text)
    result = extract_invoice(path, client=client)

    assert result.extraction is None
    assert result.missing_fields == ()
    assert [r.kind for r in result.refusals] == [kind]
    assert result.meta.pages == [PageMeta(page=1, method="native")]
    assert result.meta.field_provenance == {}
    # The refusal message is the actionable output, so it has to reach the warnings.
    refusal_warnings = [w for w in result.meta.warnings if w.field == field]
    assert len(refusal_warnings) == 1
    assert result.refusals[0].message in refusal_warnings[0].message
    # And no money is spent asking a model about a document build 1 already refused.
    assert client.calls == []


# --- OCR routes -----------------------------------------------------------------


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    """One scanned page carrying the whole clean invoice. OCR runs once for the module."""
    path = _mixed_pdf(tmp_path_factory.mktemp("scan") / "scanned.pdf", ("image", CLEAN_TEXT))
    client = FakeClient(CLEAN_RESPONSE)
    return path, extract_invoice(path, client=client), client


def test_a_derived_field_does_not_also_claim_it_was_left_empty(tmp_path):
    """Stage 2's absence note promises the field was left empty. Deriving it breaks
    that promise, so the derivation note supersedes it rather than printing beside it."""
    text = CLEAN_TEXT.replace(" IGST 0.00", "")
    response = json.loads(json.dumps(CLEAN_RESPONSE))
    response["items"][0]["IgstAmt"] = None
    response["totals"]["IgstVal"] = None
    path = _text_pdf(tmp_path / "no_igst_column.pdf", text)
    result = extract_invoice(path, client=FakeClient(response))
    for field in ("ItemList[0].IgstAmt", "ValDtls.IgstVal"):
        notes = [w for w in result.meta.warnings if w.field == field]
        assert notes, f"{field} was derived silently"
        assert not any("was not found in the document" in w.message for w in notes)
        assert any(w.check == "pipeline" for w in notes)
        assert result.meta.field_provenance[field]["source"] == "derived"


@requires_tesseract
def test_a_scanned_page_keeps_the_ocr_required_method(scanned):
    _, result, _ = scanned
    assert result.meta.pages == [PageMeta(page=1, method="ocr_required")]


@requires_tesseract
def test_a_scanned_page_produces_the_same_payload_as_its_text_layer_twin(scanned):
    _, result, _ = scanned
    assert result.missing_fields == ()
    assert result.extraction.invoice.model_dump(exclude_none=True) == EXPECTED_CLEAN_PAYLOAD


@requires_tesseract
def test_a_scanned_page_carries_ocr_confidence_into_provenance(scanned):
    _, result, _ = scanned
    for path_name in ("SellerDtls.Gstin", "DocDtls.No", "SellerDtls.LglNm", "ItemList[0].PrdDesc"):
        confidence = result.meta.field_provenance[path_name]["ocr_confidence"]
        assert isinstance(confidence, float), path_name
        assert 0.0 <= confidence <= 1.0, path_name


@requires_tesseract
def test_the_recorded_confidence_is_the_one_tesseract_reported_for_that_word(scanned):
    # Read independently from ocr.py rather than from the pipeline's own bookkeeping:
    # the point of per-field confidence is that a shaky read is not laundered on the way
    # into provenance, so the number recorded must be the number Tesseract gave.
    path, result, _ = scanned
    page = ocr_pdf_page(path, 1)
    words = [w for w in page.words if w.text == "27AAPFU0939F1ZV"]
    assert len(words) == 1
    assert result.meta.field_provenance["SellerDtls.Gstin"]["ocr_confidence"] == words[0].confidence


@requires_tesseract
def test_a_derived_state_code_keeps_the_confidence_of_the_gstin_it_came_from(scanned):
    # "derived" is a claim about HOW the value was obtained, not about whether any
    # text was read: the two characters it was derived from were read off the scan,
    # so a shaky read of the GSTIN must still travel with the state code taken from
    # it. Recording None here would hide the one thing that can go wrong with a
    # derived state code -- the digits it was derived from being misread.
    path, result, _ = scanned
    page = ocr_pdf_page(path, 1)
    seller = [w for w in page.words if w.text == "27AAPFU0939F1ZV"]
    buyer = [w for w in page.words if w.text == "27AABCB5507N1ZJ"]
    assert len(seller) == 1 and len(buyer) == 1
    assert result.meta.field_provenance["SellerDtls.Stcd"] == {
        "source": "derived",
        "ocr_confidence": seller[0].confidence,
    }
    assert result.meta.field_provenance["BuyerDtls.Stcd"] == {
        "source": "derived",
        "ocr_confidence": buyer[0].confidence,
    }


@requires_tesseract
def test_a_bare_image_file_is_read_end_to_end(tmp_path):
    # Not a PDF at all: an accountant photographs an invoice and hands over the photo,
    # which is the commonest way a scan reaches this tool. Every fixture above wraps
    # its scan in a PDF, so nothing else exercises the path where ingest is handed an
    # image extension and the pipeline has to rasterise the file itself.
    path = _image_file(tmp_path / "invoice_photo.png", CLEAN_TEXT)
    client = FakeClient(CLEAN_RESPONSE)
    result = extract_invoice(path, client=client)

    assert result.refusals == ()
    assert result.meta.pages == [PageMeta(page=1, method="ocr_required")]
    assert result.missing_fields == ()
    assert result.extraction.invoice.model_dump(exclude_none=True) == EXPECTED_CLEAN_PAYLOAD
    # And it is genuinely the OCR route: the fields carry a confidence, which a native
    # text layer never provides.
    assert isinstance(result.meta.field_provenance["SellerDtls.Gstin"]["ocr_confidence"], float)


# Page 1 is a real text layer, page 2 is a scan: the two routes in one document.
MIXED_PAGE_ONE = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0042\n"
    "Invoice Date: 17/04/2026\n"
    "Buyer: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Page 1 of 2 continued overleaf\n"
)

MIXED_PAGE_TWO = (
    "Sl No 01 Laptop Stand\n"
    "Qty 4 NOS Rate 1500.00\n"
    "Taxable 6000.00 GST 18%\n"
    "CGST 540.00 SGST 540.00 IGST 0.00\n"
    "HSN/SAC: 8471\n"
    "Goods once sold will not be taken back or exchanged\n"
    "Total Invoice Value 7080.00\n"
)


@pytest.fixture(scope="module")
def multi_page(tmp_path_factory):
    path = _mixed_pdf(
        tmp_path_factory.mktemp("mixed") / "multi.pdf",
        ("text", MIXED_PAGE_ONE),
        ("image", MIXED_PAGE_TWO),
    )
    client = FakeClient(CLEAN_RESPONSE)
    return path, extract_invoice(path, client=client), client


@requires_tesseract
def test_a_multi_page_invoice_records_both_routing_methods(multi_page):
    _, result, _ = multi_page
    assert result.meta.pages == [
        PageMeta(page=1, method="native"),
        PageMeta(page=2, method="ocr_required"),
    ]


@requires_tesseract
def test_a_multi_page_invoice_joins_both_pages_into_one_payload(multi_page):
    _, result, _ = multi_page
    assert result.missing_fields == ()
    assert result.extraction.invoice.model_dump(exclude_none=True) == EXPECTED_CLEAN_PAYLOAD


@requires_tesseract
def test_confidence_follows_the_page_a_field_was_read_from(multi_page):
    # The invoice number and the seller's name are printed on the native page, so they
    # have no OCR confidence; the line item is on the scan, so it does. Getting the
    # page offsets wrong would swap these two.
    _, result, _ = multi_page
    provenance = result.meta.field_provenance
    assert provenance["DocDtls.No"]["ocr_confidence"] is None
    assert provenance["SellerDtls.Gstin"]["ocr_confidence"] is None
    assert provenance["SellerDtls.LglNm"]["ocr_confidence"] is None
    assert isinstance(provenance["ItemList[0].PrdDesc"]["ocr_confidence"], float)
    assert isinstance(provenance["ValDtls.TotInvVal"]["ocr_confidence"], float)


@pytest.fixture(scope="module")
def scanned_export(tmp_path_factory):
    path = _mixed_pdf(tmp_path_factory.mktemp("refuse") / "scanned_export.pdf", ("image", EXPORT_TEXT))
    client = FakeClient(CLEAN_RESPONSE)
    return path, extract_invoice(path, client=client), client


@requires_tesseract
def test_an_export_invoice_is_still_refused_when_only_the_scan_reveals_it(scanned_export):
    # Build 1 runs its detectors on native pages only and hands the scanned ones to
    # build 2. Without this, a scanned export invoice would be extracted as domestic
    # B2B -- which is the assumption the pipeline's hard-coded SupTyp rests on.
    _, result, client = scanned_export
    assert result.extraction is None
    assert [r.kind for r in result.refusals] == ["export_sez"]
    assert client.calls == []
    warning = next(w for w in result.meta.warnings if w.field == "TranDtls.SupTyp")
    assert "OCR" in warning.message


@requires_tesseract
def test_a_stage_two_field_carries_the_confidence_of_its_own_words(scanned):
    # The test above pins the stage-1 path, which knows a field's exact offsets. A
    # stage-2 field has no offsets -- it is located by searching the page for the text
    # as printed -- so it needs its own pin, to the same standard: the number recorded
    # is the one Tesseract reported for the item's own words, read here from ocr.py
    # rather than from the pipeline's bookkeeping. "Laptop Stand" is two words and a
    # field is only as good as its worst-read one, so the lower of the two is expected.
    path, result, _ = scanned
    page = ocr_pdf_page(path, 1)
    confidences = [w.confidence for w in page.words if w.text in {"Laptop", "Stand"}]
    assert len(confidences) == 2
    recorded = result.meta.field_provenance["ItemList[0].PrdDesc"]["ocr_confidence"]
    assert recorded == min(confidences)


# The quantity is 3 and the invoice number VT/2026/0311 contains a 3 as well. Stage 2
# grounds Qty against the printed token "3", so the pipeline has to find *that* token:
# a bare substring search finds the 3 inside the invoice number too, and would record
# a confidence belonging to a word the quantity was never read from.
# Arithmetic, by hand: 3 x 1500.00 = 4500.00 taxable; 18% intra-state is 405.00 CGST
# and 405.00 SGST; 4500 + 405 + 405 = 5310.00 for the row and for the invoice.
SUBSTRING_TRAP_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: VT/2026/0311\n"
    "Invoice Date: 17/04/2026\n"
    "Buyer: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Sl No 01 Laptop Stand\n"
    "Qty 3 NOS Rate 1500.00\n"
    "Taxable 4500.00 GST 18%\n"
    "CGST 405.00 SGST 405.00 IGST 0.00\n"
    "HSN/SAC: 8471\n"
    "Total Invoice Value 5310.00\n"
)

SUBSTRING_TRAP_RESPONSE = {
    "items": [
        {
            "SlNo": "01",
            "PrdDesc": "Laptop Stand",
            "HsnCd": "8471",
            "Qty": 3,
            "Unit": "NOS",
            "UnitPrice": 1500.00,
            "TotAmt": 4500.00,
            "Discount": None,
            "AssAmt": 4500.00,
            "GstRt": 18,
            "CgstAmt": 405.00,
            "SgstAmt": 405.00,
            "IgstAmt": 0.00,
            "CesAmt": None,
            "TotItemVal": 5310.00,
        }
    ],
    "seller": {
        "LglNm": "Nimbus Components Pvt Ltd",
        "Addr1": "Plot 14 MIDC Andheri East",
        "Loc": "Mumbai",
        "Pin": "400093",
    },
    "buyer": {
        "LglNm": "Kanchan Electricals LLP",
        "Addr1": "221 Laxmi Road Shivajinagar",
        "Loc": "Pune",
        "Pin": "411005",
    },
    "totals": {
        "AssVal": 4500.00,
        "CgstVal": 405.00,
        "SgstVal": 405.00,
        "IgstVal": 0.00,
        "CesVal": None,
        "TotInvVal": 5310.00,
        "RndOffAmt": None,
    },
}


@pytest.fixture(scope="module")
def substring_trap(tmp_path_factory):
    path = _mixed_pdf(tmp_path_factory.mktemp("trap") / "trap.pdf", ("image", SUBSTRING_TRAP_TEXT))
    return path, extract_invoice(path, client=FakeClient(SUBSTRING_TRAP_RESPONSE))


@requires_tesseract
def test_a_short_value_takes_the_confidence_of_the_token_it_is_printed_as(substring_trap):
    # A substring hit inside a longer word is not a place the value is printed, and
    # the confidence of an unrelated word is worse than no confidence at all: it can
    # report a clean read as shaky or -- the direction that matters -- a shaky read as
    # clean. The quantity's confidence must be the one its own "3" was read at.
    path, result = substring_trap
    page = ocr_pdf_page(path, 1)
    standalone = [w for w in page.words if w.text == "3"]
    assert len(standalone) == 1
    # The trap really is on the page: another word has a 3 inside it.
    assert any("3" in w.text and w.text != "3" for w in page.words)

    assert result.missing_fields == ()
    assert result.extraction.invoice.ItemList[0].Qty == 3
    recorded = result.meta.field_provenance["ItemList[0].Qty"]["ocr_confidence"]
    assert recorded == standalone[0].confidence


@pytest.fixture(scope="module")
def blank_second_page(tmp_path_factory):
    """A readable native page plus a scanned page OCR recovers nothing from."""
    path = _mixed_pdf(
        tmp_path_factory.mktemp("blank") / "blank.pdf",
        ("text", UNLABELLED_TEXT),
        ("image", "   \n"),
    )
    return path, extract_invoice(path, client=FakeClient(CLEAN_RESPONSE))


@requires_tesseract
def test_an_ingest_derived_note_comes_before_the_stage_warnings(blank_second_page):
    # The specified order starts with the ingest-derived notes, and the ordering test
    # above cannot see that segment: its fixture produces no such note. A blank scanned
    # page does -- and the note matters most exactly here, because it says a field
    # reported missing below may simply be printed on the page nothing was read from.
    _, result = blank_second_page
    warnings = result.meta.warnings
    ingest_note = [i for i, w in enumerate(warnings) if w.field == "page[2]"]
    assert len(ingest_note) == 1
    first_rules = min(i for i, w in enumerate(warnings) if w.check == "extract_rules")
    first_stage = min(
        i for i, w in enumerate(warnings) if w.check in {"extract_rules", "extract_llm"}
    )
    assert ingest_note[0] < first_rules
    assert ingest_note[0] < first_stage
    # It is labelled like the refusal warnings that share its segment. Labelling it
    # "pipeline" would put a pipeline warning ahead of the stage warnings for any
    # document with a blank scanned page, which is the opposite of the documented
    # order and would make that order unrecoverable from the warnings themselves.
    assert warnings[ingest_note[0]].check == "ingest"
    last_stage = max(
        i for i, w in enumerate(warnings) if w.check in {"extract_rules", "extract_llm"}
    )
    first_pipeline = min(i for i, w in enumerate(warnings) if w.check == "pipeline")
    assert last_stage < first_pipeline


@pytest.fixture(scope="module")
def scanned_reverse_charge(tmp_path_factory):
    path = _mixed_pdf(
        tmp_path_factory.mktemp("rcm") / "scanned_rcm.pdf",
        ("image", CLEAN_TEXT + "Reverse Charge Applicable: Yes\n"),
    )
    return path, extract_invoice(path, client=FakeClient(CLEAN_RESPONSE))


@requires_tesseract
def test_a_reverse_charge_declaration_only_the_scan_reveals_still_sets_regrev(
    scanned_reverse_charge,
):
    # Build 1 runs its detectors on native pages only and says in its own docstring
    # that build 2 re-runs them on OCR output. Without that, this document -- which
    # declares reverse charge on its face -- would be filed with RegRev absent, saying
    # the opposite of what it prints.
    path, result = scanned_reverse_charge
    assert ingest(path).reverse_charge is False
    assert result.missing_fields == ()
    assert result.extraction.invoice.TranDtls.RegRev == "Y"
    # RegRev="Y" makes build 1's tax-split check stand down -- but it says so itself
    # rather than falling silent, so the accountant can see which check did not run.
    skipped = [
        w
        for w in result.meta.warnings
        if w.check == "validate_tax_split" and w.field == "TranDtls.RegRev"
    ]
    assert len(skipped) == 1
    assert skipped[0].severity == "info"


@pytest.fixture(scope="module")
def scanned_foreign_currency(tmp_path_factory):
    path = _mixed_pdf(
        tmp_path_factory.mktemp("fx") / "scanned_fx.pdf",
        ("image", CLEAN_TEXT.replace("Total Invoice Value 7080.00", "Grand Total USD 7080.00")),
    )
    client = FakeClient(CLEAN_RESPONSE)
    return path, extract_invoice(path, client=client), client


@requires_tesseract
def test_a_foreign_currency_invoice_is_still_refused_when_only_the_scan_reveals_it(
    scanned_foreign_currency,
):
    # The same re-run that catches a scanned export invoice catches this one. Build 1
    # alone sees nothing here, and every amount would then be parsed as rupees.
    path, result, client = scanned_foreign_currency
    assert ingest(path).refusals == []
    assert result.extraction is None
    assert result.missing_fields == ()
    assert [r.kind for r in result.refusals] == ["multi_currency"]
    assert client.calls == []
    warning = next(w for w in result.meta.warnings if w.field == "ValDtls.TotInvValFc")
    assert "OCR" in warning.message
    assert result.refusals[0].message in warning.message
    # The evidence is the ISO code beside a number, and that is exactly the part of
    # build 1's detector that OCR'd text keeps: only the bare glyph is exempted, so a
    # scanned foreign-currency invoice is refused where its text-layer twin is.
    assert "USD" in result.refusals[0].evidence


# The same invoice as the clean fixture, scanned, plus a footer line carrying a bare
# currency glyph. Rupee amounts throughout: the glyph is the whole of the evidence.
GLYPH_FOOTER_TEXT = CLEAN_TEXT + "Bank charges $ 0.00 recovered separately\n"


@pytest.fixture(scope="module")
def scanned_currency_glyph(tmp_path_factory):
    path = _mixed_pdf(
        tmp_path_factory.mktemp("glyph") / "scanned_glyph.pdf",
        ("image", GLYPH_FOOTER_TEXT),
    )
    client = FakeClient(CLEAN_RESPONSE)
    return path, extract_invoice(path, client=client), client


@requires_tesseract
def test_a_currency_glyph_on_a_scan_warns_instead_of_discarding_the_invoice(
    scanned_currency_glyph,
):
    # Build 1 refuses a document on a bare currency glyph, which is right for a text
    # layer, where the glyph really is printed. On OCR'd text it is not evidence of
    # anything: Tesseract returns "$55" for a printed "S55" and "\u00a350.00" for a
    # printed "850.00", and nothing here can tell an invented glyph from the printed one
    # in this fixture. Refusing would hand the accountant no payload plus a statement,
    # asserted as fact, that an invoice the tool read correctly quotes a foreign
    # currency. So the glyph is reported and the invoice is still extracted.
    path, result, client = scanned_currency_glyph
    assert detect_multi_currency("Bank charges $ 0.00 recovered separately") is not None
    assert [w.text for w in ocr_pdf_page(path, 1).words if "$" in w.text] == ["$"]

    assert ingest(path).refusals == []
    assert result.refusals == ()
    assert result.missing_fields == ()
    assert client.calls != []
    assert result.extraction is not None
    invoice = result.extraction.invoice
    assert invoice.SellerDtls.Gstin == "27AAPFU0939F1ZV"
    assert invoice.BuyerDtls.Gstin == "27AABCB5507N1ZJ"
    assert invoice.DocDtls.No == "INV-2026-0042"
    assert invoice.ValDtls.TotInvVal == 7080.00

    # The glyph is still reported -- in the ingest-derived segment, as the caveat it is.
    note = [w for w in result.meta.warnings if w.field == "ValDtls.TotInvValFc"]
    assert len(note) == 1
    assert note[0].check == "ingest"
    assert note[0].severity == "warning"
    assert '"$"' in note[0].message
    assert "parsed as rupees" in note[0].message
    # Wording, an exchange rate or an ISO code beside an amount still refuses outright:
    # that is what the USD fixture above pins, on the same OCR path.


@pytest.fixture(scope="module")
def refused_with_a_glyph(tmp_path_factory):
    """A text layer that refuses outright, plus a scanned page carrying a bare glyph."""
    path = _mixed_pdf(
        tmp_path_factory.mktemp("refused_glyph") / "refused_glyph.pdf",
        ("text", FOREIGN_CURRENCY_TEXT),
        ("image", GLYPH_FOOTER_TEXT),
    )
    return path, extract_invoice(path, client=FakeClient(CLEAN_RESPONSE))


@requires_tesseract
def test_a_blanked_glyph_is_reported_even_when_the_document_is_refused_anyway(
    refused_with_a_glyph,
):
    # The narrowing above is the one place this module withholds evidence from a build 1
    # detector, so the note saying so must not be conditional on anything: whenever a
    # glyph was blanked it is reported, including on a document something else already
    # refused. Otherwise the exemption's footprint is unrecoverable from the output, and
    # a reader cannot tell a page whose glyph was set aside from one that had none.
    path, result = refused_with_a_glyph
    assert [w.text for w in ocr_pdf_page(path, 2).words if "$" in w.text] == ["$"]
    assert [r.kind for r in result.refusals] == ["multi_currency"]

    reported = [w for w in result.meta.warnings if w.field == "ValDtls.TotInvValFc"]
    glyph_notes = [w for w in reported if "OCR read a currency symbol" in w.message]
    assert len(glyph_notes) == 1
    assert glyph_notes[0].check == "ingest"
    assert '"$"' in glyph_notes[0].message
    assert "page 2" in glyph_notes[0].message
    # Build 1's refusal is still there beside it, and is still the actionable output.
    assert any(result.refusals[0].message in w.message for w in reported)
    # Nothing was parsed from this document, so the note does not claim any figure was
    # read as rupees -- that sentence would be false on a document with no payload.
    assert "parsed as rupees" not in glyph_notes[0].message


# --- derived per-line total on a single-row invoice --------------------------------

# CLEAN_TEXT prints a per-item taxable value, the tax split and a document-level total,
# but no per-row total column -- the shape a real single-line invoice usually has, since
# the per-row total and the invoice total are the same number and the template prints it
# once. CLEAN_RESPONSE answers TotItemVal anyway, so these fixtures blank it to model what
# a model reading this document actually returns: null for a column that is not there.
#
# Arithmetic transcribed by hand from CLEAN_TEXT: taxable 6000.00, CGST 540.00, SGST
# 540.00, IGST 0.00, no cess, so AssAmt + CgstAmt + SgstAmt + IgstAmt + CesAmt +
# StateCesAmt + OthChrg = 6000 + 540 + 540 + 0 + 0 + 0 + 0 = 7080.00, and the document
# prints "Total Invoice Value 7080.00". The two agree, which is what licenses the rule.


def _clean_response_without_line_total():
    response = json.loads(json.dumps(CLEAN_RESPONSE))
    response["items"][0]["TotItemVal"] = None
    return response


@pytest.fixture
def unprinted_line_total(tmp_path):
    path = _text_pdf(tmp_path / "no_line_total.pdf", CLEAN_TEXT)
    return extract_invoice(path, client=FakeClient(_clean_response_without_line_total()))


def test_a_single_row_invoice_derives_its_unprinted_line_total(unprinted_line_total):
    result = unprinted_line_total
    assert result.missing_fields == ()
    assert isinstance(result.extraction, ExtractionResult)
    # 6000.00 + 540.00 + 540.00 + 0.00, hand-added from CLEAN_TEXT above.
    assert result.extraction.invoice.ItemList[0].TotItemVal == 7080.00
    # The derived value is the whole payload's only difference from the printed-total run.
    assert result.extraction.invoice.model_dump(exclude_none=True) == EXPECTED_CLEAN_PAYLOAD
    # And it satisfies the identity the validator enforces, so nothing is flagged.
    assert [w for w in result.meta.warnings if w.check in VALIDATOR_CHECKS] == []


def test_the_derived_line_total_is_recorded_as_derived_not_read(unprinted_line_total):
    result = unprinted_line_total
    assert result.meta.field_provenance["ItemList[0].TotItemVal"]["source"] == "derived"
    # The amounts it was derived from are still stage 2 readings, so they stay "llm".
    for path in ("ItemList[0].AssAmt", "ItemList[0].CgstAmt", "ValDtls.TotInvVal"):
        assert result.meta.field_provenance[path]["source"] == "llm"


def test_the_derived_line_total_carries_a_note_naming_its_identity(unprinted_line_total):
    notes = [
        w for w in unprinted_line_total.meta.warnings
        if w.field == "ItemList[0].TotItemVal" and w.check == "pipeline"
    ]
    assert len(notes) == 1
    note = notes[0]
    assert note.severity == "info"
    assert "TotItemVal = AssAmt + CgstAmt + SgstAmt + IgstAmt" in note.message
    assert "derived, not read from the page" in note.message
    # The note must say the reconciliation happened and what it reconciled against,
    # because that is the condition the reader has to be able to check.
    assert "exactly one line item" in note.message
    assert "ValDtls.TotInvVal" in note.message
    assert "7080.00" in note.message


def test_the_derivation_note_supersedes_stage_twos_absence_note(unprinted_line_total):
    """Stage 2's note promises the field was left empty rather than filled with a guess.
    Deriving it makes that promise false, so the note is superseded, not printed beside."""
    notes = [
        w for w in unprinted_line_total.meta.warnings
        if w.field == "ItemList[0].TotItemVal"
    ]
    assert notes, "the field was derived silently"
    assert not any("was not found in the document" in w.message for w in notes)


def test_a_line_total_that_does_not_reconcile_is_not_derived(tmp_path):
    """A single-row invoice whose derived total disagrees with its own printed total is
    evidence the reading is wrong. The rule stands down and the refusal stands."""
    # The document now foots to 9000.00 while its own line arithmetic makes 7080.00.
    text = CLEAN_TEXT.replace("Total Invoice Value 7080.00", "Total Invoice Value 9000.00")
    response = _clean_response_without_line_total()
    response["totals"]["TotInvVal"] = 9000.00
    path = _text_pdf(tmp_path / "bad_foot.pdf", text)
    result = extract_invoice(path, client=FakeClient(response))

    assert result.extraction is None
    assert result.missing_fields == ("ItemList[0].TotItemVal",)
    # Nothing may claim the field was derived, in the payload metadata or the warnings.
    assert "ItemList[0].TotItemVal" not in result.meta.field_provenance
    derivation_notes = [
        w for w in result.meta.warnings
        if w.field == "ItemList[0].TotItemVal" and "derived, not read" in w.message
    ]
    assert derivation_notes == []


def test_a_two_item_invoice_never_derives_a_missing_line_total(tmp_path):
    """The case that must not regress. On a multi-row invoice, a per-row total the
    document never states is a split across rows it never states either, however
    obvious the arithmetic looks from the totals block."""
    response = json.loads(json.dumps(MIXED_RATES_RESPONSE))
    response["items"][1]["TotItemVal"] = None
    path = _text_pdf(tmp_path / "mixed_no_line_total.pdf", MIXED_RATES_TEXT)
    result = extract_invoice(path, client=FakeClient(response))

    assert result.extraction is None
    assert result.missing_fields == ("ItemList[1].TotItemVal",)
    assert "ItemList[1].TotItemVal" not in result.meta.field_provenance
    # Row 0 keeps its printed total and is not dragged into the refusal.
    assert result.meta.field_provenance["ItemList[0].TotItemVal"]["source"] == "llm"


def test_a_printed_line_total_is_read_and_never_derived(clean_result):
    """The fourth case: when the document does print the column, it is read normally
    and the rule does not fire. Provenance is the stage that read it."""
    result, _ = clean_result
    assert result.meta.field_provenance["ItemList[0].TotItemVal"]["source"] == "llm"
    assert result.extraction.invoice.ItemList[0].TotItemVal == 7080.00
    pipeline_notes = [
        w for w in result.meta.warnings
        if w.field == "ItemList[0].TotItemVal" and w.check == "pipeline"
    ]
    assert pipeline_notes == []
