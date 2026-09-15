"""Tests for gst_einvoice.extract_rules (stage 1, deterministic).

Expected values are transcribed by hand from the fixture text; spans are located
with str.index on the fixture, never with the module under test.
"""

import dataclasses

import pytest

from gst_einvoice.extract_rules import (
    DATE_LABEL_TIERS,
    DOC_NO_LABEL_PATTERN,
    DOC_NO_SERIAL_SEPARATORS,
    DOC_NO_TOKEN_PATTERN,
    HSN_LABEL_PATTERN,
    RuleExtraction,
    RuleField,
    Span,
    _BUYER_PATTERNS,
    _SELLER_PATTERNS,
    extract_rules,
    remaining_text,
)
from gst_einvoice.gstin import validate_gstin
from gst_einvoice.schema import ExtractionWarning

# --- fixture GSTINs (build 1 ground truth, plus three computed once and pinned) ---

MAHARASHTRA = "27AAPFU0939F1ZV"  # state code 27
DELHI = "07AAGFF2194N1Z1"  # state code 07
GUJARAT = "24AAACC1206D1ZM"  # state code 24
UTTAR_PRADESH = "09AAACH7409R1ZZ"  # state code 09
BAD_CHECKSUM = "29AAACT2727Q1ZW"  # structurally fine, checksum wrong
DISCONTINUED_25 = "25AAPFU0939F1ZZ"  # checksum-valid, state code 25 (discontinued)
UNKNOWN_88 = "88AAPFU0939F1ZN"  # checksum-valid, state code 88 (not a GST state code)
LEGACY_28 = "28AAGFF2194N1ZX"  # checksum-valid, state code 28 (legacy Andhra Pradesh)

CHECKSUM_VALID = (
    MAHARASHTRA,
    DELHI,
    GUJARAT,
    UTTAR_PRADESH,
    DISCONTINUED_25,
    UNKNOWN_88,
    LEGACY_28,
)


def test_fixture_gstins_are_what_the_tests_assume():
    """Pin the fixtures against build 1's validator before relying on them."""
    for gstin in CHECKSUM_VALID:
        assert validate_gstin(gstin).is_valid, gstin
    bad = validate_gstin(BAD_CHECKSUM)
    assert bad.structural_ok is True
    assert bad.checksum_ok is False


def warnings_for(result: RuleExtraction, field: str) -> list[ExtractionWarning]:
    return [w for w in result.warnings if w.field == field]


WRAPPED_LABEL_PHRASE = "wrapped across a line break"


def wrapped_label_warnings(result: RuleExtraction) -> list[ExtractionWarning]:
    """The DocDtls.No warnings reporting an invoice-number label split across lines."""
    return [w for w in warnings_for(result, "DocDtls.No") if WRAPPED_LABEL_PHRASE in w.message]


# --- a clean, fully labelled invoice ----------------------------------------

CLEAN_INVOICE = (
    "TAX INVOICE\n"
    "Seller: Umang Fabrics Private Limited\n"
    "12 Kalbadevi Road, Mumbai, Maharashtra 400002\n"
    "GSTIN: 27AAPFU0939F1ZV\n"
    "\n"
    "Buyer: Farheen Traders\n"
    "9 Chandni Chowk, Delhi 110006\n"
    "GSTIN: 07AAGFF2194N1Z1\n"
    "\n"
    "Invoice No: INV/2026-27/0042\n"
    "Invoice Date: 25/04/2026\n"
    "\n"
    "Sl  Description       Qty      Rate        Amount\n"
    "1   Consulting fee    HSN 998313    2 Nos    5,000.00    10,000.00\n"
    "2   Cotton fabric     HSN 52081190  10 Mtr   1,200.00    12,000.00\n"
    "3   Consulting fee    HSN 998313    1 Nos    5,000.00     5,000.00\n"
)


def test_clean_invoice_fields():
    result = extract_rules(CLEAN_INVOICE)
    assert result.seller_gstin == RuleField(
        "27AAPFU0939F1ZV",
        Span(CLEAN_INVOICE.index("27AAPFU0939F1ZV"), CLEAN_INVOICE.index("27AAPFU0939F1ZV") + 15),
        "label",
    )
    assert result.buyer_gstin == RuleField(
        "07AAGFF2194N1Z1",
        Span(CLEAN_INVOICE.index("07AAGFF2194N1Z1"), CLEAN_INVOICE.index("07AAGFF2194N1Z1") + 15),
        "label",
    )
    assert result.doc_no == RuleField(
        "INV/2026-27/0042",
        Span(
            CLEAN_INVOICE.index("INV/2026-27/0042"),
            CLEAN_INVOICE.index("INV/2026-27/0042") + 16,
        ),
        "regex",
    )
    assert result.doc_date == RuleField(
        "25/04/2026",
        Span(CLEAN_INVOICE.index("25/04/2026"), CLEAN_INVOICE.index("25/04/2026") + 10),
        "regex",
    )
    assert [f.value for f in result.hsn_codes] == ["998313", "52081190"]
    assert result.seller_stcd == RuleField(
        "27", Span(CLEAN_INVOICE.index("27AAPFU0939F1ZV"), CLEAN_INVOICE.index("27AAPFU0939F1ZV") + 2), "derived"
    )
    assert result.buyer_stcd == RuleField(
        "07", Span(CLEAN_INVOICE.index("07AAGFF2194N1Z1"), CLEAN_INVOICE.index("07AAGFF2194N1Z1") + 2), "derived"
    )


def test_clean_invoice_raises_no_warnings():
    """Labels resolved both parties, the date is unambiguous, both state codes are active."""
    assert extract_rules(CLEAN_INVOICE).warnings == ()


def test_clean_invoice_spans_index_back_to_the_text():
    result = extract_rules(CLEAN_INVOICE)
    for field in (result.seller_gstin, result.buyer_gstin, result.doc_no, *result.hsn_codes):
        assert CLEAN_INVOICE[field.span.start : field.span.end] == field.value
    # The date span covers the raw text; the value is the normalised form.
    assert CLEAN_INVOICE[result.doc_date.span.start : result.doc_date.span.end] == "25/04/2026"
    assert CLEAN_INVOICE[result.seller_stcd.span.start : result.seller_stcd.span.end] == "27"


def test_clean_invoice_pin_codes_are_not_mistaken_for_hsn():
    values = [f.value for f in extract_rules(CLEAN_INVOICE).hsn_codes]
    assert "400002" not in values
    assert "110006" not in values


# --- result types -----------------------------------------------------------


def test_result_types_are_frozen_dataclasses_with_the_contract_fields():
    assert [f.name for f in dataclasses.fields(Span)] == ["start", "end"]
    assert [f.name for f in dataclasses.fields(RuleField)] == ["value", "span", "method"]
    assert [f.name for f in dataclasses.fields(RuleExtraction)] == [
        "seller_gstin",
        "buyer_gstin",
        "seller_stcd",
        "buyer_stcd",
        "doc_no",
        "doc_date",
        "hsn_codes",
        "consumed",
        "warnings",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Span(0, 1).start = 5


def test_every_warning_is_tagged_for_this_stage():
    texts = (
        CLEAN_INVOICE,
        "",
        f"GSTIN {BAD_CHECKSUM}\n",
        f"GSTIN {DISCONTINUED_25}\nInvoice Date 03/04/2026\n",
    )
    for text in texts:
        for warning in extract_rules(text).warnings:
            assert warning.check == "extract_rules"
            assert warning.severity == "warning"
            assert warning.message


# --- GSTIN candidates -------------------------------------------------------


def rejected_candidate_warnings(result: RuleExtraction, field: str) -> list[ExtractionWarning]:
    return [w for w in warnings_for(result, field) if "failed validation" in w.message]


def test_checksum_failure_is_not_used_and_is_reported():
    text = (
        "Seller: Acme\n"
        f"GSTIN: {BAD_CHECKSUM}\n"
        "Buyer: Beta\n"
        f"GSTIN: {DELHI}\n"
    )
    result = extract_rules(text)
    assert result.seller_gstin is None or result.seller_gstin.value != BAD_CHECKSUM
    assert result.buyer_gstin is None or result.buyer_gstin.value != BAD_CHECKSUM
    # The candidate was never attributed to a party, so it is reported against both.
    for field in ("SellerDtls.Gstin", "BuyerDtls.Gstin"):
        rejected = rejected_candidate_warnings(result, field)
        assert len(rejected) == 1
        message = rejected[0].message
        assert BAD_CHECKSUM in message
        assert str(text.index(BAD_CHECKSUM)) in message
        assert "checksum mismatch" in message


def test_every_warning_names_an_inv01_field_path():
    """A warning filed under a leaf like 'Gstin' is invisible to a consumer grouping by path."""
    texts = (
        CLEAN_INVOICE,
        "",
        f"Seller: Acme\nGSTIN: {BAD_CHECKSUM}\nBuyer: Beta\nGSTIN: {DELHI}\n",
        f"GSTIN {DISCONTINUED_25}\nInvoice Date 03/04/2026\nInvoice No.   Date\n",
    )
    allowed = {
        "SellerDtls.Gstin",
        "BuyerDtls.Gstin",
        "SellerDtls.Stcd",
        "BuyerDtls.Stcd",
        "DocDtls.No",
        "DocDtls.Dt",
    }
    for text in texts:
        for warning in extract_rules(text).warnings:
            assert warning.field in allowed, warning.field


def test_a_rejected_candidate_leaves_the_remaining_gstin_as_the_only_party():
    text = (
        f"GSTIN: {BAD_CHECKSUM}\n"
        "Seller: Acme\n"
        f"GSTIN: {MAHARASHTRA}\n"
        "Buyer: Beta\n"
        f"GSTIN: {DELHI}\n"
    )
    result = extract_rules(text)
    assert result.seller_gstin.value == MAHARASHTRA
    assert result.buyer_gstin.value == DELHI


@pytest.mark.parametrize(
    "text",
    [
        "GSTIN: 27aapfu0939f1zv\n",  # lowercase is not a GSTIN
        "GSTIN: 127AAPFU0939F1ZV\n",  # 16 characters, no word boundary
        "GSTIN: 27AAPFU0939F1Z\n",  # 14 characters
        "GSTIN: 27AAPFU0939F1YV\n",  # position 14 is not 'Z'
    ],
)
def test_non_gstin_shaped_text_is_not_a_candidate(text):
    result = extract_rules(text)
    assert result.seller_gstin is None
    assert result.buyer_gstin is None
    assert rejected_candidate_warnings(result, "SellerDtls.Gstin") == []
    assert rejected_candidate_warnings(result, "BuyerDtls.Gstin") == []


# --- party disambiguation by label ------------------------------------------


@pytest.mark.parametrize("label", ["Seller", "From", "Supplier", "Sold By", "Consignor", "Vendor"])
def test_every_seller_label(label):
    text = f"{label}: Umang Fabrics\nGSTIN: {MAHARASHTRA}\nBill To: Farheen\nGSTIN: {DELHI}\n"
    result = extract_rules(text)
    assert result.seller_gstin == RuleField(
        MAHARASHTRA, Span(text.index(MAHARASHTRA), text.index(MAHARASHTRA) + 15), "label"
    )
    assert result.buyer_gstin.value == DELHI
    assert warnings_for(result, "SellerDtls.Gstin") == []


@pytest.mark.parametrize(
    "label",
    ["Buyer", "Bill To", "Billed To", "Ship To", "Shipped To", "Recipient", "Consignee", "Customer", "To"],
)
def test_every_buyer_label(label):
    text = f"Seller: Umang Fabrics\nGSTIN: {MAHARASHTRA}\n{label}: Farheen\nGSTIN: {DELHI}\n"
    result = extract_rules(text)
    assert result.seller_gstin.value == MAHARASHTRA
    assert result.buyer_gstin == RuleField(
        DELHI, Span(text.index(DELHI), text.index(DELHI) + 15), "label"
    )
    assert warnings_for(result, "SellerDtls.Gstin") == []


NEAREST_LABEL_WINS = (
    "Supplier: Umang Fabrics\n"
    "Bill To: Farheen Traders\n"
    f"GSTIN: {DELHI}\n" + "-" * 140 + "\n"
    f"Supplier GSTIN: {MAHARASHTRA}\n"
)


def test_nearest_label_wins_even_when_it_reverses_document_order():
    """'Bill To' ends nearer the first GSTIN than 'Supplier', so that GSTIN is the buyer."""
    result = extract_rules(NEAREST_LABEL_WINS)
    assert result.seller_gstin.value == MAHARASHTRA
    assert result.seller_gstin.method == "label"
    assert result.buyer_gstin.value == DELHI
    assert result.buyer_gstin.method == "label"
    assert warnings_for(result, "SellerDtls.Gstin") == []


def test_bare_to_at_a_line_start_is_a_buyer_label():
    text = (
        f"Seller: Umang Fabrics\nGSTIN: {MAHARASHTRA}\n" + "x" * 140 + "\n"
        f"To: Farheen Traders\nGSTIN: {DELHI}\n"
    )
    result = extract_rules(text)
    assert result.buyer_gstin.method == "label"
    assert result.buyer_gstin.value == DELHI


def test_bare_to_inside_prose_is_not_a_buyer_label():
    text = (
        f"Seller: Umang Fabrics\nGSTIN: {MAHARASHTRA}\n" + "x" * 140 + "\n"
        f"Goods delivered to the site\nGSTIN: {DELHI}\n"
    )
    result = extract_rules(text)
    assert result.buyer_gstin.method == "position"
    assert len(warnings_for(result, "SellerDtls.Gstin")) == 1


def test_from_inside_prose_is_not_a_seller_label():
    """'collected from our depot' is prose; taking it as a label inverts who owes the tax."""
    text = (
        "Farheen Traders\n"
        "9 Chandni Chowk, Delhi 110006\n"
        "Goods to be collected from our depot\n"
        f"GSTIN: {DELHI}\n"
        "Bill To: Umang Fabrics Pvt Ltd\n"
        "12 Kalbadevi Road, Mumbai 400002\n"
        f"GSTIN: {MAHARASHTRA}\n"
    )
    result = extract_rules(text)
    assert result.seller_gstin.method == "position"
    assert result.buyer_gstin.method == "position"
    assert len(warnings_for(result, "SellerDtls.Gstin")) == 1


def test_a_bare_to_after_a_pipe_separator_is_a_buyer_label():
    text = (
        f"Sold By: Umang Fabrics\nGSTIN: {MAHARASHTRA}\n" + "x" * 140 + "\n"
        f"Consignment | To: Farheen Traders\nGSTIN: {DELHI}\n"
    )
    result = extract_rules(text)
    assert result.buyer_gstin == RuleField(
        DELHI, Span(text.index(DELHI), text.index(DELHI) + 15), "label"
    )
    assert warnings_for(result, "SellerDtls.Gstin") == []


@pytest.mark.parametrize(
    ("seller_line", "buyer_line"),
    [
        ("Sold By: Umang Fabrics", "Despatch: To: Farheen Traders"),
        ("Sold By: Umang Fabrics", "Despatch:To: Farheen Traders"),
        ("Ref: From: Umang Fabrics", "Bill To: Farheen Traders"),
    ],
)
def test_a_bare_label_after_a_colon_separator_qualifies(seller_line, buyer_line):
    """The contract names ':' beside '|' and the line start; only ':' had no test.

    Losing ':' from the separators would leave 'Despatch: To: Farheen' resolving by
    position instead of by label — silently swapping the parties on any document whose
    party labels sit after a colon rather than at a line start.
    """
    text = (
        f"{seller_line}\nGSTIN: {MAHARASHTRA}\n" + "x" * 140 + f"\n{buyer_line}\nGSTIN: {DELHI}\n"
    )
    result = extract_rules(text)
    assert result.seller_gstin == RuleField(
        MAHARASHTRA, Span(text.index(MAHARASHTRA), text.index(MAHARASHTRA) + 15), "label"
    )
    assert result.buyer_gstin == RuleField(
        DELHI, Span(text.index(DELHI), text.index(DELHI) + 15), "label"
    )
    assert warnings_for(result, "SellerDtls.Gstin") == []


def _seller_label_at_distance(distance: int) -> str:
    """Fixture whose 'Seller:' label starts exactly `distance` characters before the GSTIN."""
    label = "Seller:"
    return (
        label
        + "." * (distance - len(label))
        + MAHARASHTRA
        + f"\nBill To: Farheen Traders\nGSTIN: {DELHI}\n"
    )


@pytest.mark.parametrize(("distance", "method"), [(120, "label"), (121, "position")])
def test_a_label_counts_only_within_a_hundred_and_twenty_characters(distance, method):
    """The window is 120 characters from the label's start to the GSTIN's start."""
    text = _seller_label_at_distance(distance)
    assert text.index(MAHARASHTRA) - text.index("Seller:") == distance
    result = extract_rules(text)
    assert result.seller_gstin.value == MAHARASHTRA
    assert result.seller_gstin.method == method
    assert len(warnings_for(result, "SellerDtls.Gstin")) == (0 if method == "label" else 1)


def test_to_inside_a_longer_label_does_not_count_as_a_bare_to():
    """'Bill To' must resolve as the buyer label, not as two competing labels."""
    text = f"Seller: Umang\nGSTIN: {MAHARASHTRA}\nBill To: Farheen\nGSTIN: {DELHI}\n"
    result = extract_rules(text)
    assert result.buyer_gstin.method == "label"


def test_both_gstins_labelled_seller_falls_back_to_position():
    text = f"Seller: Umang\nGSTIN: {MAHARASHTRA}\nSupplier: Farheen\nGSTIN: {DELHI}\n"
    result = extract_rules(text)
    assert result.seller_gstin.method == "position"
    assert result.buyer_gstin.method == "position"
    assert len(warnings_for(result, "SellerDtls.Gstin")) == 1


def test_the_same_gstin_labelled_both_ways_falls_back_to_position():
    text = f"Seller: Umang\nGSTIN: {MAHARASHTRA}\nBill To: Umang\nGSTIN: {MAHARASHTRA}\n"
    result = extract_rules(text)
    assert result.seller_gstin.value == MAHARASHTRA
    assert result.seller_gstin.method == "position"
    assert result.buyer_gstin is None


# --- party disambiguation by position ---------------------------------------

POSITIONAL_INVOICE = (
    "Umang Fabrics Private Limited\n"
    f"GSTIN {MAHARASHTRA}\n"
    "Farheen Traders\n"
    f"GSTIN {DELHI}\n"
    "Invoice No 7788\n"
    "Invoice Date 25/04/2026\n"
)


def test_positional_fallback_assigns_reading_order():
    result = extract_rules(POSITIONAL_INVOICE)
    assert result.seller_gstin.value == MAHARASHTRA
    assert result.seller_gstin.method == "position"
    assert result.buyer_gstin.value == DELHI
    assert result.buyer_gstin.method == "position"


def test_positional_fallback_always_warns_including_on_success():
    """The success path is the dangerous one: it must never resolve silently."""
    result = extract_rules(POSITIONAL_INVOICE)
    seller_warnings = warnings_for(result, "SellerDtls.Gstin")
    assert len(seller_warnings) == 1
    message = seller_warnings[0].message
    assert MAHARASHTRA in message and "seller" in message
    assert DELHI in message and "buyer" in message
    assert "position" in message
    assert "label" in message
    assert "inverts who owes the tax" in message
    assert "confirm" in message.lower()


def test_one_gstin_only_goes_to_the_seller_and_warns_twice():
    text = f"Umang Fabrics Private Limited\nGSTIN {MAHARASHTRA}\nInvoice No 7788\n"
    result = extract_rules(text)
    assert result.seller_gstin.value == MAHARASHTRA
    assert result.seller_gstin.method == "position"
    assert result.buyer_gstin is None
    assert result.buyer_stcd is None
    assert len(warnings_for(result, "SellerDtls.Gstin")) == 1
    buyer_warnings = warnings_for(result, "BuyerDtls.Gstin")
    assert len(buyer_warnings) == 1
    assert "one checksum-valid GSTIN" in buyer_warnings[0].message


def test_the_same_gstin_twice_is_one_distinct_gstin():
    text = f"Umang Fabrics\nGSTIN {MAHARASHTRA}\nAlso GSTIN {MAHARASHTRA}\n"
    result = extract_rules(text)
    assert result.seller_gstin.value == MAHARASHTRA
    assert result.buyer_gstin is None
    assert len(warnings_for(result, "BuyerDtls.Gstin")) == 1


def test_a_lone_valid_gstin_labelled_buyer_goes_to_the_seller_but_the_warning_says_so():
    """Contract step 4 assigns it to the seller; the warning must not hide the 'Buyer' label."""
    text = f"Seller: Acme\nGSTIN: {BAD_CHECKSUM}\nBuyer: Beta\nGSTIN: {DELHI}\n"
    result = extract_rules(text)
    assert result.seller_gstin.value == DELHI
    assert result.seller_gstin.method == "position"
    seller_warnings = [
        w for w in warnings_for(result, "SellerDtls.Gstin") if "from position" in w.message
    ]
    assert len(seller_warnings) == 1
    message = seller_warnings[0].message
    assert f"labels GSTIN {DELHI} as the buyer" in message
    assert "contradicts" in message


def test_no_valid_gstin_leaves_both_parties_unset_with_one_warning_each():
    text = f"Umang Fabrics\nGSTIN {BAD_CHECKSUM}\nInvoice No 7788\n"
    result = extract_rules(text)
    assert result.seller_gstin is None
    assert result.buyer_gstin is None
    assert result.seller_stcd is None
    assert result.buyer_stcd is None
    for field in ("SellerDtls.Gstin", "BuyerDtls.Gstin"):
        missing = [
            w for w in warnings_for(result, field) if "No checksum-valid GSTIN" in w.message
        ]
        assert len(missing) == 1
        # plus the rejected candidate that explains why there is none
        assert len(rejected_candidate_warnings(result, field)) == 1
        assert len(warnings_for(result, field)) == 2


def test_three_gstins_take_the_first_two_distinct_by_position():
    text = f"A\nGSTIN {MAHARASHTRA}\nB\nGSTIN {DELHI}\nC\nGSTIN {GUJARAT}\n"
    result = extract_rules(text)
    assert result.seller_gstin.value == MAHARASHTRA
    assert result.buyer_gstin.value == DELHI


# --- invoice number ---------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    ["Invoice No", "Invoice Number", "Bill No", "Bill Number", "Tax Invoice No", "Inv No"],
)
def test_every_doc_no_label(label):
    text = f"{label}: INV-0042\n"
    result = extract_rules(text)
    assert result.doc_no == RuleField(
        "INV-0042", Span(text.index("INV-0042"), text.index("INV-0042") + 8), "regex"
    )
    assert warnings_for(result, "DocDtls.No") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice No: INV-0042\n", "INV-0042"),
        ("Invoice No. 0042\n", "0042"),
        ("Invoice No.: INV/2026-27/0042\n", "INV/2026-27/0042"),
        ("Invoice No - 5\n", "5"),
        ("Invoice No #A7\n", "A7"),
        ("Invoice # 0042\n", "0042"),
        ("Invoice#42\n", "42"),
        ("INVOICE NUMBER : ABC123\n", "ABC123"),
    ],
)
def test_doc_no_separators_and_token(text, expected):
    assert extract_rules(text).doc_no.value == expected


def test_doc_no_token_stops_at_a_space():
    result = extract_rules("Invoice No: INV-0042 dated 25/04/2026\n")
    assert result.doc_no.value == "INV-0042"


def test_doc_no_takes_the_first_labelled_match():
    text = "Invoice No: FIRST-1\nBill No: SECOND-2\n"
    result = extract_rules(text)
    assert result.doc_no.value == "FIRST-1"


# --- a reference to another document above the invoice's own header ---------

# Every one of these prints a reference to ANOTHER document on a line above the
# invoice's own header, and every reference line carries an invoice-number label from
# the contract's own list. The contract takes the FIRST label in the document, so the
# value filed as the mandatory DocDtls.No is the referenced document's number. The
# values below are transcribed by hand from the fixture text.
REFERENCE_LINE_ABOVE_HEADER = (
    (
        "Ref: Your Bill No. 7788 dated 01-Apr-2026\n"
        "Umang Fabrics Private Limited\n"
        "Invoice No. INV-0042        Dated 25-Apr-2026\n",
        "7788",
        "INV-0042",
    ),
    (
        "Against Invoice No. 0031 dated 01-Apr-2026\n"
        "Tax Invoice No. INV-0042\n"
        "Invoice Date: 25-Apr-2026\n",
        "0031",
        "INV-0042",
    ),
    (
        "Ref our Inv No SFPL/24-25/0700 of 01/04/2026\n"
        "Invoice No: SFPL/24-25/0731\n"
        "Invoice Date: 25/04/2026\n",
        "SFPL/24-25/0700",
        "SFPL/24-25/0731",
    ),
    (
        # the e-Way bill number, printed above the header by a large share of Indian
        # templates: another document's number under a label from the same list
        "e-Way Bill No. 391000123456\nInvoice No. INV-0042   Dated 25-Apr-2026\n",
        "391000123456",
        "INV-0042",
    ),
    (
        "Tax Invoice\ne-Way Bill No 391000123456    Invoice No INV-0042\n",
        "391000123456",
        "INV-0042",
    ),
    (
        "P.O. Bill No 9001 dated 20-Mar-2026\nTax Invoice No 4455\n",
        "9001",
        "4455",
    ),
)


@pytest.mark.parametrize(("text", "taken", "competing"), REFERENCE_LINE_ABOVE_HEADER)
def test_a_second_invoice_number_label_carrying_a_different_value_is_never_silent(
    text, taken, competing
):
    """The invoice-number half of the competing-date warning, for the same document shape.

    On this document the date side already warns that the date may be the referenced
    document's. Filing the referenced document's NUMBER with nothing to show for it is
    the same wrong value in the same mandatory field, so it must be just as visible.
    """
    result = extract_rules(text)
    assert result.doc_no.value == taken
    messages = [w.message for w in warnings_for(result, "DocDtls.No")]
    assert any(f"'{taken}'" in m and f"'{competing}'" in m for m in messages), messages
    assert any("Confirm which of these is this invoice's own number" in m for m in messages)


@pytest.mark.parametrize(
    "text",
    [
        # a repeated multi-page header carries the same value, so nothing competes
        "Invoice No. INV-0042   Dated 25-Apr-2026\npage 1 of 2\n"
        "Invoice No. INV-0042   Dated 25-Apr-2026\n",
        # 'e-Way Bill No.' under an 'Invoice No.' header: the number taken is right,
        # and a warning here would cost the warnings that matter their credibility
        "Invoice No. INV-0042   Dated 25-Apr-2026\ne-Way Bill No. 391000123456\n",
        "Tax Invoice No. INV-0042\ne-Way Bill No. 391000123456\n",
        "Inv No. INV-0042\ne-Way Bill No. 391000123456\n",
        # a label that supplies no value of its own competes with nothing
        "Invoice No. INV-0042\nBill No.\n",
        # another document's number under a label that is NOT in the contract's list
        "Delivery Challan No. 55 dated 01-Apr-2026\nInvoice No. INV-0042\n",
    ],
)
def test_a_second_label_that_settles_nothing_does_not_warn(text):
    """A warning raised on a correct reading costs the ones that matter their weight."""
    result = extract_rules(text)
    assert result.doc_no.value in ("INV-0042", "391000123456")
    assert warnings_for(result, "DocDtls.No") == []


def test_the_competing_label_warning_names_both_labels_and_both_values():
    text = "Ref: Your Bill No. 7788 dated 01-Apr-2026\nInvoice No. INV-0042\n"
    result = extract_rules(text)
    assert result.doc_no.value == "7788"
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert "'Bill No'" in message and "'7788'" in message
    assert "'Invoice No'" in message and "'INV-0042'" in message
    assert f"character {text.index('Invoice No')}" in message
    assert doc_no_warnings[0].field == "DocDtls.No"
    assert doc_no_warnings[0].check == "extract_rules"


def test_a_doc_no_label_with_no_value_below_it_is_reported_not_guessed():
    """The label may not reach down the page: two lines below is a company name."""
    text = "Tax Invoice\nInvoice No:\n\n\nUmang Fabrics Pvt Ltd\n"
    result = extract_rules(text)
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert "Invoice No" in message
    assert "not followed by a value on its own line or in its column on the line below" in message


def test_an_over_long_doc_no_token_is_truncated_and_says_so():
    text = "Invoice Number: INV-" + "7" * 35 + "\n"
    result = extract_rules(text)
    assert result.doc_no.value == "INV-" + "7" * 26
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    assert "30 characters" in doc_no_warnings[0].message


# --- invoice number in a column layout (label row above value row) ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "TAX INVOICE\n"
            "Invoice No.            Invoice Date            Place of Supply\n"
            "INV/2026-27/0042       25/04/2026              Maharashtra\n",
            "INV/2026-27/0042",
        ),
        ("Invoice No.      Dated\nINV-0042         25/04/2026\n", "INV-0042"),
        ("Invoice No   Date   Amount\n0042   25/04/2026   11800.00\n", "0042"),
        (
            "Invoice Number       Invoice Date        Due Date\n"
            "AB/2026/119          25/04/2026          30/05/2026\n",
            "AB/2026/119",
        ),
        ("Bill No     Bill Date\n7788        25/04/2026\n", "7788"),
        # OCR collapses column runs to a single space; the columns still line up.
        ("Invoice No. Invoice Date\nINV/2026-27/0042 25/04/2026\n", "INV/2026-27/0042"),
        # the label is the second column: the value under it, not the first one
        ("Date         Invoice No.\n25/04/2026   INV-0042\n", "INV-0042"),
        ("Date Invoice No.\n25/04/2026 INV-0042\n", "INV-0042"),
    ],
)
def test_a_column_header_layout_takes_the_value_from_the_row_below(text, expected):
    result = extract_rules(text)
    assert result.doc_no is not None, "the invoice number was not found at all"
    assert result.doc_no.value == expected
    assert text[result.doc_no.span.start : result.doc_no.span.end] == expected
    assert result.doc_no.method == "column"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # a label with nothing under it, and a second label that does have a value
        ("Invoice No.   Place of Supply\nMaharashtra\nBill No: 7788\n", "7788"),
        # the label's own line still wins over the line below
        ("Invoice No: INV-1\n9988 something\n", "INV-1"),
    ],
)
def test_a_value_beside_its_own_label_is_method_regex_and_silent(text, expected):
    """The same-line path is the safe one: the label names the value it sits beside."""
    result = extract_rules(text)
    assert result.doc_no.value == expected
    assert result.doc_no.method == "regex"
    assert warnings_for(result, "DocDtls.No") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice No.      Dated\nINV-0042         25/04/2026\n", "INV-0042"),
        ("Bill No     Bill Date\n7788        25/04/2026\n", "7788"),
    ],
)
def test_the_row_below_always_warns_including_when_it_lands_on_the_right_value(text, expected):
    """Deliberately the parties' positional-fallback rule: layout chose it, so say so."""
    result = extract_rules(text)
    assert result.doc_no.value == expected
    assert result.doc_no.method == "column"
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert expected in message
    assert "row below" in message
    assert "confirm" in message.lower()


def test_a_value_from_the_row_below_is_distinguishable_from_one_beside_the_label():
    """Same value, same label, two layouts: the method and the warning must differ."""
    beside = extract_rules("Invoice No: 7788\n")
    below = extract_rules("Invoice No\n7788\n")
    assert beside.doc_no.value == below.doc_no.value == "7788"
    assert beside.doc_no.method == "regex"
    assert below.doc_no.method == "column"
    assert warnings_for(beside, "DocDtls.No") == []
    assert len(warnings_for(below, "DocDtls.No")) == 1


@pytest.mark.parametrize(
    "text",
    [
        "Invoice No.            Invoice Date            Place of Supply\nINV-1   25/04/2026\n",
        "Invoice No.      Dated\nINV-0042         25/04/2026\n",
        "Invoice No   Date   Amount\n0042   25/04/2026   11800.00\n",
        "Invoice No:      Date: 25/04/2026\n",
        "Invoice Number       Invoice Date        Due Date\nAB/2026/119   25/04/2026\n",
        "Bill No     Bill Date\n7788        25/04/2026\n",
    ],
)
def test_a_neighbouring_column_heading_is_never_the_invoice_number(text):
    """'Invoice', 'Date', 'Dated', 'Bill' filed as DocDtls.No is a confidently wrong value."""
    doc_no = extract_rules(text).doc_no
    value = None if doc_no is None else doc_no.value
    assert value not in {"Invoice", "Date", "Dated", "Bill", "No", "Number", "Due", "Place"}


def test_a_heading_with_no_number_under_it_is_reported_not_guessed():
    text = "Invoice No:      Date: 25/04/2026\n"
    result = extract_rules(text)
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert "'Date'" in message
    assert "column heading" in message
    assert "enter it manually" in message


@pytest.mark.parametrize(
    "text",
    [
        "Invoice No.\n\nINV-0042\n",  # two lines down is out of reach
        "Invoice No.\nUmang Fabrics Pvt Ltd\n",  # a company name is not a number
        "Invoice No.\n              25/04/2026\n",  # a different column
        "Invoice No.\n25/04/2026\n",  # a date is not an invoice number
    ],
)
def test_the_line_below_is_not_scraped_for_anything_that_is_not_a_number(text):
    result = extract_rules(text)
    assert result.doc_no is None
    assert len(warnings_for(result, "DocDtls.No")) == 1


@pytest.mark.parametrize(
    ("text", "why"),
    [
        (f"Invoice No.\nGSTIN: {MAHARASHTRA}\n", "already matched as a GSTIN"),
        ("Invoice No.\n25.04.2026\n", "a component of a dotted date"),
        ("Invoice No.\n11800.00\n", "a component of a decimal amount"),
        ("Invoice No.\n2 Nos\n", "a quantity carrying a unit"),
    ],
)
def test_the_row_below_refuses_a_value_the_document_identifies_as_something_else(text, why):
    """DocDtls.No is mandatory: a GSTIN, half a number or a quantity filed there is wrong."""
    result = extract_rules(text)
    assert result.doc_no is None, why
    assert len(warnings_for(result, "DocDtls.No")) == 1


@pytest.mark.parametrize(
    ("text", "token", "why"),
    [
        (f"Invoice No.\nGSTIN: {MAHARASHTRA}\n", MAHARASHTRA, "already matched as a GSTIN"),
        ("Invoice No.\n25/04/2026\n", "25/04/2026", "is a date"),
        ("Invoice No.\n11800.00\n", "11800", "one run of a longer number"),
        ("Invoice No.\n2 Nos\n", "2", "quantity carrying a unit"),
    ],
)
def test_the_row_below_refusal_names_what_it_found_rather_than_an_empty_column(text, token, why):
    """A value WAS printed in that column and refused; saying it was empty misdirects."""
    result = extract_rules(text)
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert f"'{token}'" in message
    assert why in message
    assert "no value was printed in the label's column" not in message
    assert "enter it manually" in message


def test_a_column_that_really_is_empty_is_still_reported_as_empty():
    """The other half of the discrimination: nothing under the label means nothing.

    The company name is printed to the RIGHT of the label's own column, which is what
    makes that column empty. Printed under the label it is a value in the cell — not
    one that can be taken, but the message must not call the cell empty either.
    """
    message = warnings_for(
        extract_rules("Invoice No.\n              Umang Fabrics Pvt Ltd\n"), "DocDtls.No"
    )[0].message
    assert "not followed by a value on its own line or in its column on the line below" in message
    assert "row below prints" not in message


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("Invoice No.\nSFPL-EXP\n", "SFPL-EXP"),  # a letters-only serial, refused but printed
        ("Invoice No.   Dated\nSFPL-EXP      25-Apr-2026\n", "SFPL-EXP"),
        ("Invoice No.\nUmang Fabrics Pvt Ltd\n", "Umang"),  # a company name, likewise printed
    ],
)
def test_a_word_in_the_labels_column_is_reported_rather_than_called_an_empty_cell(text, token):
    """A digit-free token under the label is refused, but the cell is not empty.

    The row below is searched for a number because nothing on that row names the
    value, so a word there can never be taken. It IS printed in the label's own
    column, though, and an accountant told that column was empty is sent to look at a
    cell that holds something.
    """
    result = extract_rules(text)
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert f"row below prints '{token}' in the label's column" in message
    assert "carries no digit" in message
    assert "no value was printed in the label's column" not in message


def test_a_refused_number_under_the_label_outranks_a_word_beside_it():
    """'GSTIN: 27AAPFU0939F1ZV' under the label is reported by the GSTIN, not by 'GSTIN'."""
    message = warnings_for(
        extract_rules(f"Invoice No.\nGSTIN: {MAHARASHTRA}\n"), "DocDtls.No"
    )[0].message
    assert f"'{MAHARASHTRA}'" in message
    assert "already matched as a GSTIN" in message


def test_both_halves_of_the_refusal_name_what_they_found():
    """Beside the label and under it, each half reports its own token and reason."""
    text = "Invoice No.   Dated\n11,800.00     25/04/2026\n"
    result = extract_rules(text)
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert "'Dated'" in message and "column heading" in message
    assert "'11'" in message and "one run of a longer number" in message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice No.\n400002 Mumbai\n", "400002"),
        ("Invoice No.\n9820012345\n", "9820012345"),
    ],
)
def test_a_doubtful_value_from_the_row_below_is_emitted_but_never_silently(text, expected):
    """A PIN code and a phone number are not refusable by rule, so they must be visible."""
    result = extract_rules(text)
    assert result.doc_no.value == expected
    assert result.doc_no.method == "column"
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    assert "row below" in doc_no_warnings[0].message


@pytest.mark.parametrize(("text", "expected"), [
    ("Invoice No: ABC/XYZ\n", "ABC/XYZ"),
    ("Invoice No: SFPL-EXP\n", "SFPL-EXP"),
])
def test_a_digit_free_serial_is_accepted(text, expected):
    """Rule 46(b) allows letters and special characters, so no digit is not a refusal."""
    result = extract_rules(text)
    assert result.doc_no == RuleField(
        expected, Span(text.index(expected), text.index(expected) + len(expected)), "regex"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice No: ABC/XYZ\n", "ABC/XYZ"),
        ("Invoice No: SFPL-EXP\n", "SFPL-EXP"),
    ],
)
def test_a_digit_free_serial_is_accepted_but_never_silently(text, expected):
    """Nothing on the line tells a letters-only serial from the next column's heading."""
    result = extract_rules(text)
    assert result.doc_no.value == expected
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert f"'{expected}'" in message
    assert "carries no digit" in message
    assert "confirm" in message.lower()


@pytest.mark.parametrize(
    ("text", "printed", "why"),
    [
        ("Invoice No: ABC/XYZ\n25/04/2026\n", "25/04/2026", "is a date"),
        ("Invoice No: ABC/XYZ\n11,800.00 total\n", "11", "one run of a longer number"),
        (f"Invoice No: ABC/XYZ\n{MAHARASHTRA}\n", MAHARASHTRA, "already matched as a GSTIN"),
    ],
)
def test_the_digit_free_refusal_names_what_the_row_below_printed(text, printed, why):
    """The same discrimination as the row-below refusal, on the path that takes a value.

    This message told the accountant that no number was printed in the label's column
    on the line below, on documents where one WAS printed there and refused. Sending
    them to look at a cell that is not empty is the misdirection the row-below half
    already refuses to make.
    """
    result = extract_rules(text)
    assert result.doc_no.value == "ABC/XYZ"
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert f"row below prints '{printed}' in the label's own column" in message
    assert why in message
    assert "no number was printed" not in message
    assert "no value was printed in the label's own column" not in message


def test_a_digit_free_serial_with_a_genuinely_empty_column_below_still_says_so():
    """The other half again: nothing under the label must not be reported as something."""
    message = warnings_for(extract_rules("Invoice No: ABC/XYZ\n"), "DocDtls.No")[0].message
    assert "no value was printed in the label's own column on the line below" in message
    assert "row below prints" not in message


# A dozen words Indian invoices print in the column beside the invoice number,
# transcribed from the headers of Tally, Zoho and Vyapar layouts rather than from
# the module's own list of refusal words: the point is the words it does NOT know.
COLUMN_HEADINGS_BESIDE_THE_INVOICE_NUMBER = (
    "e-Way Bill No.",
    "Transport Mode",
    "Party Name",
    "Terms of Payment",
    "Buyer Order No.",
    "Delivery Note",
    "Dispatched Through",
    "Destination",
    "Mode/Terms of Payment",
    "Currency",
    "Status",
    "GST-Rate",
)


@pytest.mark.parametrize("heading", COLUMN_HEADINGS_BESIDE_THE_INVOICE_NUMBER)
def test_a_blank_cell_beside_the_label_never_files_a_heading_silently(heading):
    """A heading the refusal list cannot know is either not taken, or taken out loud."""
    text = f"Invoice No.   {heading}\nSFPL/24-25/0731   something\n"
    result = extract_rules(text)
    value = None if result.doc_no is None else result.doc_no.value
    assert value != heading.split()[0]
    assert len(warnings_for(result, "DocDtls.No")) >= 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # the header a Tally e-way-bill layout prints; the serial is in the label's
        # own column on the next line, and 'e-Way' is the next column's heading
        (
            "Invoice No.              e-Way Bill No.             Dated\n"
            "SFPL/24-25/0731          3312 4455 6677             25-Apr-2026\n",
            "SFPL/24-25/0731",
        ),
        ("Invoice No.        Transport Mode\nINV-0042           By Road\n", "INV-0042"),
        ("Bill No.  Party Name\n7788   Umang Fabrics\n", "7788"),
    ],
)
def test_a_digit_free_heading_never_beats_the_number_in_the_labels_own_column(text, expected):
    """The blank-cell layout, on the invoice-number side: the serial below wins."""
    result = extract_rules(text)
    assert result.doc_no is not None, "the invoice number was not found at all"
    assert result.doc_no.value == expected
    assert text[result.doc_no.span.start : result.doc_no.span.end] == expected
    assert result.doc_no.method == "column"
    assert len(warnings_for(result, "DocDtls.No")) == 1


@pytest.mark.parametrize(
    "text",
    [
        "Invoice No: ABC/XYZ\n",
        "Invoice No.   e-Way Bill No.   Dated\nSFPL/24-25/0731   3312   25-Apr-2026\n",
        "Invoice No.   Transport Mode\nINV-0042   By Road\n",
        "Invoice No.   Currency\nINR\n",
        "Invoice No: SFPL-EXP\n",
    ],
)
def test_a_doc_no_carrying_no_digit_is_never_emitted_without_a_warning(text):
    """The structural rule behind both fixes, independent of any word list."""
    result = extract_rules(text)
    doc_no = result.doc_no
    if doc_no is not None and not any(char.isdigit() for char in doc_no.value):
        assert warnings_for(result, "DocDtls.No") != []


@pytest.mark.parametrize("text", ["Invoice No.   Dated\n", "Invoice No: INV 0042\n"])
def test_a_label_word_or_a_truncated_word_is_still_refused(text):
    """'Dated' is the next heading; 'INV' is the first half of 'INV 0042', not the whole of it."""
    result = extract_rules(text)
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert "was not accepted as the invoice number" in message
    assert "enter it manually" in message


def test_the_refusal_warning_does_not_assert_the_token_is_a_neighbouring_heading():
    """It may be the number itself, truncated; the warning must not claim otherwise."""
    message = warnings_for(extract_rules("Invoice No: INV 0042\n"), "DocDtls.No")[0].message
    assert "'INV'" in message
    assert "neighbouring heading" not in message
    assert "truncated word" in message


# --- a value beside the label is held to the same tests as the row below ----


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("Invoice No.   25/04/2026\n", "a date"),
        ("Invoice No: 25-Apr-2026\n", "a date with a month name"),
        (f"Invoice No: {MAHARASHTRA}\n", "already matched as a GSTIN"),
        ("Invoice No: 11800.00\n", "one component of a decimal amount"),
        ("Invoice No: 2 Nos\n", "a quantity carrying a unit"),
    ],
)
def test_a_value_beside_the_label_is_refused_when_it_is_something_else(text, why):
    """An empty invoice-number cell leaves the next cell printed where the value goes."""
    result = extract_rules(text)
    assert result.doc_no is None, why
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert "was not accepted as the invoice number" in message
    assert "enter it manually" in message


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("Invoice No: 11800.00\n", "11800"),
        ("Invoice No: 11,800.00\n", "11"),
        ("Invoice No.   1,23,456.78\n", "1"),
        ("Invoice No.   12,000\n", "12"),
        ("Invoice No.      5,000.00\n", "5"),
    ],
)
def test_a_comma_grouped_amount_beside_the_label_is_refused_not_truncated(text, token):
    """Indian grouping splits the amount, so '11,800.00' otherwise files DocDtls.No = '11'."""
    result = extract_rules(text)
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert f"'{token}'" in message
    assert "one run of a longer number" in message
    assert "enter it manually" in message


def test_a_comma_grouped_amount_in_a_blank_cell_is_not_the_invoice_number():
    """The layout the guard exists for, in the commonest amount format on the document."""
    text = "TAX INVOICE\nInvoice No.                      11,800.00\nInvoice Date: 25/04/2026\n"
    result = extract_rules(text)
    assert result.doc_no is None
    assert len(warnings_for(result, "DocDtls.No")) == 1
    assert result.doc_date.value == "25/04/2026"


def test_an_hsn_code_on_a_row_of_comma_grouped_amounts_is_still_a_code():
    """The same guard serves HSN: widening it must not start dropping real codes."""
    text = "1  Cotton fabric  HSN 52081190  10 Mtr  1,200.00  12,000.00\n"
    assert [f.value for f in extract_rules(text).hsn_codes] == ["52081190"]


def test_the_same_date_is_refused_beside_the_label_and_below_it():
    """The two layouts must agree: a date is not the invoice number in either of them."""
    beside = extract_rules("Invoice No.   25/04/2026\n")
    below = extract_rules("Invoice No.\n25/04/2026\n")
    assert beside.doc_no is None and below.doc_no is None
    assert len(warnings_for(beside, "DocDtls.No")) == 1
    assert len(warnings_for(below, "DocDtls.No")) == 1


def test_a_date_beside_the_label_does_not_become_both_mandatory_fields():
    result = extract_rules("Invoice No.   25/04/2026\n")
    assert result.doc_date.value == "25/04/2026"
    assert result.doc_no is None


# --- a spaced date beside the label is a date, not the day filed as a serial ---


@pytest.mark.parametrize(
    ("text", "day", "printed"),
    [
        ("Invoice No.   25 Apr 2026\n", "25", "25 Apr 2026"),
        ("Invoice No: 5 April 2026\n", "5", "5 April 2026"),
        ("Bill No.   1 May 2026\n", "1", "1 May 2026"),
    ],
)
def test_a_spaced_date_beside_the_label_is_not_filed_as_the_invoice_number(text, day, printed):
    """The token alphabet stops at a space, so '25 Apr 2026' tokenises as the day alone.

    Filing that day as DocDtls.No is the worst shape of wrong value: a short, entirely
    plausible serial in a mandatory field, with nothing anywhere to show it came from
    the date column. The refusal names the whole date the document printed, not just
    the component it was offered.
    """
    result = extract_rules(text)
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    message = doc_no_warnings[0].message
    assert f"'{day}'" in message
    assert f"component of the date '{printed}'" in message
    assert "enter it manually" in message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice No.   25 Apr 2026\n", "25/04/2026"),
        ("Invoice No: 5 April 2026\n", "05/04/2026"),
        ("Bill No.   1 May 2026\n", "01/05/2026"),
    ],
)
def test_the_spaced_date_beside_the_label_still_reaches_the_date_field(text, expected):
    """Refusing it as the serial must not lose it: it is the only date the page prints."""
    assert extract_rules(text).doc_date.value == expected


# --- a spaced date on the ROW BELOW the label is a date, not a serial -------

SPACED_DATE_BELOW = (
    # a header row whose invoice-number cell is blank, the date printed underneath
    ("Invoice No. Invoice Date\n 25 Apr 2026\n", "25 Apr 2026", "25/04/2026"),
    ("Invoice No.\n7 Apr 2026\n", "7 Apr 2026", "07/04/2026"),
    ("Tax Invoice No. Dated\n15 Jan 2026\n", "15 Jan 2026", "15/01/2026"),
    # the label indented, so its column lands on the YEAR rather than on the day
    ("        Invoice No.\n25 Apr 2026\n", "25 Apr 2026", "25/04/2026"),
    ("Bill No.   Date\n1 May 2026\n", "1 May 2026", "01/05/2026"),
)


@pytest.mark.parametrize(
    ("text", "printed", "date"),
    SPACED_DATE_BELOW,
)
def test_no_component_of_a_spaced_date_below_the_label_becomes_the_invoice_number(
    text, printed, date
):
    """"25 Apr 2026" tokenises as "25", "Apr" and "2026": none of the three is a serial.

    Refusing only a token a date STARTS at catches the day and lets the YEAR through,
    and the year is a four-digit number in the label's own column — filed as the
    mandatory DocDtls.No by the row-below path. The damage lands on the date: the
    chosen date overlaps the invoice number's span and is dropped, so the accountant
    is told "No date was found in the document" about a document that prints one, and
    the consumed span blanks the year out of stage 2's input as well.
    """
    result = extract_rules(text)
    assert result.doc_no is None, "a component of the printed date was filed as DocDtls.No"
    assert result.doc_date is not None, "the date the document prints was lost with it"
    assert result.doc_date.value == date
    dt_messages = [w.message for w in warnings_for(result, "DocDtls.Dt")]
    assert not [m for m in dt_messages if "No date was found" in m]
    message = warnings_for(result, "DocDtls.No")[0].message
    assert f"component of the date '{printed}'" in message
    assert "enter it manually" in message


def test_the_row_below_refusal_names_the_year_when_the_year_is_in_the_column():
    """The refusal reports the token it actually found, day or year, and the whole date."""
    message = warnings_for(
        extract_rules("        Invoice No.\n25 Apr 2026\n"), "DocDtls.No"
    )[0].message
    assert "the row below prints '2026' in the label's column" in message
    assert "component of the date '25 Apr 2026'" in message


@pytest.mark.parametrize(("text", "printed", "date"), SPACED_DATE_BELOW)
def test_the_spaced_date_below_the_label_reaches_the_date_field_whole(text, printed, date):
    """The span covers the date as printed, so nothing of it is left behind for stage 2."""
    result = extract_rules(text)
    assert text[result.doc_date.span.start : result.doc_date.span.end] == printed
    assert printed not in remaining_text(text, result.consumed)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # a financial-year serial in the label's column is not a date in either order
        ("Invoice No.\n24-25/0731\n", "24-25/0731"),
        # a genuine serial beside the date column on the row below keeps its value
        ("Invoice No.   Invoice Date\nINV-0042      25 Apr 2026\n", "INV-0042"),
        ("Bill No.      Date\n7788          1 May 2026\n", "7788"),
    ],
)
def test_the_row_below_guard_does_not_refuse_a_serial_beside_a_spaced_date(text, expected):
    """Refusing a real serial drops a mandatory field; only an overlapping date counts."""
    assert extract_rules(text).doc_no.value == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # the financial-year middle of a Tally serial is not a real date in either order
        ("Invoice No: 24-25/0731\n", "24-25/0731"),
        ("Invoice No.   24-25/0731\n", "24-25/0731"),
        # a genuine serial printed beside a date column keeps its own value
        ("Invoice No. 7788    Date   25 Apr 2026\n", "7788"),
        ("Invoice No: INV/2026-27/0042    Dated 1 May 2026\n", "INV/2026-27/0042"),
    ],
)
def test_the_spaced_date_guard_does_not_refuse_a_genuine_serial(text, expected):
    """The guard fires on a date COVERING the token, never on one printed beside it."""
    assert extract_rules(text).doc_no.value == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 24-25 is a financial year, not a day and a month: it is a serial, not a date.
        ("Invoice No: 24-25/0731\n", "24-25/0731"),
        ("Invoice No.\n24-25/0731\n", "24-25/0731"),
        # the unit test looks at the next word only, so an unrelated "No." further
        # along the line does not refuse a correctly labelled number
        ("Invoice No. 42    Vehicle No. MH-12-AB-1234\n", "42"),
    ],
)
def test_a_date_shaped_serial_is_still_an_invoice_number(text, expected):
    """Refusing a real serial drops a mandatory field; the pair has to be a real date."""
    assert extract_rules(text).doc_no.value == expected


def test_every_digit_free_serial_separator_is_reachable():
    """A rule listing a character the contract's token alphabet cannot produce never fires."""
    for char in DOC_NO_SERIAL_SEPARATORS:
        assert DOC_NO_TOKEN_PATTERN.fullmatch(f"A{char}B"), char


@pytest.mark.parametrize(
    ("text", "wrapped"),
    [
        # a number IS printed beside the wrapped label: the claim that matters is that a
        # label the rules cannot read supplies no value at all
        ("Invoice\nNo.   7788\n", "'Invoice' above 'No'"),
        # 'Tax Invoice No' wrapped at both of its spaces, with no value anywhere
        ("Tax\nInvoice\nNo\n", "'Tax' above 'Invoice' above 'No'"),
    ],
)
def test_a_label_wrapped_across_two_lines_is_reported_rather_than_read(text, wrapped):
    """OCR wraps a header column: the label starts on one line and ends on the next.

    Expectations transcribed by hand from the fixture and the contract: a label whose
    own words straddle a line break is not a label, so nothing beside it or below it
    may be taken — '7788' sits beside the wrapped label and must not reach DocDtls.No —
    and the document nonetheless prints one, so the wrap is named rather than passed
    over in silence.
    """
    result = extract_rules(text)
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    # Two: the label was not found, and the document nonetheless prints a wrapped one.
    assert len(doc_no_warnings) == 2
    assert "No invoice number label" in doc_no_warnings[0].message
    wrapped_warnings = wrapped_label_warnings(result)
    assert len(wrapped_warnings) == 1
    message = wrapped_warnings[0].message
    assert wrapped in message
    assert "could not be read as a label" in message


@pytest.mark.parametrize(
    "text",
    [
        "Bill\nNo 7788\n",
        "Invoice\nNo.        Dated\nINV-0042   25/04/2026\n",
        "Bill\nNo.      Date\n7788     25/04/2026\n",
        "Tax Invoice\nNo.        Dated\nINV-0042   25/04/2026\n",
        "Tax Invoice\nNo: 42\n",
        "Invoice\nNumber: 42\n",
        "Invoice\nNo.\nUmang Fabrics Pvt Ltd\n",
    ],
)
def test_a_label_ocr_wrapped_mid_label_is_not_found_and_says_so(text):
    """The accepted cost of keeping a label's own words on one line.

    A label whose words are split across a line break is no longer a label, so an
    invoice number the document really did print under an OCR-wrapped header is now
    reported as missing instead of read. That is the deliberate trade: the same
    line-spanning whitespace that found those values also read an invoice title
    printed above a door-number address line as a label and filed the house number
    as the mandatory DocDtls.No with nothing to show for it (see the title-above-
    address tests below). A mandatory field reported missing is recoverable; a
    plausible wrong one filed silently is not.
    """
    result = extract_rules(text)  # must not raise
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    # Two: the label was not found, and the document nonetheless prints a wrapped one.
    assert len(doc_no_warnings) == 2
    assert "No invoice number label" in doc_no_warnings[0].message
    assert len(wrapped_label_warnings(result)) == 1


@pytest.mark.parametrize(
    "text",
    [
        "Bill\nNo 7788\n",
        "Invoice\nNo.        Dated\nINV-0042   25/04/2026\n",
        "Tax Invoice\nNo: 42\n",
        "Invoice\nNumber: 42\n",
    ],
)
def test_the_missing_label_message_admits_that_a_wrapped_label_is_not_recognised(text):
    """"No label was found" is a claim about the DOCUMENT, and these documents print one.

    The rule that a label's words must share a line is what stops a house number under
    an invoice title becoming DocDtls.No, and it is not being widened again. But an
    accountant reading that no invoice-number label was found, while looking at a page
    that prints 'Invoice' above 'No.', is told something false about the document. The
    message says which rule refused it instead.
    """
    message = warnings_for(extract_rules(text), "DocDtls.No")[0].message
    assert "No invoice number label" in message
    assert "own words are printed on one line" in message
    assert "read it from the document" in message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # the same header-row layout, with the label NOT wrapped: still read
        ("Invoice No.        Dated\nINV-0042   25/04/2026\n", "INV-0042"),
        ("Bill No.      Date\n7788     25/04/2026\n", "7788"),
        ("Tax Invoice No.        Dated\nINV-0042   25/04/2026\n", "INV-0042"),
    ],
)
def test_an_unwrapped_label_still_reads_its_column_on_the_row_below(text, expected):
    """Restricting the label's INTERNAL whitespace must not touch the row-below path."""
    result = extract_rules(text)
    assert result.doc_no is not None, "the value in the label's own column was not found"
    assert result.doc_no.value == expected
    assert text[result.doc_no.span.start : result.doc_no.span.end] == expected
    assert result.doc_no.method == "column"
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    assert "row below" in doc_no_warnings[0].message


def test_an_unwrapped_label_with_an_empty_column_below_is_reported_as_empty():
    """The other half: the fix must not start inventing a value where there is none."""
    result = extract_rules("Invoice No.\n              Umang Fabrics Pvt Ltd\n")
    assert result.doc_no is None
    message = warnings_for(result, "DocDtls.No")[0].message
    assert "not followed by a value on its own line or in its column on the line below" in message
    assert "row below prints" not in message


# --- a wrapped own label lets ANOTHER document's label supply DocDtls.No -----

WRAPPED_OWN_LABEL_WITH_EWAY = (
    # the invoice's own label is OCR-wrapped; the e-Way bill's is not
    "Invoice\n"
    "No. 4242\n"
    "e-Way Bill No. 7788\n"
    "Invoice Date: 25/04/2026\n"
)


def test_a_wrapped_own_label_lets_another_labels_number_through_but_never_silently():
    """The reviewer's blocker: refusing the wrapped label is not the end of the story.

    A label whose words straddle a line break is not read as a label — that is what
    keeps a house number out of DocDtls.No. But the contract takes the FIRST
    invoice-number label in the document, so when the invoice's OWN label is the
    wrapped one, the first label the rules can read is some other document's. An
    e-Way bill number is printed on a large share of Indian invoices; filed as
    DocDtls.No it is a well-formed serial nothing downstream can question, and the
    path that files it — a digit-carrying token beside its own label — is the one
    path in this stage that emits no warning at all.
    """
    result = extract_rules(WRAPPED_OWN_LABEL_WITH_EWAY)
    # transcribed by hand from the fixture: 7788 is the e-WAY bill's number, and 4242
    # is the invoice's own, printed under the label the rules cannot read.
    assert result.doc_no is not None
    assert result.doc_no.value == "7788"
    wrapped = wrapped_label_warnings(result)
    assert len(wrapped) == 1, "the wrapped own label was not reported"
    message = wrapped[0].message
    assert "'Invoice' above 'No'" in message
    assert "could not be read as a label" in message
    assert "may belong to that other label" in message


@pytest.mark.parametrize(
    "text",
    [
        "Tax Invoice\nNo. 42\n",
        "Invoice\nNumber: 42\n",
        "Invoice\nNo. 42\n",
        "Invoice\n#42\n",
        "Bill\nNumber: 42\n",
        "Bill\nNo. 42\n",
        "Inv\nNo. 42\n",
    ],
)
def test_every_wrapped_invoice_number_label_shape_is_reported(text):
    """Each label the contract lists, wrapped where the document would print a space."""
    assert len(wrapped_label_warnings(extract_rules(text))) == 1


@pytest.mark.parametrize(
    "text",
    [
        # the same document with the own label unwrapped: the e-Way number is not taken
        # and nothing is reported as wrapped
        "Invoice No. 4242\ne-Way Bill No. 7788\nInvoice Date: 25/04/2026\n",
        "Tax Invoice No: 42\n",
        "Invoice No.        Dated\nINV-0042   25/04/2026\n",
    ],
)
def test_an_unwrapped_label_is_never_reported_as_wrapped(text):
    """A warning raised on a correct reading costs the warnings that matter their credibility."""
    assert wrapped_label_warnings(extract_rules(text)) == []


def test_the_clean_invoice_reports_no_wrapped_label():
    """'TAX INVOICE' above 'Seller:' is a title above a name, not a wrapped label."""
    assert wrapped_label_warnings(extract_rules(CLEAN_INVOICE)) == []


@pytest.mark.parametrize(
    "text",
    [
        # The longest label alternation reaches back over the line break for a preceding
        # word, so a heading whose last word is "Tax" spans into a correctly printed
        # "Invoice No." on the next line. The label was read from its own line, beside
        # its value, on this stage's one silent path: nothing may cast doubt on it.
        "Nimbus Tax\nInvoice No. 4242\n",
        "Maharashtra Tax\nInvoice No: SFPL/26-27/0042\n",
        "Goods and Service Tax\nInvoice Number 7788\n",
    ],
)
def test_a_heading_ending_in_tax_does_not_make_the_label_below_it_wrapped(text):
    result = extract_rules(text)
    assert result.doc_no is not None
    assert wrapped_label_warnings(result) == []


def test_a_non_breaking_space_in_a_label_is_not_reported_as_a_line_break():
    """PyMuPDF's text layer emits U+00A0 inside a label printed on one line.

    The value is withheld either way, but the wrapped-label message names a line break
    the document does not print, which sends the reader hunting for something absent.
    """
    result = extract_rules("Invoice No. 4242\n")
    assert wrapped_label_warnings(result) == []


# --- an invoice title above a door-number address is not a label ------------

TITLE_ABOVE_ADDRESS = (
    # the reviewer's repro: an Indian door-number line under the document's title
    "TAX INVOICE\nNo. 42, MG Road\nBengaluru 560001\n",
    "INVOICE\nNo. 7/B, Anna Salai\nChennai 600002\n",
    "BILL\nNo. 118, Sector 5\nNoida 201301\n",
    "INV\nNo. 3, Park Street\nKolkata 700016\n",
    "INVOICE\nNumber 12, Brigade Road\nBengaluru 560025\n",
    "INVOICE\n#42, Church Street\nBengaluru 560001\n",
    "TAX INVOICE\nNo. 5-A, Nehru Nagar\nPune 411014\n",
)
# the house numbers above, transcribed by hand from the fixture text
TITLE_ABOVE_ADDRESS_HOUSE_NUMBERS = ("42", "7/B", "118", "3", "12", "42", "5-A")


@pytest.mark.parametrize(
    ("text", "house_number"),
    list(zip(TITLE_ABOVE_ADDRESS, TITLE_ABOVE_ADDRESS_HOUSE_NUMBERS, strict=True)),
)
def test_a_house_number_under_the_invoice_title_is_never_the_invoice_number(
    text, house_number
):
    """Indian addresses begin "No. 42, MG Road", and invoices are titled "TAX INVOICE".

    Printed one above the other, whitespace that spans a line break reads the title's
    last word plus the address line's first word as an invoice-number label, and the
    house number sitting right after it is filed as the mandatory DocDtls.No beside
    its own apparent label — the one path that emits no warning at all. An accountant
    would see a clean extraction reporting the seller's street number as the invoice
    number.
    """
    result = extract_rules(text)
    assert result.doc_no is None, f"house number {house_number!r} was filed as DocDtls.No"
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    # Two: no label was found, and the title-over-address shape is reported as a
    # label that reads as wrapped. Neither of them may quote the house number as a
    # candidate — that is the value this whole fixture exists to keep out of the field.
    assert len(doc_no_warnings) == 2
    assert "No invoice number label" in doc_no_warnings[0].message
    assert len(wrapped_label_warnings(result)) == 1
    assert house_number not in "".join(w.message for w in doc_no_warnings)


@pytest.mark.parametrize("text", TITLE_ABOVE_ADDRESS)
def test_a_house_number_under_the_invoice_title_is_not_consumed_either(text):
    """Stage 2 reads the address as the document prints it, house number included."""
    result = extract_rules(text)
    assert remaining_text(text, result.consumed) == text


def test_a_label_below_the_invoice_title_is_still_read():
    """The guard is the line break inside the label, not the title above it."""
    text = "TAX INVOICE\nInvoice No. INV-42\n"
    result = extract_rules(text)
    assert result.doc_no == RuleField("INV-42", Span(24, 30), "regex")
    assert warnings_for(result, "DocDtls.No") == []


@pytest.mark.parametrize(
    "text",
    [
        "Tax Invoice\nNo",
        "Invoice\nNumber",
        "Invoice\nNo",
        "Invoice\n#",
        "Bill\nNumber",
        "Bill\nNo",
        "Inv\nNo",
    ],
)
def test_no_labels_own_words_may_straddle_a_line_break(text):
    """Pin the property directly, so a later '\\s+' cannot creep back in unnoticed."""
    assert DOC_NO_LABEL_PATTERN.search(text) is None
    assert DOC_NO_LABEL_PATTERN.search(text.replace("\n", " ")) is not None


# --- the same property, over EVERY label pattern in the module ---------------

# Every pattern in the module that recognises a label the document PRINTS. The
# wrapped-shape detector is deliberately not here: its whole job is to match across a
# line break, and it never supplies a value.
LABEL_PATTERNS = (
    ("DOC_NO_LABEL_PATTERN", DOC_NO_LABEL_PATTERN),
    *((f"DATE_LABEL_TIERS[{tier}]", pattern) for tier, pattern in enumerate(DATE_LABEL_TIERS)),
    *((f"seller label {label!r}", pattern) for label, pattern in _SELLER_PATTERNS),
    *((f"buyer label {label!r}", pattern) for label, pattern in _BUYER_PATTERNS),
    ("HSN_LABEL_PATTERN", HSN_LABEL_PATTERN),
)

# Every multi-word label the module recognises, transcribed by hand from the contract's
# own lists rather than generated from the module's constants.
MULTI_WORD_LABELS = (
    "Tax Invoice No",
    "Invoice Number",
    "Invoice No",
    "Invoice #",
    "Bill Number",
    "Bill No",
    "Inv No",
    "Invoice Date",
    "Invoice Dt",
    "Sold By",
    "Bill To",
    "Billed To",
    "Ship To",
    "Shipped To",
)


@pytest.mark.parametrize("label", MULTI_WORD_LABELS)
def test_no_label_patterns_own_words_may_straddle_a_line_break(label):
    """The property the whole class of defects reduces to.

    A label is a thing PRINTED on the document, and its words sit on one line. A
    pattern whose internal whitespace spans a line break invents labels out of a line's
    last word and the next line's first: an invoice TITLE over an address line becomes
    'Invoice No' and files a house number; a title over a ledger column becomes
    'Invoice Date' and files a supply date, silently, because the most specific date
    label is exactly the tier that settles the date without a word. Asserting it here,
    over every label pattern at once, is what stops the next '\\s' from reintroducing it
    somewhere new.
    """
    words = label.split()
    assert any(pattern.search(label) for _, pattern in LABEL_PATTERNS), label
    for cut in range(1, len(words)):
        wrapped = " ".join(words[:cut]) + "\n" + " ".join(words[cut:])
        for name, pattern in LABEL_PATTERNS:
            for match in pattern.finditer(wrapped):
                assert "\n" not in match.group(0), f"{name} matched {match.group(0)!r}"


def test_no_label_pattern_is_written_with_line_spanning_whitespace():
    """The same property read off the patterns, so an untested label shape cannot slip in."""
    for name, pattern in LABEL_PATTERNS:
        assert r"\s" not in pattern.pattern, name


def test_missing_doc_no_is_reported():
    result = extract_rules(f"Umang Fabrics\nGSTIN {MAHARASHTRA}\n")
    assert result.doc_no is None
    doc_no_warnings = warnings_for(result, "DocDtls.No")
    assert len(doc_no_warnings) == 1
    assert "invoice number" in doc_no_warnings[0].message


# --- invoice date -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("25/04/2026", "25/04/2026"),
        ("25-04-2026", "25/04/2026"),
        ("25.04.2026", "25/04/2026"),
        ("25/04/26", "25/04/2026"),
        ("5/4/2026", "05/04/2026"),
        ("5-Apr-2026", "05/04/2026"),
        ("5 Apr 2026", "05/04/2026"),
        ("5 April 2026", "05/04/2026"),
        ("05-APR-26", "05/04/2026"),
        ("31 December 2026", "31/12/2026"),
    ],
)
def test_doc_date_formats_normalise_to_day_first(raw, expected):
    result = extract_rules(f"Invoice Date: {raw}\n")
    assert result.doc_date.value == expected
    assert result.doc_date.method == "regex"
    assert f"Invoice Date: {raw}\n"[result.doc_date.span.start : result.doc_date.span.end] == raw


@pytest.mark.parametrize("label", ["Invoice Date", "Date", "Dated", "Dt"])
def test_every_date_label(label):
    result = extract_rules(f"{label}: 25/04/2026\n")
    assert result.doc_date.value == "25/04/2026"


def test_a_labelled_date_beats_an_earlier_unlabelled_one():
    text = "Purchase order 01/02/2026 received\nInvoice Date: 25/04/2026\n"
    assert extract_rules(text).doc_date.value == "25/04/2026"


def test_the_first_date_is_used_when_no_label_is_present():
    text = "Dispatched 03/04/2026 by road\n"
    assert extract_rules(text).doc_date.value == "03/04/2026"


def test_ambiguous_date_warns_with_both_readings():
    text = "Invoice Date: 03/04/2026\n"
    result = extract_rules(text)
    assert result.doc_date.value == "03/04/2026"
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    message = date_warnings[0].message
    assert "03/04/2026" in message
    assert "3 April 2026" in message
    assert "4 March 2026" in message
    assert "DD/MM/YYYY" in message


def test_equal_components_are_not_ambiguous():
    result = extract_rules("Invoice Date: 05/05/2026\n")
    assert result.doc_date.value == "05/05/2026"
    assert warnings_for(result, "DocDtls.Dt") == []


def test_a_day_over_twelve_is_not_ambiguous():
    result = extract_rules("Invoice Date: 25/04/2026\n")
    assert warnings_for(result, "DocDtls.Dt") == []


def test_another_day_first_date_resolves_the_ambiguity_silently():
    text = "Invoice Date: 03/04/2026\nDue Date: 25/05/2026\n"
    result = extract_rules(text)
    assert result.doc_date.value == "03/04/2026"
    assert warnings_for(result, "DocDtls.Dt") == []


def test_a_month_first_looking_document_still_emits_day_first_but_warns():
    text = "Invoice Date: 03/04/2026\nShipped 05/25/2026 by road\n"
    result = extract_rules(text)
    assert result.doc_date.value == "03/04/2026"
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    message = date_warnings[0].message
    assert "month-first" in message
    assert "MM/DD" in message
    assert "4 March 2026" in message


def test_a_month_name_date_is_never_ambiguous():
    text = "Invoice Date: 03-Apr-2026\nShipped 05/25/2026 by road\n"
    result = extract_rules(text)
    assert result.doc_date.value == "03/04/2026"
    assert warnings_for(result, "DocDtls.Dt") == []


def test_a_second_component_over_twelve_cannot_be_a_month_and_warns():
    result = extract_rules("Invoice Date: 05/25/2026\n")
    assert result.doc_date.value == "05/25/2026"
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "cannot be a month" in date_warnings[0].message


def test_a_first_component_that_cannot_be_a_day_warns():
    result = extract_rules("Invoice Date: 00/03/2026\n")
    assert result.doc_date.value == "00/03/2026"
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "cannot be a day" in date_warnings[0].message


def test_a_due_date_printed_above_the_invoice_date_does_not_win():
    """A bare 'Date' matches inside 'Due Date'; the due date filed as DocDtls.Dt is wrong."""
    text = "Invoice No: INV/2026-27/0042\nDue Date: 30/05/2026\nInvoice Date: 03/02/2026\n"
    result = extract_rules(text)
    assert result.doc_date.value == "03/02/2026"
    assert warnings_for(result, "DocDtls.Dt") == []


def test_a_supply_date_label_does_not_beat_an_invoice_date_label():
    text = "Date of Supply: 20/03/2026\nInvoice Date: 25/04/2026\n"
    assert extract_rules(text).doc_date.value == "25/04/2026"


def test_competing_labelled_dates_with_no_invoice_date_label_warn():
    text = "Due Date: 30/05/2026\nDelivery Date: 01/06/2026\n"
    result = extract_rules(text)
    assert result.doc_date.value == "30/05/2026"
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    message = date_warnings[0].message
    assert "Due Date" in message
    assert "Delivery Date" in message
    assert "01/06/2026" in message


@pytest.mark.parametrize("raw", ["31/02/2026", "30/02/2026", "31/04/2026", "29/02/2027"])
def test_an_impossible_calendar_date_warns(raw):
    """31 February is provably a misread; emitting it clean is the failure mode to avoid."""
    result = extract_rules(f"Invoice Date: {raw}\n")
    assert result.doc_date.value == raw
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "not a real calendar date" in date_warnings[0].message


def test_the_impossible_date_warning_names_the_month_and_year():
    result = extract_rules("Invoice Date: 31/04/2026\n")
    assert "April 2026 has no day 31" in warnings_for(result, "DocDtls.Dt")[0].message


def test_a_leap_day_in_a_leap_year_is_a_real_date():
    result = extract_rules("Invoice Date: 29/02/2028\n")
    assert result.doc_date.value == "29/02/2028"
    assert warnings_for(result, "DocDtls.Dt") == []


# --- a month name settles the ORDER, not whether the day exists --------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("31-Feb-2026", "31/02/2026"),
        ("31 April 2026", "31/04/2026"),
        ("29-Feb-2027", "29/02/2027"),  # 2027 is not a leap year
        ("32 May 2026", "32/05/2026"),
        ("00 May 2026", "00/05/2026"),
        ("99-Dec-2026", "99/12/2026"),
        ("31 Apr 26", "31/04/2026"),
    ],
)
def test_an_impossible_month_name_date_warns_like_its_numeric_twin(raw, expected):
    """'31/04/2026' warns; '31-Feb-2026' filed clean is the same wrong mandatory field.

    A month name settles which component is the month, so the DD-Mon form is never
    ambiguous — but it is not thereby a real date, and emitting one with nothing
    anywhere on the document to show it was doubted is the failure mode this part
    exists to prevent.
    """
    result = extract_rules(f"Invoice Date: {raw}\n")
    assert result.doc_date.value == expected
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "not a real calendar date" in date_warnings[0].message


def test_the_impossible_month_name_warning_names_the_month_and_year():
    result = extract_rules("Invoice Date: 31-Feb-2026\n")
    assert "February 2026 has no day 31" in warnings_for(result, "DocDtls.Dt")[0].message


def test_an_impossible_month_name_date_is_not_clean_anywhere_on_the_document():
    """The whole document carried zero warnings, so nothing flagged the date at all."""
    text = (
        "TAX INVOICE\n"
        "Seller: Umang Fabrics\n"
        f"GSTIN: {MAHARASHTRA}\n"
        "Buyer: Farheen Traders\n"
        f"GSTIN: {DELHI}\n"
        "Invoice No: INV-0042\n"
        "Invoice Date: 31-Feb-2026\n"
        "1  Consulting  HSN 998313  2 Nos  5,000.00  10,000.00\n"
    )
    result = extract_rules(text)
    assert result.doc_date.value == "31/02/2026"
    assert [w.field for w in result.warnings] == ["DocDtls.Dt"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("25 Apr 2026", "25/04/2026"),
        ("05-APR-26", "05/04/2026"),
        ("15-Jan-2026", "15/01/2026"),
        ("29-Feb-2028", "29/02/2028"),  # 2028 is a leap year
        ("30 June 2026", "30/06/2026"),
    ],
)
def test_a_real_month_name_date_is_still_silent(raw, expected):
    """The day check must not start warning about the dates invoices actually print."""
    result = extract_rules(f"Invoice Date: {raw}\n")
    assert result.doc_date.value == expected
    assert warnings_for(result, "DocDtls.Dt") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice No: 42 May 2026 charges\n", "42"),
        ("Bill No: 99 Dec 2026 statement\n", "99"),
        ("Invoice No: 35 Jan 2026\n", "35"),
    ],
)
def test_a_token_before_a_month_name_that_is_no_real_day_is_still_the_serial(text, expected):
    """The ruling refuses the day of a REAL date; '42 May 2026' is not one.

    Refusing it drops the mandatory invoice number on the ground that the text
    resembles a date it cannot possibly be.
    """
    result = extract_rules(text)
    assert result.doc_no.value == expected
    assert warnings_for(result, "DocDtls.No") == []


@pytest.mark.parametrize(
    "row",
    [
        "1   Aprons        50    Nos",
        "3   Marble    10   Sqft",
        "2   Marketing    12    Nos",
        "4   Decorative tiles    20    Box",
        "5   Octane booster    6    Ltr",
        "6   Junction box    2    Nos",
    ],
)
def test_an_item_row_is_not_a_date_just_because_a_word_starts_like_a_month(row):
    """'Aprons' is not April: an open month suffix files an item row as the invoice date."""
    result = extract_rules(row)
    assert result.doc_date is None
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "No date" in date_warnings[0].message


@pytest.mark.parametrize("row", ["1   Aprons        50    Nos", "3   Marble    10   Sqft"])
def test_an_item_row_misread_as_a_date_is_not_consumed_either(row):
    """Stage 2 must still see the row: a consumed span is blanked out of its input."""
    result = extract_rules(row)
    assert result.consumed == ()
    assert remaining_text(row, result.consumed) == row


@pytest.mark.parametrize("row", ["1   May   10   Nos", "2   March   12   Pcs", "3 June 26 Nos"])
def test_an_item_row_whose_description_is_a_month_word_is_not_a_date(row):
    """A two-digit year is allowed, so '1 May 10 Nos' otherwise files 01/05/2010 clean."""
    result = extract_rules(row)
    assert result.doc_date is None
    assert result.consumed == ()
    assert remaining_text(row, result.consumed) == row
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "No date" in date_warnings[0].message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice Date: 5 Apr 26\n", "05/04/2026"),
        ("Invoice Date: 5 Apr 26  2 Nos 1200.00\n", "05/04/2026"),
        ("Invoice Date: 05-APR-26\n1  Cotton  10  Mtr\n", "05/04/2026"),
    ],
)
def test_a_two_digit_year_date_is_still_a_date_when_no_unit_follows_it(text, expected):
    """Only a unit word immediately after the year marks the line as an item row."""
    assert extract_rules(text).doc_date.value == expected


@pytest.mark.parametrize(
    "text",
    [
        # a quantity ending an item row, with the description wrapped onto the next line
        "Sl  Particulars                 Rate    Qty\n"
        "1   Annual maintenance          50000     1\n"
        "    April 2026 to March 2027\n",
        # Part D joins pages with "\n", so a page boundary has exactly this shape
        "Balance carried forward   10\nApril 2026 statement, page 2 of 2\n",
        "Total 12\nMarch 2026 statement enclosed\n",
        "Qty 10\n5 April\n2026\n",  # the month-to-year gap spans the break too
    ],
)
def test_a_date_is_never_read_across_a_line_break(text):
    """A printed date's day, month and year are on one line; two rows read as one are not."""
    result = extract_rules(text)
    assert result.doc_date is None
    assert result.consumed == ()
    assert remaining_text(text, result.consumed) == text
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "No date" in date_warnings[0].message


@pytest.mark.parametrize(
    "row",
    [
        "1   May 2026 hosting charges   1   Nos   5,000.00\n",
        "2   March 2026 subscription    1   Nos  12,000.00\n",
        "1   April 2026 to March 2027 annual maintenance\n",
    ],
)
def test_a_column_gap_before_a_month_is_an_item_row_not_a_date(row):
    """'1   May 2026 ...' is an Sl.No beside a description, not the first of May."""
    result = extract_rules(row)
    assert result.doc_date is None
    assert result.consumed == ()
    assert remaining_text(row, result.consumed) == row
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "No date" in date_warnings[0].message


# --- an item row read as the invoice date must say so ------------------------

ITEM_ROW_NATIVE = "1 May 2026    hosting charges    1    5,000.00\n"
ITEM_ROW_OCR = "1 May 2026 hosting charges 1 5,000.00\n"


@pytest.mark.parametrize("text", [ITEM_ROW_NATIVE, ITEM_ROW_OCR])
def test_an_unlabelled_date_opening_an_item_row_is_kept_but_never_silent(text):
    """The column gap is the only thing separating the two readings, and OCR removes it.

    Tesseract joins a line's words with single spaces, so the native row and the OCR'd
    row arrive at DATE_PATTERN identically. The value stays — dropping a mandatory
    field on a guess is worse — but "1" as an item row's serial number filed as the
    first of May is exactly the confidently wrong reading that has to be visible.
    """
    result = extract_rules(text)
    assert result.doc_date.value == "01/05/2026"  # kept, not dropped
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    message = date_warnings[0].message
    assert "item row" in message
    assert "01/05/2026" in message
    assert "Confirm" in message


@pytest.mark.parametrize("text", [ITEM_ROW_NATIVE, ITEM_ROW_OCR])
def test_the_row_warned_about_as_an_item_row_is_left_whole_for_stage_two(text):
    """On the reading the warning raises, this text is a line item, not a date.

    Consuming it blanks the row's serial number and the first words of its
    description out of stage 2's input, so stage 2 reads "hosting charges" where the
    document prints "May 2026 hosting charges" — a plausible line item that is not
    the one on the page, and Part C's grounding check cannot see it because the
    truncated string really is a substring of what it was given.
    """
    result = extract_rules(text)
    assert result.doc_date.value == "01/05/2026"  # still confirmed to Part D
    assert result.consumed == ()
    assert remaining_text(text, result.consumed) == text
    assert remaining_text(text, result.consumed).startswith("1 May 2026")


def test_a_date_no_item_row_reading_touched_is_still_consumed():
    """The exception is the item-row reading alone; an ordinary date is still stripped."""
    text = "Invoice Date: 25/04/2026\n"
    result = extract_rules(text)
    assert result.consumed == (Span(text.index("25/04/2026"), text.index("25/04/2026") + 10),)
    assert "25/04/2026" not in remaining_text(text, result.consumed)


def test_the_item_row_date_warning_is_not_raised_when_a_label_named_the_date():
    """A date label naming the date is the document's own say-so; it stays silent."""
    text = "Invoice Date\n1 May 2026 hosting charges 1 5,000.00\n"
    result = extract_rules(text)
    assert result.doc_date.value == "01/05/2026"
    assert warnings_for(result, "DocDtls.Dt") == []


def test_a_labelled_date_is_still_read_when_an_item_row_starts_with_a_month():
    text = "Invoice Date: 30-Jun-2026\n1   May 2026 hosting charges   1   Nos   5,000.00\n"
    result = extract_rules(text)
    assert result.doc_date.value == "30/06/2026"
    assert warnings_for(result, "DocDtls.Dt") == []
    assert "May 2026 hosting charges" in remaining_text(text, result.consumed)


@pytest.mark.parametrize(
    ("text", "doc_no", "date"),
    [
        # the date column printed before the invoice-number column, native spacing
        ("25/04/2026    Invoice No. 42\n", "42", "25/04/2026"),
        # the same header line as OCR returns it, columns joined by single spaces
        ("25/04/2026 Invoice No. 42\n", "42", "25/04/2026"),
        ("25/04/2026  Bill No. A7\n", "A7", "25/04/2026"),
        ("25-04-2026   Tax Invoice No. 7\n", "7", "25/04/2026"),
        ("25-Apr-2026    Invoice No. 42\n", "42", "25/04/2026"),
        # the module's own documented example, and the serials Indian invoices print:
        # the runs inside one serial are not the row's numeric columns
        ("03/06/2026    Invoice No. INV/2026/17\n", "INV/2026/17", "03/06/2026"),
        ("25/04/2026    Invoice No. SFPL/24-25/0731\n", "SFPL/24-25/0731", "25/04/2026"),
        ("25/04/2026 Invoice No. INV/2026-27/0042\n", "INV/2026-27/0042", "25/04/2026"),
        ("25/04/2026 Invoice No. 2026-42\n", "2026-42", "25/04/2026"),
        # a header line that also prints a total: the label is what settles it
        ("25/04/2026   Invoice No. 42   Total 11,800.00\n", "42", "25/04/2026"),
        ("25/04/2026   Bill No. 7788   Amount 5,000.00\n", "7788", "25/04/2026"),
    ],
)
def test_a_header_line_printing_the_date_before_the_label_is_not_an_item_row(
    text, doc_no, date
):
    """"No." after a date is the invoice-number label, not a unit of quantity.

    The item-row signature test counts a unit word as evidence that the line is a
    line item and its leading date is really a serial number. "No." is in the unit
    vocabulary for the invoice-number guard, where "2 No." really is a quantity —
    but on a header line it is the label the date is printed next to, and reading
    that as an item row warns the accountant about a date field that is exactly
    where it belongs. A warning raised on a correct reading costs the warnings that
    matter their credibility.
    """
    result = extract_rules(text)
    assert result.doc_date is not None
    assert result.doc_date.value == date
    assert result.doc_no is not None
    assert result.doc_no.value == doc_no
    assert not [w for w in warnings_for(result, "DocDtls.Dt") if "item row" in w.message]


def test_a_real_item_row_is_still_warned_about_after_the_unit_word_split():
    """The branch still fires on the rows it exists for: "Nos" is untouched."""
    result = extract_rules("1 May 2026 hosting charges 2 Nos\n")
    assert result.doc_date.value == "01/05/2026"
    assert [w for w in warnings_for(result, "DocDtls.Dt") if "item row" in w.message]


@pytest.mark.parametrize(
    ("text", "date"),
    [
        # no unit word: the numeric-column route is the only thing that can fire
        ("1 May 2026 hosting charges 1 5,000.00\n", "01/05/2026"),
        ("25/04/2026 Cotton Fabric 10 1200 12000\n", "25/04/2026"),
        ("1 Jun 2026 annual licence 1 12,000.00 12,000.00\n", "01/06/2026"),
    ],
)
def test_a_real_item_row_with_only_numeric_columns_is_still_warned_about(text, date):
    """Counting whole fields instead of digit runs must not silence a genuine item row.

    These rows print their quantity, rate and amount as separate whitespace-delimited
    fields, which is what a column is; the runs welded inside one invoice-number token
    never were.
    """
    result = extract_rules(text)
    assert result.doc_date.value == date
    assert [w for w in warnings_for(result, "DocDtls.Dt") if "item row" in w.message]


@pytest.mark.parametrize(
    "text", ["Invoice No: 2 No.\n", "Invoice No.\n2 No.\n"]
)
def test_the_invoice_number_guard_still_refuses_a_quantity_written_no_dot(text):
    """"No." stays in DOC_NO_UNIT_WORDS: splitting the vocabularies must not weaken it."""
    result = extract_rules(text)
    assert result.doc_no is None
    assert "quantity carrying a unit" in warnings_for(result, "DocDtls.No")[0].message


# --- the invoice number's financial year is not the invoice date ------------

FY_SERIAL_COLUMN_HEADER = (
    "Umang Fabrics Private Limited          Invoice No.           Dated\n"
    "12 Kalbadevi Road                      SFPL/24-25/0731       25-Apr-2026\n"
)


def test_a_financial_year_serial_does_not_become_the_invoice_date():
    """'SFPL/24-25/0731' -> DocDtls.Dt '24/25/0731' files a wrong mandatory field."""
    result = extract_rules(FY_SERIAL_COLUMN_HEADER)
    assert result.doc_no.value == "SFPL/24-25/0731"
    assert result.doc_date.value == "25/04/2026"
    assert (
        FY_SERIAL_COLUMN_HEADER[result.doc_date.span.start : result.doc_date.span.end]
        == "25-Apr-2026"
    )


def test_the_date_printed_below_a_financial_year_serial_is_the_one_taken():
    text = "Invoice No: SFPL/24-25/0731\n25-Apr-2026\n"
    result = extract_rules(text)
    assert result.doc_no.value == "SFPL/24-25/0731"
    assert result.doc_date.value == "25/04/2026"
    assert warnings_for(result, "DocDtls.Dt") == []


@pytest.mark.parametrize(
    "text",
    [
        "Invoice No: SFPL/24-25/0731\n",
        "Invoice No: GST/24-25/12\n",
        "Invoice No: UF/23-24/0007\n",
        "Ref our bill SFPL/24-25/0731 enclosed\n",  # no label, so no span to exclude
    ],
)
def test_a_serial_with_no_date_beside_it_reports_no_date_rather_than_its_middle(text):
    result = extract_rules(text)
    assert result.doc_date is None
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "No date" in date_warnings[0].message


@pytest.mark.parametrize("raw", ["Dt.25/04/2026", "(25/04/2026)", "Date :- 25/04/2026"])
def test_a_date_after_a_dot_bracket_or_colon_dash_is_still_a_date(raw):
    """The serial guard looks for '/' or '-' after an alphanumeric, nothing else."""
    assert extract_rules(f"{raw}\n").doc_date.value == "25/04/2026"


# --- another document's 'dated' must not take DocDtls.Dt silently -----------

CHALLAN_REFERENCE_ABOVE_A_WIDE_HEADER = (
    "Ref: Delivery Challan No. 77 dated 01-Apr-2026\n"
    "Umang Fabrics Private Limited          Invoice No.           Dated\n"
    "12 Kalbadevi Road                      INV-0042              25-Apr-2026\n"
)


def test_a_reference_lines_dated_beating_the_header_date_is_never_silent():
    """The header's 'Dated' is 62 characters from its value, out of the label window."""
    result = extract_rules(CHALLAN_REFERENCE_ABOVE_A_WIDE_HEADER)
    assert result.doc_date.value == "01/04/2026"  # the challan's, chosen by its label
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    message = date_warnings[0].message
    assert "01-Apr-2026" in message
    assert "25/04/2026" in message  # the date the header actually prints
    assert "confirm" in message.lower()


def test_an_unlabelled_competing_date_warns_when_no_invoice_date_label_settled_it():
    text = "Date: 25/04/2026\nTerms: interest charged from 30/05/2026\n"
    result = extract_rules(text)
    assert result.doc_date.value == "25/04/2026"
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "30/05/2026" in date_warnings[0].message


def test_an_invoice_date_label_still_settles_it_silently():
    """The warning exists for documents with no 'Invoice Date' label; this one has it."""
    text = "Invoice Date: 25/04/2026\nTerms: interest charged from 30/05/2026\n"
    result = extract_rules(text)
    assert result.doc_date.value == "25/04/2026"
    assert warnings_for(result, "DocDtls.Dt") == []


# --- an invoice TITLE above a "Date" line is not an invoice-date label -------

TITLE_ABOVE_A_DATE_LINE = (
    # (fixture, the invoice date transcribed by hand, whether the stage must warn)
    #
    # The title's last word plus the next line's first word read as 'Invoice Date',
    # the most specific label there is, so the supply date wins over the invoice's
    # own date printed two lines down.
    ("TAX INVOICE\nDate of Supply: 25/07/2026\nInvoice Date: 30/07/2026\n", "30/07/2026", False),
    # the same with the abbreviation the contract also lists
    ("INVOICE\nDt: 22/05/2026\nInvoice Date: 30/05/2026\n", "30/05/2026", False),
    # no invoice-date label anywhere: the title must not manufacture one, because tier 0
    # is the tier that silences the competing-date warning
    ("TAX INVOICE\nDate of Supply: 25/07/2026\nDue Date: 30/08/2026\n", "25/07/2026", True),
    # a ledger-style table under the title: its first column is dates
    (
        "TAX INVOICE\nDate       Particulars        Amount\n"
        "25/07/2026  Opening balance   1,000.00\n"
        "Invoice Date: 31/07/2026\n",
        "31/07/2026",
        False,
    ),
)


@pytest.mark.parametrize(("text", "expected", "warns"), TITLE_ABOVE_A_DATE_LINE)
def test_the_invoice_title_above_a_date_line_is_not_an_invoice_date_label(text, expected, warns):
    """The date-side twin of the house-number defect, and the quieter of the two.

    'TAX INVOICE' printed above a line beginning 'Date' matched the tier-0
    invoice-date label across the line break. Tier 0 is the tier that settles the
    date SILENTLY, so a supply date, a due date or a ledger column's first cell was
    filed as the mandatory DocDtls.Dt with nothing at all to show for it — and a
    well-formed wrong date is exactly the extraction no later check can question.
    """
    result = extract_rules(text)
    assert result.doc_date is not None, "no date was extracted at all"
    assert result.doc_date.value == expected
    assert bool(warnings_for(result, "DocDtls.Dt")) is warns


def test_a_title_over_a_date_line_no_longer_silences_the_competing_date_warning():
    """The document prints no invoice-date label, so the choice between its two dates warns."""
    result = extract_rules("TAX INVOICE\nDate of Supply: 25/07/2026\nDue Date: 30/08/2026\n")
    assert result.doc_date.value == "25/07/2026"
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "30/08/2026" in date_warnings[0].message


@pytest.mark.parametrize("text", [fixture for fixture, _, _ in TITLE_ABOVE_A_DATE_LINE])
def test_the_invoice_date_label_never_matches_across_the_title_line_break(text):
    """Pin the tier-0 pattern itself on the four documents that exposed it."""
    for match in DATE_LABEL_TIERS[0].finditer(text):
        assert "\n" not in match.group(0)


@pytest.mark.parametrize(
    "text",
    [
        # the invoice-date cell is blank, so the next cell's label and value sit where
        # a date after its own label would
        "Invoice Date:        Due Date: 30/05/2026\n",
        # the column-header form of the same layout
        "Invoice No.   Invoice Date   Due Date\nINV-0042                     30/05/2026\n",
    ],
)
def test_an_empty_invoice_date_cell_never_files_the_due_date_silently(text):
    """The date-side analogue of the blank invoice-number cell, and the more dangerous:
    a due date is a well-formed date that no downstream validator can question."""
    result = extract_rules(text)
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert date_warnings != [], "the due date was filed as the invoice date in silence"
    message = date_warnings[0].message
    assert "Invoice Date" in message
    assert "the 'Due Date' label" in message
    assert "30/05/2026" in message
    assert "confirm" in message.lower()


def test_an_invoice_date_label_cannot_claim_a_date_across_another_date_label():
    """'Invoice Date:   Due Date: 30/05/2026' is one blank cell and one filled one."""
    result = extract_rules("Invoice Date:        Due Date: 30/05/2026\n")
    assert result.doc_date.value == "30/05/2026"  # the only date the document prints
    assert len(warnings_for(result, "DocDtls.Dt")) == 1


def test_an_invoice_date_with_its_own_value_is_unaffected_by_the_due_date_beside_it():
    """The guard must not start warning about documents whose own label is satisfied."""
    result = extract_rules("Invoice Date: 25/04/2026    Due Date: 30/05/2026\n")
    assert result.doc_date.value == "25/04/2026"
    assert warnings_for(result, "DocDtls.Dt") == []


def test_the_first_date_with_no_label_at_all_does_not_warn_about_competition():
    text = "Dispatched 25/04/2026 by road\nDelivered 30/05/2026\n"
    result = extract_rules(text)
    assert result.doc_date.value == "25/04/2026"
    assert warnings_for(result, "DocDtls.Dt") == []


@pytest.mark.parametrize(
    "text",
    [
        "Bill No. 7788   Date 03/04/2026\nDue 30/05/2026\n",  # a native column gap
        "Bill No. 7788 Date 03/04/2026\nDue 30/05/2026\n",  # the same page through OCR
    ],
)
def test_a_collapsed_column_gap_does_not_put_the_cell_beside_the_label_into_its_name(text):
    """OCR joins a line's cells with single spaces, so the gap cannot end the label.

    Quoting 'Bill No. 7788 Date' names a label the document does not print, and puts
    the invoice number into a warning about the date.
    """
    message = warnings_for(extract_rules(text), "DocDtls.Dt")[0].message
    assert "the 'Date' label" in message
    assert "7788" not in message


def test_a_word_qualifying_the_label_is_still_part_of_it():
    """The trim must not reduce 'Due Date' to 'Date': which label supplied it is the point."""
    message = warnings_for(
        extract_rules("Due Date: 30/05/2026\nDelivery Date: 01/06/2026\n"), "DocDtls.Dt"
    )[0].message
    assert "the 'Due Date' label" in message
    assert "'Delivery Date' -> 01/06/2026" in message


def test_missing_date_is_reported():
    result = extract_rules(f"Umang Fabrics\nGSTIN {MAHARASHTRA}\nInvoice No 7788\n")
    assert result.doc_date is None
    date_warnings = warnings_for(result, "DocDtls.Dt")
    assert len(date_warnings) == 1
    assert "No date" in date_warnings[0].message


# --- HSN / SAC --------------------------------------------------------------


def test_hsn_on_a_labelled_line():
    result = extract_rules("1  Consulting  HSN 998313  2 Nos\n")
    assert [f.value for f in result.hsn_codes] == ["998313"]
    assert result.hsn_codes[0].method == "regex"


def test_hsn_within_forty_characters_after_a_label():
    text = "HSN/SAC Code\n998313\n"
    assert [f.value for f in extract_rules(text).hsn_codes] == ["998313"]


def test_hsn_beyond_forty_characters_after_a_label_is_not_taken():
    text = "HSN/SAC Code\n" + "." * 45 + "\n998313\n"
    assert extract_rules(text).hsn_codes == ()


@pytest.mark.parametrize(
    ("digits", "taken"),
    [("8471", True), ("998313", True), ("52081190", True), ("12345", False), ("1234567", False), ("847", False)],
)
def test_only_four_six_and_eight_digit_codes_are_taken(digits, taken):
    result = extract_rules(f"Item  HSN {digits}  Qty 1\n")
    assert [f.value for f in result.hsn_codes] == ([digits] if taken else [])


def test_hsn_deduplicates_by_value_and_keeps_document_order():
    text = (
        "1  Fabric   HSN 52081190\n"
        "2  Service  HSN 998313\n"
        "3  Fabric   HSN 52081190\n"
    )
    assert [f.value for f in extract_rules(text).hsn_codes] == ["52081190", "998313"]


def test_a_repeated_hsn_code_is_deduplicated_but_every_occurrence_is_consumed():
    text = "1  Fabric  HSN 52081190\n2  Fabric  HSN 52081190\n"
    result = extract_rules(text)
    assert len(result.hsn_codes) == 1
    assert "52081190" not in remaining_text(text, result.consumed)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1   Cotton fabric     HSN 52081190  10 Mtr  1200.00  12000.00\n", ["52081190"]),
        ("1   Consulting  HSN 8471  Rate 3.4500\n", ["8471"]),
    ],
)
def test_an_amount_on_an_hsn_line_is_not_taken_as_a_code(text, expected):
    """'1200' in '1200.00' is half an amount, not a standalone code."""
    assert [f.value for f in extract_rules(text).hsn_codes] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1   Cotton fabric   HSN 52081190   10 Mtr   1200   12000", ["52081190"]),
        ("1   Consulting      SAC 998313      2 Nos   5000   10000", ["998313"]),
    ],
)
def test_a_bare_rate_or_amount_on_an_hsn_row_is_not_a_code(text, expected):
    """A rate of 1200 is not an HSN code just because the row also carries the word HSN."""
    assert [f.value for f in extract_rules(text).hsn_codes] == expected


def test_a_column_header_table_yields_no_codes_rather_than_the_wrong_ones():
    """Part C reads HsnCd per item; a missed code is a miss, a swept-up rate is a wrong value."""
    text = "Sl Desc HSN Qty Rate Amount\n1 Fabric 5208 10 1200 12000"
    assert extract_rules(text).hsn_codes == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # a services row prints no unit, so nothing but spaces stands between the code
        # and the money columns
        ("1   Consulting fee   SAC 998313   1   5000   5000\n", ["998313"]),
        # the same row through OCR, which joins a line's cells with single spaces
        ("1 Consulting fee SAC 998313 1 5000 5000\n", ["998313"]),
        ("1 Cotton fabric HSN 52081190 10 1200 12000\n", ["52081190"]),
        ("2 Annual maintenance SAC 998313 1 12000 12000\n", ["998313"]),
    ],
)
def test_a_row_with_no_unit_column_does_not_file_its_rate_as_a_code(text, expected):
    """The unit word is what ended the code run, and these rows do not print one.

    'SAC 998313 1 5000 5000' otherwise files 5000 as a confirmed HsnCd — a plausible
    six-figure-looking code that was never a code — and consumes it, so stage 2 reads
    the row with its rate and its total blanked out.
    """
    assert [f.value for f in extract_rules(text).hsn_codes] == expected


@pytest.mark.parametrize(
    "text",
    [
        "1   Consulting fee   SAC 998313   1   5000   5000\n",
        "1 Consulting fee SAC 998313 1 5000 5000\n",
    ],
)
def test_the_money_columns_of_a_unit_less_row_survive_into_stage_twos_input(text):
    remaining = remaining_text(text, extract_rules(text).consumed)
    assert remaining.count("5000") == 2
    assert "998313" not in remaining


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # the word before the number names its column; the HSN label does not
        ("Amount 1200 HSN 998313\n", ["998313"]),
        ("Total 1200 HSN wise summary\n", []),
        ("HSN wise summary   Amount 5000\n", []),
        ("Taxable value 11800 HSN 998313\n", ["998313"]),
        # and the legitimate readings carry no such word
        ("52081190 - HSN of Cotton Fabric\n", ["52081190"]),
        ("Taxable value under 998313 (SAC)\n", ["998313"]),
    ],
)
def test_a_number_a_money_column_names_is_not_a_code_beside_the_label(text, expected):
    """'Amount 1200  HSN 998313' prints two cells, not a code before its label."""
    assert [f.value for f in extract_rules(text).hsn_codes] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # more than one code after a single label
        ("HSN/SAC : 998313 / 52081190", ["998313", "52081190"]),
        ("HSN/SAC Code : 52081190, 998313, 8471", ["52081190", "998313", "8471"]),
        # the code is printed before the label on the line, across punctuation
        ("Taxable value under 998313 (SAC)", ["998313"]),
        ("52081190 - HSN of Cotton Fabric", ["52081190"]),
        ("HSN of Cotton Fabric: 52081190", ["52081190"]),
    ],
)
def test_a_code_on_an_hsn_line_is_taken_whichever_side_of_the_label_it_sits(text, expected):
    """The contract takes a code from a line containing HSN or SAC, not only after it.

    Only the first code after the label was read, so every later one on the row was
    dropped and one printed before the label was never seen — silently, with a short
    hsn_codes list as the only evidence.
    """
    assert [f.value for f in extract_rules(text).hsn_codes] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # the numeric columns of an item row: a word separates them from the code
        ("1   Cotton fabric   HSN 52081190   10 Mtr   1200   12000", ["52081190"]),
        ("1   Consulting      SAC 998313      2 Nos   5000   10000", ["998313"]),
        ("1  Fabric  HSN 52081190  10 Mtr  1,200.00  12,000.00  400002 Mumbai", ["52081190"]),
        # a date's year is four standalone digits, and a serial's middle is not a code
        ("Invoice Date 25 Apr 2026  HSN 998313", ["998313"]),
        ("HSN 2026-27/0042", []),
        ("Invoice No. 24-25/0731  HSN 998313", ["998313"]),
    ],
)
def test_reading_along_the_hsn_line_stops_at_the_next_word(text, expected):
    """A rate, a quantity, a PIN code or a date's year filed as HsnCd is a wrong value."""
    result = extract_rules(text)
    assert [f.value for f in result.hsn_codes] == expected


def test_the_numeric_columns_of_an_hsn_row_survive_into_stage_twos_input():
    """A swept-up rate would be blanked out of the row stage 2 has to read."""
    text = "1   Cotton fabric   HSN 52081190   10 Mtr   1200   12000\n"
    remaining = remaining_text(text, extract_rules(text).consumed)
    assert "10 Mtr   1200   12000" in remaining
    assert "52081190" not in remaining


def test_every_occurrence_of_a_code_read_before_its_label_is_consumed():
    text = "52081190 - HSN of Cotton Fabric\n"
    result = extract_rules(text)
    assert [f.value for f in result.hsn_codes] == ["52081190"]
    assert "52081190" not in remaining_text(text, result.consumed)


PIN_BEFORE_AN_HSN_LABEL = (
    # an address cell and an HSN cell sharing one row
    ("Mumbai 400002 HSN 998313\n", "400002", ["998313"]),
    ("Place of Supply: Bengaluru 560001  HSN 998313  2 Nos\n", "560001", ["998313"]),
    ("Ship to Delhi 110006 SAC 998313\n", "110006", ["998313"]),
    ("Delivery at Pune 411019 HSN/SAC 52081190\n", "411019", ["52081190"]),
    # OCR merging the address line with the item-table header: the PIN is the only
    # "code" on the line, so taking it makes it the document's only extracted code
    ("12 Kalbadevi Road, Mumbai 400002    HSN/SAC   Qty   Rate\n", "400002", []),
)


@pytest.mark.parametrize(
    ("text", "pin", "expected"),
    PIN_BEFORE_AN_HSN_LABEL,
)
def test_a_pin_code_printed_before_an_hsn_label_is_never_taken_as_a_code(text, pin, expected):
    """The contract names this number: "PIN codes and invoice numbers would be swept in".

    An Indian PIN is six digits, the shape of a six-digit HSN, so a PIN filed as a
    confirmed HsnCd is unquestionable by anything downstream. Only punctuation between
    the number and the label makes the label name it; a bare column gap does not.
    """
    assert [f.value for f in extract_rules(text).hsn_codes] == expected
    assert pin not in [f.value for f in extract_rules(text).hsn_codes]


@pytest.mark.parametrize(("text", "pin", "expected"), PIN_BEFORE_AN_HSN_LABEL)
def test_a_pin_code_survives_into_stage_twos_input(text, pin, expected):
    """SellerDtls.Pin and BuyerDtls.Pin are mandatory and stage 2 reads them from here."""
    result = extract_rules(text)
    assert pin in remaining_text(text, result.consumed)


def test_a_code_before_the_label_across_a_bare_column_gap_is_the_accepted_cost():
    """The other side of the trade, pinned so it is a decision rather than a surprise.

    "Goods 52081190 HSN 10 Mtr" prints a genuine code before the label with nothing but
    a space in between — structurally identical to "Mumbai 400002 HSN 998313", which is
    a PIN. Nothing distinguishes them, so the module errs the way it errs everywhere
    else: a missed code is a miss stage 2 still sees in the text, a wrong one is a
    confirmed wrong value that also blanks the real number out of stage 2's input.
    """
    text = "Goods 52081190 HSN 10 Mtr\n"
    result = extract_rules(text)
    assert result.hsn_codes == ()
    assert "52081190" in remaining_text(text, result.consumed)


def test_no_hsn_label_means_no_hsn_codes():
    text = "Umang Fabrics\nMumbai 400002\nInvoice No 7788\nPart number 8471 fitted\n"
    assert extract_rules(text).hsn_codes == ()


# --- state codes ------------------------------------------------------------


def test_state_codes_are_derived_not_read_from_the_address():
    text = (
        "Seller: Umang Fabrics\n"
        "Mumbai, State Code: 07\n"
        f"GSTIN: {MAHARASHTRA}\n"
        "Buyer: Farheen Traders\n"
        "Delhi, State Code: 27\n"
        f"GSTIN: {DELHI}\n"
    )
    result = extract_rules(text)
    assert result.seller_stcd.value == "27"
    assert result.buyer_stcd.value == "07"
    assert result.seller_stcd.method == "derived"
    assert result.buyer_stcd.method == "derived"


def test_a_legacy_state_code_is_derived_without_a_warning():
    text = f"Seller: A\nGSTIN: {LEGACY_28}\nBuyer: B\nGSTIN: {DELHI}\n"
    result = extract_rules(text)
    assert result.seller_stcd.value == "28"
    assert warnings_for(result, "SellerDtls.Stcd") == []


def test_a_discontinued_state_code_warns_with_the_lookup_note():
    text = f"Seller: A\nGSTIN: {DISCONTINUED_25}\nBuyer: B\nGSTIN: {DELHI}\n"
    result = extract_rules(text)
    assert result.seller_stcd.value == "25"
    stcd_warnings = warnings_for(result, "SellerDtls.Stcd")
    assert len(stcd_warnings) == 1
    assert DISCONTINUED_25 in stcd_warnings[0].message
    assert "discontinued in 2020" in stcd_warnings[0].message


def test_an_unknown_state_code_warns_with_the_lookup_note():
    text = f"Seller: A\nGSTIN: {MAHARASHTRA}\nBuyer: B\nGSTIN: {UNKNOWN_88}\n"
    result = extract_rules(text)
    assert result.buyer_stcd.value == "88"
    stcd_warnings = warnings_for(result, "BuyerDtls.Stcd")
    assert len(stcd_warnings) == 1
    assert "no GST state code 88" in stcd_warnings[0].message


def test_no_gstin_means_no_state_code():
    result = extract_rules("Umang Fabrics\nInvoice No 7788\n")
    assert result.seller_stcd is None
    assert result.buyer_stcd is None


# --- consumed spans ---------------------------------------------------------


def test_consumed_is_sorted_merged_and_non_overlapping():
    consumed = extract_rules(CLEAN_INVOICE).consumed
    assert list(consumed) == sorted(consumed, key=lambda s: s.start)
    for earlier, later in zip(consumed, consumed[1:]):
        assert earlier.end < later.start
        assert earlier.start < earlier.end


def test_consumed_covers_every_matched_span():
    result = extract_rules(CLEAN_INVOICE)
    covered = "".join(CLEAN_INVOICE[s.start : s.end] for s in result.consumed)
    for expected in (
        MAHARASHTRA,
        DELHI,
        "INV/2026-27/0042",
        "25/04/2026",
        "998313",
        "52081190",
    ):
        assert expected in covered
    assert covered.count("HSN") == 3  # the HSN labels that anchored a code


def test_a_rejected_gstin_candidate_is_still_consumed():
    text = f"GSTIN {BAD_CHECKSUM}\n"
    result = extract_rules(text)
    assert Span(text.index(BAD_CHECKSUM), text.index(BAD_CHECKSUM) + 15) in result.consumed


# --- remaining_text ---------------------------------------------------------


def test_remaining_text_blanks_each_span_without_moving_the_rest():
    text = "GSTIN: 27AAPFU0939F1ZV end"
    assert remaining_text(text, [Span(7, 22)]) == "GSTIN: " + " " * 15 + " end"


def test_remaining_text_sorts_merges_and_clamps_its_input():
    assert remaining_text("abcdefgh", [Span(5, 7), Span(1, 3), Span(2, 4)]) == "a   e  h"
    assert remaining_text("abc", [Span(-5, 100)]) == "   "
    assert remaining_text("abc", [Span(2, 2), Span(3, 1)]) == "abc"
    assert remaining_text("abc", []) == "abc"


def test_remaining_text_preserves_length_and_every_offset():
    """Part D maps an offset in this text back to the page it came from."""
    result = extract_rules(CLEAN_INVOICE)
    remaining = remaining_text(CLEAN_INVOICE, result.consumed)
    assert len(remaining) == len(CLEAN_INVOICE)
    for marker in ("Farheen Traders", "Cotton fabric", "5,000.00"):
        assert remaining.index(marker) == CLEAN_INVOICE.index(marker)


def test_remaining_text_strips_what_stage_one_confirmed():
    result = extract_rules(CLEAN_INVOICE)
    remaining = remaining_text(CLEAN_INVOICE, result.consumed)
    for gone in (MAHARASHTRA, DELHI, "INV/2026-27/0042", "25/04/2026", "998313", "52081190"):
        assert gone not in remaining
    for kept in ("Umang Fabrics Private Limited", "Farheen Traders", "Consulting fee", "5,000.00"):
        assert kept in remaining


# --- empty input ------------------------------------------------------------


def test_empty_text_reports_everything_as_missing():
    result = extract_rules("")
    assert result == RuleExtraction(
        seller_gstin=None,
        buyer_gstin=None,
        seller_stcd=None,
        buyer_stcd=None,
        doc_no=None,
        doc_date=None,
        hsn_codes=(),
        consumed=(),
        warnings=result.warnings,
    )
    assert {w.field for w in result.warnings} == {
        "SellerDtls.Gstin",
        "BuyerDtls.Gstin",
        "DocDtls.No",
        "DocDtls.Dt",
    }
