"""Tests for gst_einvoice.extract_llm.

Every LLM response here comes from a fake client: no network, no API key, no
patching. Expected values are transcribed by hand from the fixture text above,
never computed with the module under test.
"""

import dataclasses
import inspect
import json
from types import SimpleNamespace

import pytest

from gst_einvoice import extract_llm
from gst_einvoice.extract_llm import (
    DEFAULT_MODEL,
    TOTAL_KEYS,
    LlmExtraction,
    LlmItem,
    build_prompt,
    extract_with_llm,
    is_grounded,
    make_client,
)

# --------------------------------------------------------------------------- #
# Fixtures: the text stage 1 left behind, and what stage 1 confirmed
# --------------------------------------------------------------------------- #

# GSTINs, the invoice number, the date and the HSN codes have been consumed by
# stage 1, so they are NOT in this text -- they arrive only through `confirmed`.
INVOICE_TEXT = """\
TAX INVOICE
Sunrise Traders Pvt Ltd
14 Marine Drive, Nariman Point
Mumbai 400021

Bill To: Deccan Hardware LLP
9 Connaught Circus, Janpath
New Delhi 110001

Sl  Description           Qty  Unit  Rate     Amount   GST%  CGST    SGST    Total
1   Steel Bracket 12mm    10   NOS   150.00   1500.00  18    135.00  135.00  1770.00
2   Mounting Service      1    NOS   800.00   800.00   5     20.00   20.00   840.00

Taxable Value    2,300.00
CGST             155.00
SGST             155.00
Round Off        0.00
Grand Total      2,610.00
"""

CONFIRMED = {
    "SellerDtls.Gstin": "27AAPFU0939F1ZV",
    "BuyerDtls.Gstin": "07AAGFF2194N1Z1",
    "SellerDtls.Stcd": "27",
    "BuyerDtls.Stcd": "07",
    "DocDtls.No": "STP/2026/0417",
    "DocDtls.Dt": "09/04/2026",
    "HsnCd": ["7326", "998729"],
}

# The same document numbered the way Tally, Vyapar and a great many hand-written
# books number one: a bare running number. "1042" is an invoice number here, not
# a figure anybody could read off the page as money.
NUMERIC_DOC_NO_CONFIRMED = dict(CONFIRMED, **{"DocDtls.No": "1042"})

# Transcribed by hand from INVOICE_TEXT above, not produced by any code here.
GOOD_PAYLOAD = {
    "items": [
        {
            "SlNo": "1",
            "PrdDesc": "Steel Bracket 12mm",
            "HsnCd": "7326",
            "Qty": 10,
            "Unit": "NOS",
            "UnitPrice": 150.00,
            "TotAmt": 1500.00,
            "Discount": None,
            "AssAmt": 1500.00,
            "GstRt": 18,
            "CgstAmt": 135.00,
            "SgstAmt": 135.00,
            "IgstAmt": None,
            "CesAmt": None,
            "TotItemVal": 1770.00,
        },
        {
            "SlNo": "2",
            "PrdDesc": "Mounting Service",
            "HsnCd": "998729",
            "Qty": 1,
            "Unit": "NOS",
            "UnitPrice": 800.00,
            "TotAmt": 800.00,
            "Discount": None,
            "AssAmt": 800.00,
            "GstRt": 5,
            "CgstAmt": 20.00,
            "SgstAmt": 20.00,
            "IgstAmt": None,
            "CesAmt": None,
            "TotItemVal": 840.00,
        },
    ],
    "seller": {
        "LglNm": "Sunrise Traders Pvt Ltd",
        "Addr1": "14 Marine Drive, Nariman Point",
        "Loc": "Mumbai",
        "Pin": "400021",
    },
    "buyer": {
        "LglNm": "Deccan Hardware LLP",
        "Addr1": "9 Connaught Circus, Janpath",
        "Loc": "New Delhi",
        "Pin": "110001",
    },
    "totals": {
        "AssVal": 2300.00,
        "CgstVal": 155.00,
        "SgstVal": 155.00,
        "IgstVal": None,
        "CesVal": None,
        "TotInvVal": 2610.00,
        "RndOffAmt": 0.00,
    },
}

# Indian invoices print credits in accounting brackets: this document's discount
# is MINUS 250.00 and its round off is MINUS 0.40. The bare digits of both are
# printed on the page, so the grounding check alone would wave them through.
CREDIT_TEXT = """\
1   Steel Bracket 12mm    10   NOS   150.00   1500.00
Discount                  (250.00)
Round Off                 (0.40)
Grand Total               1250.00
"""

# The same credit in the other notation Indian invoices use.
CR_TEXT = """\
1   Steel Bracket 12mm    10   NOS   150.00   1500.00
Round Off                 0.40 CR
Grand Total               1499.60
"""

# The two remaining ways a printed figure carries a minus: a Unicode minus
# (U+2212, what a typeset PDF emits) and a trailing minus (Tally, SAP). In both
# the discount below is MINUS 250.00. A detached ASCII hyphen is NOT one of
# them -- see HYPHEN_TEXT.
MINUS_TEXTS = {
    "unicode": "Discount                  −250.00\nGrand Total               1250.00\n",
    "trailing": "Discount                  250.00-\nGrand Total               1250.00\n",
}

# The same trailing minus, but as one column among several -- Tally prints a
# running balance beside the figure. The minus no longer ends the line, so this
# stage does not read it as a sign: the printed credit of 250.00 is read as a
# positive charge. That is a documented limitation, not an accident, and the
# tests below pin it so it cannot drift silently.
TALLY_COLUMN_TEXT = """\
Discount                  250.00-      1,250.00
Grand Total               1250.00
"""

# A detached ASCII hyphen is a label separator at least as often as it is a
# minus, and on this layout -- the common one -- every figure is a positive
# charge. Reading the hyphen as a sign would ground the inverted reading of each
# of these and drop the correct one.
HYPHEN_TEXT = """\
Freight - 200.00
Sub Total - 1500.00
CGST @ 9% - 135.00
SGST @ 9% - 135.00
Grand Total - 1770.00
"""

# A minus between two figures is a subtraction, not a sign: 250.00 is a charge
# here and the model reading it as +250.00 is reading it correctly.
SUBTRACTION_TEXT = """\
1   Steel Bracket 12mm    10   NOS   150.00   1500.00
Net of discount           1500.00 − 250.00
"""

# A document with no buyer block and no tax split at all: the fields the model
# cannot find here are genuinely absent, not merely hard to read.
SPARSE_TEXT = """\
TAX INVOICE
Sunrise Traders Pvt Ltd
Mumbai 400021

1   Steel Bracket 12mm   10   NOS   150.00   1500.00
Grand Total   1500.00
"""


# --------------------------------------------------------------------------- #
# Fake Groq client
# --------------------------------------------------------------------------- #


class FakeCompletions:
    """Records every call and replays one canned response body."""

    def __init__(self, content):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Exposes exactly the ``client.chat.completions.create`` surface the contract uses."""

    def __init__(self, content):
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict]:
        return self.completions.calls


def run(payload, *, remaining=INVOICE_TEXT, confirmed=None, **kwargs):
    """Run the stage against a canned response; return (extraction, client)."""
    client = FakeClient(payload)
    result = extract_with_llm(
        remaining,
        CONFIRMED if confirmed is None else confirmed,
        client=client,
        **kwargs,
    )
    return result, client


def item_payload(**overrides) -> dict:
    """One line-item row: the first row of GOOD_PAYLOAD with fields replaced."""
    row = dict(GOOD_PAYLOAD["items"][0])
    row.update(overrides)
    return {"items": [row]}


def warnings_for(result: LlmExtraction, path: str) -> list:
    return [w for w in result.warnings if w.field == path]


def message_for(result: LlmExtraction, path: str) -> str:
    matches = warnings_for(result, path)
    assert len(matches) == 1, f"expected exactly one warning on {path}, got {matches}"
    return matches[0].message


# --------------------------------------------------------------------------- #
# Module contract
# --------------------------------------------------------------------------- #


def test_default_model_is_the_one_the_live_validation_ran_against():
    """The default must be a model that exists. The previous one had been retired by
    Groq and returned HTTP 404, breaking the first run for anyone who did not override
    it; this is the model the twelve-invoice live validation actually used."""
    assert DEFAULT_MODEL == "openai/gpt-oss-120b"


def test_total_keys_are_the_seven_valdtls_fields():
    assert TOTAL_KEYS == (
        "AssVal",
        "CgstVal",
        "SgstVal",
        "IgstVal",
        "CesVal",
        "TotInvVal",
        "RndOffAmt",
    )


def test_client_is_keyword_only_with_no_default():
    """The module must never be able to build a client implicitly."""
    parameter = inspect.signature(extract_with_llm).parameters["client"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_calling_without_a_client_is_an_error():
    with pytest.raises(TypeError):
        extract_with_llm(INVOICE_TEXT, CONFIRMED)


# --------------------------------------------------------------------------- #
# The prompt: enforcement point 1
# --------------------------------------------------------------------------- #


def flat(text: str) -> str:
    """Whitespace-collapsed prompt, so assertions do not depend on line wrapping."""
    return " ".join(text.split())


def test_prompt_states_the_structure_not_invent_constraint():
    prompt = flat(build_prompt(INVOICE_TEXT, CONFIRMED))
    assert "you may STRUCTURE text, you may never INVENT values" in prompt
    assert "Every value you output must already appear in the INVOICE TEXT" in prompt
    assert "Never infer, complete, correct, translate or invent a value" in prompt
    assert "never calculate one that is not printed" in prompt


def test_prompt_requires_null_for_anything_not_found():
    prompt = flat(build_prompt(INVOICE_TEXT, CONFIRMED))
    assert "Any field you cannot find in the INVOICE TEXT must be null" in prompt
    assert "Do not guess" in prompt


def test_prompt_warns_that_ungrounded_values_are_dropped():
    prompt = flat(build_prompt(INVOICE_TEXT, CONFIRMED))
    assert "Every value is checked against the INVOICE TEXT after you answer" in prompt


def test_prompt_marks_confirmed_fields_as_settled():
    prompt = flat(build_prompt(INVOICE_TEXT, CONFIRMED))
    assert "do not re-derive, re-check or contradict them" in prompt
    assert "SellerDtls.Gstin: 27AAPFU0939F1ZV" in prompt
    assert "DocDtls.Dt: 09/04/2026" in prompt
    assert "HsnCd: 7326, 998729" in prompt


def test_prompt_carries_the_remaining_text_and_the_json_shape():
    prompt = build_prompt(INVOICE_TEXT, CONFIRMED)
    assert INVOICE_TEXT in prompt
    for key in ('"items"', '"PrdDesc"', '"HsnCd"', '"seller"', '"buyer"', '"totals"'):
        assert key in prompt
    for key in TOTAL_KEYS:
        assert f'"{key}"' in prompt


def test_prompt_without_confirmed_fields_says_nothing_was_confirmed():
    prompt = build_prompt(INVOICE_TEXT, {})
    assert "(nothing was confirmed by stage 1)" in prompt


def test_prompt_skips_confirmed_fields_that_are_none():
    prompt = build_prompt(INVOICE_TEXT, {"DocDtls.No": None, "DocDtls.Dt": "09/04/2026"})
    assert "DocDtls.No" not in prompt
    assert "DocDtls.Dt: 09/04/2026" in prompt


# --------------------------------------------------------------------------- #
# The call itself
# --------------------------------------------------------------------------- #


def test_call_uses_the_documented_client_shape():
    _, client = run(GOOD_PAYLOAD)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"] == [
        {"role": "user", "content": build_prompt(INVOICE_TEXT, CONFIRMED)}
    ]


def test_model_can_be_overridden():
    _, client = run(GOOD_PAYLOAD, model="llama-3.1-8b-instant")
    assert client.calls[0]["model"] == "llama-3.1-8b-instant"


def test_an_unparsable_response_is_not_retried():
    _, client = run("I am sorry, I cannot help with that.")
    assert len(client.calls) == 1


# --------------------------------------------------------------------------- #
# is_grounded
# --------------------------------------------------------------------------- #


def test_grounded_string_is_a_substring_of_the_source():
    assert is_grounded("Steel Bracket 12mm", INVOICE_TEXT) is True
    assert is_grounded("Deccan Hardware LLP", INVOICE_TEXT) is True


def test_string_grounding_collapses_whitespace_and_folds_case():
    assert is_grounded("steel   bracket\n12MM", INVOICE_TEXT) is True
    assert is_grounded("NARIMAN POINT", INVOICE_TEXT) is True


def test_absent_string_is_not_grounded():
    assert is_grounded("Premium Extended Warranty", INVOICE_TEXT) is False


def test_a_one_character_string_grounds_only_as_a_token_of_its_own():
    """Substring presence proves nothing of one character; a token of its own is the test."""
    # Transcribed by hand from INVOICE_TEXT: the Sl column prints "1" and "2" as
    # tokens of their own, and no "3" is printed anywhere except inside 2,300.00.
    assert is_grounded("1", INVOICE_TEXT) is True
    assert is_grounded("2", INVOICE_TEXT) is True
    assert is_grounded("3", INVOICE_TEXT) is False
    # "S" occurs in "Sl", "Steel" and "SGST"; none of them is an "S" of its own.
    assert is_grounded("S", INVOICE_TEXT) is False
    assert is_grounded("", INVOICE_TEXT) is False
    assert is_grounded("  ", INVOICE_TEXT) is False


def test_two_character_and_longer_grounding_is_unchanged_by_that_rule():
    """The floor moved for one-character values only: longer ones still ground as substrings."""
    assert is_grounded("10", INVOICE_TEXT) is True
    assert is_grounded("NOS", INVOICE_TEXT) is True
    # "os" is printed nowhere as a token, only inside "NOS" -- and still grounds,
    # because substring grounding is what the contract specifies for a value this
    # long. Tightening that was never the ruling.
    assert is_grounded("os", INVOICE_TEXT) is True
    assert is_grounded("zz", INVOICE_TEXT) is False


def test_numbers_are_grounded_by_a_matching_numeric_token():
    assert is_grounded(1500.00, INVOICE_TEXT) is True
    assert is_grounded(1500, INVOICE_TEXT) is True
    assert is_grounded(9999.00, INVOICE_TEXT) is False


def test_number_grounding_tolerates_half_a_paisa_and_no_more():
    assert is_grounded(1500.004, INVOICE_TEXT) is True
    assert is_grounded(1500.01, INVOICE_TEXT) is False


def test_number_grounding_strips_indian_thousands_separators():
    assert is_grounded(2300.00, INVOICE_TEXT) is True
    assert is_grounded(2610.00, INVOICE_TEXT) is True
    assert is_grounded(123456.78, "Grand Total 1,23,456.78") is True


def test_gst_rate_is_grounded_by_a_percent_sign_or_trailing_zeros():
    assert is_grounded(18, "GST Rate 18%") is True
    assert is_grounded(18, "GST Rate 18.00") is True
    assert is_grounded(18, "GST Rate 12%") is False


def test_non_scalar_and_boolean_values_are_never_grounded():
    assert is_grounded(None, INVOICE_TEXT) is False
    assert is_grounded(True, INVOICE_TEXT) is False
    assert is_grounded(["Steel Bracket 12mm"], INVOICE_TEXT) is False


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_grounded_items_survive_with_their_values():
    result, _ = run(GOOD_PAYLOAD)
    assert len(result.items) == 2
    first, second = result.items
    assert first.PrdDesc == "Steel Bracket 12mm"
    assert first.Qty == 10.0
    assert first.Unit == "NOS"
    assert first.UnitPrice == 150.00
    assert first.AssAmt == 1500.00
    assert first.GstRt == 18.0
    assert first.CgstAmt == 135.00
    assert first.SgstAmt == 135.00
    assert first.TotItemVal == 1770.00
    assert second.PrdDesc == "Mounting Service"
    assert second.GstRt == 5.0
    assert second.TotItemVal == 840.00


def test_grounded_parties_and_totals_survive():
    result, _ = run(GOOD_PAYLOAD)
    assert result.seller_name == "Sunrise Traders Pvt Ltd"
    assert result.seller_addr1 == "14 Marine Drive, Nariman Point"
    assert result.seller_loc == "Mumbai"
    assert result.seller_pin == "400021"
    assert result.buyer_name == "Deccan Hardware LLP"
    assert result.buyer_addr1 == "9 Connaught Circus, Janpath"
    assert result.buyer_loc == "New Delhi"
    assert result.buyer_pin == "110001"
    assert result.totals["AssVal"] == 2300.00
    assert result.totals["CgstVal"] == 155.00
    assert result.totals["SgstVal"] == 155.00
    assert result.totals["TotInvVal"] == 2610.00
    assert result.totals["RndOffAmt"] == 0.00


def test_totals_always_carry_every_key():
    result, _ = run(GOOD_PAYLOAD)
    assert tuple(result.totals) == TOTAL_KEYS
    assert result.totals["IgstVal"] is None
    assert result.totals["CesVal"] is None


def test_mixed_gst_rates_across_line_items_are_kept_per_item():
    result, _ = run(GOOD_PAYLOAD)
    assert [item.GstRt for item in result.items] == [18.0, 5.0]


def test_a_value_confirmed_by_stage_one_grounds_even_though_it_was_stripped():
    """The HSN codes live only in `confirmed`; a model echoing them is not hallucinating."""
    assert "7326" not in INVOICE_TEXT
    result, _ = run(GOOD_PAYLOAD)
    assert result.items[0].HsnCd == "7326"
    assert result.items[1].HsnCd == "998729"
    assert result.provenance["ItemList[0].HsnCd"] == {"source": "llm", "ocr_confidence": None}


def test_an_hsn_copied_from_the_confirmed_list_says_the_row_assignment_is_unverified():
    """Which confirmed code belongs to which row is the model's call; grounding cannot check it."""
    result, _ = run(GOOD_PAYLOAD)
    swapped, _ = run(
        {
            "items": [
                dict(GOOD_PAYLOAD["items"][0], HsnCd="998729"),
                dict(GOOD_PAYLOAD["items"][1], HsnCd="7326"),
            ]
        }
    )
    # Both readings survive grounding -- both codes are in `confirmed` -- so the
    # warning is the only thing standing between a swapped SAC and a filed return.
    assert swapped.items[0].HsnCd == "998729"
    for extraction in (result, swapped):
        for index in (0, 1):
            message = message_for(extraction, f"ItemList[{index}].HsnCd")
            assert "stage 1 confirmed" in message
            assert "could not be corroborated" in message
            assert "confirm it against the invoice" in message
            assert "dropped rather than kept" not in message


def test_the_hsn_row_assignment_note_is_info_not_a_warning():
    """It fires on every successful extraction, so as a warning it carries no information.

    Stage 1 strips the codes it confirms, so stage 2 structurally cannot
    corroborate which row a code belongs to: the note reports a limitation of the
    two-stage design rather than anything wrong with this document. A warning on
    100% of documents teaches an accountant to skim the warning stream, which
    costs the warnings that matter their credibility.
    """
    result, _ = run(GOOD_PAYLOAD)
    notes = warnings_for(result, "ItemList[0].HsnCd") + warnings_for(result, "ItemList[1].HsnCd")
    assert len(notes) == 2
    assert all(note.severity == "info" for note in notes)
    # The text is unchanged: only the severity moved.
    for note in notes:
        assert "could not be corroborated" in note.message
        assert "confirm it against the invoice" in note.message


def test_an_hsn_printed_on_the_row_itself_needs_no_assignment_warning():
    text = "1   Steel Bracket 12mm   7326   10   NOS   150.00   1500.00\n"
    result, _ = run(item_payload(HsnCd="7326"), remaining=text, confirmed={})
    assert result.items[0].HsnCd == "7326"
    assert warnings_for(result, "ItemList[0].HsnCd") == []


def test_a_value_only_in_the_stripped_text_is_not_grounded_when_confirmed_is_empty():
    result, _ = run(GOOD_PAYLOAD, confirmed={})
    assert result.items[0].HsnCd is None
    assert "7326" in message_for(result, "ItemList[0].HsnCd")


def test_an_hsn_that_is_only_part_of_a_longer_number_still_warns_about_its_row():
    """The PIN 400021 is printed here; "4000" is not a code of its own anywhere on it."""
    text = "Sunrise Traders Pvt Ltd\nMumbai 400021\n1   Steel Bracket 12mm   10   NOS\n"
    result, _ = run(item_payload(HsnCd="4000"), remaining=text, confirmed={})
    message = message_for(result, "ItemList[0].HsnCd")
    assert "not printed as a code of its own" in message
    assert "confirm it against the invoice" in message


def test_a_pin_grounded_only_inside_a_longer_number_says_so():
    """110001 is printed here only inside an account number, and Pin is mandatory in INV-01."""
    text = "Sunrise Traders\nA/c No 1100015678\n1   Steel Bracket 12mm   10   NOS\n"
    payload = {"items": [], "seller": {"Pin": "110001"}}
    result, _ = run(payload, remaining=text, confirmed={})
    assert result.seller_pin == "110001"
    message = message_for(result, "SellerDtls.Pin")
    assert "not printed as a number of its own" in message
    assert "confirm it against the invoice" in message


def test_a_pin_corroborated_only_by_a_confirmed_identifier_says_so():
    """A PIN-shaped invoice number is not a PIN code, and no PIN is printed here.

    The string grounding source is the remaining text PLUS the confirmed values,
    so a confirmed identifier grounds this value all by itself. That corroborates
    that the digits are an identifier, never that they are this document's PIN,
    so the warning must fire exactly as it does for the HsnCd twin.
    """
    text = "Sunrise Traders Pvt Ltd\n1   Steel Bracket 12mm   10   NOS   150.00   1500.00\n"
    assert "400021" not in text
    payload = {"items": [], "seller": {"Pin": "400021"}}
    result, _ = run(payload, remaining=text, confirmed={"DocDtls.No": "400021"})
    assert result.seller_pin == "400021"
    message = message_for(result, "SellerDtls.Pin")
    assert "not printed as a number of its own" in message
    assert "confirm it against the invoice" in message


def test_a_pin_corroborated_only_by_a_confirmed_state_code_says_so():
    """PartyDtls.Pin is an int in INV-01, so a state code would ship as a PIN code."""
    text = "Sunrise Traders Pvt Ltd\n1   Steel Bracket 12mm   10   NOS   150.00   1500.00\n"
    assert "27" not in text
    payload = {"items": [], "seller": {"Pin": "27"}}
    result, _ = run(payload, remaining=text, confirmed={"SellerDtls.Stcd": "27"})
    assert result.seller_pin == "27"
    assert "not printed as a number of its own" in message_for(result, "SellerDtls.Pin")


def test_the_pin_and_hsn_corroboration_checks_ask_the_same_text_the_same_question():
    """The twins must not disagree about what corroborates a digit-only field."""
    text = "Sunrise Traders Pvt Ltd\n1   Steel Bracket 12mm   10   NOS   150.00   1500.00\n"
    confirmed = {"HsnCd": ["400021"], "DocDtls.No": "400021"}
    payload = {"items": [{"HsnCd": "400021"}], "seller": {"Pin": "400021"}}
    result, _ = run(payload, remaining=text, confirmed=confirmed)
    assert result.items[0].HsnCd == "400021"
    assert result.seller_pin == "400021"
    assert len(warnings_for(result, "ItemList[0].HsnCd")) == 1
    assert len(warnings_for(result, "SellerDtls.Pin")) == 1


def test_a_pin_printed_as_a_number_of_its_own_needs_no_such_warning():
    result, _ = run(GOOD_PAYLOAD)
    assert result.seller_pin == "400021"
    assert result.buyer_pin == "110001"
    assert warnings_for(result, "SellerDtls.Pin") == []
    assert warnings_for(result, "BuyerDtls.Pin") == []


def test_a_number_cannot_be_grounded_by_digits_inside_a_gstin_or_an_invoice_number():
    """417 and 939 are printed nowhere: they are digit runs inside STP/2026/0417 and the GSTIN."""
    assert "417" not in INVOICE_TEXT and "939" not in INVOICE_TEXT
    assert "2026" not in INVOICE_TEXT
    result, _ = run(item_payload(UnitPrice=417.00, Qty=939, TotAmt=2026))
    assert result.items[0].UnitPrice is None
    assert result.items[0].Qty is None
    assert result.items[0].TotAmt is None
    for name in ("UnitPrice", "Qty", "TotAmt"):
        assert f"ItemList[0].{name}" not in result.provenance
        assert "dropped rather than kept" in message_for(result, f"ItemList[0].{name}")


def test_digits_lost_to_the_identifier_narrowing_are_diagnosed_as_such():
    """The digits ARE in the confirmed context, so claiming they never appeared is untrue."""
    result, _ = run(item_payload(UnitPrice=417.00, Qty=939, TotAmt=2026))
    for name in ("UnitPrice", "Qty", "TotAmt"):
        message = message_for(result, f"ItemList[0].{name}")
        assert "did not appear in the source text" not in message
        assert "only inside a confirmed identifier" in message
        assert "GSTIN, invoice number or date" in message
        assert "could not be corroborated" in message
        assert "dropped rather than kept" in message
    # A figure that is in neither the text nor the confirmed context keeps the
    # other diagnosis, and the two stay distinct.
    absent, _ = run(item_payload(TotItemVal=1777.00))
    assert "did not appear in the source text" in message_for(absent, "ItemList[0].TotItemVal")


def test_a_confirmed_state_code_is_an_identifier_and_cannot_ground_a_quantity():
    """27 is Maharashtra's code, not a figure printed on the invoice for anyone to read."""
    assert "27" not in INVOICE_TEXT
    result, _ = run(item_payload(Qty=27))
    assert result.items[0].Qty is None
    assert "ItemList[0].Qty" not in result.provenance
    assert "only inside a confirmed identifier" in message_for(result, "ItemList[0].Qty")


def test_a_purely_numeric_invoice_number_cannot_ground_an_amount():
    """Indian invoice numbers are commonly bare digits; 1042 is a document number, not money."""
    assert "1042" not in INVOICE_TEXT
    result, _ = run(item_payload(TotAmt=1042.00), confirmed=NUMERIC_DOC_NO_CONFIRMED)
    assert result.items[0].TotAmt is None
    assert "ItemList[0].TotAmt" not in result.provenance
    message = message_for(result, "ItemList[0].TotAmt")
    assert "only inside a confirmed identifier" in message
    assert "dropped rather than kept" in message


def test_the_numeric_invoice_number_still_grounds_a_string_field():
    """Narrowed for numbers only: a model echoing the confirmed number as text is fine."""
    result, _ = run(item_payload(SlNo="1042"), confirmed=NUMERIC_DOC_NO_CONFIRMED)
    assert result.items[0].SlNo == "1042"


def test_a_confirmed_hsn_code_cannot_ground_an_amount_either():
    """7326 classifies the goods; it is not a figure the document prints as money."""
    assert "7326" not in INVOICE_TEXT
    result, _ = run(item_payload(TotAmt=7326.00))
    assert result.items[0].TotAmt is None
    assert "ItemList[0].TotAmt" not in result.provenance
    assert "only inside a confirmed identifier" in message_for(result, "ItemList[0].TotAmt")


def test_numbers_are_grounded_against_the_remaining_text_directly():
    """No confirmed value is a figure, so there is no partition of them to make."""
    assert not hasattr(extract_llm, "_partition_confirmed")
    assert not hasattr(extract_llm, "_NUMERIC_CONFIRMED_KEYS")


def test_an_amount_printed_on_the_document_is_unaffected_by_that_narrowing():
    """The narrowing must cost nothing that is actually printed: 1500.00 is on the page."""
    result, _ = run(item_payload(TotAmt=1500.00), confirmed=NUMERIC_DOC_NO_CONFIRMED)
    assert result.items[0].TotAmt == 1500.00
    assert result.provenance["ItemList[0].TotAmt"] == {"source": "llm", "ocr_confidence": None}


def test_every_warning_is_checked_by_this_stage():
    result, _ = run(GOOD_PAYLOAD)
    assert result.warnings
    assert all(w.check == "extract_llm" for w in result.warnings)
    assert all(w.severity in ("warning", "info") for w in result.warnings)


# Transcribed by hand: the four INV-01 fields with a schema default. GOOD_PAYLOAD
# leaves all of them null, and an invoice with no discount and no cess is an
# ordinary invoice, so their absence is `info`, not a gap in the reading.
EXPECTED_INFO_PATHS = {
    "ItemList[0].Discount",
    "ItemList[0].CesAmt",
    "ItemList[1].Discount",
    "ItemList[1].CesAmt",
    "ValDtls.CesVal",
    "ValDtls.RndOffAmt",
}

# The per-row HsnCd assignment note is `info` too, for a different reason: it
# reports a limitation of the two-stage design rather than anything wrong with
# this document. Both of GOOD_PAYLOAD's rows carry one.
EXPECTED_ASSIGNMENT_INFO_PATHS = {"ItemList[0].HsnCd", "ItemList[1].HsnCd"}


def test_only_the_optional_fields_report_their_absence_as_info():
    payload = dict(GOOD_PAYLOAD)
    payload["totals"] = dict(GOOD_PAYLOAD["totals"], RndOffAmt=None)
    result, _ = run(payload)
    absence_info = {
        w.field
        for w in result.warnings
        if w.severity == "info" and "was not found in the document" in w.message
    }
    assert absence_info == EXPECTED_INFO_PATHS
    assert {w.field for w in result.warnings if w.severity == "info"} == (
        EXPECTED_INFO_PATHS | EXPECTED_ASSIGNMENT_INFO_PATHS
    )
    # A mandatory field the model could not find still shouts.
    assert all(
        w.severity == "warning"
        for w in result.warnings
        if w.field in ("ItemList[0].IgstAmt", "ValDtls.IgstVal")
    )


def test_a_dropped_optional_field_is_a_warning_not_an_info():
    """`info` is for a field the document does not print, never for one we refused."""
    result, _ = run(item_payload(Discount=99.99))
    assert result.items[0].Discount is None
    assert warnings_for(result, "ItemList[0].Discount")[0].severity == "warning"


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

# Transcribed by hand: every field of GOOD_PAYLOAD that is both non-null and
# present in the source text. Discount/IgstAmt/CesAmt/IgstVal/CesVal are absent
# because they are null. SlNo is present: "1" and "2" are each printed as a token
# of their own in the Sl column, which is what a one-character value needs.
EXPECTED_PROVENANCE_PATHS = {
    "ItemList[0].SlNo",
    "ItemList[1].SlNo",
    "ItemList[0].PrdDesc",
    "ItemList[0].HsnCd",
    "ItemList[0].Qty",
    "ItemList[0].Unit",
    "ItemList[0].UnitPrice",
    "ItemList[0].TotAmt",
    "ItemList[0].AssAmt",
    "ItemList[0].GstRt",
    "ItemList[0].CgstAmt",
    "ItemList[0].SgstAmt",
    "ItemList[0].TotItemVal",
    "ItemList[1].PrdDesc",
    "ItemList[1].HsnCd",
    "ItemList[1].Qty",
    "ItemList[1].Unit",
    "ItemList[1].UnitPrice",
    "ItemList[1].TotAmt",
    "ItemList[1].AssAmt",
    "ItemList[1].GstRt",
    "ItemList[1].CgstAmt",
    "ItemList[1].SgstAmt",
    "ItemList[1].TotItemVal",
    "SellerDtls.LglNm",
    "SellerDtls.Addr1",
    "SellerDtls.Loc",
    "SellerDtls.Pin",
    "BuyerDtls.LglNm",
    "BuyerDtls.Addr1",
    "BuyerDtls.Loc",
    "BuyerDtls.Pin",
    "ValDtls.AssVal",
    "ValDtls.CgstVal",
    "ValDtls.SgstVal",
    "ValDtls.TotInvVal",
    "ValDtls.RndOffAmt",
}


def test_provenance_covers_exactly_the_surviving_fields_by_inv01_path():
    result, _ = run(GOOD_PAYLOAD)
    assert set(result.provenance) == EXPECTED_PROVENANCE_PATHS


def test_every_provenance_entry_names_the_llm_as_source():
    result, _ = run(GOOD_PAYLOAD)
    assert all(entry["source"] == "llm" for entry in result.provenance.values())
    assert all(set(entry) == {"source", "ocr_confidence"} for entry in result.provenance.values())


def test_ocr_confidence_is_none_without_a_lookup():
    result, _ = run(GOOD_PAYLOAD)
    assert all(entry["ocr_confidence"] is None for entry in result.provenance.values())


def test_dropped_and_missing_fields_get_no_provenance_entry():
    result, _ = run(GOOD_PAYLOAD)
    for path in ("ItemList[0].IgstAmt", "ValDtls.CesVal"):
        assert path not in result.provenance
    dropped, _ = run(item_payload(PrdDesc="Premium Extended Warranty"))
    assert "ItemList[0].PrdDesc" not in dropped.provenance


def test_low_ocr_confidence_survives_into_provenance_unrounded():
    """A shaky read must stay visible, not be laundered into a clean-looking value."""
    confidences = {"Steel Bracket 12mm": 0.71, "2,300.00": 0.42}
    result, _ = run(GOOD_PAYLOAD, confidence_lookup=confidences.get)
    assert result.provenance["ItemList[0].PrdDesc"]["ocr_confidence"] == 0.71
    assert result.provenance["ValDtls.AssVal"]["ocr_confidence"] == 0.42
    assert result.provenance["ItemList[0].Unit"]["ocr_confidence"] is None


def test_confidence_lookup_is_asked_about_the_source_text_of_the_value():
    """Numbers are looked up by the token as printed, so a page search can find it."""
    seen: list[str] = []

    def lookup(text):
        seen.append(text)
        return None

    run(GOOD_PAYLOAD, confidence_lookup=lookup)
    assert "2,300.00" in seen
    assert "1500.00" in seen
    assert "Steel Bracket 12mm" in seen


def test_confidence_lookup_is_asked_about_the_source_spelling_not_the_models():
    """Grounding tolerates re-casing and re-spacing, so the lookup must not see either."""
    seen: list[str] = []

    def lookup(text):
        seen.append(text)
        return None

    run(item_payload(PrdDesc="STEEL   BRACKET 12MM", Unit="nos"), confidence_lookup=lookup)
    assert "Steel Bracket 12mm" in seen
    assert "STEEL   BRACKET 12MM" not in seen
    assert "NOS" in seen
    assert "nos" not in seen


def test_a_renormalised_echo_does_not_launder_away_a_low_confidence_read():
    """The whole point of Part C: a shaky OCR read must not come back looking clean."""
    confidences = {"Steel Bracket 12mm": 0.41, "Mumbai": 0.38}
    result, _ = run(
        item_payload(PrdDesc="STEEL   BRACKET 12MM"),
        confidence_lookup=confidences.get,
    )
    # The confidence is looked up against the matched source text, so the shaky
    # read survives even though the kept value is the document's own spelling.
    assert result.items[0].PrdDesc == "Steel Bracket 12mm"
    assert result.provenance["ItemList[0].PrdDesc"]["ocr_confidence"] == 0.41


# The buyer's name printed twice, as a scanned invoice prints it: once in a
# clean header line, once in a smudged Bill To block. Transcribed by hand: the
# two printings differ in spacing, so an OCR-backed lookup tells them apart and
# reads them at 0.93 and 0.41.
TWICE_PRINTED_TEXT = """\
Deccan  Hardware LLP
9 Connaught Circus, Janpath

Bill To: Deccan Hardware LLP
New Delhi
"""
TWICE_PRINTED_CONFIDENCES = {"Deccan  Hardware LLP": 0.93, "Deccan Hardware LLP": 0.41}


def test_the_lowest_confidence_of_two_printings_is_the_one_recorded():
    """A clean second printing must not mask a shaky first one -- min, as confidence_for_span does."""
    payload = {"items": [], "buyer": {"LglNm": "Deccan Hardware LLP"}}
    result, _ = run(
        payload,
        remaining=TWICE_PRINTED_TEXT,
        confirmed={},
        confidence_lookup=TWICE_PRINTED_CONFIDENCES.get,
    )
    assert result.buyer_name == "Deccan Hardware LLP"
    assert result.provenance["BuyerDtls.LglNm"]["ocr_confidence"] == 0.41


def test_both_printings_are_asked_about_not_just_the_first():
    seen: list[str] = []

    def lookup(text):
        seen.append(text)
        return TWICE_PRINTED_CONFIDENCES.get(text)

    payload = {"items": [], "buyer": {"LglNm": "Deccan Hardware LLP"}}
    run(payload, remaining=TWICE_PRINTED_TEXT, confirmed={}, confidence_lookup=lookup)
    assert "Deccan  Hardware LLP" in seen
    assert "Deccan Hardware LLP" in seen


def test_an_occurrence_the_lookup_knows_nothing_about_does_not_erase_a_known_one():
    """None is no information, not a low confidence: it must neither win nor lower the record."""
    payload = {"items": [], "buyer": {"LglNm": "Deccan Hardware LLP"}}
    result, _ = run(
        payload,
        remaining=TWICE_PRINTED_TEXT,
        confirmed={},
        confidence_lookup={"Deccan Hardware LLP": 0.41}.get,
    )
    assert result.provenance["BuyerDtls.LglNm"]["ocr_confidence"] == 0.41


# The same figure printed twice and in two forms, which is how nearly every
# Indian invoice prints a single row's amount: once in the row, once in the
# totals block with thousands separators. Transcribed by hand: the row prints
# "1500.00" and the two totals lines print "1,500.00", so an OCR-backed lookup
# tells the printings apart and reads them at 0.95 and 0.20.
TWICE_PRINTED_FIGURE = """\
1  Steel Bracket 12mm  10  NOS  150.00  1500.00
Taxable Value                           1,500.00
Grand Total                             1,500.00
"""
TWICE_PRINTED_FIGURE_REVERSED = """\
Taxable Value                           1,500.00
1  Steel Bracket 12mm  10  NOS  150.00  1500.00
Grand Total                             1,500.00
"""
FIGURE_CONFIDENCES = {"1500.00": 0.95, "1,500.00": 0.20}


@pytest.mark.parametrize("text", [TWICE_PRINTED_FIGURE, TWICE_PRINTED_FIGURE_REVERSED])
def test_the_lowest_confidence_of_two_printed_forms_of_a_figure_is_recorded(text):
    """A clean row must not launder the smudged totals line the figure was read from.

    The recorded number must not depend on which printing comes first in the
    document: an AssVal whose only totals-line printing was read at 0.20 cannot
    be published with the row's 0.95.
    """
    payload = {"items": [], "totals": {"AssVal": 1500.00}}
    result, _ = run(
        payload,
        remaining=text,
        confirmed={},
        confidence_lookup=FIGURE_CONFIDENCES.get,
    )
    assert result.totals["AssVal"] == 1500.00
    assert result.provenance["ValDtls.AssVal"]["ocr_confidence"] == 0.20


def test_every_printing_of_a_figure_is_asked_about_not_just_the_first():
    seen: list[str] = []

    def lookup(text):
        seen.append(text)
        return FIGURE_CONFIDENCES.get(text)

    payload = {"items": [], "totals": {"AssVal": 1500.00}}
    run(payload, remaining=TWICE_PRINTED_FIGURE, confirmed={}, confidence_lookup=lookup)
    assert "1500.00" in seen
    assert "1,500.00" in seen


def test_the_confidence_comes_from_the_token_not_an_earlier_word_containing_it():
    """"Nosy Distributors" is not the unit: a loose first match reads the wrong occurrence."""
    text = "Bill To: Nosy Distributors\n1  Steel Bracket 12mm  10  NOS  150.00  1500.00\n"
    confidences = {"NOS": 0.35, "Nos": 0.99}
    result, _ = run(
        item_payload(Unit="NOS"),
        remaining=text,
        confirmed={},
        confidence_lookup=confidences.get,
    )
    assert result.items[0].Unit == "NOS"
    assert result.provenance["ItemList[0].Unit"]["ocr_confidence"] == 0.35


def test_a_recased_value_matches_the_standalone_token_not_the_middle_of_a_word():
    """The model answered "nos"; the "nos" inside "Diagnosis" is not the unit either."""
    text = "Diagnosis Fee\n1  Steel Bracket 12mm  10  NOS  150.00  1500.00\n"
    confidences = {"NOS": 0.30, "nos": 0.99}
    result, _ = run(
        item_payload(Unit="nos"),
        remaining=text,
        confirmed={},
        confidence_lookup=confidences.get,
    )
    assert result.items[0].Unit == "NOS"
    assert result.provenance["ItemList[0].Unit"]["ocr_confidence"] == 0.30


def test_an_exact_case_token_is_preferred_over_a_recased_one():
    text = "SUPPLY OF NOS\n1  Steel Bracket 12mm  10  Nos  150.00  1500.00\n"
    confidences = {"NOS": 0.88, "Nos": 0.52}
    result, _ = run(
        item_payload(Unit="Nos"),
        remaining=text,
        confirmed={},
        confidence_lookup=confidences.get,
    )
    assert result.items[0].Unit == "Nos"
    assert result.provenance["ItemList[0].Unit"]["ocr_confidence"] == 0.52


def test_a_value_printed_only_inside_a_longer_number_still_grounds_loosely():
    """The last tier is unchanged: a digit-only field grounds inside a longer number and warns."""
    text = "Sunrise Traders\nA/c No 1100015678\n1  Steel Bracket 12mm  10  NOS\n"
    payload = {"items": [], "seller": {"Pin": "110001"}}
    result, _ = run(
        payload,
        remaining=text,
        confirmed={},
        confidence_lookup={"110001": 0.77}.get,
    )
    assert result.seller_pin == "110001"
    assert result.provenance["SellerDtls.Pin"]["ocr_confidence"] == 0.77
    assert "not printed as a number of its own" in message_for(result, "SellerDtls.Pin")


# "NOS" is printed nowhere on this page as a unit of its own -- the unit column
# reads KGS -- and the only occurrence of those three letters is inside the
# buyer's name.
LOOSE_WORD_TEXT = "Bill To: Nosy Distributors\n1  Steel Bracket 12mm  10  KGS  150.00  1500.00\n"


def test_a_value_grounded_only_inside_a_longer_word_keeps_the_models_spelling():
    """GSTN's UQC list has no "Nos": re-casing an uncorroborated unit to a fragment invents an invalid code."""
    result, _ = run(
        item_payload(Unit="NOS"),
        remaining=LOOSE_WORD_TEXT,
        confirmed={},
        confidence_lookup={"Nos": 0.99}.get,
    )
    # Kept, because substring grounding is what the contract specifies -- but in
    # the model's own spelling, never rewritten into the fragment of the word it
    # was found inside.
    assert result.items[0].Unit == "NOS"
    # The lookup is still asked about the text it was grounded against, so the
    # confidence recorded belongs to that word: the known limitation.
    assert result.provenance["ItemList[0].Unit"]["ocr_confidence"] == 0.99


@pytest.mark.parametrize(
    "text, unit",
    [
        ("Bill To: Nosy Distributors\n", "NOS"),
        ("Freight boxes charged separately\n", "BOX"),
        ("Settlement terms: 30 days\n", "SET"),
    ],
)
def test_a_unit_found_only_inside_a_longer_word_is_never_recased_into_another_code(text, unit):
    """A 2-3 character UQC is short enough to hide in an ordinary word; it must survive as written."""
    result, _ = run(item_payload(Unit=unit), remaining=text, confirmed={})
    assert result.items[0].Unit == unit


def test_the_module_docstring_records_the_short_alphabetic_field_limitation():
    """Build 3 publishes this block: the substring consequence is not confined to digit-only fields."""
    doc = " ".join(extract_llm.__doc__.split())
    assert "a short value of ANY string field" in doc
    assert "Nosy Distributors" in doc


def test_a_kept_string_carries_the_documents_spelling_not_the_models():
    """GSTN enumerates Unit codes: "nos" where the invoice prints "NOS" is a wrong value."""
    result, _ = run(item_payload(PrdDesc="steel bracket 12MM", Unit="nos"))
    assert result.items[0].Unit == "NOS"
    assert result.items[0].PrdDesc == "Steel Bracket 12mm"


def test_a_kept_string_carries_the_documents_spacing_not_the_models():
    result, _ = run(item_payload(PrdDesc="STEEL   BRACKET  12mm"))
    assert result.items[0].PrdDesc == "Steel Bracket 12mm"


def test_party_fields_are_returned_in_the_documents_spelling_too():
    payload = dict(GOOD_PAYLOAD)
    payload["seller"] = dict(GOOD_PAYLOAD["seller"], LglNm="SUNRISE TRADERS PVT LTD", Loc="mumbai")
    result, _ = run(payload)
    assert result.seller_name == "Sunrise Traders Pvt Ltd"
    assert result.seller_loc == "Mumbai"


def test_a_value_the_model_merged_across_two_printed_lines_enters_the_payload_on_one_line():
    """The document's spelling is kept; its layout is not. Addr1 is a single-line field."""
    text = "14 Marine Drive, Nariman Point\nMumbai 400021\n"
    payload = {"items": [], "seller": {"Addr1": "14 Marine Drive, Nariman Point Mumbai"}}
    result, _ = run(payload, remaining=text, confirmed={})
    assert result.seller_addr1 == "14 Marine Drive, Nariman Point Mumbai"
    assert "\n" not in result.seller_addr1


def test_column_padding_in_the_source_is_collapsed_out_of_the_kept_string():
    """A PrdDesc of "Steel   Bracket 12mm" is not a value an accountant can submit."""
    text = "1   Steel   Bracket 12mm    10   NOS   150.00   1500.00\n"
    result, _ = run(item_payload(PrdDesc="Steel Bracket 12mm"), remaining=text, confirmed={})
    assert result.items[0].PrdDesc == "Steel Bracket 12mm"


def test_the_lookup_still_sees_the_padding_the_payload_does_not():
    """Collapsing is for the payload only: the page prints the padding, so the lookup gets it."""
    text = "1   Steel   Bracket 12mm    10   NOS   150.00   1500.00\n"
    seen: list[str] = []

    def lookup(value):
        seen.append(value)
        return 0.33

    result, _ = run(
        item_payload(PrdDesc="Steel Bracket 12mm"),
        remaining=text,
        confirmed={},
        confidence_lookup=lookup,
    )
    assert "Steel   Bracket 12mm" in seen
    assert result.provenance["ItemList[0].PrdDesc"]["ocr_confidence"] == 0.33


def test_a_string_grounded_only_by_a_confirmed_value_keeps_that_spelling():
    """The HSN codes live only in `confirmed`; the confirmed spelling is the document's."""
    result, _ = run(item_payload(HsnCd="7326"))
    assert result.items[0].HsnCd == "7326"


def test_confidence_lookup_is_not_consulted_for_dropped_values():
    seen: list[str] = []

    def lookup(text):
        seen.append(text)
        return 0.9

    run(item_payload(PrdDesc="Premium Extended Warranty"), confidence_lookup=lookup)
    assert "Premium Extended Warranty" not in seen


# --------------------------------------------------------------------------- #
# The two behaviours the user called out
# --------------------------------------------------------------------------- #


def test_field_genuinely_absent_from_the_document_is_null_with_a_warning():
    """No buyer block exists in SPARSE_TEXT; the model says null and we keep it null."""
    payload = {
        "items": [],
        "seller": {"LglNm": "Sunrise Traders Pvt Ltd", "Addr1": None, "Loc": "Mumbai",
                   "Pin": "400021"},
        "buyer": {"LglNm": None, "Addr1": None, "Loc": None, "Pin": None},
        "totals": {"AssVal": None, "CgstVal": None, "SgstVal": None, "IgstVal": None,
                   "CesVal": None, "TotInvVal": 1500.00, "RndOffAmt": None},
    }
    result, _ = run(payload, remaining=SPARSE_TEXT, confirmed={})
    assert result.buyer_name is None
    assert result.buyer_addr1 is None
    assert result.buyer_loc is None
    assert result.buyer_pin is None
    for path in ("BuyerDtls.LglNm", "BuyerDtls.Addr1", "BuyerDtls.Loc", "BuyerDtls.Pin"):
        assert "was not found in the document" in message_for(result, path)
        assert path not in result.provenance
    assert result.totals["TotInvVal"] == 1500.00


def test_a_plausible_but_absent_value_is_dropped_not_kept():
    payload = item_payload(PrdDesc="Stainless Steel Mounting Bracket, 12 mm, IS 2062")
    result, _ = run(payload)
    assert result.items[0].PrdDesc is None
    message = message_for(result, "ItemList[0].PrdDesc")
    assert "Stainless Steel Mounting Bracket, 12 mm, IS 2062" in message
    assert "did not appear in the source text" in message
    assert "dropped rather than kept" in message
    assert "ItemList[0].PrdDesc" not in result.provenance


def test_an_invented_number_is_dropped_however_plausible():
    payload = item_payload(TotItemVal=1777.00)
    result, _ = run(payload)
    assert result.items[0].TotItemVal is None
    assert "1777.0" in message_for(result, "ItemList[0].TotItemVal")


def test_a_wholly_invented_line_item_keeps_none_of_its_identifying_values():
    """Everything that identifies the invented row is dropped; five bare figures are not.

    The name says "identifying" rather than "no values at all" because that is
    what the check is: grounding corroborates that a value is PRINTED, never that
    it belongs to this row, so a figure the page prints elsewhere survives on a
    row that does not exist. The survivors are asserted below by name.
    """
    row = {
        "SlNo": "3",
        "PrdDesc": "Premium Extended Warranty",
        "HsnCd": "998599",
        "Qty": 1,
        "Unit": "YRS",
        "UnitPrice": 4999.00,
        "TotAmt": 4999.00,
        "Discount": 0,
        "AssAmt": 4999.00,
        "GstRt": 18,
        "CgstAmt": 449.91,
        "SgstAmt": 449.91,
        "IgstAmt": 0,
        "CesAmt": 0,
        "TotItemVal": 5898.82,
    }
    result, _ = run({"items": [row]})
    invented = result.items[0]
    for name in ("PrdDesc", "HsnCd", "Unit", "UnitPrice", "TotAmt", "AssAmt",
                 "CgstAmt", "SgstAmt", "TotItemVal"):
        assert getattr(invented, name) is None, name
        assert f"ItemList[0].{name}" not in result.provenance
    assert "Premium Extended Warranty" in message_for(result, "ItemList[0].PrdDesc")
    # SlNo "3" is ungrounded: INVOICE_TEXT prints no 3 as a token of its own.
    assert invented.SlNo is None
    assert "did not appear in the source text" in message_for(result, "ItemList[0].SlNo")
    # The five that DO survive, and why each one does. Transcribed by hand from
    # INVOICE_TEXT: it prints "1" in the Sl and Qty columns, "18" in the GST%
    # column of row 1, and "0.00" on the Round Off line. Every one of these is a
    # figure printed somewhere on the page, so presence-based grounding keeps it
    # even on a row that does not exist -- the limitation the module docstring
    # records, pinned here rather than skipped over.
    survivors = {
        name: value
        for name, value in dataclasses.asdict(invented).items()
        if value is not None
    }
    assert survivors == {
        "Qty": 1.0,        # "1" is printed in the Sl and Qty columns
        "Discount": 0.0,   # "0.00" is printed on the Round Off line
        "GstRt": 18.0,     # "18" is printed in row 1's GST% column
        "IgstAmt": 0.0,    # same "0.00"
        "CesAmt": 0.0,     # same "0.00"
    }
    for name in survivors:
        path = f"ItemList[0].{name}"
        assert result.provenance[path] == {"source": "llm", "ocr_confidence": None}
        assert warnings_for(result, path) == []


def test_missing_and_dropped_warnings_are_distinguishable():
    result, _ = run(item_payload(PrdDesc="Premium Extended Warranty", Unit=None))
    dropped = message_for(result, "ItemList[0].PrdDesc")
    missing = message_for(result, "ItemList[0].Unit")
    assert dropped != missing
    assert "dropped rather than kept" in dropped
    assert "dropped rather than kept" not in missing
    assert "returned no value for it" in missing
    assert "returned no value for it" not in dropped


def test_a_field_the_model_omitted_entirely_is_treated_as_not_found():
    row = dict(GOOD_PAYLOAD["items"][0])
    del row["Unit"]
    result, _ = run({"items": [row]})
    assert result.items[0].Unit is None
    assert "was not found in the document" in message_for(result, "ItemList[0].Unit")


def test_an_absent_field_is_never_said_to_have_been_returned_as_null():
    """The model returned no such thing when it omitted the key or malformed the block."""
    row = dict(GOOD_PAYLOAD["items"][0])
    del row["Unit"]
    omitted, _ = run({"items": [row], "seller": ["Sunrise Traders Pvt Ltd"]})
    for path in ("ItemList[0].Unit", "SellerDtls.LglNm"):
        message = message_for(omitted, path)
        assert "was not found in the document" in message
        assert "returned null" not in message
    # An explicit JSON null reads the same way, and is just as true of it.
    explicit, _ = run(item_payload(Unit=None))
    assert "returned no value for it" in message_for(explicit, "ItemList[0].Unit")


def test_rows_numbered_one_to_nine_keep_their_slno():
    """The ordinary Indian invoice: rows numbered 1..9, and SlNo is mandatory in INV-01."""
    result, _ = run(GOOD_PAYLOAD)
    # Transcribed by hand from INVOICE_TEXT's Sl column.
    assert [item.SlNo for item in result.items] == ["1", "2"]
    for index in (0, 1):
        path = f"ItemList[{index}].SlNo"
        assert result.provenance[path] == {"source": "llm", "ocr_confidence": None}


def test_a_single_character_kept_this_way_says_the_corroboration_is_weak():
    """One character on the page proves very little: kept, but never kept silently."""
    result, _ = run(GOOD_PAYLOAD)
    message = message_for(result, "ItemList[0].SlNo")
    assert '"1"' in message
    assert "is a single character" in message
    assert "weak corroboration" in message
    assert "row numbering" in message
    assert "Confirm it against the invoice by eye" in message
    # Kept, not dropped: the two diagnoses must not be confusable.
    assert "dropped rather than kept" not in message


def test_a_single_character_not_printed_as_a_token_is_still_dropped():
    """No standalone occurrence, no grounding: the ordinary ungrounded diagnosis applies."""
    # Transcribed by hand: the only "3" characters in INVOICE_TEXT sit inside
    # "2,300.00" and the two "135.00" figures. None of them is a printed 3.
    result, _ = run(item_payload(SlNo="3"))
    assert result.items[0].SlNo is None
    assert "ItemList[0].SlNo" not in result.provenance
    message = message_for(result, "ItemList[0].SlNo")
    assert '"3"' in message
    assert "did not appear in the source text" in message
    assert "dropped rather than kept" in message


def test_a_one_character_unit_printed_as_its_own_token_survives_too():
    """The rule is about value length, not about SlNo: Unit "L" is printed right there."""
    text = "1   Steel Bracket 12mm    10   L     150.00   1500.00\n"
    result, _ = run(item_payload(SlNo="1", Unit="L"), remaining=text, confirmed={})
    assert result.items[0].SlNo == "1"
    assert result.items[0].Unit == "L"
    for path in ("ItemList[0].SlNo", "ItemList[0].Unit"):
        message = message_for(result, path)
        assert "is a single character" in message
        assert "did not appear in the source text" not in message
    # A value genuinely absent keeps the other diagnosis, and the two stay distinct.
    absent, _ = run(item_payload(PrdDesc="Premium Extended Warranty"))
    assert "did not appear in the source text" in message_for(absent, "ItemList[0].PrdDesc")


def test_a_one_character_value_buried_in_a_longer_token_is_not_corroborated():
    """The floor's purpose survives: "5" inside "150.00" is not a printed 5."""
    text = "1   Steel Bracket 12mm    10   NOS   150.00   1500.00\n"
    assert "5" in text
    result, _ = run(item_payload(SlNo="5"), remaining=text, confirmed={})
    assert result.items[0].SlNo is None
    assert "did not appear in the source text" in message_for(result, "ItemList[0].SlNo")


# --------------------------------------------------------------------------- #
# The documented limit of grounding: presence, not assignment
# --------------------------------------------------------------------------- #


def test_a_swapped_seller_and_buyer_block_is_kept_silently():
    """Grounding says a name is printed, never that it is the SELLER's name.

    Both party blocks are printed on this document, so a response that swaps them
    passes every check. This is the largest known limitation of the stage and the
    module docstring must say so: an accountant reading extraction_meta sees the
    same clean provenance either way.
    """
    swapped = dict(GOOD_PAYLOAD)
    swapped["seller"] = GOOD_PAYLOAD["buyer"]
    swapped["buyer"] = GOOD_PAYLOAD["seller"]
    result, _ = run(swapped)

    # Transcribed from INVOICE_TEXT: Sunrise is the seller, Deccan the buyer.
    # The stage keeps them the wrong way round, because both are printed.
    assert result.seller_name == "Deccan Hardware LLP"
    assert result.seller_loc == "New Delhi"
    assert result.buyer_name == "Sunrise Traders Pvt Ltd"
    assert result.buyer_loc == "Mumbai"
    for path in (
        "SellerDtls.LglNm", "SellerDtls.Addr1", "SellerDtls.Loc", "SellerDtls.Pin",
        "BuyerDtls.LglNm", "BuyerDtls.Addr1", "BuyerDtls.Loc", "BuyerDtls.Pin",
    ):
        assert result.provenance[path] == {"source": "llm", "ocr_confidence": None}
        assert warnings_for(result, path) == []


def test_one_rows_figures_moved_onto_another_row_are_kept_silently():
    """Row 2's figures are printed, so row 1 carrying them grounds just as well."""
    stolen = dict(GOOD_PAYLOAD["items"][0])
    # Transcribed by hand from row 2 of INVOICE_TEXT.
    stolen.update({"UnitPrice": 800.00, "TotAmt": 800.00, "GstRt": 5, "TotItemVal": 840.00})
    result, _ = run({"items": [stolen]})

    row = result.items[0]
    assert row.PrdDesc == "Steel Bracket 12mm"
    assert (row.UnitPrice, row.TotAmt, row.GstRt, row.TotItemVal) == (800.0, 800.0, 5.0, 840.0)
    for name in ("UnitPrice", "TotAmt", "GstRt", "TotItemVal"):
        path = f"ItemList[0].{name}"
        assert result.provenance[path] == {"source": "llm", "ocr_confidence": None}
        assert warnings_for(result, path) == []


def test_the_module_docstring_records_the_assignment_limitation():
    """HsnCd is not the only uncorroborated assignment; the docstring must not say it is."""
    doc = " ".join(extract_llm.__doc__.split())  # the docstring is hard-wrapped
    assert "is the exception" not in doc
    assert "never which party or which row it belongs to" in doc
    assert "SellerDtls" in doc and "BuyerDtls" in doc


def test_a_digit_run_glued_to_letters_is_a_numeric_token_like_any_other():
    """The contract's token pattern finds a digit run wherever it sits, letters included."""
    assert is_grounded(12, "Steel Bracket 12mm") is True
    assert is_grounded(2.5, "Cable 2.5sqmm") is True
    assert is_grounded(150, "Model MS150 pump") is True
    assert is_grounded(4, "A4 Paper Ream") is True
    assert is_grounded(12, "Vehicle MH12AB1234") is True
    assert is_grounded(1234, "Vehicle MH12AB1234") is True
    # The other side of the same rule, and the reason it is not narrowed to
    # digits with no letter beside them: these are ordinary printings of a real
    # figure, and refusing them would drop a correctly read amount.
    assert is_grounded(1500.00, "Grand Total Rs1500.00") is True
    assert is_grounded(1500.00, "Grand Total INR1500.00") is True


def test_a_figure_matching_digits_inside_a_word_is_kept_silently():
    """The documented limitation, pinned: this page prints no money, yet three figures survive."""
    text = (
        "Sl  Description          Qty  Unit\n"
        "1   Steel Bracket 12mm   ??   NOS\n"
        "2   Cable 2.5sqmm        ??   MTR\n"
    )
    # Transcribed by hand: the only digit runs on that page are 1, 12, 2 and 2.5.
    result, _ = run(
        {"items": [{"Qty": 12, "UnitPrice": 2.5, "TotAmt": 12.0}]},
        remaining=text,
        confirmed={},
    )
    row = result.items[0]
    assert (row.Qty, row.UnitPrice, row.TotAmt) == (12.0, 2.5, 12.0)
    for name in ("Qty", "UnitPrice", "TotAmt"):
        path = f"ItemList[0].{name}"
        assert result.provenance[path] == {"source": "llm", "ocr_confidence": None}
        assert warnings_for(result, path) == []


def test_the_module_docstring_records_the_digits_inside_a_word_limitation():
    """Build 3 publishes this block, and it must not claim numbers are safe from this."""
    doc = " ".join(extract_llm.__doc__.split())
    assert "digits inside a longer NUMBER, not of digits inside a WORD" in doc
    assert "12mm" in doc and "Rs1500.00" in doc


# --------------------------------------------------------------------------- #
# Value coercion at the boundary
# --------------------------------------------------------------------------- #


def test_numeric_strings_from_the_model_are_parsed_and_grounded():
    result, _ = run(item_payload(AssAmt="1,500.00", GstRt="18%"))
    assert result.items[0].AssAmt == 1500.00
    assert result.items[0].GstRt == 18.0


def test_a_bracketed_credit_is_never_read_as_a_positive_number():
    """(250.00) is MINUS 250.00. Reading it as +250.00 would silently inflate AssVal."""
    payload = {
        "items": [{"Discount": "(250.00)"}],
        "totals": {"RndOffAmt": "(0.40)"},
    }
    result, _ = run(payload, remaining=CREDIT_TEXT, confirmed={})
    assert result.items[0].Discount is None
    assert result.totals["RndOffAmt"] is None
    assert "ItemList[0].Discount" not in result.provenance
    assert "ValDtls.RndOffAmt" not in result.provenance
    message = message_for(result, "ItemList[0].Discount")
    assert "(250.00)" in message
    assert "dropped rather than kept" in message
    # The bare digits ARE printed here, so claiming absence would be a lie.
    assert "did not appear in the source text" not in message


def test_a_plain_number_is_not_grounded_by_digits_the_page_prints_as_a_credit():
    """The prompt asks for JSON numbers, so 250.0 -- not "(250.00)" -- is the likely answer."""
    assert is_grounded(250.0, CREDIT_TEXT) is False
    assert is_grounded(-250.0, CREDIT_TEXT) is True
    assert is_grounded(0.40, CR_TEXT) is False
    assert is_grounded(-0.40, CR_TEXT) is True


def test_a_json_number_matching_a_bracketed_credit_is_dropped_with_the_right_diagnosis():
    result, _ = run(item_payload(Discount=250.0), remaining=CREDIT_TEXT, confirmed={})
    assert result.items[0].Discount is None
    assert "ItemList[0].Discount" not in result.provenance
    message = message_for(result, "ItemList[0].Discount")
    assert "(250.00)" in message
    assert "credit" in message
    assert "dropped rather than kept" in message
    # The digits ARE printed here, so the hallucination diagnosis would be a lie.
    assert "did not appear in the source text" not in message


def test_the_negative_reading_of_a_credit_is_kept_and_looked_up_as_printed():
    seen: list[str] = []

    def lookup(text):
        seen.append(text)
        return 0.55

    result, _ = run(
        item_payload(Discount=-250.0),
        remaining=CREDIT_TEXT,
        confirmed={},
        confidence_lookup=lookup,
    )
    assert result.items[0].Discount == -250.0
    assert result.provenance["ItemList[0].Discount"] == {"source": "llm", "ocr_confidence": 0.55}
    assert "(250.00)" in seen
    assert warnings_for(result, "ItemList[0].Discount") == []


def test_a_cr_suffix_carries_the_same_sign_as_accounting_brackets():
    seen: list[str] = []
    dropped, _ = run(
        {"items": [], "totals": dict.fromkeys(TOTAL_KEYS) | {"RndOffAmt": 0.40}},
        remaining=CR_TEXT,
        confirmed={},
    )
    assert dropped.totals["RndOffAmt"] is None
    assert "0.40 CR" in message_for(dropped, "ValDtls.RndOffAmt")
    kept, _ = run(
        {"items": [], "totals": dict.fromkeys(TOTAL_KEYS) | {"RndOffAmt": -0.40}},
        remaining=CR_TEXT,
        confirmed={},
        confidence_lookup=lambda text: seen.append(text),
    )
    assert kept.totals["RndOffAmt"] == -0.40
    assert seen == ["0.40 CR"]


def test_a_detached_minus_sign_in_the_models_answer_is_not_a_plain_number():
    """About the MODEL's own string, not the page: a detached sign is refused, not read."""
    result, _ = run(item_payload(Discount="- 250.00"), remaining=CREDIT_TEXT, confirmed={})
    assert result.items[0].Discount is None


@pytest.mark.parametrize("notation", sorted(MINUS_TEXTS))
def test_every_printed_minus_carries_its_sign_the_way_brackets_do(notation):
    """A minus the page prints is a minus, whether it is Unicode or trailing."""
    text = MINUS_TEXTS[notation]
    assert is_grounded(250.0, text) is False
    assert is_grounded(-250.0, text) is True
    dropped, _ = run(item_payload(Discount=250.0), remaining=text, confirmed={})
    assert dropped.items[0].Discount is None
    assert "ItemList[0].Discount" not in dropped.provenance
    message = message_for(dropped, "ItemList[0].Discount")
    assert "credit" in message
    assert "dropped rather than kept" in message
    # The digits ARE printed here, so the hallucination diagnosis would be a lie.
    assert "did not appear in the source text" not in message
    kept, _ = run(item_payload(Discount=-250.0), remaining=text, confirmed={})
    assert kept.items[0].Discount == -250.0
    assert warnings_for(kept, "ItemList[0].Discount") == []


@pytest.mark.parametrize("notation", sorted(MINUS_TEXTS))
def test_a_printed_minus_is_looked_up_as_printed(notation):
    """Provenance must point at the text on the page, sign and all."""
    text = MINUS_TEXTS[notation]
    printed = {"unicode": "−250.00", "trailing": "250.00-"}[notation]
    seen: list[str] = []
    result, _ = run(
        item_payload(Discount=-250.0),
        remaining=text,
        confirmed={},
        confidence_lookup=lambda value: seen.append(value),
    )
    assert result.items[0].Discount == -250.0
    assert printed in seen


def test_a_trailing_minus_carries_its_sign_only_at_the_end_of_the_line():
    """The documented rule, both sides: end of line is a sign, mid-line is not."""
    # Transcribed by hand from the fixtures: both print a credit of 250.00, and
    # only the one whose minus ends its line is read as one.
    assert is_grounded(-250.0, MINUS_TEXTS["trailing"]) is True
    assert is_grounded(250.0, MINUS_TEXTS["trailing"]) is False
    assert is_grounded(250.0, TALLY_COLUMN_TEXT) is True
    assert is_grounded(-250.0, TALLY_COLUMN_TEXT) is False


def test_a_credit_column_followed_by_another_column_is_read_as_a_charge():
    """The limitation pinned end to end: this credit reaches the payload as a charge."""
    charge, _ = run(item_payload(Discount=250.00), remaining=TALLY_COLUMN_TEXT, confirmed={})
    assert charge.items[0].Discount == 250.00
    assert charge.provenance["ItemList[0].Discount"] == {"source": "llm", "ocr_confidence": None}
    assert warnings_for(charge, "ItemList[0].Discount") == []
    # And the correct reading of that printed credit is the one that is refused.
    credit, _ = run(item_payload(Discount=-250.00), remaining=TALLY_COLUMN_TEXT, confirmed={})
    assert credit.items[0].Discount is None
    assert "did not appear in the source text" in message_for(credit, "ItemList[0].Discount")


def test_the_trailing_minus_is_not_widened_to_a_minus_followed_by_a_column_gap():
    """Why the limitation stands: widening it would invert these, a new sign error."""
    # A printed range and a printed subtraction. Every figure here is a charge,
    # and a minus followed by a gap would make each one negative instead.
    assert is_grounded(150.00, "Rate band   150.00-200.00\n") is True
    assert is_grounded(-150.00, "Rate band   150.00-200.00\n") is False
    assert is_grounded(1500.00, "Net of discount   1500.00 - 250.00\n") is True
    assert is_grounded(-1500.00, "Net of discount   1500.00 - 250.00\n") is False


def test_the_module_docstring_records_the_trailing_minus_limitation():
    """Build 3 publishes this block; it must not claim the trailing minus is read everywhere."""
    doc = " ".join(extract_llm.__doc__.split())  # the docstring is hard-wrapped
    assert "only when it ENDS THE LINE" in doc
    assert "read as a POSITIVE CHARGE" in doc
    assert "carries its sign only at the END OF A LINE" in doc
    assert "150.00-200.00" in doc


def test_a_notation_at_the_end_of_the_line_above_still_carries_its_sign():
    """The look-behind is bounded to the figure's own line, and one line up is still in reach."""
    assert is_grounded(-250.00, "Discount (\n250.00)\nGrand Total 1250.00\n") is True
    assert is_grounded(250.00, "Discount (\n250.00)\nGrand Total 1250.00\n") is False
    assert is_grounded(-250.00, "Net of discount −\n250.00\n") is True


def test_a_detached_hyphen_in_the_document_is_not_read_as_a_minus():
    """"Freight - 200.00" prints a charge of 200.00: the hyphen separates a label."""
    assert is_grounded(200.0, "Freight - 200.00") is True
    assert is_grounded(-200.0, "Freight - 200.00") is False


def test_a_hyphen_separated_total_block_keeps_every_figure_positive():
    """The whole layout, end to end: four correct positives, kept, with no warnings."""
    payload = {
        "items": [],
        "totals": {
            "AssVal": 1500.00,
            "CgstVal": 135.00,
            "SgstVal": 135.00,
            "IgstVal": None,
            "CesVal": None,
            "TotInvVal": 1770.00,
            "RndOffAmt": None,
        },
    }
    result, _ = run(payload, remaining=HYPHEN_TEXT, confirmed={})
    # Transcribed by hand from HYPHEN_TEXT: every printed figure is a charge.
    assert result.totals["AssVal"] == 1500.00
    assert result.totals["CgstVal"] == 135.00
    assert result.totals["SgstVal"] == 135.00
    assert result.totals["TotInvVal"] == 1770.00
    for key in ("AssVal", "CgstVal", "SgstVal", "TotInvVal"):
        assert warnings_for(result, f"ValDtls.{key}") == []
        assert result.provenance[f"ValDtls.{key}"] == {"source": "llm", "ocr_confidence": None}


def test_the_inverted_reading_of_a_hyphen_separated_figure_is_refused():
    """The other half of the same bug: -200.00 must not ground off "Freight - 200.00"."""
    text = "Freight - 200.00\nGrand Total   200.00\n"
    result, _ = run(item_payload(TotAmt=-200.00), remaining=text, confirmed={})
    assert result.items[0].TotAmt is None
    assert "ItemList[0].TotAmt" not in result.provenance
    assert "did not appear in the source text" in message_for(result, "ItemList[0].TotAmt")


def test_a_hyphen_separated_figure_is_looked_up_without_the_hyphen():
    """Provenance points at the figure, not at the label's separator."""
    seen: list[str] = []
    result, _ = run(
        item_payload(TotAmt=200.00),
        remaining=HYPHEN_TEXT,
        confirmed={},
        confidence_lookup=lambda text: seen.append(text),
    )
    assert result.items[0].TotAmt == 200.00
    assert "200.00" in seen
    assert "- 200.00" not in seen


def test_an_attached_minus_the_model_can_rely_on_is_still_negative():
    """Removing the detached rule does not touch a sign printed against its digits."""
    text = "Discount   -250.00\nGrand Total   1250.00\n"
    assert is_grounded(-250.0, text) is True
    assert is_grounded(250.0, text) is False


def test_a_minus_between_two_figures_is_a_subtraction_not_a_sign():
    """"1500.00 − 250.00" prints a charge of 250.00; dropping it would be the wrong call."""
    assert is_grounded(250.0, SUBTRACTION_TEXT) is True
    assert is_grounded(-250.0, SUBTRACTION_TEXT) is False
    result, _ = run(item_payload(TotAmt=250.0), remaining=SUBTRACTION_TEXT, confirmed={})
    assert result.items[0].TotAmt == 250.0
    assert warnings_for(result, "ItemList[0].TotAmt") == []


def test_a_bracketed_token_that_is_not_money_is_still_read_as_a_credit():
    """A known consequence of the bracket rule, pinned so its direction stays the safe one.

    "(022)" is a phone area code, not a credit, so this diagnosis is wrong about
    that token -- but the value is DROPPED, never inverted or invented, and only
    when the figure is printed nowhere else on the page.
    """
    text = "Sunrise Traders Pvt Ltd\nPhone (022) 2222 3333\nGrand Total 1500.00\n"
    result, _ = run(item_payload(Qty=22), remaining=text, confirmed={})
    assert result.items[0].Qty is None
    assert "ItemList[0].Qty" not in result.provenance


def test_a_bracketed_figure_without_a_decimal_point_is_still_a_credit():
    """Why the rule above is kept: requiring a decimal would make "(1500)" ground +1500."""
    text = "Discount                  (1500)\nGrand Total               100.00\n"
    assert is_grounded(-1500.0, text) is True
    assert is_grounded(1500.0, text) is False


def test_a_numeric_string_may_carry_only_currency_or_percent_decoration():
    kept, _ = run(item_payload(AssAmt="₹ 1,500.00", GstRt="18%"))
    assert kept.items[0].AssAmt == 1500.00
    assert kept.items[0].GstRt == 18.0
    # Anything else is refused rather than read through: stray text may carry a sign.
    dropped, _ = run(item_payload(TotAmt="Rs. 1500.00 CR"))
    assert dropped.items[0].TotAmt is None
    assert "not a plain number" in message_for(dropped, "ItemList[0].TotAmt")


def test_numbers_returned_for_string_fields_become_strings():
    result, _ = run(item_payload(HsnCd=7326))
    assert result.items[0].HsnCd == "7326"


def test_a_numeric_field_holding_no_number_is_dropped():
    result, _ = run(item_payload(Qty="N/A"))
    assert result.items[0].Qty is None
    assert "N/A" in message_for(result, "ItemList[0].Qty")


def test_structured_and_boolean_junk_in_a_field_is_dropped():
    result, _ = run(item_payload(PrdDesc=["Steel Bracket 12mm"], Qty=True))
    assert result.items[0].PrdDesc is None
    assert result.items[0].Qty is None
    assert "dropped rather than kept" in message_for(result, "ItemList[0].PrdDesc")
    assert "dropped rather than kept" in message_for(result, "ItemList[0].Qty")


def test_a_list_in_a_string_field_is_not_called_a_hallucination():
    """The text inside the list is printed right there; only the JSON shape was wrong."""
    result, _ = run(item_payload(PrdDesc=["Steel Bracket 12mm"]))
    assert "Steel Bracket 12mm" in INVOICE_TEXT
    message = message_for(result, "ItemList[0].PrdDesc")
    assert "JSON list" in message
    assert "did not appear in the source text" not in message
    assert "dropped rather than kept" in message


# --------------------------------------------------------------------------- #
# Malformed responses
# --------------------------------------------------------------------------- #


def assert_empty_extraction(result: LlmExtraction) -> None:
    assert result.items == ()
    assert result.seller_name is None and result.buyer_name is None
    assert result.seller_addr1 is None and result.buyer_addr1 is None
    assert result.seller_loc is None and result.buyer_loc is None
    assert result.seller_pin is None and result.buyer_pin is None
    assert result.totals == {key: None for key in TOTAL_KEYS}
    assert result.provenance == {}
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert warning.field == "ItemList"
    assert warning.check == "extract_llm"
    assert "could not be parsed as a JSON object" in warning.message


def test_response_that_is_not_json_yields_one_warning_and_nothing_else():
    result, _ = run("Here is the invoice table you asked for!")
    assert_empty_extraction(result)


def test_response_that_is_a_json_array_is_rejected():
    result, _ = run("[{\"PrdDesc\": \"Steel Bracket 12mm\"}]")
    assert_empty_extraction(result)


def test_response_that_is_a_json_string_is_rejected():
    result, _ = run('"Steel Bracket 12mm"')
    assert_empty_extraction(result)


def test_response_with_no_content_at_all_is_rejected():
    result, _ = run(None)
    assert_empty_extraction(result)


def test_a_response_carrying_no_choices_is_reported_not_crashed():
    """A provider that stops on a content filter returns none; that is an answer, not a crash."""

    class NoChoices:
        def __init__(self):
            create = lambda **kwargs: SimpleNamespace(choices=[])  # noqa: E731
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    result = extract_with_llm(INVOICE_TEXT, CONFIRMED, client=NoChoices())
    assert_empty_extraction(result)


def test_missing_item_list_warns_that_no_line_items_were_found():
    result, _ = run({"seller": GOOD_PAYLOAD["seller"]})
    assert result.items == ()
    assert "No line items were found in the document" in message_for(result, "ItemList")


def test_an_absent_item_list_is_never_said_to_have_been_returned_empty():
    """The model returned no ItemList at all; claiming it returned an empty one is untrue."""
    result, _ = run({"seller": GOOD_PAYLOAD["seller"]})
    message = message_for(result, "ItemList")
    assert "empty ItemList" not in message
    assert "the model returned none" in message


def test_empty_item_list_warns_that_no_line_items_were_found():
    result, _ = run({"items": []})
    assert result.items == ()
    assert "No line items were found in the document" in message_for(result, "ItemList")


def test_item_list_that_is_not_a_list_is_reported():
    result, _ = run({"items": {"PrdDesc": "Steel Bracket 12mm"}})
    assert result.items == ()
    assert "rather than a list of rows" in message_for(result, "ItemList")


def test_a_row_that_is_not_an_object_is_dropped_and_the_rest_keep_contiguous_paths():
    result, _ = run({"items": ["Steel Bracket 12mm", GOOD_PAYLOAD["items"][1]]})
    assert len(result.items) == 1
    assert result.items[0].PrdDesc == "Mounting Service"
    assert "ItemList[0].PrdDesc" in result.provenance
    assert any("Row 1 of the model's ItemList" in w.message for w in warnings_for(result, "ItemList"))


def test_malformed_party_block_is_treated_as_absent_fields():
    result, _ = run({"items": GOOD_PAYLOAD["items"], "seller": "Sunrise Traders Pvt Ltd"})
    assert result.seller_name is None
    assert "was not found in the document" in message_for(result, "SellerDtls.LglNm")


# --------------------------------------------------------------------------- #
# Dataclass shapes
# --------------------------------------------------------------------------- #


def test_llm_item_fields_are_all_optional():
    item = LlmItem()
    for name in ("SlNo", "PrdDesc", "HsnCd", "Qty", "Unit", "UnitPrice", "TotAmt",
                 "Discount", "AssAmt", "GstRt", "CgstAmt", "SgstAmt", "IgstAmt",
                 "CesAmt", "TotItemVal"):
        assert getattr(item, name) is None


def test_extraction_is_frozen():
    result, _ = run(GOOD_PAYLOAD)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.items = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.items[0].PrdDesc = "something else"


# --------------------------------------------------------------------------- #
# make_client
# --------------------------------------------------------------------------- #


def test_make_client_without_the_env_var_names_it_in_the_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        make_client()
    assert "GROQ_API_KEY" in str(excinfo.value)


def test_make_client_with_an_empty_env_var_still_refuses(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(RuntimeError):
        make_client()


def test_extract_with_llm_never_reaches_make_client(monkeypatch):
    """The stage must work with an injected client in an environment with no key."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result, client = run(GOOD_PAYLOAD)
    assert len(client.calls) == 1
    assert result.items[0].PrdDesc == "Steel Bracket 12mm"
