"""Tests for gst_einvoice.schema: INV-01 models and the ExtractionResult wrapper."""

import pytest
from pydantic import ValidationError

from gst_einvoice.schema import (
    BuyerDtls,
    DocDtls,
    ExtractionMeta,
    ExtractionResult,
    ExtractionWarning,
    Invoice,
    Item,
    PageMeta,
    SellerDtls,
    TranDtls,
    ValDtls,
)

# --- fixtures -------------------------------------------------------------------


def tran_data() -> dict:
    return {"TaxSch": "GST", "SupTyp": "B2B", "RegRev": "N", "IgstOnIntra": "N"}


def doc_data() -> dict:
    return {"Typ": "INV", "No": "INV-2024-001", "Dt": "15/03/2024"}


def seller_data() -> dict:
    return {
        "Gstin": "29AAACR5055K1Z5",
        "LglNm": "Acme Traders Private Limited",
        "TrdNm": "Acme Traders",
        "Addr1": "12 MG Road",
        "Addr2": "Shanthala Nagar",
        "Loc": "Bengaluru",
        "Pin": 560001,
        "Stcd": "29",
    }


def buyer_data() -> dict:
    return {
        "Gstin": "29AABCT1332L1ZU",
        "LglNm": "Beta Industries Limited",
        "TrdNm": "Beta Industries",
        "Addr1": "45 Residency Road",
        "Addr2": "Ashok Nagar",
        "Loc": "Bengaluru",
        "Pin": 560025,
        "Stcd": "29",
        "Pos": "29",
    }


def item_data() -> dict:
    return {
        "SlNo": "1",
        "PrdDesc": "Steel bolts M10",
        "IsServc": "N",
        "HsnCd": "73181500",
        "Qty": 2,
        "Unit": "NOS",
        "UnitPrice": 1000,
        "TotAmt": 2000,
        "Discount": 0,
        "AssAmt": 2000,
        "GstRt": 18,
        "CgstAmt": 180,
        "SgstAmt": 180,
        "IgstAmt": 0,
        "CesAmt": 0,
        "StateCesAmt": 0,
        "OthChrg": 0,
        "TotItemVal": 2360,
    }


def val_data() -> dict:
    return {
        "AssVal": 2000,
        "CgstVal": 180,
        "SgstVal": 180,
        "IgstVal": 0,
        "CesVal": 0,
        "StCesVal": 0,
        "RndOffAmt": 0,
        "TotInvVal": 2360,
    }


def invoice_data() -> dict:
    """A fully populated INV-01 payload (every optional field present)."""
    return {
        "Version": "1.1",
        "TranDtls": tran_data(),
        "DocDtls": doc_data(),
        "SellerDtls": seller_data(),
        "BuyerDtls": buyer_data(),
        "ItemList": [item_data()],
        "ValDtls": val_data(),
    }


def minimal_invoice_data() -> dict:
    """Only the mandatory fields; every optional field omitted."""
    return {
        "TranDtls": {"SupTyp": "B2B"},
        "DocDtls": doc_data(),
        "SellerDtls": {
            "Gstin": "29AAACR5055K1Z5",
            "LglNm": "Acme Traders Private Limited",
            "Addr1": "12 MG Road",
            "Loc": "Bengaluru",
            "Pin": 560001,
            "Stcd": "29",
        },
        "BuyerDtls": {
            "Gstin": "29AABCT1332L1ZU",
            "LglNm": "Beta Industries Limited",
            "Addr1": "45 Residency Road",
            "Loc": "Bengaluru",
            "Pin": 560025,
            "Stcd": "29",
            "Pos": "29",
        },
        "ItemList": [
            {
                "SlNo": "1",
                "PrdDesc": "Steel bolts M10",
                "IsServc": "N",
                "HsnCd": "73181500",
                "Qty": 2,
                "Unit": "NOS",
                "UnitPrice": 1000,
                "TotAmt": 2000,
                "AssAmt": 2000,
                "GstRt": 18,
                "CgstAmt": 180,
                "SgstAmt": 180,
                "IgstAmt": 0,
                "TotItemVal": 2360,
            }
        ],
        "ValDtls": {
            "AssVal": 2000,
            "CgstVal": 180,
            "SgstVal": 180,
            "IgstVal": 0,
            "TotInvVal": 2360,
        },
    }


def result_data() -> dict:
    return {
        "invoice": invoice_data(),
        "extraction_meta": {
            "pages": [
                {"page": 1, "method": "native"},
                {"page": 2, "method": "ocr_required"},
            ],
            "field_provenance": {
                "DocDtls.No": {"page": 1, "source": "regex"},
                "SellerDtls.Gstin": {"page": 1, "source": "regex"},
            },
            "warnings": [
                {
                    "field": "ItemList[0].TotItemVal",
                    "message": "does not equal AssAmt + taxes",
                    "severity": "warning",
                    "check": "item_total_arithmetic",
                },
                {
                    "field": "DocDtls.Dt",
                    "message": "date parsed from DD-MM-YYYY",
                    "severity": "info",
                    "check": "date_format",
                },
            ],
        },
    }


# --- expected key sets (the spec, written out literally) ------------------------

TRAN_KEYS = {"TaxSch", "SupTyp", "RegRev", "IgstOnIntra"}
DOC_KEYS = {"Typ", "No", "Dt"}
SELLER_KEYS = {"Gstin", "LglNm", "TrdNm", "Addr1", "Addr2", "Loc", "Pin", "Stcd"}
BUYER_KEYS = {"Gstin", "LglNm", "TrdNm", "Addr1", "Addr2", "Loc", "Pin", "Stcd", "Pos"}
ITEM_KEYS = {
    "SlNo", "PrdDesc", "IsServc", "HsnCd", "Qty", "Unit", "UnitPrice", "TotAmt",
    "Discount", "AssAmt", "GstRt", "CgstAmt", "SgstAmt", "IgstAmt", "CesAmt",
    "StateCesAmt", "OthChrg", "TotItemVal",
}
VAL_KEYS = {
    "AssVal", "CgstVal", "SgstVal", "IgstVal", "CesVal", "StCesVal", "RndOffAmt",
    "TotInvVal",
}
INVOICE_KEYS = {
    "Version", "TranDtls", "DocDtls", "SellerDtls", "BuyerDtls", "ItemList", "ValDtls",
}

# Mandatory-only key sets: what exclude_none yields when no optional is supplied.
# Optional numerics stay (they default to 0); optional strings drop out.
TRAN_MANDATORY_KEYS = {"TaxSch", "SupTyp"}
SELLER_MANDATORY_KEYS = {"Gstin", "LglNm", "Addr1", "Loc", "Pin", "Stcd"}
BUYER_MANDATORY_KEYS = {"Gstin", "LglNm", "Addr1", "Loc", "Pin", "Stcd", "Pos"}


# --- building valid models ------------------------------------------------------


def test_full_invoice_builds():
    inv = Invoice.model_validate(invoice_data())
    assert inv.Version == "1.1"
    assert inv.TranDtls.SupTyp == "B2B"
    assert inv.DocDtls.No == "INV-2024-001"
    assert inv.SellerDtls.Gstin == "29AAACR5055K1Z5"
    assert inv.BuyerDtls.Pos == "29"
    assert len(inv.ItemList) == 1
    assert inv.ItemList[0].TotItemVal == 2360
    assert inv.ValDtls.TotInvVal == 2360


def test_invoice_builds_from_model_instances():
    inv = Invoice(
        TranDtls=TranDtls(**tran_data()),
        DocDtls=DocDtls(**doc_data()),
        SellerDtls=SellerDtls(**seller_data()),
        BuyerDtls=BuyerDtls(**buyer_data()),
        ItemList=[Item(**item_data())],
        ValDtls=ValDtls(**val_data()),
    )
    assert inv == Invoice.model_validate(invoice_data())


def test_minimal_invoice_builds():
    inv = Invoice.model_validate(minimal_invoice_data())
    assert inv.Version == "1.1"
    assert inv.TranDtls.TaxSch == "GST"
    assert inv.SellerDtls.TrdNm is None
    assert inv.BuyerDtls.Addr2 is None


def test_full_extraction_result_builds():
    res = ExtractionResult.model_validate(result_data())
    assert res.invoice.DocDtls.No == "INV-2024-001"
    assert [p.page for p in res.extraction_meta.pages] == [1, 2]
    assert res.extraction_meta.pages[1].method == "ocr_required"
    assert res.extraction_meta.field_provenance["DocDtls.No"] == {"page": 1, "source": "regex"}
    assert len(res.extraction_meta.warnings) == 2
    assert res.extraction_meta.warnings[0].field == "ItemList[0].TotItemVal"
    assert res.extraction_meta.warnings[1].severity == "info"


def test_extraction_result_with_empty_meta():
    res = ExtractionResult(invoice=Invoice.model_validate(invoice_data()), extraction_meta=ExtractionMeta())
    assert res.extraction_meta.pages == []
    assert res.extraction_meta.field_provenance == {}
    assert res.extraction_meta.warnings == []


def test_extraction_meta_defaults_are_not_shared():
    a = ExtractionMeta()
    b = ExtractionMeta()
    a.pages.append(PageMeta(page=1, method="native"))
    a.field_provenance["x"] = 1
    a.warnings.append(ExtractionWarning(field="x", message="m", check="c"))
    assert b.pages == []
    assert b.field_provenance == {}
    assert b.warnings == []


# --- exact key sets on model_dump(exclude_none=True) ----------------------------


def test_tran_dtls_keys():
    assert set(TranDtls.model_validate(tran_data()).model_dump(exclude_none=True)) == TRAN_KEYS


def test_doc_dtls_keys():
    assert set(DocDtls.model_validate(doc_data()).model_dump(exclude_none=True)) == DOC_KEYS


def test_seller_dtls_keys():
    assert set(SellerDtls.model_validate(seller_data()).model_dump(exclude_none=True)) == SELLER_KEYS


def test_buyer_dtls_keys():
    assert set(BuyerDtls.model_validate(buyer_data()).model_dump(exclude_none=True)) == BUYER_KEYS


def test_item_keys():
    assert set(Item.model_validate(item_data()).model_dump(exclude_none=True)) == ITEM_KEYS


def test_val_dtls_keys():
    assert set(ValDtls.model_validate(val_data()).model_dump(exclude_none=True)) == VAL_KEYS


def test_invoice_top_level_keys():
    assert set(Invoice.model_validate(invoice_data()).model_dump(exclude_none=True)) == INVOICE_KEYS


def test_full_invoice_dump_round_trips_input_exactly():
    """With every optional present, the submission form equals the input dict."""
    inv = Invoice.model_validate(invoice_data())
    dumped = inv.model_dump(exclude_none=True)
    assert dumped == invoice_data()
    assert set(dumped["ItemList"][0]) == ITEM_KEYS
    assert set(dumped["ValDtls"]) == VAL_KEYS


def test_minimal_invoice_dump_omits_absent_optionals_but_keeps_numerics():
    inv = Invoice.model_validate(minimal_invoice_data())
    dumped = inv.model_dump(exclude_none=True)
    assert set(dumped) == INVOICE_KEYS
    assert set(dumped["TranDtls"]) == TRAN_MANDATORY_KEYS
    assert set(dumped["DocDtls"]) == DOC_KEYS
    assert set(dumped["SellerDtls"]) == SELLER_MANDATORY_KEYS
    assert set(dumped["BuyerDtls"]) == BUYER_MANDATORY_KEYS
    assert set(dumped["ItemList"][0]) == ITEM_KEYS
    assert set(dumped["ValDtls"]) == VAL_KEYS
    assert "null" not in inv.model_dump_json(exclude_none=True)


def test_plain_dump_keeps_none_for_absent_optional_strings():
    """Without exclude_none the None is present; the submission form removes it."""
    data = seller_data()
    del data["TrdNm"]
    assert SellerDtls.model_validate(data).model_dump()["TrdNm"] is None


# --- extra="forbid" on every block ----------------------------------------------


def test_unknown_key_in_tran_dtls_rejected():
    with pytest.raises(ValidationError):
        TranDtls.model_validate({**tran_data(), "Bogus": "x"})


def test_unknown_key_in_doc_dtls_rejected():
    with pytest.raises(ValidationError):
        DocDtls.model_validate({**doc_data(), "Bogus": "x"})


def test_unknown_key_in_seller_dtls_rejected():
    with pytest.raises(ValidationError):
        SellerDtls.model_validate({**seller_data(), "Bogus": "x"})


def test_pos_is_not_a_seller_field():
    with pytest.raises(ValidationError):
        SellerDtls.model_validate({**seller_data(), "Pos": "29"})


def test_unknown_key_in_buyer_dtls_rejected():
    with pytest.raises(ValidationError):
        BuyerDtls.model_validate({**buyer_data(), "Bogus": "x"})


def test_unknown_key_in_item_rejected():
    with pytest.raises(ValidationError):
        Item.model_validate({**item_data(), "Bogus": "x"})


def test_unknown_key_in_val_dtls_rejected():
    with pytest.raises(ValidationError):
        ValDtls.model_validate({**val_data(), "Bogus": "x"})


def test_unknown_key_at_invoice_top_level_rejected():
    with pytest.raises(ValidationError):
        Invoice.model_validate({**invoice_data(), "Bogus": "x"})


@pytest.mark.parametrize("block", ["TranDtls", "DocDtls", "SellerDtls", "BuyerDtls", "ValDtls"])
def test_unknown_key_nested_inside_invoice_rejected(block):
    data = invoice_data()
    data[block]["Bogus"] = "x"
    with pytest.raises(ValidationError) as exc:
        Invoice.model_validate(data)
    assert exc.value.errors()[0]["type"] == "extra_forbidden"
    assert exc.value.errors()[0]["loc"] == (block, "Bogus")


def test_unknown_key_nested_inside_item_list_rejected():
    data = invoice_data()
    data["ItemList"][0]["Bogus"] = "x"
    with pytest.raises(ValidationError) as exc:
        Invoice.model_validate(data)
    assert exc.value.errors()[0]["type"] == "extra_forbidden"
    assert exc.value.errors()[0]["loc"] == ("ItemList", 0, "Bogus")


def test_extraction_meta_cannot_leak_into_invoice():
    """The wrapper's keys are not valid inside the invoice block."""
    with pytest.raises(ValidationError):
        Invoice.model_validate({**invoice_data(), "extraction_meta": {}})
    with pytest.raises(ValidationError):
        Invoice.model_validate({**invoice_data(), "warnings": []})


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (ExtractionWarning, {"field": "DocDtls.Dt", "message": "m", "check": "c"}),
        (PageMeta, {"page": 1, "method": "native"}),
        (ExtractionMeta, {}),
        (ExtractionResult, {"invoice": invoice_data(), "extraction_meta": {}}),
    ],
)
def test_unknown_key_in_wrapper_models_rejected(model, data):
    assert model.model_validate(data)
    with pytest.raises(ValidationError):
        model.model_validate({**data, "Bogus": "x"})


# --- optional numerics default to 0, never None ---------------------------------


def test_item_optional_numerics_default_to_zero():
    item = Item.model_validate(minimal_invoice_data()["ItemList"][0])
    for name in ("Discount", "CesAmt", "StateCesAmt", "OthChrg"):
        value = getattr(item, name)
        assert value is not None
        assert value == 0


def test_val_dtls_optional_numerics_default_to_zero():
    val = ValDtls.model_validate(minimal_invoice_data()["ValDtls"])
    for name in ("CesVal", "StCesVal", "RndOffAmt"):
        value = getattr(val, name)
        assert value is not None
        assert value == 0


def test_optional_numerics_survive_exclude_none():
    item = Item.model_validate(minimal_invoice_data()["ItemList"][0]).model_dump(exclude_none=True)
    assert item["Discount"] == 0 and item["CesAmt"] == 0
    val = ValDtls.model_validate(minimal_invoice_data()["ValDtls"]).model_dump(exclude_none=True)
    assert val["RndOffAmt"] == 0 and val["StCesVal"] == 0


def test_optional_numerics_reject_none():
    with pytest.raises(ValidationError):
        Item.model_validate({**item_data(), "Discount": None})
    with pytest.raises(ValidationError):
        ValDtls.model_validate({**val_data(), "RndOffAmt": None})


def test_numeric_fields_are_floats():
    item = Item.model_validate(item_data())
    assert isinstance(item.Qty, float) and isinstance(item.TotItemVal, float)
    assert Item.model_validate({**item_data(), "UnitPrice": 12.5}).UnitPrice == 12.5
    with pytest.raises(ValidationError):
        Item.model_validate({**item_data(), "Qty": "two"})


# --- literal enumerations -------------------------------------------------------


@pytest.mark.parametrize("value", ["B2B", "SEZWP", "SEZWOP", "EXPWP", "EXPWOP", "DEXP"])
def test_sup_typ_accepts_enumeration(value):
    assert TranDtls(SupTyp=value).SupTyp == value


@pytest.mark.parametrize("value", ["B2C", "b2b", "", "EXP", None])
def test_sup_typ_rejects_other_values(value):
    with pytest.raises(ValidationError):
        TranDtls(SupTyp=value)


def test_tax_sch_defaults_to_gst_and_rejects_others():
    assert TranDtls(SupTyp="B2B").TaxSch == "GST"
    with pytest.raises(ValidationError):
        TranDtls(SupTyp="B2B", TaxSch="VAT")


@pytest.mark.parametrize("field", ["RegRev", "IgstOnIntra"])
def test_reg_rev_and_igst_on_intra(field):
    assert getattr(TranDtls(SupTyp="B2B"), field) is None
    assert getattr(TranDtls(SupTyp="B2B", **{field: "Y"}), field) == "Y"
    with pytest.raises(ValidationError):
        TranDtls(SupTyp="B2B", **{field: "yes"})


@pytest.mark.parametrize("value", ["INV", "CRN", "DBN"])
def test_doc_typ_accepts_enumeration(value):
    assert DocDtls(Typ=value, No="1", Dt="01/01/2024").Typ == value


@pytest.mark.parametrize("value", ["inv", "INVOICE", "BOE", "", None])
def test_doc_typ_rejects_other_values(value):
    with pytest.raises(ValidationError):
        DocDtls(Typ=value, No="1", Dt="01/01/2024")


@pytest.mark.parametrize("value", ["Y", "N"])
def test_is_servc_accepts_enumeration(value):
    assert Item.model_validate({**item_data(), "IsServc": value}).IsServc == value


@pytest.mark.parametrize("value", ["y", "Yes", True, 1, "", None])
def test_is_servc_rejects_other_values(value):
    with pytest.raises(ValidationError):
        Item.model_validate({**item_data(), "IsServc": value})


@pytest.mark.parametrize("value", ["scanned", "OCR", "", None])
def test_page_method_rejects_other_values(value):
    with pytest.raises(ValidationError):
        PageMeta(page=1, method=value)


@pytest.mark.parametrize("value", ["error", "WARNING", "", None])
def test_warning_severity_rejects_other_values(value):
    with pytest.raises(ValidationError):
        ExtractionWarning(field="f", message="m", check="c", severity=value)


# --- mandatory fields and list constraints --------------------------------------


def test_doc_dt_format_is_not_validated():
    """The model carries whatever was read; format problems are for the warnings block."""
    assert DocDtls(Typ="INV", No="1", Dt="2024-03-15").Dt == "2024-03-15"
    assert DocDtls(Typ="INV", No="1", Dt="not a date").Dt == "not a date"


def test_tran_dtls_sup_typ_mandatory():
    with pytest.raises(ValidationError):
        TranDtls.model_validate({"TaxSch": "GST"})


@pytest.mark.parametrize("field", ["Typ", "No", "Dt"])
def test_doc_dtls_mandatory(field):
    with pytest.raises(ValidationError):
        DocDtls.model_validate({k: v for k, v in doc_data().items() if k != field})


@pytest.mark.parametrize("field", ["Gstin", "LglNm", "Addr1", "Loc", "Pin", "Stcd"])
def test_seller_dtls_mandatory(field):
    with pytest.raises(ValidationError):
        SellerDtls.model_validate({k: v for k, v in seller_data().items() if k != field})


@pytest.mark.parametrize("field", ["Gstin", "LglNm", "Addr1", "Loc", "Pin", "Stcd", "Pos"])
def test_buyer_dtls_mandatory(field):
    with pytest.raises(ValidationError):
        BuyerDtls.model_validate({k: v for k, v in buyer_data().items() if k != field})


@pytest.mark.parametrize(
    "field",
    ["SlNo", "PrdDesc", "IsServc", "HsnCd", "Qty", "Unit", "UnitPrice", "TotAmt",
     "AssAmt", "GstRt", "CgstAmt", "SgstAmt", "IgstAmt", "TotItemVal"],
)
def test_item_mandatory(field):
    with pytest.raises(ValidationError):
        Item.model_validate({k: v for k, v in item_data().items() if k != field})


@pytest.mark.parametrize("field", ["AssVal", "CgstVal", "SgstVal", "IgstVal", "TotInvVal"])
def test_val_dtls_mandatory(field):
    with pytest.raises(ValidationError):
        ValDtls.model_validate({k: v for k, v in val_data().items() if k != field})


@pytest.mark.parametrize("field", ["TranDtls", "DocDtls", "SellerDtls", "BuyerDtls", "ItemList", "ValDtls"])
def test_invoice_blocks_mandatory(field):
    with pytest.raises(ValidationError):
        Invoice.model_validate({k: v for k, v in invoice_data().items() if k != field})


def test_item_list_requires_at_least_one_item():
    with pytest.raises(ValidationError) as exc:
        Invoice.model_validate({**invoice_data(), "ItemList": []})
    assert exc.value.errors()[0]["type"] == "too_short"


def test_item_list_accepts_multiple_items():
    second = {**item_data(), "SlNo": "2"}
    inv = Invoice.model_validate({**invoice_data(), "ItemList": [item_data(), second]})
    assert [i.SlNo for i in inv.ItemList] == ["1", "2"]


def test_version_defaults_and_is_a_plain_string():
    assert Invoice.model_validate(minimal_invoice_data()).Version == "1.1"
    assert Invoice.model_validate({**invoice_data(), "Version": "1.0"}).Version == "1.0"


def test_pin_is_int():
    assert SellerDtls.model_validate(seller_data()).Pin == 560001
    with pytest.raises(ValidationError):
        SellerDtls.model_validate({**seller_data(), "Pin": "not a pin"})


def test_extraction_result_requires_both_parts():
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate({"invoice": invoice_data()})
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate({"extraction_meta": {}})


@pytest.mark.parametrize("field", ["field", "message", "check"])
def test_extraction_warning_mandatory(field):
    data = {"field": "DocDtls.Dt", "message": "m", "check": "c"}
    del data[field]
    with pytest.raises(ValidationError):
        ExtractionWarning.model_validate(data)


# --- serialisation --------------------------------------------------------------


def test_extraction_result_round_trips_through_json():
    original = ExtractionResult.model_validate(result_data())
    restored = ExtractionResult.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.model_dump() == original.model_dump()


def test_extraction_result_json_shape():
    dumped = ExtractionResult.model_validate(result_data()).model_dump(exclude_none=True)
    assert set(dumped) == {"invoice", "extraction_meta"}
    assert set(dumped["invoice"]) == INVOICE_KEYS
    assert set(dumped["extraction_meta"]) == {"pages", "field_provenance", "warnings"}


def test_minimal_extraction_result_round_trips_through_json():
    original = ExtractionResult(invoice=Invoice.model_validate(minimal_invoice_data()), extraction_meta=ExtractionMeta())
    restored = ExtractionResult.model_validate_json(original.model_dump_json(exclude_none=True))
    assert restored == original


def test_warnings_serialise_as_expected():
    warning = ExtractionWarning(field="ItemList[2].TotItemVal", message="mismatch", check="item_total_arithmetic")
    assert warning.model_dump() == {
        "field": "ItemList[2].TotItemVal",
        "message": "mismatch",
        "severity": "warning",
        "check": "item_total_arithmetic",
    }
    info = ExtractionWarning(field="DocDtls.Dt", message="normalised", severity="info", check="date_format")
    assert info.model_dump()["severity"] == "info"
    meta = ExtractionMeta(warnings=[warning, info])
    assert [w["field"] for w in meta.model_dump()["warnings"]] == ["ItemList[2].TotItemVal", "DocDtls.Dt"]


def test_pages_serialise_as_expected():
    meta = ExtractionMeta(pages=[PageMeta(page=1, method="native"), PageMeta(page=2, method="ocr_required")])
    assert meta.model_dump()["pages"] == [
        {"page": 1, "method": "native"},
        {"page": 2, "method": "ocr_required"},
    ]


def test_field_provenance_accepts_arbitrary_values():
    meta = ExtractionMeta(field_provenance={"DocDtls.No": {"page": 1, "bbox": [1, 2, 3, 4]}, "count": 3})
    assert meta.model_dump()["field_provenance"] == {"DocDtls.No": {"page": 1, "bbox": [1, 2, 3, 4]}, "count": 3}
