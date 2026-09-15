"""Tests for gst_einvoice.validators."""

import pytest

from gst_einvoice.schema import (
    BuyerDtls,
    DocDtls,
    ExtractionWarning,
    Invoice,
    Item,
    SellerDtls,
    TranDtls,
    ValDtls,
)
from gst_einvoice.validators import (
    TAX_SPLIT_LIMITATION,
    validate_invoice,
    validate_invoice_total,
    validate_item_total,
    validate_tax_split,
    validate_val_dtls_sums,
)

# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #


def make_item(
    ass_amt: float,
    gst_rt: float,
    *,
    cgst: float = 0,
    sgst: float = 0,
    igst: float = 0,
    ces: float = 0,
    state_ces: float = 0,
    oth: float = 0,
    tot_item_val: float | None = None,
    sl_no: str = "1",
) -> Item:
    """Build an Item; TotItemVal defaults to the arithmetically correct value."""
    if tot_item_val is None:
        tot_item_val = ass_amt + cgst + sgst + igst + ces + state_ces + oth
    return Item(
        SlNo=sl_no,
        PrdDesc="Widget",
        IsServc="N",
        HsnCd="8471",
        Qty=1,
        Unit="NOS",
        UnitPrice=ass_amt,
        TotAmt=ass_amt,
        AssAmt=ass_amt,
        GstRt=gst_rt,
        CgstAmt=cgst,
        SgstAmt=sgst,
        IgstAmt=igst,
        CesAmt=ces,
        StateCesAmt=state_ces,
        OthChrg=oth,
        TotItemVal=tot_item_val,
    )


def intra_item(ass_amt: float, gst_rt: float, **kw) -> Item:
    """Item with a correct intra-state split (CGST + SGST)."""
    half = ass_amt * gst_rt / 200
    return make_item(ass_amt, gst_rt, cgst=half, sgst=half, **kw)


def inter_item(ass_amt: float, gst_rt: float, **kw) -> Item:
    """Item with a correct inter-state split (IGST only)."""
    return make_item(ass_amt, gst_rt, igst=ass_amt * gst_rt / 100, **kw)


def make_val_dtls(items: list[Item], *, rnd_off: float = 0, **overrides) -> ValDtls:
    """ValDtls summed from items; any field may be overridden."""
    fields = {
        "AssVal": sum(i.AssAmt for i in items),
        "CgstVal": sum(i.CgstAmt for i in items),
        "SgstVal": sum(i.SgstAmt for i in items),
        "IgstVal": sum(i.IgstAmt for i in items),
        "CesVal": sum(i.CesAmt for i in items),
        "StCesVal": sum(i.StateCesAmt for i in items),
        "RndOffAmt": rnd_off,
    }
    fields.update(overrides)
    fields.setdefault(
        "TotInvVal",
        fields["AssVal"] + fields["CgstVal"] + fields["SgstVal"] + fields["IgstVal"]
        + fields["CesVal"] + fields["StCesVal"] + fields["RndOffAmt"],
    )
    return ValDtls(**fields)


def make_invoice(
    items: list[Item],
    *,
    seller_stcd: str = "29",
    buyer_stcd: str = "29",
    pos: str | None = None,
    reg_rev: str | None = None,
    val_dtls: ValDtls | None = None,
) -> Invoice:
    """Invoice around ``items``; intra-state (29/29) by default, Pos defaults to buyer Stcd."""
    return Invoice(
        TranDtls=TranDtls(SupTyp="B2B", RegRev=reg_rev),
        DocDtls=DocDtls(Typ="INV", No="INV-001", Dt="01/04/2026"),
        SellerDtls=SellerDtls(
            Gstin="29AAAAA0000A1Z5", LglNm="Seller Ltd", Addr1="1 Main St",
            Loc="Bengaluru", Pin=560001, Stcd=seller_stcd,
        ),
        BuyerDtls=BuyerDtls(
            Gstin="29BBBBB0000B1Z5", LglNm="Buyer Ltd", Addr1="2 High St",
            Loc="Mysuru", Pin=570001, Stcd=buyer_stcd,
            Pos=buyer_stcd if pos is None else pos,
        ),
        ItemList=items,
        ValDtls=val_dtls if val_dtls is not None else make_val_dtls(items),
    )


def intra_invoice(**kw) -> Invoice:
    return make_invoice([intra_item(1000, 18, sl_no="1"), intra_item(500, 12, sl_no="2")], **kw)


def inter_invoice(**kw) -> Invoice:
    return make_invoice(
        [inter_item(1000, 18, sl_no="1"), inter_item(500, 12, sl_no="2")],
        seller_stcd="29", buyer_stcd="27", **kw,
    )


def fields_of(warnings: list[ExtractionWarning]) -> list[str]:
    return [w.field for w in warnings]


# --------------------------------------------------------------------------- #
# validate_item_total
# --------------------------------------------------------------------------- #


class TestValidateItemTotal:
    def test_correct_items_produce_no_warnings(self):
        assert validate_item_total(intra_invoice()) == []

    def test_delta_0_02_passes(self):
        inv = make_invoice([intra_item(1000, 18, tot_item_val=1180.02)])
        assert validate_item_total(inv) == []

    def test_delta_5_00_fails(self):
        inv = make_invoice([intra_item(1000, 18, tot_item_val=1185.00)])
        warnings = validate_item_total(inv)
        assert len(warnings) == 1
        w = warnings[0]
        assert w.field == "ItemList[0].TotItemVal"
        assert w.severity == "warning"
        assert w.check == "validate_item_total"

    def test_delta_exactly_0_05_passes(self):
        inv = make_invoice([intra_item(1000, 18, tot_item_val=1180.05)])
        assert validate_item_total(inv) == []

    def test_delta_exactly_tolerance_passes_despite_float_representation(self):
        # 1.05 - 1.0 == 0.050000000000000044 in IEEE-754; the epsilon must absorb it.
        assert 1.05 - 1.0 > 0.05
        inv = make_invoice([make_item(1.0, 0, tot_item_val=1.05)])
        assert validate_item_total(inv) == []

    def test_negative_delta_fails_too(self):
        inv = make_invoice([intra_item(1000, 18, tot_item_val=1175.00)])
        assert fields_of(validate_item_total(inv)) == ["ItemList[0].TotItemVal"]

    def test_tolerance_parameter_is_honoured(self):
        inv = make_invoice([intra_item(1000, 18, tot_item_val=1185.00)])
        assert validate_item_total(inv, tolerance=10.0) == []
        assert len(validate_item_total(inv, tolerance=4.99)) == 1

    def test_message_format_matches_spec_example(self):
        item = make_item(1000, 18, cgst=90, sgst=90, oth=5, tot_item_val=1180.00)
        w = validate_item_total(make_invoice([item]))[0]
        assert w.message == (
            "TotItemVal is 1180.00 but AssAmt + CgstAmt + SgstAmt + IgstAmt + CesAmt "
            "+ StateCesAmt + OthChrg = 1185.00 (delta 5 exceeds tolerance 0.05)"
        )

    def test_message_reports_custom_tolerance(self):
        inv = make_invoice([intra_item(1000, 18, tot_item_val=1190.00)])
        w = validate_item_total(inv, tolerance=1.5)[0]
        assert "delta 10 exceeds tolerance 1.5" in w.message

    def test_message_never_shows_delta_equal_to_tolerance(self):
        # A delta of 0.055 fails the 0.05 tolerance; rendering both at two decimals
        # would print the self-contradictory "delta 0.05 exceeds tolerance 0.05".
        inv = make_invoice([intra_item(1000, 18, tot_item_val=1180.055)])
        w = validate_item_total(inv)[0]
        assert "(delta 0.055 exceeds tolerance 0.05)" in w.message
        assert "delta 0.05 exceeds" not in w.message

    def test_message_keeps_paise_and_avoids_exponents_for_large_deltas(self):
        # A dropped digit or a missing line item gives a large delta. It must print
        # in full ("123456.78", "10000000"), never rounded to "123457" or "1e+07",
        # which would contradict the actual and expected values in the same sentence.
        inv = make_invoice([intra_item(1000, 18, tot_item_val=1180.00 + 123456.78)])
        assert "(delta 123456.78 exceeds tolerance 0.05)" in validate_item_total(inv)[0].message
        inv = make_invoice([intra_item(1000, 18, tot_item_val=1180.00 + 10_000_000)])
        assert "(delta 10000000 exceeds tolerance 0.05)" in validate_item_total(inv)[0].message

    def test_cess_state_cess_and_other_charges_count_toward_total(self):
        item = make_item(1000, 18, cgst=90, sgst=90, ces=10, state_ces=20, oth=30)
        assert item.TotItemVal == 1240
        assert validate_item_total(make_invoice([item])) == []

    def test_field_path_uses_zero_based_index_of_failing_item(self):
        items = [
            intra_item(100, 18, sl_no="1"),
            intra_item(200, 18, sl_no="2"),
            intra_item(300, 18, tot_item_val=400.00, sl_no="3"),
        ]
        warnings = validate_item_total(make_invoice(items))
        assert fields_of(warnings) == ["ItemList[2].TotItemVal"]

    def test_one_warning_per_failing_item(self):
        items = [
            intra_item(100, 18, tot_item_val=200, sl_no="1"),
            intra_item(200, 18, sl_no="2"),
            intra_item(300, 18, tot_item_val=100, sl_no="3"),
        ]
        warnings = validate_item_total(make_invoice(items))
        assert fields_of(warnings) == ["ItemList[0].TotItemVal", "ItemList[2].TotItemVal"]


# --------------------------------------------------------------------------- #
# validate_invoice_total
# --------------------------------------------------------------------------- #


class TestValidateInvoiceTotal:
    def test_correct_invoice_produces_no_warnings(self):
        assert validate_invoice_total(intra_invoice()) == []

    def test_delta_0_02_passes(self):
        items = [intra_item(1000, 18)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, TotInvVal=1180.02))
        assert validate_invoice_total(inv) == []

    def test_delta_5_00_fails(self):
        items = [intra_item(1000, 18)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, TotInvVal=1185.00))
        warnings = validate_invoice_total(inv)
        assert len(warnings) == 1
        w = warnings[0]
        assert w.field == "ValDtls.TotInvVal"
        assert w.severity == "warning"
        assert w.check == "validate_invoice_total"
        assert w.message == (
            "TotInvVal is 1185.00 but AssVal + CgstVal + SgstVal + IgstVal + CesVal "
            "+ StCesVal + RndOffAmt = 1180.00 (delta 5 exceeds tolerance 0.05)"
        )

    def test_delta_exactly_0_05_passes(self):
        items = [intra_item(1000, 18)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, TotInvVal=1180.05))
        assert validate_invoice_total(inv) == []

    def test_tolerance_parameter_is_honoured(self):
        items = [intra_item(1000, 18)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, TotInvVal=1185.00))
        assert validate_invoice_total(inv, tolerance=10.0) == []

    def test_round_off_and_cess_count_toward_total(self):
        items = [intra_item(1000, 18, ces=10, state_ces=5)]
        val = make_val_dtls(items, rnd_off=-0.45)
        assert val.TotInvVal == pytest.approx(1194.55)
        assert validate_invoice_total(make_invoice(items, val_dtls=val)) == []


# --------------------------------------------------------------------------- #
# validate_val_dtls_sums
# --------------------------------------------------------------------------- #


class TestValidateValDtlsSums:
    def test_correct_invoice_produces_no_warnings(self):
        assert validate_val_dtls_sums(intra_invoice()) == []
        assert validate_val_dtls_sums(inter_invoice()) == []

    def test_delta_0_02_passes(self):
        items = [intra_item(1000, 18), intra_item(500, 12)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, AssVal=1500.02))
        assert validate_val_dtls_sums(inv) == []

    def test_delta_5_00_fails(self):
        items = [intra_item(1000, 18), intra_item(500, 12)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, AssVal=1505.00))
        warnings = validate_val_dtls_sums(inv)
        assert len(warnings) == 1
        w = warnings[0]
        assert w.field == "ValDtls.AssVal"
        assert w.severity == "warning"
        assert w.check == "validate_val_dtls_sums"
        assert w.message == (
            "AssVal is 1505.00 but sum of ItemList[].AssAmt = 1500.00 "
            "(delta 5 exceeds tolerance 0.05)"
        )

    def test_delta_exactly_0_05_passes(self):
        items = [intra_item(1000, 18), intra_item(500, 12)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, CgstVal=120.05))
        assert validate_val_dtls_sums(inv) == []

    def test_tolerance_parameter_is_honoured(self):
        items = [intra_item(1000, 18), intra_item(500, 12)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, AssVal=1505.00))
        assert validate_val_dtls_sums(inv, tolerance=10.0) == []

    @pytest.mark.parametrize(
        "total_field, item_field",
        [
            ("AssVal", "AssAmt"),
            ("CgstVal", "CgstAmt"),
            ("SgstVal", "SgstAmt"),
            ("IgstVal", "IgstAmt"),
            ("CesVal", "CesAmt"),
            ("StCesVal", "StateCesAmt"),
        ],
    )
    def test_each_of_the_six_pairs_is_checked(self, total_field, item_field):
        items = [
            make_item(1000, 18, cgst=90, sgst=90, igst=0, ces=10, state_ces=20),
            make_item(500, 18, cgst=0, sgst=0, igst=90, ces=5, state_ces=15),
        ]
        correct = make_val_dtls(items)
        wrong = correct.model_copy(update={total_field: getattr(correct, total_field) + 5.0})
        warnings = validate_val_dtls_sums(make_invoice(items, val_dtls=wrong))
        assert fields_of(warnings) == [f"ValDtls.{total_field}"]
        assert f"sum of ItemList[].{item_field}" in warnings[0].message

    def test_one_warning_per_failing_field_in_schema_order(self):
        items = [intra_item(1000, 18), intra_item(500, 12)]
        inv = make_invoice(
            items,
            val_dtls=make_val_dtls(items, AssVal=1600, SgstVal=100, StCesVal=7),
        )
        assert fields_of(validate_val_dtls_sums(inv)) == [
            "ValDtls.AssVal", "ValDtls.SgstVal", "ValDtls.StCesVal",
        ]

    def test_only_the_six_component_totals_are_checked_not_tot_inv_val(self):
        # TotInvVal is validate_invoice_total's job; a wrong TotInvVal alone is silent here.
        items = [intra_item(1000, 18)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, TotInvVal=9999))
        assert validate_val_dtls_sums(inv) == []


# --------------------------------------------------------------------------- #
# validate_tax_split
# --------------------------------------------------------------------------- #


class TestValidateTaxSplit:
    def test_correct_intra_state_invoice_produces_no_warnings(self):
        assert validate_tax_split(intra_invoice()) == []

    def test_correct_inter_state_invoice_produces_no_warnings(self):
        assert validate_tax_split(inter_invoice()) == []

    def test_delta_0_02_passes(self):
        inv = make_invoice([make_item(1000, 18, cgst=90.02, sgst=89.98)])
        assert validate_tax_split(inv) == []

    def test_delta_5_00_fails(self):
        inv = make_invoice([make_item(1000, 18, cgst=95.00, sgst=90.00)])
        warnings = validate_tax_split(inv)
        assert len(warnings) == 1
        w = warnings[0]
        assert w.field == "ItemList[0].CgstAmt"
        assert w.severity == "warning"
        assert w.check == "validate_tax_split"
        assert w.message.startswith(
            "CgstAmt is 95.00 but AssAmt x GstRt / 2 / 100 = 90.00 "
            "(delta 5 exceeds tolerance 0.05)"
        )

    def test_delta_exactly_0_05_passes(self):
        inv = make_invoice([make_item(1000, 18, cgst=90.05, sgst=90.00)])
        assert validate_tax_split(inv) == []

    def test_tolerance_parameter_is_honoured(self):
        inv = make_invoice([make_item(1000, 18, cgst=95.00, sgst=90.00)])
        assert validate_tax_split(inv, tolerance=10.0) == []
        assert len(validate_tax_split(inv, tolerance=4.0)) == 1

    def test_intra_state_with_igst_populated_is_flagged_on_all_three_fields(self):
        # Seller 29, buyer 29, but the item carries IGST instead of CGST/SGST.
        inv = make_invoice([inter_item(1000, 18)], seller_stcd="29", buyer_stcd="29")
        warnings = validate_tax_split(inv)
        assert fields_of(warnings) == [
            "ItemList[0].CgstAmt", "ItemList[0].SgstAmt", "ItemList[0].IgstAmt",
        ]
        by_field = {w.field: w.message for w in warnings}
        assert "CgstAmt is 0.00 but AssAmt x GstRt / 2 / 100 = 90.00" in by_field["ItemList[0].CgstAmt"]
        assert "SgstAmt is 0.00 but AssAmt x GstRt / 2 / 100 = 90.00" in by_field["ItemList[0].SgstAmt"]
        assert "IgstAmt is 180.00 but 0 (intra-state supply) = 0.00" in by_field["ItemList[0].IgstAmt"]

    def test_inter_state_with_cgst_sgst_populated_is_flagged_on_all_three_fields(self):
        inv = make_invoice([intra_item(1000, 18)], seller_stcd="29", buyer_stcd="27")
        warnings = validate_tax_split(inv)
        assert fields_of(warnings) == [
            "ItemList[0].CgstAmt", "ItemList[0].SgstAmt", "ItemList[0].IgstAmt",
        ]
        by_field = {w.field: w.message for w in warnings}
        assert "CgstAmt is 90.00 but 0 (inter-state supply) = 0.00" in by_field["ItemList[0].CgstAmt"]
        assert "SgstAmt is 90.00 but 0 (inter-state supply) = 0.00" in by_field["ItemList[0].SgstAmt"]
        assert "IgstAmt is 0.00 but AssAmt x GstRt / 100 = 180.00" in by_field["ItemList[0].IgstAmt"]

    def test_only_the_wrong_field_is_flagged(self):
        inv = make_invoice([make_item(1000, 18, cgst=90, sgst=80)])
        assert fields_of(validate_tax_split(inv)) == ["ItemList[0].SgstAmt"]

    def test_field_path_uses_zero_based_index_of_failing_item(self):
        items = [
            intra_item(100, 18, sl_no="1"),
            intra_item(200, 18, sl_no="2"),
            make_item(300, 18, cgst=27, sgst=27, igst=54, sl_no="3"),
        ]
        warnings = validate_tax_split(make_invoice(items))
        assert fields_of(warnings) == ["ItemList[2].IgstAmt"]

    def test_wrong_rate_applied_is_flagged(self):
        # 12% charged on an item whose GstRt says 18%.
        inv = make_invoice([make_item(1000, 18, cgst=60, sgst=60)])
        assert fields_of(validate_tax_split(inv)) == ["ItemList[0].CgstAmt", "ItemList[0].SgstAmt"]

    def test_every_warning_ends_with_the_limitation_sentence(self):
        intra_wrong = make_invoice([inter_item(1000, 18)], seller_stcd="29", buyer_stcd="29")
        inter_wrong = make_invoice([intra_item(1000, 18)], seller_stcd="29", buyer_stcd="27")
        warnings = validate_tax_split(intra_wrong) + validate_tax_split(inter_wrong)
        assert len(warnings) == 6
        for w in warnings:
            assert w.message.endswith(TAX_SPLIT_LIMITATION)

    def test_limitation_sentence_text_is_exact(self):
        assert TAX_SPLIT_LIMITATION == (
            "Note: this check compares the seller's and buyer's registered state codes "
            "(SellerDtls.Stcd vs BuyerDtls.Stcd) rather than the true Place of Supply, "
            "and may false-flag SEZ supplies and bill-to/ship-to cases."
        )

    def test_gst_rate_zero_with_zero_taxes_produces_no_warnings(self):
        assert validate_tax_split(make_invoice([make_item(1000, 0)])) == []
        assert validate_tax_split(
            make_invoice([make_item(1000, 0)], seller_stcd="29", buyer_stcd="27")
        ) == []

    def test_reverse_charge_skips_with_one_info_note_even_when_split_is_wrong(self):
        inv = make_invoice([inter_item(1000, 18)], seller_stcd="29", buyer_stcd="29", reg_rev="Y")
        result = validate_tax_split(inv)
        assert [w.severity for w in result] == ["info"]
        # Prove the split really is wrong: the same invoice without the flag warns.
        without_flag = inv.model_copy(update={"TranDtls": TranDtls(SupTyp="B2B", RegRev="N")})
        assert len(validate_tax_split(without_flag)) == 3

    def test_reverse_charge_note_shape(self):
        inv = make_invoice([inter_item(1000, 18)], seller_stcd="29", buyer_stcd="29", reg_rev="Y")
        notes = validate_tax_split(inv)
        assert len(notes) == 1
        note = notes[0]
        assert note.severity == "info"
        assert note.field == "TranDtls.RegRev"
        assert note.check == "validate_tax_split"
        assert "reverse charge" in note.message
        assert "skipped" in note.message.lower()
        assert [n for n in notes if n.severity == "warning"] == []

    def test_reg_rev_n_does_not_skip(self):
        inv = make_invoice([inter_item(1000, 18)], seller_stcd="29", buyer_stcd="29", reg_rev="N")
        assert all(w.severity == "warning" for w in validate_tax_split(inv))
        assert len(validate_tax_split(inv)) == 3

    def test_pos_differing_from_buyer_stcd_skips_with_one_info_note(self):
        inv = make_invoice(
            [inter_item(1000, 18)], seller_stcd="29", buyer_stcd="29", pos="27",
        )
        notes = validate_tax_split(inv)
        assert len(notes) == 1
        note = notes[0]
        assert note.severity == "info"
        assert note.field == "BuyerDtls.Pos"
        assert note.check == "validate_tax_split"
        assert "place of supply" in note.message.lower()
        assert "cannot evaluate" in note.message
        assert [n for n in notes if n.severity == "warning"] == []

    def test_pos_equal_to_buyer_stcd_does_not_skip(self):
        inv = make_invoice([inter_item(1000, 18)], seller_stcd="29", buyer_stcd="29", pos="29")
        assert len(validate_tax_split(inv)) == 3

    def test_reverse_charge_takes_precedence_over_pos_mismatch(self):
        inv = make_invoice([intra_item(1000, 18)], pos="27", reg_rev="Y")
        notes = validate_tax_split(inv)
        assert fields_of(notes) == ["TranDtls.RegRev"]

    def test_skip_notes_do_not_carry_the_limitation_sentence(self):
        # The limitation applies to warnings the rule actually emits; a skip note is not one.
        for inv in (
            make_invoice([intra_item(1000, 18)], reg_rev="Y"),
            make_invoice([intra_item(1000, 18)], pos="27"),
        ):
            (note,) = validate_tax_split(inv)
            assert note.severity == "info"
            assert TAX_SPLIT_LIMITATION not in note.message


# --------------------------------------------------------------------------- #
# validate_invoice
# --------------------------------------------------------------------------- #


class TestValidateInvoice:
    def test_fully_correct_invoice_produces_nothing(self):
        assert validate_invoice(intra_invoice()) == []
        assert validate_invoice(inter_invoice()) == []

    def test_concatenates_all_four_validators_in_order(self):
        # Item 1 total wrong; ValDtls.TotInvVal wrong; ValDtls.AssVal wrong; item 0 IGST wrong.
        items = [
            make_item(1000, 18, cgst=90, sgst=90, igst=180, sl_no="1"),
            intra_item(500, 12, tot_item_val=600, sl_no="2"),
        ]
        val = make_val_dtls(items, AssVal=1600, TotInvVal=5000)
        inv = make_invoice(items, val_dtls=val)
        warnings = validate_invoice(inv)
        assert [(w.check, w.field) for w in warnings] == [
            ("validate_item_total", "ItemList[1].TotItemVal"),
            ("validate_invoice_total", "ValDtls.TotInvVal"),
            ("validate_val_dtls_sums", "ValDtls.AssVal"),
            ("validate_tax_split", "ItemList[0].IgstAmt"),
        ]
        assert warnings == (
            validate_item_total(inv)
            + validate_invoice_total(inv)
            + validate_val_dtls_sums(inv)
            + validate_tax_split(inv)
        )

    def test_passes_tolerance_through_to_every_validator(self):
        items = [
            make_item(1000, 18, cgst=95, sgst=90, tot_item_val=1190, sl_no="1"),
        ]
        val = make_val_dtls(items, AssVal=1005, TotInvVal=1195)
        inv = make_invoice(items, val_dtls=val)
        assert len(validate_invoice(inv)) == 4
        assert validate_invoice(inv, tolerance=10.0) == []

    def test_includes_tax_split_info_note(self):
        inv = intra_invoice(reg_rev="Y")
        result = validate_invoice(inv)
        assert len(result) == 1
        assert result[0].severity == "info"
        assert result[0].field == "TranDtls.RegRev"

    def test_every_result_is_an_extraction_warning(self):
        items = [make_item(1000, 18, cgst=95, sgst=90, tot_item_val=1190)]
        inv = make_invoice(items, val_dtls=make_val_dtls(items, AssVal=1005, TotInvVal=1195))
        for w in validate_invoice(inv):
            assert isinstance(w, ExtractionWarning)
            assert w.severity == "warning"
            assert w.field
            assert w.message
