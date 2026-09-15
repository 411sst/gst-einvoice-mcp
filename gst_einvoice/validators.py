"""Arithmetic and tax-split consistency checks over an INV-01 ``Invoice``.

Every check takes a tolerance (default 0.05 rupees) and returns structured
``ExtractionWarning`` entries naming the exact field path -- never a bare
boolean. Comparisons are never exact float equality: real invoices round to
the nearest rupee, so ``a ~= b`` means ``abs(a - b) <= tolerance`` (plus a
tiny epsilon so a delta of exactly ``tolerance`` passes despite float
representation).
"""

from gst_einvoice.schema import ExtractionWarning, Invoice

_EPSILON = 1e-9

_ITEM_TOTAL_FORMULA = "AssAmt + CgstAmt + SgstAmt + IgstAmt + CesAmt + StateCesAmt + OthChrg"
_INVOICE_TOTAL_FORMULA = "AssVal + CgstVal + SgstVal + IgstVal + CesVal + StCesVal + RndOffAmt"

TAX_SPLIT_LIMITATION = (
    "Note: this check compares the seller's and buyer's registered state codes "
    "(SellerDtls.Stcd vs BuyerDtls.Stcd) rather than the true Place of Supply, "
    "and may false-flag SEZ supplies and bill-to/ship-to cases."
)


def _compare(
    field: str,
    actual: float,
    formula: str,
    expected: float,
    tolerance: float,
    check: str,
    suffix: str = "",
) -> ExtractionWarning | None:
    """Return a warning if ``actual`` is not within ``tolerance`` of ``expected``, else None."""
    delta = abs(actual - expected)
    if delta <= tolerance + _EPSILON:
        return None
    leaf = field.rsplit(".", 1)[-1]
    return ExtractionWarning(
        field=field,
        check=check,
        severity="warning",
        message=(
            f"{leaf} is {actual:.2f} but {formula} = {expected:.2f} "
            f"(delta {round(delta, 8):.10g} exceeds tolerance {tolerance:.10g}){suffix}"
        ),
    )


def validate_item_total(invoice: Invoice, tolerance: float = 0.05) -> list[ExtractionWarning]:
    """Per item: TotItemVal ~= AssAmt + CgstAmt + SgstAmt + IgstAmt + CesAmt + StateCesAmt + OthChrg."""
    warnings: list[ExtractionWarning] = []
    for i, item in enumerate(invoice.ItemList):
        expected = (
            item.AssAmt + item.CgstAmt + item.SgstAmt + item.IgstAmt
            + item.CesAmt + item.StateCesAmt + item.OthChrg
        )
        warning = _compare(
            f"ItemList[{i}].TotItemVal", item.TotItemVal, _ITEM_TOTAL_FORMULA,
            expected, tolerance, "validate_item_total",
        )
        if warning is not None:
            warnings.append(warning)
    return warnings


def validate_invoice_total(invoice: Invoice, tolerance: float = 0.05) -> list[ExtractionWarning]:
    """TotInvVal ~= AssVal + CgstVal + SgstVal + IgstVal + CesVal + StCesVal + RndOffAmt."""
    val = invoice.ValDtls
    expected = (
        val.AssVal + val.CgstVal + val.SgstVal + val.IgstVal
        + val.CesVal + val.StCesVal + val.RndOffAmt
    )
    warning = _compare(
        "ValDtls.TotInvVal", val.TotInvVal, _INVOICE_TOTAL_FORMULA,
        expected, tolerance, "validate_invoice_total",
    )
    return [] if warning is None else [warning]


def validate_val_dtls_sums(invoice: Invoice, tolerance: float = 0.05) -> list[ExtractionWarning]:
    """Each ValDtls total ~= sum of the corresponding ItemList field. One warning per failing field."""
    val = invoice.ValDtls
    items = invoice.ItemList
    pairs = (
        ("AssVal", val.AssVal, "AssAmt"),
        ("CgstVal", val.CgstVal, "CgstAmt"),
        ("SgstVal", val.SgstVal, "SgstAmt"),
        ("IgstVal", val.IgstVal, "IgstAmt"),
        ("CesVal", val.CesVal, "CesAmt"),
        ("StCesVal", val.StCesVal, "StateCesAmt"),
    )
    warnings: list[ExtractionWarning] = []
    for total_name, actual, item_field in pairs:
        expected = sum(getattr(item, item_field) for item in items)
        warning = _compare(
            f"ValDtls.{total_name}", actual, f"sum of ItemList[].{item_field}",
            expected, tolerance, "validate_val_dtls_sums",
        )
        if warning is not None:
            warnings.append(warning)
    return warnings


def validate_tax_split(invoice: Invoice, tolerance: float = 0.05) -> list[ExtractionWarning]:
    """Check CGST/SGST vs IGST per item against the seller/buyer state codes.

    Skipped (one ``info`` note, no warnings) under reverse charge (``RegRev == "Y"``)
    and when the buyer's ``Pos`` differs from its ``Stcd``. Intra-state
    (``SellerDtls.Stcd == BuyerDtls.Stcd``) expects CGST and SGST each ~=
    AssAmt x GstRt / 2 / 100 and IGST ~= 0; inter-state expects IGST ~=
    AssAmt x GstRt / 100 and CGST, SGST ~= 0.
    """
    check = "validate_tax_split"
    if invoice.TranDtls.RegRev == "Y":
        return [ExtractionWarning(
            field="TranDtls.RegRev",
            severity="info",
            check=check,
            message=(
                "Tax-split check skipped: RegRev is 'Y' (reverse charge), so tax is "
                "legitimately absent from the invoice face and the CGST/SGST/IGST split "
                "cannot be evaluated."
            ),
        )]
    buyer = invoice.BuyerDtls
    if buyer.Pos != buyer.Stcd:
        return [ExtractionWarning(
            field="BuyerDtls.Pos",
            severity="info",
            check=check,
            message=(
                f"Tax-split check skipped: place of supply (BuyerDtls.Pos = {buyer.Pos}) "
                f"differs from the buyer's registered state (BuyerDtls.Stcd = {buyer.Stcd}); "
                "this rule cannot evaluate that case."
            ),
        )]

    intra = invoice.SellerDtls.Stcd == buyer.Stcd
    suffix = ". " + TAX_SPLIT_LIMITATION
    warnings: list[ExtractionWarning] = []
    for i, item in enumerate(invoice.ItemList):
        if intra:
            half = item.AssAmt * item.GstRt / 2 / 100
            expectations = (
                ("CgstAmt", item.CgstAmt, "AssAmt x GstRt / 2 / 100", half),
                ("SgstAmt", item.SgstAmt, "AssAmt x GstRt / 2 / 100", half),
                ("IgstAmt", item.IgstAmt, "0 (intra-state supply)", 0.0),
            )
        else:
            expectations = (
                ("CgstAmt", item.CgstAmt, "0 (inter-state supply)", 0.0),
                ("SgstAmt", item.SgstAmt, "0 (inter-state supply)", 0.0),
                ("IgstAmt", item.IgstAmt, "AssAmt x GstRt / 100", item.AssAmt * item.GstRt / 100),
            )
        for name, actual, formula, expected in expectations:
            warning = _compare(
                f"ItemList[{i}].{name}", actual, formula, expected, tolerance, check, suffix,
            )
            if warning is not None:
                warnings.append(warning)
    return warnings


def validate_invoice(invoice: Invoice, tolerance: float = 0.05) -> list[ExtractionWarning]:
    """Run all four validators in order and concatenate their output."""
    return (
        validate_item_total(invoice, tolerance)
        + validate_invoice_total(invoice, tolerance)
        + validate_val_dtls_sums(invoice, tolerance)
        + validate_tax_split(invoice, tolerance)
    )
