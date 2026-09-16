"""End-to-end assembly: the one code path that produces a real INV-01 payload.

``ingest`` routes the pages, :mod:`gst_einvoice.ocr` supplies the text build 1
deliberately left empty, stage 1 confirms what is structurally certain, stage 2
structures the rest, and this module puts the two together, runs every build 1
validator and reports what it could not read.

The central guarantee lives here: **a mandatory INV-01 field that could not be
read is reported as missing, never fabricated.** No empty string, no zero, no
plausible placeholder is substituted to force ``Invoice`` to validate. When any
mandatory field is absent, ``extraction`` is ``None``, ``missing_fields`` names
the exact INV-01 paths, and the warnings say why each one is missing. A payload
that validates because a field was invented is precisely the "confidently wrong
extraction" this tool exists to prevent, so it is never produced.

Two things this module reads that build 1 could not. Build 1 runs its
export/SEZ, multi-currency and reverse-charge detectors on native pages only --
``ingest``'s own docstring says so and hands the OCR'd pages to build 2 ("build
2 re-runs the detectors on OCR output") -- so all three are re-run here on the
text OCR recovered. That is not an extra feature: the contract's licence to
hard-code ``SupTyp="B2B"`` rests on "build 1 refuses export/SEZ, so anything
that reaches here is domestic B2B", and on a scanned export invoice that
sentence is only true if the detectors get to see the scan. The same holds for
the other two: a foreign-currency invoice whose evidence is on a scan would
otherwise be parsed as rupees, and ``RegRev`` would say reverse charge does not
apply to a scanned invoice that declares on its face that it does. A
``RegRev="Y"`` reached this way does make ``validate_tax_split`` step aside, but
build 1's validator says so itself with an ``info`` warning rather than falling
silent.

One detector is narrowed on OCR'd text. Build 1's multi-currency detector fires
on a bare ``$``/``€``/``£``/``¥`` with nothing around it, and a single mis-read
character invents one: Tesseract returns ``"$55"`` for a printed ``"S55"`` and
``"£50.00"`` for a printed ``"850.00"``. On native text a glyph really is
printed, so build 1 is right to refuse; on OCR'd text the same hit refuses a
rupee invoice that was read perfectly and tells the accountant, as fact, that
the invoice states non-rupee amounts. So currency glyphs are blanked out of the
OCR'd text before the detector sees it, and a glyph is reported as a warning on
``ValDtls.TotInvValFc`` instead. A foreign currency named in words, an exchange
rate, or an ISO code beside an amount -- wording OCR noise cannot synthesise --
still refuses the document outright, on the scan as on the text layer. Whenever a
glyph is blanked the ``ValDtls.TotInvValFc`` warning is emitted, so no payload
can quote rupee figures after currency evidence was set aside without saying so.

One deviation from the contract's wording, in the direction the contract's own
reasoning points. The contract writes stage 1's provenance entries as
``{"source": "regex", ...}`` wholesale, but two of the six are not regex hits:
``SellerDtls.Stcd`` and ``BuyerDtls.Stcd`` are never matched on the page at all,
because stage 1 derives them from the GSTIN's first two characters through
``lookup_state_code`` and labels them ``method="derived"`` itself. They are
recorded here as ``"derived"`` to match, the same label ``IsServc`` already
carries. These are the two fields that decide whether tax splits CGST/SGST or
falls to IGST, so "how was this obtained" is exactly the question an accountant
asks of them when the derived code disagrees with the printed address. Their
``ocr_confidence`` is unchanged -- it is the confidence of those two characters
of the GSTIN, which genuinely were read off the page -- so a shaky read still
travels with the code taken from it.

One known imprecision, stated here because it is a property of the whole module
rather than of any one function. Stage 2 returns a value's text, not its offsets,
so ``_confidence_for_text`` locates a value by searching every OCR'd page for it
as a printed token, and takes the lowest confidence it finds. On a document that
mixes routes, a value that stage 2 actually read from a native text layer is
still given a scanned page's confidence when the same text happens to be printed
on the scan too -- a total repeated in a footer, a description repeated in a
summary line. The error only ever runs one way: the number recorded can be lower
than the read deserved, never higher, because a native reading contributes no
confidence of its own and only genuine printings of that same text are
considered. A shaky read therefore cannot be laundered into a clean-looking one,
which is the direction per-field confidence exists to guard.
"""

import os
import re
from dataclasses import dataclass
from typing import Any

from gst_einvoice.extract_llm import DEFAULT_MODEL, LlmExtraction, extract_with_llm
from gst_einvoice.extract_rules import RuleExtraction, RuleField, extract_rules, remaining_text
from gst_einvoice.ingest import (
    IngestResult,
    Refusal,
    detect_export_or_sez,
    detect_multi_currency,
    detect_reverse_charge,
    ingest,
)
from gst_einvoice.ocr import OcrPage, confidence_for_span, ocr_pdf_page
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
# ``_EPSILON`` is imported rather than restated so the reconciliation guarding the
# TotItemVal derivation below compares amounts on exactly the terms the validators
# do; a second copy of the constant here could drift from the one that is enforced.
from gst_einvoice.validators import _EPSILON, validate_invoice

CHECK = "pipeline"
#: Warnings carrying a build 1 refusal, or a detector re-run on OCR'd text.
INGEST_CHECK = "ingest"

#: INV-01 item fields that are mandatory and must come from stage 2.
#: ``IsServc`` is absent on purpose: the pipeline derives it, so it is never missing.
MANDATORY_ITEM_FIELDS: tuple[str, ...] = (
    "SlNo",
    "PrdDesc",
    "HsnCd",
    "Qty",
    "Unit",
    "UnitPrice",
    "TotAmt",
    "AssAmt",
    "GstRt",
    "CgstAmt",
    "SgstAmt",
    "IgstAmt",
    "TotItemVal",
)

#: INV-01 ``ValDtls`` totals that are mandatory.
MANDATORY_TOTALS: tuple[str, ...] = ("AssVal", "CgstVal", "SgstVal", "IgstVal", "TotInvVal")

#: The currency glyphs build 1's multi-currency detector fires on by themselves, and
#: the token they were read inside, for quoting back at the accountant.
CURRENCY_GLYPHS = "$€£¥"
_GLYPH_TOKEN = re.compile(rf"\S*[{re.escape(CURRENCY_GLYPHS)}]\S*")
_GLYPH_BLANKS = {ord(glyph): " " for glyph in CURRENCY_GLYPHS}

_ASSUMED = {"source": "assumed", "ocr_confidence": None}
#: A value no stage read off the page, filled from a rule that follows from values
#: that WERE read. Never a guess at what the document might have said.
_DERIVED = {"source": "derived", "ocr_confidence": None}


@dataclass(frozen=True)
class PipelineResult:
    """One document's end-to-end result.

    ``extraction`` is ``None`` whenever the document was refused or a mandatory
    INV-01 field could not be read; ``meta`` is populated either way, because the
    pages, the provenance and the warnings are the actionable output when there
    is no payload.
    """

    extraction: ExtractionResult | None
    missing_fields: tuple[str, ...]
    meta: ExtractionMeta
    refusals: tuple[Refusal, ...]


@dataclass(frozen=True)
class _Page:
    """One page's text and where it sits in the joined document text."""

    page: int
    method: str
    text: str
    start: int
    ocr: OcrPage | None


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _read_pages(path: str | os.PathLike[str], ingested: IngestResult) -> list[_Page]:
    """Native text as build 1 read it; ``ocr_required`` pages through Tesseract.

    ``start`` is the page's offset into ``"\\n".join(page texts)``, which is what
    lets a stage 1 span be traced back to the page it came from and to that
    page's OCR confidence.
    """
    pages: list[_Page] = []
    offset = 0
    for result in ingested.pages:
        if result.method == "native":
            text, ocr_page = result.text, None
        else:
            ocr_page = ocr_pdf_page(path, result.page)
            text = ocr_page.text
        pages.append(
            _Page(page=result.page, method=result.method, text=text, start=offset, ocr=ocr_page)
        )
        offset += len(text) + 1  # the "\n" the join inserts after this page
    return pages


def _confidence_for_offsets(pages: list[_Page], start: int, end: int) -> float | None:
    """OCR confidence for a span of the joined text, or None on a native page.

    The span is attributed to the page its first character falls on and clipped
    to that page, so a span that runs over a page break is answered by the page
    it started on rather than silently dropped.
    """
    for page in pages:
        page_end = page.start + len(page.text)
        if page.start <= start < page_end:
            if page.ocr is None:
                return None
            return confidence_for_span(page.ocr, start - page.start, min(end, page_end) - page.start)
    return None


def _word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _printed_at(text: str, value: str) -> list[int]:
    """Offsets where ``value`` is printed in ``text`` as a token, not inside a longer one.

    A bare ``find`` answers with any character sequence: a ``Qty`` of ``3``
    matches the ``3`` inside the invoice number ``VT/2026/0311``, and the
    confidence that comes back then belongs to a word the field was never read
    from. A hit is kept only when the characters on either side are not word
    characters -- the same boundary ``\\b`` applies, so a value whose own first or
    last character is punctuation (the printed credit ``(250.00)``) is not
    rejected for the side it cannot have a boundary on.
    """
    hits: list[int] = []
    index = text.find(value)
    while index != -1:
        end = index + len(value)
        before = text[index - 1] if index else ""
        after = text[end] if end < len(text) else ""
        starts_clean = not (_word_char(value[0]) and _word_char(before))
        ends_clean = not (_word_char(value[-1]) and _word_char(after))
        if starts_clean and ends_clean:
            hits.append(index)
        index = text.find(value, index + 1)
    return hits


def _confidence_for_text(pages: list[_Page], value: str) -> float | None:
    """Lowest OCR confidence among the places ``value`` is printed on an OCR'd page.

    Stage 2 hands back the source's own spelling of a grounded value, not an
    offset, so the page it came from has to be found by searching -- and the
    search has to be for the value as a printed token, because a substring of a
    longer word is not a place the value is printed. When the value really is
    printed more than once the lowest confidence wins: which occurrence the model
    read cannot be known, and reporting the best of them is exactly the
    laundering of a shaky read that per-field confidence exists to stop. That
    direction is the one that matters -- a genuine printing read badly can lower
    what is recorded, but no unrelated word can raise it. ``None`` means the text
    was found only in native page text, which has no OCR confidence, or was not
    found at all.
    """
    if not value:
        return None
    found: list[float] = []
    for page in pages:
        if page.ocr is None:
            continue
        for index in _printed_at(page.text, value):
            confidence = confidence_for_span(page.ocr, index, index + len(value))
            if confidence is not None:
                found.append(confidence)
    return min(found) if found else None


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #


def _note(field: str, message: str, severity: str = "warning") -> ExtractionWarning:
    return ExtractionWarning(field=field, message=message, severity=severity, check=CHECK)


def _ingest_note(field: str, message: str, severity: str = "warning") -> ExtractionWarning:
    """A note about what reading the pages produced, labelled like the refusals beside it.

    ``meta.warnings`` opens with the ingest-derived segment, and a consumer that
    groups by ``check`` can only see where that segment ends if every warning in
    it is labelled the same way. A note built with ``_note`` would read as a
    pipeline note sitting ahead of the stage warnings, which is exactly the
    ordering the contract says cannot happen.
    """
    return ExtractionWarning(field=field, message=message, severity=severity, check=INGEST_CHECK)


def _currency_glyph_note(page: int, evidence: str, *, refused: bool) -> ExtractionWarning:
    """A currency glyph OCR read, reported rather than treated as a refusal.

    Emitted on every page a glyph was blanked out of the detector's input, refused
    document or not, so evidence that was set aside is never set aside silently.
    What the setting-aside cost differs between the two cases, though, and the note
    says which one it is: on a document refused for some other reason there are no
    parsed figures to caveat, and claiming there were would be its own small lie.
    """
    outcome = (
        "This document was refused for another reason, so nothing on it was parsed into "
        "amounts at all and there are no figures below for this to cast doubt on."
        if refused
        else "So every amount below was parsed as rupees. If this invoice really does quote a "
        "foreign currency, those figures are wrong -- check the quoted text on the page."
    )
    return _ingest_note(
        "ValDtls.TotInvValFc",
        f'OCR read a currency symbol on page {page}: "{evidence}". On its own that is not '
        f"enough to refuse this document as a foreign-currency invoice, because a single "
        f'mis-read character invents one: Tesseract returns "$55" for a printed "S55" and '
        f'"£50.00" for a printed "850.00". Refusing on that evidence would discard an '
        f"invoice the tool read correctly and tell you, as fact, that it states non-rupee "
        f"amounts. {outcome} A currency named in words, an exchange rate, or an ISO code "
        f"beside an amount still refuses the document outright, on a scan as on a text layer.",
    )


def _refusal_warning(refusal: Refusal, *, from_ocr: bool) -> ExtractionWarning:
    """A build 1 refusal restated as a warning, so it reaches ``meta.warnings``."""
    prefix = (
        "Found in text recovered by OCR from a page whose text layer build 1 could not read. "
        if from_ocr
        else ""
    )
    return ExtractionWarning(
        field=refusal.field,
        message=f"{prefix}{refusal.message} Evidence: {refusal.evidence}.",
        severity="warning",
        check=INGEST_CHECK,
    )


def _missing_note(path: str) -> ExtractionWarning:
    return _note(
        path,
        f"{path} is mandatory in INV-01 but could not be read from this document, so no "
        f"invoice payload was produced. It was left missing rather than filled with an "
        f"empty string, a zero or a plausible-looking placeholder: a fabricated mandatory "
        f"field is worse than an absent one. The warnings from the extraction stages above "
        f"say what was seen at this field's position.",
    )


def _unprinted_tax_note(path: str, intra: bool, seller: str, buyer: str) -> ExtractionWarning:
    """Report a tax head filled with zero because the supply cannot attract it.

    INV-01 makes all three heads mandatory on every line, but an invoice template
    prints only the heads that apply: an intra-state invoice has no IGST column and
    an inter-state one has no CGST/SGST column. Stage 2 answers null for a column
    that is not on the page, which is correct, and without this rule the ordinary
    case would produce no payload at all.
    """
    kind = "intra-state" if intra else "inter-state"
    other = "CGST and SGST" if intra else "IGST"
    return _note(
        path,
        f"{path} is mandatory in INV-01 but is not printed on this document. It was set "
        f"to zero because the supply is {kind}: the seller's state code ({seller}) and the "
        f"buyer's ({buyer}) say the tax falls to {other}, so this head cannot apply and no "
        f"invoice prints a column for it. The zero was derived from those two state codes, "
        f"not read from the page. Note that the state codes are the registered states, not "
        f"the true place of supply, so confirm the split if this is an SEZ supply or a "
        f"bill-to/ship-to case.",
        severity="info",
    )


def _gross_from_taxable_note(path: str, assamt: float, discount: float) -> ExtractionWarning:
    """Report a gross amount filled from the taxable value and the discount."""
    return _note(
        path,
        f"{path} (gross amount before discount) is mandatory in INV-01 but is not printed "
        f"as a column of its own on this document, which prints only the taxable value. It "
        f"was set to {assamt + discount:.2f} from the INV-01 identity TotAmt = AssAmt + "
        f"Discount, using the taxable value {assamt:.2f} and the discount {discount:.2f} "
        f"that were read. It was derived, not read from the page: if this invoice does "
        f"carry a gross figure that differs, the printed one is the correct value.",
        severity="info",
    )


def _single_row_total_note(
    path: str, value: float, printed_total: float, tolerance: float
) -> ExtractionWarning:
    """Report a per-line total filled from the INV-01 item identity.

    INV-01 makes ``TotItemVal`` mandatory on every line, but on an invoice with a
    single line item the per-row total and the document total are the same number,
    so the template routinely prints it once at the foot rather than twice. The
    rule is confined to that case, and to a derived value that reconciles with the
    printed document total: on a multi-row invoice a per-row total the document
    never states would assert a split across rows it never states either.
    """
    return _note(
        path,
        f"{path} (per-line total) is mandatory in INV-01 but is not printed as a column "
        f"of its own on this document. It was set to {value:.2f} from the INV-01 identity "
        f"TotItemVal = AssAmt + CgstAmt + SgstAmt + IgstAmt + CesAmt + StateCesAmt + "
        f"OthChrg -- the identity validate_item_total enforces -- using the amounts that "
        f"were read. It was derived, not read from the page. The derivation was allowed "
        f"only because this invoice carries exactly one line item and the derived value "
        f"reconciles with the printed document total ValDtls.TotInvVal "
        f"({printed_total:.2f}) within the {tolerance:.10g} tolerance. On a multi-line "
        f"invoice, or where that reconciliation fails, this field is reported missing "
        f"instead and no payload is produced.",
        severity="info",
    )


# --------------------------------------------------------------------------- #
# Assembly helpers
# --------------------------------------------------------------------------- #


def _rule_value(field: RuleField | None) -> str | None:
    return None if field is None else field.value


def _is_service_hsn(hsn: str | None) -> str:
    """``"Y"`` only for a 6-digit SAC beginning 99; everything else is goods."""
    if hsn is None:
        return "N"
    code = hsn.strip()
    return "Y" if len(code) == 6 and code.isdigit() and code.startswith("99") else "N"


def _leaf_paths(value: Any, prefix: str = "") -> list[str]:
    """Every leaf path of a dumped payload, as ``ItemList[0].SlNo`` / ``ValDtls.AssVal``."""
    if isinstance(value, dict):
        paths: list[str] = []
        for key, sub in value.items():
            paths.extend(_leaf_paths(sub, f"{prefix}.{key}" if prefix else key))
        return paths
    if isinstance(value, list):
        paths = []
        for index, sub in enumerate(value):
            paths.extend(_leaf_paths(sub, f"{prefix}[{index}]"))
        return paths
    return [prefix]


def _record_structural_provenance(invoice: Invoice, provenance: dict[str, Any]) -> None:
    """Mark every emitted field the pipeline supplied rather than read as ``assumed``.

    Stage 1, stage 2 and the ``IsServc`` derivation each record their own fields
    as they produce them, and a mandatory field that was not produced never
    reaches an ``Invoice`` at all. So whatever is left in the payload is
    structural -- ``Version``, ``TaxSch``, ``SupTyp``, ``Typ``, ``Pos`` -- or a
    schema default the document did not print, and ``assumed`` is the honest
    label for all of it. The invariant this maintains is that no field is ever
    emitted without a provenance entry saying where it came from.
    """
    for path in _leaf_paths(invoice.model_dump(exclude_none=True)):
        provenance.setdefault(path, dict(_ASSUMED))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def extract_invoice(
    path: str | os.PathLike[str],
    *,
    client: Any,
    model: str = DEFAULT_MODEL,
    tolerance: float = 0.05,
) -> PipelineResult:
    """Read one invoice end to end and return its INV-01 payload, or what is missing.

    ``client`` has no default: stage 2 never builds one, so this path cannot
    reach the network by accident.
    """
    ingested = ingest(path)
    page_metas = [PageMeta(page=p.page, method=p.method) for p in ingested.pages]
    pages = _read_pages(path, ingested)

    ingest_notes: list[ExtractionWarning] = []
    refusals = list(ingested.refusals)
    refusal_warnings = [_refusal_warning(r, from_ocr=False) for r in refusals]
    reverse_charge = ingested.reverse_charge

    glyph_hits: list[tuple[int, str]] = []
    for page in pages:
        if page.ocr is None:
            continue
        if not page.text.strip():
            ingest_notes.append(
                _ingest_note(
                    f"page[{page.page}]",
                    f"Page {page.page} had no usable text layer and OCR recovered no text from "
                    f"it either, so nothing on that page could contribute to the invoice. Any "
                    f"field reported missing below may simply be printed on this page.",
                )
            )
            continue
        # The currency detector sees the page with its currency glyphs blanked out
        # (blanked, not deleted, so every other pattern still matches at the same
        # offsets): a lone glyph is one OCR character error away from any capital S
        # or 8, and a refusal is the end of the document. What is left of the
        # detector is the wording noise cannot synthesise.
        for detector, subject in (
            (detect_export_or_sez, page.text),
            (detect_multi_currency, page.text.translate(_GLYPH_BLANKS)),
        ):
            found = detector(subject, page=page.page)
            if found is not None and not any(r.kind == found.kind for r in refusals):
                refusals.append(found)
                refusal_warnings.append(_refusal_warning(found, from_ocr=True))
        glyph = _GLYPH_TOKEN.search(page.text)
        if glyph is not None:
            glyph_hits.append((page.page, " ".join(glyph.group(0).split())))
        reverse_charge = reverse_charge or detect_reverse_charge(page.text)

    # A glyph the detector was not allowed to see is reported on every page it was
    # blanked from, with no exception for a document some other evidence already
    # refused. The exemption is narrow by design -- it exists so a mis-read character
    # cannot discard a rupee invoice -- and the only thing that keeps it honest is
    # that every use of it is stated. Suppressing the note wherever a refusal happens
    # to say something similar would make the exemption's footprint unrecoverable
    # from the output, which is the one thing it must never be.
    ingest_notes.extend(
        _currency_glyph_note(page_no, ev, refused=bool(refusals)) for page_no, ev in glyph_hits
    )

    if refusals:
        # Build 1 already decided these documents are out of scope, and its
        # refusal messages are the actionable output. Extracting anyway would
        # produce a payload built on assumptions the document contradicts.
        meta = ExtractionMeta(
            pages=page_metas, field_provenance={}, warnings=refusal_warnings + ingest_notes
        )
        return PipelineResult(
            extraction=None, missing_fields=(), meta=meta, refusals=tuple(refusals)
        )

    text = "\n".join(page.text for page in pages)
    rules: RuleExtraction = extract_rules(text)
    stripped = remaining_text(text, rules.consumed)

    provenance: dict[str, Any] = {}
    # The two state codes are not regex hits. Nothing on the page was matched to
    # produce them: stage 1 takes the GSTIN's first two characters through
    # ``lookup_state_code`` and records the result as ``method="derived"``, the same
    # label this module already gives ``IsServc``. Calling them ``"regex"`` would
    # tell an accountant the state was read off the document, on exactly the two
    # fields that decide whether the tax splits CGST/SGST or falls to IGST -- so a
    # state code that contradicts the printed address would look like evidence of
    # what the page says rather than a restatement of the GSTIN. Their
    # ``ocr_confidence`` stays as it is: the span is those two characters of the
    # GSTIN, and the GSTIN genuinely was read off the page, so a shaky read of it
    # still travels with the code derived from it.
    for path, field, source in (
        ("SellerDtls.Gstin", rules.seller_gstin, "regex"),
        ("BuyerDtls.Gstin", rules.buyer_gstin, "regex"),
        ("SellerDtls.Stcd", rules.seller_stcd, "derived"),
        ("BuyerDtls.Stcd", rules.buyer_stcd, "derived"),
        ("DocDtls.No", rules.doc_no, "regex"),
        ("DocDtls.Dt", rules.doc_date, "regex"),
    ):
        if field is not None:
            provenance[path] = {
                "source": source,
                "ocr_confidence": _confidence_for_offsets(pages, field.span.start, field.span.end),
            }

    confirmed = {
        "SellerDtls.Gstin": _rule_value(rules.seller_gstin),
        "BuyerDtls.Gstin": _rule_value(rules.buyer_gstin),
        "SellerDtls.Stcd": _rule_value(rules.seller_stcd),
        "BuyerDtls.Stcd": _rule_value(rules.buyer_stcd),
        "DocDtls.No": _rule_value(rules.doc_no),
        "DocDtls.Dt": _rule_value(rules.doc_date),
        "HSN/SAC codes found on the document": [f.value for f in rules.hsn_codes] or None,
    }
    llm: LlmExtraction = extract_with_llm(
        stripped,
        confirmed,
        client=client,
        model=model,
        confidence_lookup=lambda value: _confidence_for_text(pages, value),
    )
    provenance.update(llm.provenance)

    missing: list[str] = []
    pipeline_notes: list[ExtractionWarning] = []
    #: Paths this module filled from a rule after stage 2 reported them absent. Stage 2's
    #: own note on such a path says the field was "left empty rather than filled with a
    #: guess", which stops being true the moment the value is derived, so that note is
    #: superseded by the derivation note rather than printed beside it.
    derived_paths: set[str] = set()

    def record_missing(path: str, note: ExtractionWarning) -> None:
        """Report a mandatory field as missing, and take back any provenance for it.

        A path in ``missing_fields`` must not also be in ``field_provenance``:
        the entry would tell a consumer the field was read from a stage while the
        same result says it could not be read at all, and the metadata is the
        only output there is when no payload is produced.
        """
        missing.append(path)
        pipeline_notes.append(note)
        provenance.pop(path, None)

    def require(path: str, value: Any) -> Any:
        """Keep a mandatory value, or record it as missing. Never substitutes one."""
        if value is None or (isinstance(value, str) and not value.strip()):
            record_missing(path, _missing_note(path))
            return None
        return value

    def require_pin(path: str, raw: str | None) -> int | None:
        if raw is None:
            return require(path, None)
        digits = raw.strip()
        # ``str.isdigit`` and ``int`` do not agree: ``isdigit`` is true for "²³"
        # (superscripts), which ``int`` refuses, and for Arabic-Indic digits, which
        # ``int`` silently converts into a PIN nobody printed. Either way the text at
        # that field was not an INV-01 PIN code, so it is reported missing.
        if not (digits.isascii() and digits.isdigit()):
            record_missing(
                path,
                _note(
                    path,
                    f"{path} is mandatory in INV-01 and must be a number, but the text read at "
                    f"that field was {raw!r}. It was left missing rather than coerced into a "
                    f"number, because a PIN code invented from non-numeric text would look "
                    f"exactly like one that was read.",
                )
            )
            return None
        return int(digits)

    seller = {
        "Gstin": require("SellerDtls.Gstin", _rule_value(rules.seller_gstin)),
        "LglNm": require("SellerDtls.LglNm", llm.seller_name),
        "Addr1": require("SellerDtls.Addr1", llm.seller_addr1),
        "Loc": require("SellerDtls.Loc", llm.seller_loc),
        "Pin": require_pin("SellerDtls.Pin", llm.seller_pin),
        "Stcd": require("SellerDtls.Stcd", _rule_value(rules.seller_stcd)),
    }
    buyer = {
        "Gstin": require("BuyerDtls.Gstin", _rule_value(rules.buyer_gstin)),
        "LglNm": require("BuyerDtls.LglNm", llm.buyer_name),
        "Addr1": require("BuyerDtls.Addr1", llm.buyer_addr1),
        "Loc": require("BuyerDtls.Loc", llm.buyer_loc),
        "Pin": require_pin("BuyerDtls.Pin", llm.buyer_pin),
        "Stcd": require("BuyerDtls.Stcd", _rule_value(rules.buyer_stcd)),
    }
    doc_no = require("DocDtls.No", _rule_value(rules.doc_no))
    doc_dt = require("DocDtls.Dt", _rule_value(rules.doc_date))

    item_values: list[dict[str, Any]] = []
    if not llm.items:
        record_missing(
            "ItemList",
            _note(
                "ItemList",
                "ItemList is mandatory in INV-01 and needs at least one line item, but stage 2 "
                "returned no usable rows. No row was invented to satisfy the schema.",
            ),
        )
    # Which tax heads this supply can attract, from the two state codes stage 1 derived
    # from validated GSTINs. An invoice prints only the heads that apply, so the other
    # one comes back null from stage 2 and has to be derived or nothing is produced.
    seller_stcd_raw = _rule_value(rules.seller_stcd)
    buyer_stcd_raw = _rule_value(rules.buyer_stcd)
    split_known = seller_stcd_raw is not None and buyer_stcd_raw is not None
    intra = split_known and seller_stcd_raw == buyer_stcd_raw
    unprinted_heads = ("IgstAmt",) if intra else ("CgstAmt", "SgstAmt")
    unprinted_totals = ("IgstVal",) if intra else ("CgstVal", "SgstVal")

    for index, row in enumerate(llm.items):
        raw_values = {name: getattr(row, name) for name in MANDATORY_ITEM_FIELDS}
        discount = row.Discount or 0.0
        if raw_values["TotAmt"] is None and raw_values["AssAmt"] is not None:
            raw_values["TotAmt"] = raw_values["AssAmt"] + discount
            derived_paths.add(f"ItemList[{index}].TotAmt")
            provenance[f"ItemList[{index}].TotAmt"] = dict(_DERIVED)
            pipeline_notes.append(_gross_from_taxable_note(
                f"ItemList[{index}].TotAmt", raw_values["AssAmt"], discount
            ))
        # Only a row that carries real content gets a derived zero. A row that came back
        # wholly empty has nothing to derive from, and reporting every field missing is
        # the honest output there.
        row_has_content = any(
            raw_values[name] is not None
            for name in MANDATORY_ITEM_FIELDS
            if name not in unprinted_heads
        )
        if split_known and row_has_content:
            for head in unprinted_heads:
                if raw_values[head] is None:
                    raw_values[head] = 0.0
                    derived_paths.add(f"ItemList[{index}].{head}")
                    provenance[f"ItemList[{index}].{head}"] = dict(_DERIVED)
                    pipeline_notes.append(_unprinted_tax_note(
                        f"ItemList[{index}].{head}", intra, seller_stcd_raw, buyer_stcd_raw
                    ))
        # A single-row invoice routinely prints its per-line total once, at the foot, as
        # the document total. Derive it back from the INV-01 item identity, but only when
        # the document itself corroborates the result: exactly one line item, and a
        # derived value that reconciles with the printed ValDtls.TotInvVal on the same
        # terms the validators compare on. Both conditions are checkable from amounts
        # already read; neither involves guessing. On a multi-line invoice a per-row total
        # the document never states would assert a split across rows it never states
        # either, so the refusal stands there unchanged, however obvious the arithmetic
        # looks. A reconciliation that fails is evidence the reading is wrong, not licence
        # to paper over it.
        if raw_values["TotItemVal"] is None and len(llm.items) == 1:
            printed_total = llm.totals.get("TotInvVal")
            identity_heads = ("AssAmt", "CgstAmt", "SgstAmt", "IgstAmt")
            if printed_total is not None and all(
                raw_values[name] is not None for name in identity_heads
            ):
                # StateCesAmt and OthChrg are not extracted in build 2 and default to zero
                # in the schema, so they contribute nothing to the identity here.
                candidate = sum(raw_values[name] for name in identity_heads) + (row.CesAmt or 0.0)
                if abs(candidate - printed_total) <= tolerance + _EPSILON:
                    raw_values["TotItemVal"] = candidate
                    derived_paths.add(f"ItemList[{index}].TotItemVal")
                    provenance[f"ItemList[{index}].TotItemVal"] = dict(_DERIVED)
                    pipeline_notes.append(_single_row_total_note(
                        f"ItemList[{index}].TotItemVal", candidate, printed_total, tolerance
                    ))
        values = {
            name: require(f"ItemList[{index}].{name}", raw_values[name])
            for name in MANDATORY_ITEM_FIELDS
        }
        values["IsServc"] = _is_service_hsn(row.HsnCd)
        # Only a row that keeps every mandatory field becomes an ``Item``, so only such
        # a row's ``IsServc`` is ever emitted. Recording the derivation for a row that
        # was dropped would leave ``field_provenance`` describing a field no payload
        # contains -- the same mismatch ``record_missing`` takes back for the paths it
        # reports, and the metadata is the only output there is when no payload is
        # produced.
        if all(values[name] is not None for name in MANDATORY_ITEM_FIELDS):
            provenance[f"ItemList[{index}].IsServc"] = {"source": "derived", "ocr_confidence": None}
        for optional in ("Discount", "CesAmt"):
            if getattr(row, optional) is not None:
                values[optional] = getattr(row, optional)
        item_values.append(values)

    raw_totals = {name: llm.totals.get(name) for name in MANDATORY_TOTALS}
    # As with an item row: derive only when the totals block carries real content.
    totals_have_content = any(
        raw_totals[name] is not None
        for name in MANDATORY_TOTALS
        if name not in unprinted_totals
    )
    if split_known and totals_have_content:
        for total in unprinted_totals:
            if raw_totals[total] is None:
                raw_totals[total] = 0.0
                derived_paths.add(f"ValDtls.{total}")
                provenance[f"ValDtls.{total}"] = dict(_DERIVED)
                pipeline_notes.append(_unprinted_tax_note(
                    f"ValDtls.{total}", intra, seller_stcd_raw, buyer_stcd_raw
                ))
    totals = {name: require(f"ValDtls.{name}", raw_totals[name]) for name in MANDATORY_TOTALS}
    for optional in ("CesVal", "RndOffAmt"):
        if llm.totals.get(optional) is not None:
            totals[optional] = llm.totals[optional]

    if not missing:
        # Place of supply is not extracted in build 2. Assuming it equals the buyer's
        # registered state is right for an ordinary domestic supply and wrong for a
        # bill-to/ship-to one, so it is recorded as an assumption rather than a reading.
        buyer["Pos"] = buyer["Stcd"]
        pipeline_notes.append(
            _note(
                "BuyerDtls.Pos",
                f"BuyerDtls.Pos (place of supply) was not read from the document -- build 2 "
                f"does not extract it -- so it was assumed equal to the buyer's registered "
                f"state code ({buyer['Stcd']}). A genuine bill-to/ship-to supply, where the "
                f"goods go to a different state from the one the buyer is registered in, has "
                f"a different place of supply, and the CGST/SGST-versus-IGST split follows "
                f"the place of supply. Confirm it against the document before filing.",
                severity="info",
            )
        )

    # A stage 2 note reporting a field absent is superseded when this module then
    # derived that field: the note's own wording promises the value was left empty.
    stage_two = [
        w for w in llm.warnings
        if not (w.field in derived_paths and "was not found in the document" in w.message)
    ]
    warnings = (
        refusal_warnings
        + ingest_notes
        + list(rules.warnings)
        + stage_two
        + list(pipeline_notes)
    )

    if missing:
        meta = ExtractionMeta(pages=page_metas, field_provenance=provenance, warnings=warnings)
        return PipelineResult(
            extraction=None, missing_fields=tuple(missing), meta=meta, refusals=()
        )

    invoice = Invoice(
        TranDtls=TranDtls(
            TaxSch="GST", SupTyp="B2B", RegRev="Y" if reverse_charge else None
        ),
        DocDtls=DocDtls(Typ="INV", No=doc_no, Dt=doc_dt),
        SellerDtls=SellerDtls(**seller),
        BuyerDtls=BuyerDtls(**buyer),
        ItemList=[Item(**values) for values in item_values],
        ValDtls=ValDtls(**totals),
    )
    _record_structural_provenance(invoice, provenance)

    warnings.extend(validate_invoice(invoice, tolerance))
    meta = ExtractionMeta(pages=page_metas, field_provenance=provenance, warnings=warnings)
    return PipelineResult(
        extraction=ExtractionResult(invoice=invoice, extraction_meta=meta),
        missing_fields=(),
        meta=meta,
        refusals=(),
    )
