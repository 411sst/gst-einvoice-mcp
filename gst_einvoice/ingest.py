"""Document ingestion: per-page native-text routing plus detect-and-refuse checks.

Build 1 performs no OCR. Pages whose text layer fails a gate are marked
``ocr_required`` and their text is left empty for build 2. The routing
function and the three detectors are pure functions on text so build 2 can
re-run them on OCR output.
"""

import os
import re
from dataclasses import dataclass
from typing import Literal

import pymupdf

Method = Literal["native", "ocr_required"]

ACCEPTED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Gate 1: fewer stripped characters than this means there is no usable text layer.
MIN_TEXT_CHARS = 100
# Gate 2: at least one of these must appear as a whole word (case-insensitive).
KEYWORD_RE = re.compile(r"\b(?:GSTIN|HSN|SGST|CGST|IGST|TAX\s+INVOICE)\b", re.IGNORECASE)

# Export/SEZ phrases, each tagged with the SupTyp family ("EXP"/"SEZ") and payment
# variant ("WP"/"WOP") it implies, when the phrase implies one. Bare "export"/"SEZ",
# "deemed export" (SupTyp DEXP, a domestic buyer) and "Special Economic Zone" in an
# address line deliberately do not match: precision beats recall here.
_EXPORT_SEZ_PATTERNS: tuple[tuple[re.Pattern[str], str | None, str | None], ...] = (
    (re.compile(r"\bsupply\s+meant\s+for\s+export\b", re.I), "EXP", None),
    (re.compile(r"\bexport\s+invoice\b", re.I), "EXP", None),
    (re.compile(r"\bbill\s+of\s+export\b", re.I), "EXP", None),
    # Same line only, and not the "Shipping Bill To:" / "Mode of Shipping\nBill To:" labels.
    (re.compile(r"\bshipping[ \t]+bill\b(?![ \t]+to\b)", re.I), "EXP", None),
    (re.compile(r"\b(?:with|on)\s+payment\s+of\s+(?:IGST|integrated\s+tax)\b", re.I), None, "WP"),
    (re.compile(r"\bwithout\s+payment\s+of\s+(?:IGST|integrated\s+tax)\b", re.I), None, "WOP"),
    (re.compile(r"\bunder\s+LUT\b", re.I), None, "WOP"),
    (re.compile(r"\bletter\s+of\s+undertaking\b", re.I), None, "WOP"),
    (re.compile(r"\bunder\s+bond\b", re.I), None, "WOP"),
    (re.compile(r"\bSEZ\s+unit\b", re.I), "SEZ", None),
    (re.compile(r"\bSEZ\s+developer\b", re.I), "SEZ", None),
    (re.compile(r"\bsupp(?:ly|lies)\s+to\s+SEZ\b", re.I), "SEZ", None),
    (re.compile(r"\bsupply\s+meant\s+for\s+SEZ\b", re.I), "SEZ", None),
)
# Why build 1 refuses, per SupTyp family: a foreign buyer has no GSTIN; an SEZ buyer has
# one, but the supply is zero-rated and inter-state whatever the buyer's state says.
_EXPORT_SEZ_WHY = {
    "EXP": "for an export the buyer has no GSTIN by design",
    "SEZ": (
        "for an SEZ supply the SEZ unit or developer holds a GSTIN but the supply is "
        "zero-rated and treated as inter-state regardless of the buyer's state"
    ),
}

# Multi-currency indicators. ISO codes are upper-case only and count only when
# adjacent to a number on the same line or following the word "currency", so a
# line item like "2 CAD drawings" or "SAR filing" does not trigger.
_ISO_CODES = (
    r"(?:USD|EUR|GBP|AED|SGD|JPY|AUD|CAD|CHF|CNY|HKD|SAR|QAR|KWD|BHD|OMR|NZD"
    r"|SEK|NOK|DKK|ZAR|MYR|THB|LKR|BDT|NPR)"
)
_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_CURRENCY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bTotInvValFc\b", re.I),
    re.compile(r"\bforeign\s+currency\b|\bexchange\s+rate\b|\bconversion\s+rate\b", re.I),
    re.compile(r"[$€£¥](?:[ \t]*" + _NUMBER + r")?"),
    re.compile(r"\b" + _ISO_CODES + r"[ \t]*" + _NUMBER + r"(?![A-Za-z\d])"),  # USD 1,200.00 / USD1200
    # 1,200.00 USD / 1,200.00 USD Only; a following word ("2 CAD drawings") disqualifies.
    re.compile(_NUMBER + r"[ \t]*" + _ISO_CODES + r"\b(?![ \t]+(?!(?i:only)\b)[A-Za-z])"),
    # Currency: USD / Currency of invoice: USD / Currency (USD)
    re.compile(r"(?i:\bcurrency\b(?:\s+of\s+(?:the\s+)?invoice)?)\s*[:\-(]?\s*" + _ISO_CODES + r"\b\)?"),
)

# Reverse charge: every mention is examined together with the rest of its line (and
# the start of the next non-blank line, for "Reverse Charge:\nYes" layouts). The tail
# is a lookahead so the scan resumes right after the phrase and a second mention
# inside the tail is still examined. Longer, assertive phrases are listed before the
# bare label so the alternation prefers them.
_RC_MENTION = re.compile(
    r"(?P<phrase>"
    r"\bsubject\s+to\s+reverse\s+charge\b"
    r"|\bsupply\s+attracts\s+reverse\s+charge\b"
    r"|\btax\s+(?:is\s+)?payable\s+(?:under|on)\s+reverse\s+charge\b"
    r"|\breverse\s+charge\s+(?:is\s+)?applicable\b"
    r"|\breverse\s+charge\b"
    r"|\bRCM\b"
    r")(?=(?P<tail>[^\n]{0,40}(?:\s*\n[^\n]{0,12})?))",
    re.I,
)
# Label words, "(RCM)" and "(Yes/No)" templates that may sit between the mention and its answer.
_RC_ANSWER_PREFIX = (
    r"^\W*(?:(?:basis|mechanism|applicability|applicable|RCM"
    r"|\(?\s*(?:yes|y)\s*/\s*(?:no|n)\s*\)?)\W*)*"
)
_RC_NO = re.compile(_RC_ANSWER_PREFIX + r"(?:no|not|non|nil|n/a|na|n|false)\b(?!\s*/)", re.I)
_RC_YES = re.compile(_RC_ANSWER_PREFIX + r"(?:yes|y|true|applicable)\b(?!\s*/)", re.I)
# Phrases that assert reverse charge by themselves, needing no explicit "Yes" ...
_RC_ASSERTIVE = re.compile(r"subject|attracts|payable|applicable", re.I)
# ... unless negated or asked as a question earlier on the same line ("not subject to",
# "No tax payable under", "Whether tax is payable on"); "No." / "No: 42" is a number label.
_RC_NEGATION = re.compile(r"\b(?:whether|not|never|no(?![.:]|\s*\d))\b", re.I)
# ... or used as a form label: followed by a separator, a column gap or a Yes/No template
# ("Reverse Charge Applicable: -", "Reverse Charge Applicable    E-Way Bill No."), by a
# value-like token even on the next line ("Amount of Tax subject to Reverse Charge\n0.00"),
# or by nothing at all (a bare grid header whose answer is out of reach).
_RC_LABEL_TAIL = re.compile(
    r"[ \t]*[:?|(\[]|[ \t]{2,}\S|\t|[ \t]*(?:yes|y)\s*/"
    r"|\s*(?:[-\u2013]|nil\b|n/?a\b|rs\.?|\u20b9|inr\b|\d|$)",
    re.I,
)


@dataclass(frozen=True)
class PageResult:
    """Routing decision for one page. ``text`` is "" when ``ocr_required``: it failed a gate."""

    page: int
    method: Method
    text: str


@dataclass(frozen=True)
class Refusal:
    """A detect-and-refuse finding. ``field`` is the INV-01 field the finding maps to."""

    kind: Literal["export_sez", "multi_currency"]
    field: str
    evidence: str
    message: str


@dataclass(frozen=True)
class IngestResult:
    """Per-page routing, refusals (both kinds may apply) and the reverse-charge flag."""

    pages: list[PageResult]
    refusals: list[Refusal]
    reverse_charge: bool


def route_page(text: str) -> Method:
    """Apply the two gates: length (< MIN_TEXT_CHARS) first, then a GST keyword."""
    if len(text.strip()) < MIN_TEXT_CHARS:
        return "ocr_required"
    if KEYWORD_RE.search(text) is None:
        return "ocr_required"
    return "native"


def _evidence(match: re.Match[str], page: int | None) -> tuple[str, str, str]:
    """Return (normalised phrase, evidence string, 'on page N' / 'in the text')."""
    phrase = " ".join(match.group(0).split())
    if page is None:
        return phrase, f'"{phrase}"', "in the document text"
    return phrase, f'"{phrase}" (page {page})', f"on page {page}"


def detect_export_or_sez(text: str, *, page: int | None = None) -> Refusal | None:
    """Refuse export/SEZ invoices (SupTyp EXPWP/EXPWOP/SEZWP/SEZWOP) on specific phrases."""
    hits = [(p.search(text), family, payment) for p, family, payment in _EXPORT_SEZ_PATTERNS]
    hits = [h for h in hits if h[0] is not None]
    if not hits:
        return None
    families = sorted({family for _, family, _ in hits if family} or {"EXP", "SEZ"})
    payments = sorted({payment for _, _, payment in hits if payment} or {"WP", "WOP"}, reverse=True)
    values = " or ".join(f + w for f in families for w in payments)
    first = min((m for m, _, _ in hits), key=lambda m: m.start())
    phrase, evidence, where = _evidence(first, page)
    why = " and ".join(_EXPORT_SEZ_WHY[f] for f in families)
    message = (
        f'Detected export/SEZ wording: "{phrase}" {where}. '
        f"This maps to TranDtls.SupTyp, most likely {values}. "
        f"Build 1 cannot process export or SEZ invoices: {why}, "
        "while BuyerDtls.Gstin, BuyerDtls.Stcd, BuyerDtls.Pos and the tax-split rule all "
        "assume a domestic B2B buyer, so the output would be a broken payload. "
        "What you can do: prepare the INV-01 through the e-invoice portal's export/SEZ flow "
        f"or the offline utility with SupTyp set to {values}. If this document is actually a "
        "domestic supply, the quoted phrase is what triggered the refusal, so check whether it "
        "appears only in printed boilerplate rather than describing this supply."
    )
    return Refusal(kind="export_sez", field="TranDtls.SupTyp", evidence=evidence, message=message)


def detect_multi_currency(text: str, *, page: int | None = None) -> Refusal | None:
    """Refuse invoices carrying non-rupee currency indicators. INR/Rs./rupee never trigger."""
    matches = [m for p in _CURRENCY_PATTERNS if (m := p.search(text)) is not None]
    if not matches:
        return None
    phrase, evidence, where = _evidence(min(matches, key=lambda m: m.start()), page)
    message = (
        f'Detected a non-rupee currency indicator: "{phrase}" {where}. '
        "This corresponds to ValDtls.TotInvValFc (invoice value in foreign currency), meaning "
        "the invoice states non-rupee amounts. Build 1 cannot process it because the extractor "
        "assumes every amount is in rupees and would silently misparse foreign-currency figures "
        "as INR. What you can do: if the invoice also shows INR amounts, use those, or convert "
        "the figures at the exchange rate printed on the invoice, then re-run. If this is an "
        "export invoice, follow the export/SEZ refusal's advice instead."
    )
    return Refusal(kind="multi_currency", field="ValDtls.TotInvValFc", evidence=evidence, message=message)


def _is_negated_or_question(text: str, start: int) -> bool:
    """True when the mention at ``start`` is preceded by a negation or "whether" on its line."""
    line_start = text.rfind("\n", 0, start) + 1
    return _RC_NEGATION.search(text[line_start:start]) is not None


def detect_reverse_charge(text: str) -> bool:
    """True only for an affirmative reverse-charge declaration.

    An explicit answer next to a mention always wins ("...: Yes" / "...: No").
    Without one, an assertive phrase counts as an affirmation only when it is
    embedded in a sentence on its line ("This supply attracts reverse charge"),
    is not negated or asked as a question earlier on that line, and is not
    followed by a form value.

    A phrase that occupies its whole line is a label or a grid header, not a
    declaration, and is unresolved without an explicit Yes or No. Row-major
    layouts, which many ERP generators emit, print the label row first and the
    answer row after it::

        Reverse Charge Applicable
        Place of Supply
        No
        Maharashtra (27)

    Reading that as an affirmation sets ``RegRev`` and stands build 1's
    ``validate_tax_split`` down for the whole invoice, silently, on a document
    that is not under reverse charge at all. The opposite error is visible and
    recoverable: a genuine reverse-charge invoice whose declaration is a bare
    label runs the tax-split check and raises a warning the accountant can
    dismiss. That is the trade this rule deliberately takes.
    """
    for m in _RC_MENTION.finditer(text):
        tail = m.group("tail")
        if _RC_NO.match(tail):
            continue
        if _RC_YES.match(tail):
            return True
        if not _RC_ASSERTIVE.search(m.group("phrase")):
            continue
        before = text[text.rfind("\n", 0, m.start()) + 1 : m.start()]
        fills_its_line = not before.strip() and not tail.partition("\n")[0].strip()
        if (
            not fills_its_line
            and not _RC_LABEL_TAIL.match(tail)
            and not _is_negated_or_question(text, m.start())
        ):
            return True
    return False


def ingest(path: str | os.PathLike[str]) -> IngestResult:
    """Open a PDF or image, route each page, and run the detectors on native pages.

    Raises ValueError for an unsupported extension or a password-protected PDF,
    and FileNotFoundError for a missing file. Detection runs per page on pages routed "native" only (so the
    evidence can cite the page), which means a refusal can be missed when the page
    that reveals it has no usable text layer; build 2 re-runs the detectors on OCR
    output.
    """
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise ValueError(
            f"unsupported file extension {ext!r}; accepted extensions are "
            + ", ".join(ACCEPTED_EXTENSIONS)
        )
    # PyMuPDF raises its own pymupdf.FileNotFoundError, which is not the builtin.
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such file: {path!r}")
    pages: list[PageResult] = []
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise ValueError(f"{path!r} is password-protected; remove the password and re-run")
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            method = route_page(text)
            pages.append(PageResult(page=number, method=method, text=text if method == "native" else ""))

    export = currency = None
    reverse_charge = False
    for result in pages:
        if result.method != "native":
            continue
        export = export or detect_export_or_sez(result.text, page=result.page)
        currency = currency or detect_multi_currency(result.text, page=result.page)
        reverse_charge = reverse_charge or detect_reverse_charge(result.text)
    refusals = [r for r in (export, currency) if r is not None]
    return IngestResult(pages=pages, refusals=refusals, reverse_charge=reverse_charge)
