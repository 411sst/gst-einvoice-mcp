"""Stage 1 extraction: regex and rules only, no LLM.

Only fields that can be confirmed structurally are returned: GSTINs validated
through build 1's ``validate_gstin``, the party each one belongs to, the
invoice number, the invoice date, HSN/SAC codes, and the state codes derived
from the GSTINs themselves. Anything this stage cannot settle is reported as an
``ExtractionWarning`` rather than guessed: a silently reversed seller/buyer pair
inverts who owes the tax, and a silently month-first date is off by months.

``consumed`` reports the character spans this stage matched so stage 2 can run
on ``remaining_text(text, consumed)`` and never re-derive what is already
confirmed. It covers every matched span with ONE deliberate exception: a date
read off what also reads as an item row is confirmed and warned about but left
in the text, because on that reading blanking it would hand stage 2 a line item
short of its serial number and the first words of its description. See
``_item_row_date_warning``. Stage 2 seeing a confirmed date twice costs nothing.
"""

import datetime
import re
from collections.abc import Iterable
from dataclasses import dataclass

from gst_einvoice.gstin import validate_gstin
from gst_einvoice.schema import ExtractionWarning
from gst_einvoice.state_codes import lookup_state_code

CHECK = "extract_rules"

# --- patterns ---------------------------------------------------------------

GSTIN_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b")

LABEL_WINDOW = 120
SELLER_LABELS = ("seller", "from", "supplier", "sold by", "consignor", "vendor")
BUYER_LABELS = (
    "buyer",
    "bill to",
    "billed to",
    "ship to",
    "shipped to",
    "recipient",
    "consignee",
    "customer",
    "to",
)
# "to" and "from" are also ordinary English prepositions. Either one only counts
# as a label at the start of a line or right after one of these separators;
# anywhere else ("goods collected from our depot") it is prose, and treating it
# as a label would silently reverse who owes the tax.
_BARE_LABEL_WORDS = ("to", "from")
_BARE_LABEL_SEPARATORS = "\n\r:|"

# The whitespace INSIDE a label is spaces and tabs, never a line break. '\s+' spans
# one, and an invoice title printed directly above an Indian door-number address
# line then reads as a label: "TAX INVOICE\nNo. 42, MG Road" matches "TAX
# INVOICE\nNo" as "tax invoice no", and the house number is filed as the mandatory
# DocDtls.No beside its own apparent label, so nothing warns. A label's own words
# are printed on one line; requiring that is what tells the two apart.
_DOC_NO_LABELS = (
    r"\btax[ \t]+invoice[ \t]+no\b"
    r"|\binvoice[ \t]+number\b"
    r"|\binvoice[ \t]+no\b"
    r"|\binvoice[ \t]*#"
    r"|\bbill[ \t]+number\b"
    r"|\bbill[ \t]+no\b"
    r"|\binv[ \t]+no\b"
)
DOC_NO_LABEL_PATTERN = re.compile(_DOC_NO_LABELS, re.IGNORECASE)
# The same label shapes with line-spanning whitespace, built from the strict source so
# the two can never drift apart. It NEVER supplies a value; it is how the module says
# out loud that the document prints an invoice-number label the strict pattern cannot
# read. Where it matches and the strict one does not, that label's own words straddle a
# line break — and because the contract takes the FIRST invoice-number label in the
# document, the number actually filed then comes from some LATER label ("e-Way Bill
# No.", a purchase order) and is another document's. See _wrapped_doc_no_label_warnings.
# Built from the strict source so the two cannot drift. The class gains the line
# terminators and nothing else: \s would also admit U+00A0, which PyMuPDF's native text
# layer emits inside a label that is printed on one line, and the warning below would
# then send the accountant looking for a line break the document does not have.
DOC_NO_WRAPPED_LABEL_PATTERN = re.compile(
    _DOC_NO_LABELS.replace(r"[ \t]", r"[ \t\r\n]"), re.IGNORECASE
)
DOC_NO_TOKEN_LIMIT = 30
# How specifically a label names THIS invoice's number, most specific first. The
# contract takes the FIRST invoice-number label in the document, and a line referring
# to ANOTHER document carries a label from the same list ("Ref: Your Bill No. 7788
# dated 01-Apr-2026", an e-Way bill number, a purchase order), so the first label is
# not always the invoice's own. Ranking them is what tells a competing label worth
# warning about from "e-Way Bill No.", which is another document's number printed on a
# large share of Indian invoices and would otherwise warn on every one of them.
DOC_NO_LABEL_RANK_WORDS = ("tax", "invoice", "inv", "bill")
# Between the label and its value the contract allows '.', ':', '#', '-' and
# whitespace. The value is looked for on the label's own line first and then in
# the label's own column on the line below, because the commonest invoice layout
# (Tally, Zoho, ClearTax) prints the labels as a header row and the values in the
# row underneath.
DOC_NO_SEPARATOR_PATTERN = re.compile(r"[ \t.:#\-]*")
DOC_NO_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9/\-]*")
DOC_NO_DIGIT = re.compile(r"\d")
# GST Rule 46(b) lets the serial be letters, numerals or special characters, so a
# same-line token carrying no digit is not refused merely for that. It is refused
# when it is one of the words invoices print as a heading or label ("Invoice
# No.   Invoice Date" hands back "Invoice"), and when it is too short to be
# anything but a truncated word ("INV" out of "Invoice No: INV 0042"): filing
# either as DocDtls.No is a confidently wrong value in a mandatory field.
_DOC_NO_LABEL_WORDS = frozenset(
    {
        "amount",
        "bill",
        "date",
        "dated",
        "description",
        "dt",
        "due",
        "gstin",
        "hsn",
        "invoice",
        "no",
        "nos",
        "number",
        "of",
        "particulars",
        "place",
        "qty",
        "rate",
        "ref",
        "reference",
        "sac",
        "sl",
        "sr",
        "supply",
        "tax",
        "total",
    }
)
# Only '/' and '-' appear here: DOC_NO_TOKEN_PATTERN is the contract's alphabet and
# cannot produce any other separator, so listing one would be a rule that can never fire.
DOC_NO_SERIAL_SEPARATORS = "/-"
DOC_NO_MIN_DIGIT_FREE_LENGTH = 4
# A quantity, not a serial: the row below a label often starts with the item's
# count ("2 Nos"), and filing that as the invoice number is silently wrong.
_UNIT_WORD_STEMS = r"nos|pcs|kgs?|mtrs?|ltrs?|sqft|sets?|boxes|box|pkt|units?"
_UNIT_WORDS = r"(?:" + _UNIT_WORD_STEMS + r"|no\.)(?![A-Za-z])"
DOC_NO_UNIT_WORDS = re.compile(r"\b" + _UNIT_WORDS, re.IGNORECASE)
# "No." is a unit only where a quantity is what the column holds. In the item-row
# SIGNATURE test it is the opposite: "No." there is overwhelmingly the invoice-number
# label itself, so a header line that prints the date before the label
# ("03/06/2026    Invoice No. INV/2026/17") reads as an item row and the date is
# warned about as an item row's serial number when it is nothing of the kind. The
# word stays in DOC_NO_UNIT_WORDS, where it still refuses "2 No." as a serial.
# More of the GST UQC vocabulary, recognised by the item-row signature ONLY.
# _UNIT_WORD_STEMS is shared with the invoice-number guard, where every added word is a
# genuine serial refused ("Bill No 4455 Bags" is an invoice number, not a quantity), so
# the widening goes here instead. The gap it closes: "1 Apr 2026 consulting 3 Hrs" is
# the same shape as "1 May 2026 hosting charges 2 Nos" — a serial in the first column
# read as a date — and warning about one but not the other makes the net look arbitrary
# to the accountant it is written for. It stays a closed list, so it carries none of the
# open-suffix risk the month vocabulary does. "can" and "day"/"days" are UQC entries
# deliberately left out: both are ordinary English printed under a date in a
# payment-terms line ("25/04/2026 Goods can be returned within 30 days"), which is not
# an item row, and a warning raised on a correct reading costs the warnings that matter
# their credibility.
_ITEM_ROW_EXTRA_UNIT_STEMS = (
    r"hrs?|gms?|tons?|doz|dzn|prs|pairs?|rolls?|rol|bags?|btl|bdl|bun|ctn"
    r"|sqm|sqy|qtl|tbs|thd|unt|cbm|kls?"
)
ITEM_ROW_UNIT_WORDS = re.compile(
    r"\b(?:" + _UNIT_WORD_STEMS + r"|" + _ITEM_ROW_EXTRA_UNIT_STEMS + r")(?![A-Za-z])",
    re.IGNORECASE,
)
# The same words, but only where they follow immediately. The row below a label is
# searched to its end, because nothing there names the value; a token printed beside
# its own label is only a quantity when the unit is the very next word, and a wider
# search would refuse a correctly labelled number over an unrelated "Vehicle No." or
# "Challan No." further along the same line.
UNIT_WORD_BESIDE = re.compile(r"[ \t]*\b" + _UNIT_WORDS, re.IGNORECASE)
# The reasons a token is something other than a serial number. They are shared
# because both paths apply the same tests, and because the row-below path now
# reports the one it refused instead of telling the accountant the column was
# empty when a value was printed in it.
DOC_NO_REFUSAL_GSTIN = "was already matched as a GSTIN, which is not an invoice number"
DOC_NO_REFUSAL_DATE = (
    "is a date, so it is the document's date column read where the invoice number "
    "should be rather than a serial number"
)
# A token is only ever ONE COMPONENT of a spaced date ("25 Apr 2026" tokenises as
# three), so the whole-date wording above cannot cover the day or the year on its own.
DOC_NO_REFUSAL_DATE_PART = (
    "is one component of the date '{printed}' the document prints in that position, so it "
    "is the date column read where the invoice number should be rather than a serial number"
)
DOC_NO_REFUSAL_NUMBER = "is one run of a longer number rather than a whole serial number"
DOC_NO_REFUSAL_QUANTITY = "is a quantity carrying a unit rather than a serial number"
# The row below a label is searched for a number only, because nothing on that row
# names the value. A word printed in the label's own column is still refused, but it
# IS printed there: reporting the column as empty sends the accountant to look at a
# cell that holds something.
DOC_NO_REFUSAL_NO_DIGIT_BELOW = (
    "carries no digit, and nothing on that row names it as the invoice number rather than "
    "a neighbouring column's value"
)

_MONTH_ABBR = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
# The month vocabulary is CLOSED and anchored by a word boundary. An open suffix
# ("apr[a-z]*") turns ordinary item-row words into months — "Aprons", "Marble",
# "Marketing", "Decorative", "Octane", "Junction" — so a quantity on the row after
# one of them ("1  Aprons  50  Nos") parses as a date and is filed as the invoice
# date with nothing to show it was ever in doubt.
_MONTH_VOCABULARY = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
# A printed date's day, month and year sit on ONE line, one space or one hyphen
# apart. '\s' includes the newline and '+' swallows a column gap, so a looser
# separator reads a table as a date: a quantity ending a row plus a wrapped
# description starting with a month ("...  1\n    April 2026 to March 2027") and
# an item row whose description begins with a month and a year ("1   May 2026
# hosting charges") both become DocDtls.Dt with nothing to show they were ever in
# doubt — and the consumed span blanks the quantity, the line break and the
# description's first words out of stage 2's input as well.
_DATE_MONTH_GAP = r"(?:[ \t]|[ \t]?-[ \t]?)"
DATE_PATTERN = re.compile(
    r"\b(?P<mday>\d{1,2})" + _DATE_MONTH_GAP + r"(?P<mon>" + _MONTH_VOCABULARY + r")\b"
    + _DATE_MONTH_GAP + r"(?P<myear>\d{4}|\d{2})\b"
    r"|\b(?P<n1>\d{1,2})[/.\-](?P<n2>\d{1,2})[/.\-](?P<nyear>\d{4}|\d{2})\b",
    re.IGNORECASE,
)
# Date labels in order of preference. A bare "date" also matches inside "Due Date"
# and "Date of Supply", so an explicit invoice-date label must win outright:
# filing a due date as the invoice date is wrong and looks perfectly clean.
#
# Tier 0's own words are printed on ONE line, for the same reason DOC_NO_LABEL_PATTERN's
# are. '\s+' spans a line break, so the document's TITLE printed above a line beginning
# with "Date" — "TAX INVOICE\nDate of Supply: 25/07/2026" — reads as the most specific
# date label there is. Tier 0 is the tier that settles the date SILENTLY: it suppresses
# the unclaimed-invoice-date-label warning, the competing-date warning and the item-row
# warning alike. So on that reading a supply date, a due date or a ledger column's date
# is filed as the mandatory DocDtls.Dt with nothing at all to show it was in doubt.
DATE_LABEL_TIERS = (
    re.compile(r"\binvoice[ \t]+(?:date|dt)\b", re.IGNORECASE),
    re.compile(r"\bdated\b", re.IGNORECASE),
    re.compile(r"\b(?:date|dt)\b", re.IGNORECASE),
)
DATE_LABEL_WINDOW = 40
DATE_LABEL_CONTEXT = 30
# The column gap in _DATE_MONTH_GAP is what keeps "1   May 2026 hosting charges" from
# reading as the first of May, and OCR destroys it: Tesseract joins the words of a
# line with a single space, so the same row arrives as "1 May 2026 hosting charges 1
# 5,000.00" and is indistinguishable from a date by its separators alone. What still
# tells them apart is the shape of the line — the day is the row's serial number in
# the first column and the rest of the row is an item row. The date is kept (dropping
# a mandatory field on a guess is worse), but on this reading it is never silent.
# A numeric COLUMN is a whole field of the row, not any run of digits. Every run
# inside one token belongs to that token: "INV/2026/17", "SFPL/24-25/0731" and
# "2026-42" are one serial each, and counting their runs as columns made every
# header line printing the date before an invoice number that carries its financial
# year — the module's own example above, "03/06/2026    Invoice No. INV/2026/17" —
# read as an item row and warn about a date field that is exactly where it belongs.
ITEM_ROW_NUMERIC_COLUMN = re.compile(
    r"(?<![A-Za-z0-9/\-])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9]|[/\-]\d)"
)
ITEM_ROW_MIN_NUMERIC_COLUMNS = 2

HSN_LABEL_PATTERN = re.compile(r"\b(?:HSN|SAC)\b", re.IGNORECASE)
HSN_DIGIT_RUN = re.compile(r"\d+")
HSN_CODE_LENGTHS = (4, 6, 8)
HSN_LABEL_WINDOW = 40
# "1200" in "1200.00" is half of an amount, not a standalone code.
HSN_DECIMAL_TAIL = re.compile(r"\.\d")
# So is "11" in "11,800.00": Indian grouping ("1,23,456.78") splits an amount into
# runs that a digit run and the invoice-number token pattern both stop at, so
# without this the commonest amount format printed where the invoice number should
# be files its first one or two digits as DocDtls.No, silently. A group is 2 or 3
# digits and nothing more, which leaves "8471,8473" (two codes) a pair of codes.
AMOUNT_GROUP_TAIL = re.compile(r",\d{2,3}(?!\d)")
# '/' and '-' join the runs of a serial ("2026-27/0042") and of a numeric date, so a
# digit run welded to another by one of them is a run of something longer rather than
# a standalone code.
HSN_JOINING_SEPARATORS = "/-"
HSN_JOINED_TAIL = re.compile(r"[/\-]\d")


# --- result types -----------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """A half-open character range ``[start, end)`` into the input text."""

    start: int
    end: int


@dataclass(frozen=True)
class RuleField:
    """One field confirmed by stage 1, with where and how it was found."""

    value: str
    span: Span
    method: str  # "label" | "position" | "regex" | "column" | "derived"


@dataclass(frozen=True)
class RuleExtraction:
    """Everything stage 1 confirmed, the spans it consumed, and its warnings."""

    seller_gstin: RuleField | None
    buyer_gstin: RuleField | None
    seller_stcd: RuleField | None
    buyer_stcd: RuleField | None
    doc_no: RuleField | None
    doc_date: RuleField | None
    hsn_codes: tuple[RuleField, ...]
    consumed: tuple[Span, ...]
    warnings: tuple[ExtractionWarning, ...]


def _warn(field: str, message: str) -> ExtractionWarning:
    return ExtractionWarning(field=field, message=message, severity="warning", check=CHECK)


# --- spans ------------------------------------------------------------------


def _merge_spans(spans: Iterable[Span], limit: int) -> tuple[Span, ...]:
    """Clamp, sort and merge spans so the result is ordered and non-overlapping."""
    clamped = []
    for span in spans:
        start = max(0, min(span.start, limit))
        end = max(0, min(span.end, limit))
        if end > start:
            clamped.append(Span(start, end))
    clamped.sort(key=lambda s: (s.start, s.end))
    merged: list[Span] = []
    for span in clamped:
        if merged and span.start <= merged[-1].end:
            if span.end > merged[-1].end:
                merged[-1] = Span(merged[-1].start, span.end)
        else:
            merged.append(span)
    return tuple(merged)


def remaining_text(text: str, consumed: Iterable[Span]) -> str:
    """Return ``text`` with every consumed span blanked out with spaces.

    Each span becomes spaces of its own length, so the text around it keeps both
    its length and its offsets: the pipeline maps a character offset back to the
    page it came from and to that page's OCR confidence, and a shortened string
    would point every later field at the wrong page.
    """
    pieces: list[str] = []
    cursor = 0
    for span in _merge_spans(consumed, len(text)):
        pieces.append(text[cursor : span.start])
        pieces.append(" " * (span.end - span.start))
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)


# --- party labels -----------------------------------------------------------


def _label_pattern(label: str) -> re.Pattern[str]:
    """A party label, its own words separated by spaces or tabs and never a line break.

    The same rule as DOC_NO_LABEL_PATTERN's, for a worse consequence. '\\s+' spans a
    line break, so a line ending in one of a label's words and the next line beginning
    with the rest reads as a printed heading the document does not have ("Goods once
    sold\\nBy order of the director" as a "Sold By" label) — and a party resolved by
    label is resolved with ``method="label"`` and no warning at all, which is the one
    path in this stage that is silent. Getting seller and buyer the wrong way round
    inverts who owes the tax. A genuinely OCR-wrapped party label is lost with it and
    falls back to position, which ALWAYS warns; that is the safe direction.
    """
    return re.compile(
        r"\b" + r"[ \t]+".join(re.escape(word) for word in label.split()) + r"\b",
        re.IGNORECASE,
    )


_SELLER_PATTERNS = tuple((label, _label_pattern(label)) for label in SELLER_LABELS)
_BUYER_PATTERNS = tuple((label, _label_pattern(label)) for label in BUYER_LABELS)


def _bare_label_qualifies(text: str, start: int) -> bool:
    """True when a bare "to"/"from" sits at a line start or right after ':' or '|'."""
    index = start - 1
    while index >= 0 and text[index] in " \t":
        index -= 1
    return index < 0 or text[index] in _BARE_LABEL_SEPARATORS


def _find_party_labels(text: str) -> list[tuple[int, int, str]]:
    """All seller/buyer label hits as (start, end, party), longer labels winning."""
    hits: list[tuple[int, int, str]] = []
    for party, patterns in (("seller", _SELLER_PATTERNS), ("buyer", _BUYER_PATTERNS)):
        for label, pattern in patterns:
            for match in pattern.finditer(text):
                if label in _BARE_LABEL_WORDS and not _bare_label_qualifies(text, match.start()):
                    continue
                hits.append((match.start(), match.end(), party))
    # Drop a hit wholly inside a longer one: the "to" of "Bill To" is not a bare "to".
    kept = [
        hit
        for hit in hits
        if not any(
            other is not hit
            and other[0] <= hit[0]
            and hit[1] <= other[1]
            and (other[1] - other[0]) > (hit[1] - hit[0])
            for other in hits
        )
    ]
    kept.sort(key=lambda hit: hit[0])
    return kept


def _nearest_party(labels: list[tuple[int, int, str]], gstin_start: int) -> str | None:
    """The party of the label ending closest before the GSTIN, within the window."""
    in_window = [
        hit for hit in labels if hit[1] <= gstin_start and hit[0] >= gstin_start - LABEL_WINDOW
    ]
    if not in_window:
        return None
    return max(in_window, key=lambda hit: (hit[1], hit[1] - hit[0]))[2]


# --- GSTINs and parties -----------------------------------------------------


def _gstin_candidates(text: str) -> tuple[list[re.Match[str]], list[ExtractionWarning]]:
    """Split GSTIN-shaped matches into checksum-valid ones and warnings for the rest."""
    valid: list[re.Match[str]] = []
    warnings: list[ExtractionWarning] = []
    for match in GSTIN_PATTERN.finditer(text):
        result = validate_gstin(match.group(0))
        if result.is_valid:
            valid.append(match)
            continue
        # The party is deliberately not guessed here: a rejected candidate was never
        # assigned to one. It is filed against both party GSTIN paths rather than a
        # party-agnostic leaf, so that whichever field an accountant is checking, the
        # reason a GSTIN is missing from it is listed under that field.
        message = (
            f"GSTIN-shaped text '{match.group(0)}' at character {match.start()} failed "
            f"validation ({result.reason}); it was not used as either party's GSTIN and could "
            f"not be attributed to one. Re-read the document: a single misread character "
            f"breaks the checksum."
        )
        warnings.append(_warn("SellerDtls.Gstin", message))
        warnings.append(_warn("BuyerDtls.Gstin", message))
    return valid, warnings


def _positional_message(
    seller: str, buyer: str | None, contradicted: list[tuple[str, str]]
) -> str:
    if buyer is None:
        assigned = f"GSTIN {seller} was assigned to the seller"
    else:
        assigned = f"GSTIN {seller} was assigned to the seller and GSTIN {buyer} to the buyer"
    message = (
        f"{assigned} from position in the document (the first GSTIN read is taken as the "
        f"seller, which is where Indian invoices conventionally print it), not from a "
        f"'Seller'/'Buyer' label. Reversing seller and buyer inverts who owes the tax, so "
        f"confirm this assignment against the document before filing."
    )
    for gstin, labelled_as in contradicted:
        message += (
            f" Note that the document labels GSTIN {gstin} as the {labelled_as}, which "
            f"contradicts this positional assignment; no label pair resolved both parties, so "
            f"position decided it anyway. Check this GSTIN first."
        )
    return message


def _resolve_parties(
    text: str, valid: list[re.Match[str]]
) -> tuple[RuleField | None, RuleField | None, list[ExtractionWarning]]:
    """Assign valid GSTINs to seller and buyer by label, else by position."""
    warnings: list[ExtractionWarning] = []
    if not valid:
        return (
            None,
            None,
            [
                _warn(
                    "SellerDtls.Gstin",
                    "No checksum-valid GSTIN was found in the document, so the seller GSTIN "
                    "could not be extracted.",
                ),
                _warn(
                    "BuyerDtls.Gstin",
                    "No checksum-valid GSTIN was found in the document, so the buyer GSTIN "
                    "could not be extracted.",
                ),
            ],
        )

    labels = _find_party_labels(text)
    by_party: dict[str, list[re.Match[str]]] = {"seller": [], "buyer": []}
    for match in valid:
        party = _nearest_party(labels, match.start())
        if party is not None:
            by_party[party].append(match)

    seller_values = {match.group(0) for match in by_party["seller"]}
    buyer_values = {match.group(0) for match in by_party["buyer"]}
    if len(seller_values) == 1 and len(buyer_values) == 1 and not (seller_values & buyer_values):
        seller_match = by_party["seller"][0]
        buyer_match = by_party["buyer"][0]
        return (
            RuleField(seller_match.group(0), Span(seller_match.start(), seller_match.end()), "label"),
            RuleField(buyer_match.group(0), Span(buyer_match.start(), buyer_match.end()), "label"),
            warnings,
        )

    # Positional fallback: reading order, first GSTIN is the seller.
    seller_match = valid[0]
    buyer_match = next((m for m in valid if m.group(0) != seller_match.group(0)), None)
    seller_field = RuleField(
        seller_match.group(0), Span(seller_match.start(), seller_match.end()), "position"
    )
    buyer_field = (
        None
        if buyer_match is None
        else RuleField(buyer_match.group(0), Span(buyer_match.start(), buyer_match.end()), "position")
    )
    contradicted = []
    if any(match is seller_match for match in by_party["buyer"]):
        contradicted.append((seller_field.value, "buyer"))
    if buyer_match is not None and any(match is buyer_match for match in by_party["seller"]):
        contradicted.append((buyer_field.value, "seller"))
    warnings.append(
        _warn(
            "SellerDtls.Gstin",
            _positional_message(
                seller_field.value,
                None if buyer_field is None else buyer_field.value,
                contradicted,
            ),
        )
    )
    if buyer_field is None:
        warnings.append(
            _warn(
                "BuyerDtls.Gstin",
                "Only one checksum-valid GSTIN was found in the document, so no buyer GSTIN "
                "could be extracted.",
            )
        )
    return seller_field, buyer_field, warnings


def _derive_stcd(
    gstin: RuleField | None, field_path: str
) -> tuple[RuleField | None, list[ExtractionWarning]]:
    """Derive a state code from a GSTIN's first two characters, never from an address."""
    if gstin is None:
        return None, []
    code = gstin.value[:2]
    result = lookup_state_code(code)
    warnings: list[ExtractionWarning] = []
    if result.status in ("discontinued", "unknown"):
        warnings.append(
            _warn(field_path, f"GSTIN {gstin.value} carries state code {code}: {result.note}")
        )
    return (
        RuleField(code, Span(gstin.span.start, gstin.span.start + 2), "derived"),
        warnings,
    )


# --- invoice number ---------------------------------------------------------


def _line_bounds(text: str, index: int) -> tuple[int, int]:
    """The ``[start, end)`` of the line holding ``index``, excluding its newline."""
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    return start, len(text) if end == -1 else end


def _doc_no_token_beside(text: str, label: re.Match[str]) -> re.Match[str] | None:
    """The token printed after the label on the label's own line.

    A label's own words share a line, so the label ends on the line it started on and
    that one line is what is searched.
    """
    line_end = _line_bounds(text, label.end())[1]
    separator = DOC_NO_SEPARATOR_PATTERN.match(text, label.end(), line_end)
    return DOC_NO_TOKEN_PATTERN.match(text, separator.end(), line_end)


def _token_identified_as_something_else(
    text: str, token: re.Match[str], gstin_spans: Iterable[Span]
) -> str | None:
    """Why the document identifies this token as something other than a serial number.

    The tests that do not depend on which side of the label the token sits: a GSTIN
    already matched as one, a date or any one component of a date printed over the
    token ("25" or "2026" out of "25 Apr 2026"), and one run of a longer number
    ("11800.00", "11,800.00"). The quantity test is left to each caller because its
    scope differs: beside a label the unit word must follow immediately, on the row
    below nothing names the value so the whole row counts.
    """
    if any(span.start < token.end() and token.start() < span.end for span in gstin_spans):
        return DOC_NO_REFUSAL_GSTIN
    printed = _date_printed_over(text, token)
    if printed is not None:
        if printed.span() == token.span():
            return DOC_NO_REFUSAL_DATE
        return DOC_NO_REFUSAL_DATE_PART.format(printed=printed.group(0))
    if _is_half_an_amount(text, token):
        return DOC_NO_REFUSAL_NUMBER
    return None


def _same_line_token_refusal(
    text: str, token: re.Match[str], gstin_spans: Iterable[Span]
) -> str | None:
    """Why the token printed beside the label cannot be the invoice number, or None.

    The tests the row below is held to apply here too. A blank invoice-number cell
    leaves the next cell printed where the value should be, so "Invoice No.
    25/04/2026" reads exactly like a value beside its own label — and filing a date,
    a GSTIN, half a decimal number or a quantity as DocDtls.No is a wrong value in a
    mandatory field whichever layout produced it.

    Beyond those, a token with a digit is accepted. A digit-free one is too — GST
    Rule 46(b) permits a serial of letters, numerals or special characters — unless
    it is one of the words invoices print as a heading, or it is short enough and
    plain enough to be a truncated word rather than a whole serial.
    """
    raw = token.group(0)
    reason = _token_identified_as_something_else(text, token, gstin_spans)
    if reason is not None:
        return reason
    if UNIT_WORD_BESIDE.match(text, token.end(), _line_bounds(text, token.start())[1]):
        return DOC_NO_REFUSAL_QUANTITY
    if DOC_NO_DIGIT.search(raw):
        return None
    if raw.casefold() in _DOC_NO_LABEL_WORDS:
        return (
            "is one of the words invoices print as a column heading or a label rather than "
            "as a serial number"
        )
    if any(char in DOC_NO_SERIAL_SEPARATORS for char in raw) or (
        len(raw) >= DOC_NO_MIN_DIGIT_FREE_LENGTH
    ):
        return None
    return (
        f"carries no digit, no '/' or '-' separator and fewer than "
        f"{DOC_NO_MIN_DIGIT_FREE_LENGTH} characters, so it reads as a truncated word rather "
        f"than a whole serial number"
    )


def _doc_no_token_below(
    text: str, label: re.Match[str], gstin_spans: Iterable[Span]
) -> tuple[re.Match[str] | None, tuple[str, str] | None]:
    """The number printed in the label's own column on the line below it, and why not.

    Only a token that carries a digit and shares columns with the label counts, and
    the one starting nearest the label's own column wins. That keeps a header row
    ("Date  Invoice No.") from handing back the neighbouring column's value.

    Anything the row below identifies as something else is refused outright: a
    GSTIN already matched as one, a date, one run of a longer number ("11800.00",
    "25.04.2026"), and a quantity carrying a unit ("2 Nos"). Each of those filed
    as DocDtls.No would be a wrong value in a mandatory field, and the caller
    warns about every value this path does return, so a refusal here is never
    silent either.

    The second element is the nearest in-column token that WAS refused and the
    reason, or None when the column really held nothing: an accountant told the
    column was empty when a value was printed in it is pointed at the wrong thing
    to check. A token carrying no digit is refused like any other but counts as
    printed — the column under "Invoice No." holding "SFPL-EXP" is not an empty one.

    A label's own words share a line, so the row below is the row below that one and
    the label's column on it is simply where the label is printed.
    """
    line_start, line_end = _line_bounds(text, label.end())
    if line_end >= len(text):
        return None, None
    next_start = line_end + 1
    next_end = _line_bounds(text, next_start)[1]
    label_first = label.start() - line_start
    label_last = label.end() - line_start
    spans = list(gstin_spans)
    best: re.Match[str] | None = None
    best_distance = 0
    refused: tuple[str, str] | None = None
    # A refused NUMBER outranks a refused word however the columns fall, because a
    # number is what this path looks for: the row below "Invoice No." reading "GSTIN:
    # 27AAPFU0939F1ZV" is reported by its GSTIN, not by the word printed left of it.
    refused_rank = (0, 0)
    for token in DOC_NO_TOKEN_PATTERN.finditer(text, next_start, next_end):
        first = token.start() - next_start
        if first >= label_last or token.end() - next_start <= label_first:
            continue  # printed in a different column
        distance = abs(first - label_first)
        has_digit = DOC_NO_DIGIT.search(token.group(0)) is not None
        if not has_digit:
            # Nothing on that row names the value, so only a number counts — but this
            # one was printed in the label's own column, so it is recorded as refused
            # rather than left to be reported as an empty cell.
            reason: str | None = DOC_NO_REFUSAL_NO_DIGIT_BELOW
        else:
            reason = _token_identified_as_something_else(text, token, spans)
            if reason is None and DOC_NO_UNIT_WORDS.search(text, token.end(), next_end):
                reason = DOC_NO_REFUSAL_QUANTITY  # a quantity ("2 Nos"), not a serial
        if reason is not None:
            rank = (0 if has_digit else 1, distance)
            if refused is None or rank < refused_rank:
                refused, refused_rank = (token.group(0), reason), rank
            continue
        if best is None or distance < best_distance:
            best, best_distance = token, distance
    return best, (None if best is not None else refused)


def _doc_no_field(
    label: re.Match[str],
    token: re.Match[str],
    method: str,
    below: tuple[str, str] | None = None,
) -> tuple[RuleField, list[ExtractionWarning]]:
    raw = token.group(0)
    value = raw[:DOC_NO_TOKEN_LIMIT]
    warnings: list[ExtractionWarning] = []
    if method == "column":
        # Deliberately the same rule as the parties' positional fallback: a value
        # taken from the row below the label was chosen by layout, not by a label
        # that names it, so it must never be indistinguishable from a value read
        # beside its own label — including when it lands on the right number.
        warnings.append(
            _warn(
                "DocDtls.No",
                f"Invoice number '{value}' was read from the row below the label "
                f"'{label.group(0)}' rather than from beside it, because the document prints "
                f"the labels as a header row and the values underneath. Nothing on that row "
                f"names it as the invoice number rather than a neighbouring column's value, so "
                f"confirm it against the document before filing.",
            )
        )
    elif not DOC_NO_DIGIT.search(value):
        # The other half of the same rule. A blank invoice-number cell leaves the NEXT
        # column's heading printed beside the label, and nothing on the line tells a
        # heading apart from a serial made of letters, which Rule 46(b) allows. The
        # closed list of heading words above catches the commonest of them; it cannot
        # catch "e-Way", "Transport" or "Party", so a digit-free value is taken but
        # never handed over as though the label had named it.
        #
        # What the line below holds is named rather than assumed to be nothing: the row
        # under the label often DID print a value that was refused (a date, an amount),
        # and telling the accountant that column was empty points them at the wrong cell.
        if below is None:
            confirmation = (
                "and no value was printed in the label's own column on the line below to "
                "confirm it"
            )
        else:
            confirmation = (
                f"and the row below prints '{below[0]}' in the label's own column, which "
                f"confirms nothing because it {below[1]}"
            )
        warnings.append(
            _warn(
                "DocDtls.No",
                f"Invoice number '{value}' was read from beside the label "
                f"'{label.group(0)}' but carries no digit, {confirmation}. GST Rule 46(b) "
                f"allows a serial of letters alone, so it was taken as the invoice number, "
                f"but a blank invoice-number cell leaves the next column's heading in "
                f"exactly this position. Confirm it against the document before filing.",
            )
        )
    if len(raw) > DOC_NO_TOKEN_LIMIT:
        warnings.append(
            _warn(
                "DocDtls.No",
                f"The invoice number token after '{label.group(0)}' runs past "
                f"{DOC_NO_TOKEN_LIMIT} characters; only its first {DOC_NO_TOKEN_LIMIT} "
                f"('{value}') were taken, so the value is shorter than the text on the "
                f"document. Confirm the invoice number against the document.",
            )
        )
    return RuleField(value, Span(token.start(), token.start() + len(value)), method), warnings


def _doc_no_refusal_message(
    label: re.Match[str],
    beside: tuple[str, str] | None,
    below: tuple[str, str] | None,
) -> str:
    """What was printed where the invoice number goes, and why it was not taken.

    Each half names the token it actually found. Reporting an empty column when the
    row below printed a GSTIN, a date or an amount in it sends the accountant to
    look at the wrong thing.
    """
    if beside is not None:
        head = (
            f"The invoice number label '{label.group(0)}' at character {label.start()} is "
            f"followed by '{beside[0]}', which was not accepted as the invoice number because "
            f"it {beside[1]}"
        )
    else:
        head = (
            f"The invoice number label '{label.group(0)}' at character {label.start()} is not "
            f"followed by a value on its own line"
        )
    if below is not None:
        tail = (
            f"; the row below prints '{below[0]}' in the label's column, which was not accepted "
            f"as the invoice number either because it {below[1]}"
        )
    else:
        tail = "; no value was printed in the label's column on the line below either"
    return (
        f"{head}{tail}. The invoice number could not be extracted; if one is visible on the "
        f"document, enter it manually."
    )


def _doc_no_candidate(
    text: str, label: re.Match[str], gstin_spans: list[Span]
) -> tuple[
    tuple[re.Match[str], str] | None, tuple[str, str] | None, tuple[str, str] | None
]:
    """What this one label supplies: ``((token, method), beside refusal, below refusal)``.

    The value is looked for beside the label first and then in the label's own column
    on the line below, with one exception. A digit-free token beside the label is what
    an EMPTY invoice-number cell leaves there — the next column's heading ("Invoice
    No.   e-Way Bill No." hands back 'e-Way') — and the closed list of heading words
    cannot enumerate the words Indian invoices actually print in that column. A number
    in the label's own column on the line below is the better reading of that layout,
    so it wins; it announces itself as a layout choice either way.
    """
    beside = _doc_no_token_beside(text, label)
    reason = None if beside is None else _same_line_token_refusal(text, beside, gstin_spans)
    beside_refusal = None if reason is None or beside is None else (beside.group(0), reason)
    accepted = None if reason is not None else beside
    if accepted is not None and DOC_NO_DIGIT.search(accepted.group(0)):
        return (accepted, "regex"), beside_refusal, None
    below, below_refusal = _doc_no_token_below(text, label, gstin_spans)
    if below is not None:
        return (below, "column"), beside_refusal, below_refusal
    if accepted is not None:
        return (accepted, "regex"), beside_refusal, below_refusal
    return None, beside_refusal, below_refusal


def _doc_no_label_rank(label: str) -> int:
    """How specifically a label names THIS invoice's number, 0 being most specific.

    'Tax Invoice No' names it outright; 'Invoice No'/'Invoice Number'/'Invoice #'
    nearly so; 'Inv No' is the abbreviation; 'Bill No' is the weakest, and it is also
    the tail of 'e-Way Bill No.', which is another document's number printed on a large
    share of Indian invoices. Only a competing label at least as specific as the one
    that supplied the value is worth a warning, which is what keeps the e-Way column
    from warning on every invoice that carries one.
    """
    words = re.findall(r"[a-z]+", label.casefold())
    return next(
        (rank for rank, word in enumerate(DOC_NO_LABEL_RANK_WORDS) if word in words),
        len(DOC_NO_LABEL_RANK_WORDS),
    )


def _competing_doc_no_warnings(
    text: str,
    label: re.Match[str],
    value: str,
    later: list[re.Match[str]],
    gstin_spans: list[Span],
) -> list[ExtractionWarning]:
    """Warn when a later, equally specific label carries a different invoice number.

    The invoice-number half of ``_competing_date_warnings``, and it exists for the same
    document shape. A reference to ANOTHER document printed above the invoice's own
    header — "Ref: Your Bill No. 7788 dated 01-Apr-2026", an e-Way bill number, a
    purchase order — carries an invoice-number label from the contract's own list, and
    the contract takes the FIRST such label in the document, so that line supplies the
    mandatory DocDtls.No while the invoice's own number is printed on the next line.
    Without this the accountant is warned that the DATE may be the referenced
    document's while the NUMBER silently is.

    A later label carrying the SAME value is a repeated multi-page header and says
    nothing. A less specific one is most often 'e-Way Bill No.' under an 'Invoice No.'
    header, where the invoice number taken is right and a warning would only cost the
    warnings that matter their credibility.
    """
    rank = _doc_no_label_rank(label.group(0))
    competing: list[str] = []
    for other in later:
        if _doc_no_label_rank(other.group(0)) > rank:
            continue
        taken = _doc_no_candidate(text, other, gstin_spans)[0]
        if taken is None:
            continue
        other_value = taken[0].group(0)[:DOC_NO_TOKEN_LIMIT]
        if other_value == value:
            continue
        entry = f"'{other.group(0)}' at character {other.start()} carrying '{other_value}'"
        if entry not in competing:
            competing.append(entry)
    if not competing:
        return []
    return [
        _warn(
            "DocDtls.No",
            f"Invoice number '{value}' was read from '{label.group(0)}' at character "
            f"{label.start()}, the first invoice-number label in the document, but the "
            f"document also prints {', '.join(competing)}. A line referring to another "
            f"document ('Ref: Your Bill No. 7788 dated ...', an e-Way bill number, a "
            f"purchase order) carries an invoice-number label of its own, and printed above "
            f"the invoice's own header it is the one read first — so the number filed would "
            f"be that other document's. Confirm which of these is this invoice's own number "
            f"before filing.",
        )
    ]


def _wrapped_doc_no_label_warnings(
    text: str, labels: list[re.Match[str]]
) -> list[ExtractionWarning]:
    """Report an invoice-number label whose own words straddle a line break.

    A label split across lines is not read as a label, and that rule is what keeps the
    house number of "TAX INVOICE\\nNo. 42, MG Road" out of the mandatory DocDtls.No. But
    refusing it is not the whole story on a document that really does print one. The
    contract takes the FIRST invoice-number label in the document, so when the invoice's
    OWN label is the wrapped one, the value filed is whatever a LATER label carries — an
    "e-Way Bill No." is another document's number and is printed on a large share of
    Indian invoices — and a digit-carrying token beside its own label is this stage's one
    silent path, so nothing else says a word. The warning is therefore emitted whether or
    not another label supplied a value: what it reports is the wrapped label, not the
    outcome.

    A wrapped match that CONTAINS a strict label is that same label read twice, not a
    wrapped one. Containment rather than an equal start is what the test has to be: the
    longest alternation reaches back over the line break for a preceding word, so a
    heading ending in "Tax" above a correctly read "Invoice No." produces a wrapped match
    that starts earlier than the strict one and encloses it.
    """
    strict_spans = [label.span() for label in labels]
    warnings: list[ExtractionWarning] = []
    for match in DOC_NO_WRAPPED_LABEL_PATTERN.finditer(text):
        if any(match.start() <= start and end <= match.end() for start, end in strict_spans):
            continue
        printed = " above ".join(f"'{part.strip()}'" for part in match.group(0).splitlines())
        warnings.append(
            _warn(
                "DocDtls.No",
                f"The document prints {printed} at character {match.start()}, which reads as an "
                f"invoice-number label wrapped across a line break. A label is only recognised "
                f"when its own words are printed on one line, so this one could not be read as "
                f"a label and no value was taken from beside it or below it. Any number "
                f"reported for DocDtls.No was read from a different label on the page and may "
                f"belong to that other label rather than to this one. Read the invoice number "
                f"from the document.",
            )
        )
    return warnings


def _extract_doc_no(
    text: str, gstin_spans: Iterable[Span]
) -> tuple[RuleField | None, list[ExtractionWarning]]:
    gstin_spans = list(gstin_spans)
    labels = list(DOC_NO_LABEL_PATTERN.finditer(text))
    # Appended to every path below, the silent one included: a wrapped label is a fact
    # about the document, not about whether some other label happened to supply a value.
    wrapped = _wrapped_doc_no_label_warnings(text, labels)
    if not labels:
        return None, [
            _warn(
                "DocDtls.No",
                "No invoice number label ('Invoice No', 'Bill No', 'Tax Invoice No', 'Inv No') "
                "was found, so the invoice number could not be extracted. A label is only "
                "recognised when its own words are printed on one line, so a label OCR split "
                "across a line break ('Invoice' above 'No.') is not found even though the "
                "document prints one; if the document shows an invoice number, read it from "
                "the document.",
            )
        ] + wrapped
    refused: tuple[re.Match[str], tuple[str, str] | None, tuple[str, str] | None] | None = None
    for index, label in enumerate(labels):
        taken, beside_refusal, below_refusal = _doc_no_candidate(text, label, gstin_spans)
        if taken is not None:
            token, method = taken
            field, warnings = _doc_no_field(label, token, method, below_refusal)
            warnings.extend(
                _competing_doc_no_warnings(
                    text, label, field.value, labels[index + 1 :], gstin_spans
                )
            )
            warnings.extend(wrapped)
            return field, warnings
        if refused is None and (beside_refusal is not None or below_refusal is not None):
            refused = (label, beside_refusal, below_refusal)
    if refused is not None:
        return None, [_warn("DocDtls.No", _doc_no_refusal_message(*refused))] + wrapped
    label = labels[0]
    return None, [
        _warn(
            "DocDtls.No",
            f"The invoice number label '{label.group(0)}' at character {label.start()} is not "
            f"followed by a value on its own line or in its column on the line below, so the "
            f"invoice number could not be extracted. Read it from the document rather than from "
            f"the text after the label.",
        )
    ] + wrapped


# --- invoice date -----------------------------------------------------------


def _date_parts(match: re.Match[str]) -> tuple[int, int, int]:
    """A matched date's day-first (day, month, year), years 00-99 as 2000-2099."""
    if match.group("mday") is not None:
        day = int(match.group("mday"))
        month = _MONTH_ABBR.index(match.group("mon")[:3].lower()) + 1
        year = int(match.group("myear"))
    else:
        day = int(match.group("n1"))
        month = int(match.group("n2"))
        year = int(match.group("nyear"))
    if year < 100:
        year += 2000
    return day, month, year


def _normalise_date(match: re.Match[str]) -> str:
    """Render a matched date as DD/MM/YYYY, day-first, years 00-99 as 2000-2099."""
    day, month, year = _date_parts(match)
    return f"{day:02d}/{month:02d}/{year:04d}"


def _is_a_calendar_date(year: int, month: int, day: int) -> bool:
    try:
        datetime.date(year, month, day)
    except ValueError:
        return False
    return True


def _is_a_real_date(match: re.Match[str]) -> bool:
    """True when a date-shaped match reads as a real calendar date in either order.

    A month name settles which component is the month, so there is only one reading
    to test — but the day must still be a real day of that month: "42 May 2026" and
    "31-Feb-2026" are date-SHAPED, not dates. A numeric match has to be a date under
    at least one of its two readings: "24-25/0731" is a financial-year serial, not a
    date, and refusing it as an invoice number would drop a mandatory field on the
    ground that it resembles something it cannot possibly be.
    """
    day, month, year = _date_parts(match)
    if match.group("mday") is not None:
        return _is_a_calendar_date(year, month, day)
    return _is_a_calendar_date(year, month, day) or _is_a_calendar_date(year, day, month)


def _overlapping_dates(text: str, start: int, end: int) -> list[re.Match[str]]:
    """Every date-shaped match on this line that overlaps ``[start, end)``.

    The search is bounded to the one line because a printed date's components sit on
    one line, and a date further along the line belongs to another column only if it
    does not actually cover these characters.
    """
    line_start, line_end = _line_bounds(text, start)
    return [
        match
        for match in DATE_PATTERN.finditer(text, line_start, line_end)
        if match.start() < end and start < match.end()
    ]


def _date_printed_over(text: str, token: re.Match[str]) -> re.Match[str] | None:
    """The real date the document prints over this token, or None.

    The invoice-number token alphabet stops at a space, so a token is never more than
    one COMPONENT of a spaced date: "Invoice No.   25 Apr 2026" tokenises as "25",
    "Apr" and "2026", and testing any of those strings on its own says it is not a
    date. Asking only whether a date STARTS at the token catches the day and misses
    the year, so a header row whose invoice-number cell is blank ("Invoice No.
    Invoice Date" over " 25 Apr 2026") files 2026 as DocDtls.No — and the date is then
    dropped for overlapping the invoice number, leaving the accountant told that no
    date was found on a document that prints one, with its year blanked out of stage
    2's input as well. Asking whether any date COVERS the token sees "25 Apr 2026" as
    the document printed it, whichever component was offered.

    The date must still be a real calendar date, so a financial-year serial
    ("24-25/0731") is not refused as one. That is stricter than
    ``_falls_inside_a_date``'s test on the HSN path, deliberately: here a wrong
    refusal drops a mandatory field, there a wrong acceptance files a wrong one.
    """
    for match in _overlapping_dates(text, token.start(), token.end()):
        if _is_a_real_date(match):
            return match
    return None


def _is_a_serial_fragment(text: str, match: re.Match[str]) -> bool:
    """True when a numeric date match is part of a longer serial ("SFPL/24-25/0731").

    A Tally-style invoice number carries the financial year, and its middle reads as
    a numeric date. Filing that as DocDtls.Dt puts a wrong value in a mandatory field
    AND drops the date actually printed on the document, so the fragment is not a
    date candidate at all. A date after '.', '(' or a space, or at the start of a
    line, is untouched.
    """
    if match.group("n1") is None:
        return False
    start = match.start()
    return start >= 2 and text[start - 1] in "/-" and text[start - 2].isalnum()


def _is_a_quantity_row(text: str, match: re.Match[str]) -> bool:
    """True when "3 June 26 Nos" is an item row rather than a two-digit-year date.

    The contract allows a two-digit year, so an item whose description is exactly a
    month word and whose quantity follows reads as a date. The unit word immediately
    after the year is what distinguishes the row; a genuine date is not followed by
    one. A wider column gap ("1   May   10   Nos") is already not a date candidate:
    DATE_PATTERN's separators are a single space or a hyphen.
    """
    if match.group("mday") is None or len(match.group("myear")) != 2:
        return False
    return UNIT_WORD_BESIDE.match(text, match.end()) is not None


def _date_candidates(text: str, exclude: Span | None) -> list[re.Match[str]]:
    """Every date match that is a date rather than part of something else."""
    candidates: list[re.Match[str]] = []
    for match in DATE_PATTERN.finditer(text):
        if exclude is not None and exclude.start < match.end() and match.start() < exclude.end:
            continue  # already matched as the invoice number
        if _is_a_serial_fragment(text, match) or _is_a_quantity_row(text, match):
            continue
        candidates.append(match)
    return candidates


def _label_context(text: str, label: re.Match[str]) -> str:
    """The label as printed, back to the start of its line ("Date" -> "Due Date").

    A run of two or more spaces is a column gap rather than part of the label, so
    "Invoice Date:      Due Date" is reported as 'Due Date': naming a pair of cells
    as one label names a label the document does not print, and here it would put
    the words "Invoice Date" into the message for a date the invoice-date label did
    not supply.

    OCR collapses that column gap to a single space, so the gap cannot be the only
    thing that ends the label. What still separates the cells is that the one before
    holds a value: only whole words of letters qualify a label ("Due Date", "Delivery
    Date"), so the context stops at the last word carrying a digit or punctuation
    ("Bill No. 7788 Date" is reported as 'Date', not as a label naming the number).
    """
    line_start = text.rfind("\n", 0, label.start()) + 1
    context = text[max(line_start, label.end() - DATE_LABEL_CONTEXT) : label.end()]
    words: list[str] = []
    for word in reversed(context.rsplit("  ", 1)[-1].split()):
        if not word.isalpha():
            break
        words.append(word)
    return " ".join(reversed(words)) or label.group(0).strip()


def _date_label_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Every date label in the text, whatever its tier, deduplicated."""
    return tuple(
        sorted(
            {
                match.span()
                for pattern in DATE_LABEL_TIERS
                for match in pattern.finditer(text)
            }
        )
    )


def _another_date_label_intervenes(
    label_spans: tuple[tuple[int, int], ...], label: re.Match[str], candidate: re.Match[str]
) -> bool:
    """True when another date label sits wholly between this label and the candidate.

    An empty invoice-date cell leaves the NEXT label and its value inside the window,
    so "Invoice Date:        Due Date: 30/05/2026" reads exactly like a date printed
    after its own label. A label only names a date it reaches without crossing another
    date label; a due date filed as DocDtls.Dt is a perfectly well-formed date that no
    downstream validator can question, which makes it the worst kind of wrong value in
    a mandatory field.
    """
    return any(label.end() <= start and end <= candidate.start() for start, end in label_spans)


def _labelled_dates(
    text: str, dates: list[re.Match[str]]
) -> dict[tuple[int, int], tuple[int, str, str]]:
    """Every labelled date's span -> (best label tier, label as printed, DD/MM/YYYY)."""
    found: dict[tuple[int, int], tuple[int, str, str]] = {}
    label_spans = _date_label_spans(text)
    for tier, pattern in enumerate(DATE_LABEL_TIERS):
        for label in pattern.finditer(text):
            for candidate in dates:
                if 0 <= candidate.start() - label.end() <= DATE_LABEL_WINDOW:
                    if not _another_date_label_intervenes(label_spans, label, candidate):
                        found.setdefault(
                            candidate.span(),
                            (tier, _label_context(text, label), _normalise_date(candidate)),
                        )
                    break
    return found


def _select_date(
    text: str, dates: list[re.Match[str]]
) -> tuple[re.Match[str] | None, int | None, dict[tuple[int, int], tuple[int, str, str]]]:
    """Prefer the date under the most specific date label, else the first date."""
    if not dates:
        return None, None, {}
    labelled = _labelled_dates(text, dates)
    if not labelled:
        return dates[0], None, labelled
    span = min(labelled, key=lambda key: (labelled[key][0], key[0]))
    chosen = next(date for date in dates if date.span() == span)
    return chosen, labelled[span][0], labelled


def _date_source(
    chosen: re.Match[str], labelled: dict[tuple[int, int], tuple[int, str, str]]
) -> str:
    """Where the chosen date came from: its own label, or its position in the text."""
    own = labelled.get(chosen.span())
    return f"the '{own[1]}' label" if own else "its position (the first date in the document)"


def _unclaimed_invoice_date_label_warning(
    text: str,
    chosen: re.Match[str],
    value: str,
    tier: int | None,
    labelled: dict[tuple[int, int], tuple[int, str, str]],
) -> list[ExtractionWarning]:
    """Warn when the document prints an invoice-date label that did not supply the date.

    The date-side analogue of the blank invoice-number cell. An 'Invoice Date' label
    whose own cell is empty, or whose value is printed out of the label's reach, leaves
    the neighbouring due date as the nearest thing the rules can claim — and an
    explicit invoice-date label is what otherwise lets the date be filed silently. The
    label being present and unsatisfied is the evidence that the value came from
    somewhere else, so it is said out loud.
    """
    if tier == 0:
        return []  # an invoice-date label claimed the date itself
    label = DATE_LABEL_TIERS[0].search(text)
    if label is None:
        return []
    return [
        _warn(
            "DocDtls.Dt",
            f"The document prints an invoice-date label ('{label.group(0)}' at character "
            f"{label.start()}) with no date of its own after it (an empty invoice-date cell, "
            f"or its value printed too far from the label to belong to it), so the invoice date "
            f"'{chosen.group(0)}' was read as {value} from {_date_source(chosen, labelled)} "
            f"instead. A blank invoice-date cell leaves the next column's date printed where the "
            f"invoice date should be, and a due date or a delivery date filed as the invoice "
            f"date is wrong in a way no later check can question. Confirm that {value} is the "
            f"invoice date before filing.",
        )
    ]


def _starts_an_item_row(text: str, match: re.Match[str]) -> bool:
    """True when this date opens a line whose remainder reads as an item row.

    Two things have to hold. The date starts the line, so its day is what an item
    row prints in its first column as the serial number; and the rest of the line
    carries an item row's signature — a unit word, or two or more further numeric
    columns (a quantity and an amount, a rate and an amount).

    The unit word here is ITEM_ROW_UNIT_WORDS, which omits "No.": a header line
    printing the date before the invoice-number label puts "No." after the date
    without an item row anywhere in sight.

    An invoice-number label on the rest of the line settles it outright, whichever
    signature the line would otherwise show. That label is printed on a header line;
    an item row does not carry one, and the serial printed after it supplies the
    "further numeric columns" ("Invoice No. 42   Total 11,800.00") that the count
    alone reads as a line item.
    """
    line_start, line_end = _line_bounds(text, match.start())
    if text[line_start : match.start()].strip():
        return False  # something is printed before it, so it is not the first column
    rest = text[match.end() : line_end]
    if DOC_NO_LABEL_PATTERN.search(rest):
        return False
    if ITEM_ROW_UNIT_WORDS.search(rest):
        return True
    return len(ITEM_ROW_NUMERIC_COLUMN.findall(rest)) >= ITEM_ROW_MIN_NUMERIC_COLUMNS


def _item_row_date_warning(
    text: str, chosen: re.Match[str], value: str, tier: int | None
) -> list[ExtractionWarning]:
    """Warn when an unlabelled date was read off what looks like an item row.

    Only the "first date in the document" fallback can reach this: a date a date
    label named is a date field by the document's own say-so and stays silent. The
    reading is kept and made visible rather than dropped, because refusing it would
    lose the invoice date of every document whose date really does start a line. On
    this reading the text is an item row, so the caller also leaves its span out of
    ``consumed``: blanking it hands stage 2 "hosting charges 1 5,000.00" where the
    document prints "1 May 2026 hosting charges 1 5,000.00", and a line item short of
    its serial number and the first words of its description is exactly the
    plausible-looking row that was never in the document. The date is already reported
    as a confirmed field, so stage 2 seeing it twice costs nothing.
    """
    if tier is not None:
        return []  # a date label named it, so it is not a guess from layout
    if not _starts_an_item_row(text, chosen):
        return []
    return [
        _warn(
            "DocDtls.Dt",
            f"Invoice date '{chosen.group(0)}' was read as {value} from its position (the "
            f"first date in the document); no date label names it, and it begins a line that "
            f"reads as an item row rather than a date field — its day sits where an item row "
            f"prints its serial number, and the rest of the line carries a unit or further "
            f"numeric columns. OCR joins a row's columns with single spaces, which removes the "
            f"column gap that otherwise tells an item row from a date. The line was left in "
            f"the text stage 2 reads, so the item row is not passed on short of its first "
            f"columns. Confirm that {value} is the invoice date and not an item row's first "
            f"column before filing.",
        )
    ]


def _competing_date_warnings(
    chosen: re.Match[str],
    value: str,
    tier: int | None,
    labelled: dict[tuple[int, int], tuple[int, str, str]],
    dates: list[re.Match[str]],
) -> list[ExtractionWarning]:
    """Warn when no invoice-date label settled it and another date on the page differs.

    EVERY other date counts, not only the labelled ones. A 'dated' on a purchase-order
    or delivery-challan reference line is a date label by the contract's own list, so
    it can win outright over the invoice's own date printed as a column value out of
    the label's reach — and that is a wrong value in a mandatory field with nothing to
    show for it. An explicit 'Invoice Date' label still settles it silently.
    """
    if tier is None or tier == 0:
        return []
    competing: list[str] = []
    for other in dates:
        if other.span() == chosen.span():
            continue
        other_value = _normalise_date(other)
        if other_value == value:
            continue
        label = labelled.get(other.span())
        entry = f"'{label[1]}' -> {other_value}" if label else f"an unlabelled {other_value}"
        if entry not in competing:
            competing.append(entry)
    if not competing:
        return []
    return [
        _warn(
            "DocDtls.Dt",
            f"Invoice date '{chosen.group(0)}' was read as {value} from "
            f"{_date_source(chosen, labelled)}; the document "
            f"carries no 'Invoice Date' label and also carries {', '.join(competing)}. A due "
            f"date, a supply date or another document's date filed as the invoice date is "
            f"wrong, so confirm which of these is the invoice date.",
        )
    ]


def _impossible_calendar_date_warning(
    chosen: re.Match[str], value: str
) -> list[ExtractionWarning]:
    """Warn when a well-formed date names a day its month does not have.

    Both components are in range but the pair is not a real date, so the text was
    misread or mistyped. For a numeric date a swapped reading cannot rescue it: if
    the day were <= 12 the day-first pair would already have been valid.
    """
    day, month, year = int(value[:2]), int(value[3:5]), int(value[6:])
    if _is_a_calendar_date(year, month, day):
        return []
    return [
        _warn(
            "DocDtls.Dt",
            f"Invoice date '{chosen.group(0)}' normalises to {value}, which is not a real "
            f"calendar date ({MONTH_NAMES[month - 1]} {year} has no day {day}). It was still "
            f"emitted as {value} because GSTN requires DD/MM/YYYY, but the date was misread or "
            f"mistyped. Confirm the date against the document.",
        )
    ]


def _date_warnings(
    dates: list[re.Match[str]], chosen: re.Match[str], value: str
) -> list[ExtractionWarning]:
    """Warn whenever a numeric date's day/month order is not settled by the document."""
    if chosen.group("mday") is not None:
        # DD-Mon-YYYY carries its own month name, so the day/month ORDER is never in
        # doubt and none of the ambiguity warnings below can apply. The day itself
        # still has to exist: "31-Feb-2026" is provably a misread, and filing it
        # clean in a mandatory field is the same confidently wrong extraction the
        # numeric path already refuses to make.
        return _impossible_calendar_date_warning(chosen, value)
    first, second = int(chosen.group("n1")), int(chosen.group("n2"))
    raw = chosen.group(0)
    if not 1 <= first <= 31:
        return [
            _warn(
                "DocDtls.Dt",
                f"Invoice date '{raw}' has {first} as its first component, which cannot be a day "
                f"of the month. It was still emitted day-first as {value} because GSTN requires "
                f"DD/MM/YYYY, but the date was probably misread. Confirm the date against the "
                f"document.",
            )
        ]
    if not 1 <= second <= 12:
        return [
            _warn(
                "DocDtls.Dt",
                f"Invoice date '{raw}' has {second} as its second component, which cannot be a "
                f"month. It was still emitted day-first as {value} because GSTN requires "
                f"DD/MM/YYYY, but the source is most likely MM/DD/YYYY. Confirm the date "
                f"against the document.",
            )
        ]
    impossible = _impossible_calendar_date_warning(chosen, value)
    if impossible:
        return impossible
    if not 1 <= first <= 12 or first == second:
        return []  # Unambiguous: the first component can only be a day, or both readings agree.

    others = [
        (int(other.group("n1")), int(other.group("n2")))
        for other in dates
        if other.group("n1") is not None and other.span() != chosen.span()
    ]
    if any(other_first > 12 for other_first, _ in others):
        return []  # Another date on the page can only be day-first; so is this one.
    if any(other_first <= 12 < other_second for other_first, other_second in others):
        return [
            _warn(
                "DocDtls.Dt",
                f"Invoice date '{raw}' was read day-first as {value} because GSTN requires "
                f"DD/MM/YYYY, but another date on the document can only be read month-first, so "
                f"this one may be MM/DD and mean "
                f"{second} {MONTH_NAMES[first - 1]} {value[6:]}. Confirm the date against the "
                f"document.",
            )
        ]
    return [
        _warn(
            "DocDtls.Dt",
            f"Invoice date '{raw}' is ambiguous: it reads as either "
            f"{first} {MONTH_NAMES[second - 1]} {value[6:]} (day-first) or "
            f"{second} {MONTH_NAMES[first - 1]} {value[6:]} (month-first), and no other date on "
            f"the document settles it. Day-first was used because GSTN requires DD/MM/YYYY. "
            f"Confirm the date against the document.",
        )
    ]


def _extract_doc_date(
    text: str, doc_no_span: Span | None = None
) -> tuple[RuleField | None, list[ExtractionWarning], bool]:
    """The invoice date, its warnings, and whether its span may be consumed.

    The span is consumed except on the item-row reading, where the module's own
    warning says the text may be a line item rather than a date: on that reading
    blanking it out mutilates the row stage 2 has to read.
    """
    dates = _date_candidates(text, doc_no_span)
    chosen, tier, labelled = _select_date(text, dates)
    if chosen is None:
        return (
            None,
            [
                _warn(
                    "DocDtls.Dt",
                    "No date was found in the document, so the invoice date could not be extracted.",
                )
            ],
            False,
        )
    value = _normalise_date(chosen)
    warnings = _unclaimed_invoice_date_label_warning(text, chosen, value, tier, labelled)
    warnings.extend(_competing_date_warnings(chosen, value, tier, labelled, dates))
    item_row = _item_row_date_warning(text, chosen, value, tier)
    warnings.extend(item_row)
    warnings.extend(_date_warnings(dates, chosen, value))
    return RuleField(value, Span(chosen.start(), chosen.end()), "regex"), warnings, not item_row


# --- HSN/SAC ----------------------------------------------------------------


def _is_half_an_amount(text: str, number: re.Match[str]) -> bool:
    """True when the digits are one run of a longer number ("1200.00", "11,800.00").

    Both separators count. The decimal point alone leaves Indian comma grouping —
    the format the module's own clean-invoice fixture and the spec both use — to
    hand back the first run of an amount as a whole value.
    """
    if HSN_DECIMAL_TAIL.match(text, number.end()) or AMOUNT_GROUP_TAIL.match(text, number.end()):
        return True
    start = number.start()
    return start >= 2 and text[start - 1] in ".," and text[start - 2].isdigit()


def _falls_inside_a_date(text: str, number: re.Match[str]) -> bool:
    """True when the digits are part of a real date printed on the same line.

    The year of "25 Apr 2026" is four standalone digits sitting next to whatever
    follows it, so a header line reading "Invoice Date 25 Apr 2026  HSN 998313"
    otherwise files 2026 as an HSN code.

    A month name is enough on its own here, whether or not its day is a real day of
    that month: "31 Feb 2026" is still a printed date, and the cost of shielding its
    year is a missed code stage 2 can still read, while the cost of not shielding it
    is 2026 filed as a confirmed HsnCd. The doc-no refusal holds date-shaped text to
    the stricter test because there the cost runs the other way — refusing a genuine
    serial drops a mandatory field.
    """
    return any(
        match.group("mday") is not None or _is_a_real_date(match)
        for match in _overlapping_dates(text, number.start(), number.end())
    )


def _is_a_standalone_code(text: str, number: re.Match[str]) -> bool:
    """True when this digit run reads as a 4/6/8 digit code and not part of something else."""
    if len(number.group(0)) not in HSN_CODE_LENGTHS:
        return False
    if number.start() > 0 and text[number.start() - 1].isalpha():
        return False
    if number.end() < len(text) and text[number.end()].isalpha():
        return False
    if _is_half_an_amount(text, number):
        return False
    if HSN_JOINED_TAIL.match(text, number.end()) or (
        number.start() >= 2
        and text[number.start() - 1] in HSN_JOINING_SEPARATORS
        and text[number.start() - 2].isdigit()
    ):
        # "2026" out of "2026-27/0042" and "0731" out of "24-25/0731" are runs of a
        # serial, not codes. A pair of codes printed with a bare "/" between them
        # ("8471/8473") is refused with them: a missed code is a miss, a serial's
        # middle filed as an HSN code is a wrong value. Spaced lists ("8471 / 8473")
        # and comma-separated ones are unaffected.
        return False
    return not _falls_inside_a_date(text, number)


def _a_word_intervenes(text: str, start: int, end: int) -> bool:
    """True when a word of letters sits between two numbers on a row."""
    return any(char.isalpha() for char in text[start:end])


def _a_list_separator_intervenes(text: str, start: int, end: int) -> bool:
    """True when punctuation, not whitespace alone, separates two numbers on a row.

    What separates two codes printed as a list is punctuation ("998313 / 52081190",
    "52081190, 998313, 8471"). What separates the columns of an item row is
    whitespace — and OCR collapses a column gap to a single space, so whitespace
    cannot tell a second code from the rate printed beside the first one. A word
    already ends the run, but a services row or an OCR'd row prints no unit word
    between the code and the money columns, which leaves nothing but spaces there.
    """
    return any(not char.isspace() for char in text[start:end])


# The word printed immediately before a number names that number's column. Where it
# names a money or a quantity column, the number is that column's value and the
# HSN/SAC label further along the row does not name it: "Amount 1200   HSN 998313"
# and "HSN wise summary   Total 5000" print two cells each, not a code beside a
# label. The legitimate readings carry no such word ("Goods 52081190 HSN 10 Mtr",
# "Taxable value under 998313 (SAC)", "HSN/SAC Code : 52081190"), so they are
# unaffected. Only a number the label would otherwise claim with nothing but
# separators in between is tested this way; everything further along the row is
# already ended by the word itself.
_MONEY_COLUMN_WORDS = frozenset(
    {"amount", "amt", "price", "qty", "quantity", "rate", "total", "val", "value"}
)
MONEY_COLUMN_WORD_BEFORE = re.compile(r"([A-Za-z]+)[ \t]*$")


def _a_money_column_names_it(text: str, start: int, end: int) -> bool:
    """True when the word right before a number names a money or quantity column."""
    word = MONEY_COLUMN_WORD_BEFORE.search(text, start, end)
    return word is not None and word.group(1).casefold() in _MONEY_COLUMN_WORDS


def _the_label_names_the_number(text: str, label: re.Match[str], start: int) -> bool:
    """True when the first number after an HSN/SAC label is the value that label names.

    The nearest-number exemption exists for two layouts. The label's own tail is a word
    and the label is still what names the value beside it ("HSN/SAC Code : 52081190"),
    so on the label's OWN LINE the exemption stands unbounded but for the 40-character
    window. And the header-over-value layout prints the label on one line with its value
    in the label's column on the next ("HSN/SAC Code\\n998313").

    Across a line break the tail is gone and the window alone names nothing, so the
    exemption is bounded there — and bounded against the same number the backward read
    is bounded against. An Indian PIN is six digits, exactly the shape of a six-digit
    HSN, and "HSN Code\\nMumbai 400002" or "HSN/SAC   Qty   Rate\\nBengaluru 560001" puts
    one on the line below with only the window in between; filed as a confirmed HsnCd it
    is unquestionable downstream, and consuming it blanks the PIN out of the text stage 2
    reads the mandatory SellerDtls.Pin and BuyerDtls.Pin from.

    Two things must hold below the break. The number is the first thing printed on the
    line IMMEDIATELY below — no blank line in between and no word before it — so it is
    that row's first cell rather than something further along a row whose first cell is
    a word ("Order 4455 dated 01-Apr-2026"). And the label starts its own line, so the
    row below answers to the label and not to some other heading: "Sl Desc Qty Rate
    HSN/SAC\\n1001 Cotton fabric 10 1200 52081190" prints the label as the LAST of five
    column headings, and the number under the FIRST of them is the item's serial number.
    That row's real code is then missed, which is the miss-over-wrong-value trade this
    module makes everywhere else — and the one it already makes for the same Tally-style
    header when the label is read backwards.
    """
    gap = text[label.end() : start]
    if "\n" not in gap:
        return True
    if gap.count("\n") != 1 or gap[gap.index("\n") + 1 :].strip():
        return False
    return not text[_line_bounds(text, label.start())[0] : label.start()].strip()


def _hsn_codes_for_label(
    text: str, label: re.Match[str], matched: Iterable[Span]
) -> list[re.Match[str]]:
    """Every code this HSN/SAC label carries, in document order.

    The contract takes a code from a line that contains HSN or SAC as well as from
    the 40 characters after the label, so both sides of the label are read: "52081190
    - HSN of Cotton Fabric" and "Taxable value under 998313 (SAC)" print the code
    before the word, and "HSN/SAC: 998313 / 52081190" prints two after it.

    What ends the row of codes is a word, or a gap that is whitespace alone. A rate,
    a quantity or an amount is not a code just because the row also carries the word
    HSN. On an invoice the numeric columns are separated by their headings and units
    ("52081190  10 Mtr  1200  12000"), and where the row prints no unit — a services
    line, or OCR that dropped it ("SAC 998313 1 5000 5000") — only a column gap
    stands between the code and the money, which a single space cannot be told from.
    A further code is therefore read outward from the label only across punctuation,
    which is how a list of codes is printed ("998313 / 52081190", "52081190, 998313").
    A missed second code is a miss stage 2 still sees in the remaining text; a rate
    filed as HsnCd is a wrong value in a confirmed field AND blanks the money column
    out of stage 2's input.

    FORWARD, the number nearest the label is exempt from that, because the label's own
    tail is a word ("HSN/SAC Code : 52081190") and the label is what names it; it is
    bounded by the 40-character window instead, and the codes following it must be on
    its line. That exemption ends at a line break, where the tail ends too — see
    ``_the_label_names_the_number``, which is what keeps a PIN or an item's serial
    printed on the line below out of a confirmed HsnCd.

    BACKWARD there is no such exemption, because the contract names the number it
    would sweep in. An Indian PIN code is six digits, which is exactly the shape of a
    six-digit HSN, and an address line merged with the item-table header by OCR ("12
    Kalbadevi Road, Mumbai 400002    HSN/SAC   Qty   Rate") or sharing a row with an
    HSN cell ("Mumbai 400002  HSN 998313") puts one immediately before the label with
    nothing but spaces in between. Filed as a confirmed HsnCd it is unquestionable
    downstream — nothing can tell a PIN from an HSN by shape — and consuming it also
    blanks the PIN out of the text stage 2 reads the mandatory SellerDtls.Pin and
    BuyerDtls.Pin from. Requiring punctuation before the label refuses it, at the cost
    of a code printed before the label across a bare column gap ("Goods 52081190 HSN
    10 Mtr"): that is a miss stage 2 still sees in the remaining text, which is the
    trade this module makes everywhere else.

    ``matched`` is what this stage has already identified as something else — the
    GSTINs and the invoice number. A four-digit serial printed to the left of the
    label ("Invoice No 7788   HSN 8471") is reached by the backward read with only
    spaces in between, and it is already filed as DocDtls.No; taking it as a code as
    well would put a number the document names as the invoice number into HsnCd.
    """
    claimed = list(matched)

    def _already_something_else(number: re.Match[str]) -> bool:
        return any(span.start < number.end() and number.start() < span.end for span in claimed)

    found: list[re.Match[str]] = []
    line_start = _line_bounds(text, label.start())[0]
    boundary = label.start()
    nearest = True
    for number in reversed(list(HSN_DIGIT_RUN.finditer(text, line_start, label.start()))):
        if _a_word_intervenes(text, number.end(), boundary):
            break
        if not _a_list_separator_intervenes(text, number.end(), boundary):
            break
        if nearest and _a_money_column_names_it(text, line_start, number.start()):
            break
        nearest = False
        boundary = number.start()
        if _is_a_standalone_code(text, number) and not _already_something_else(number):
            found.append(number)
    found.reverse()
    boundary = label.end()
    first = True
    for number in HSN_DIGIT_RUN.finditer(text, label.end()):
        if first:
            if number.start() - label.end() > HSN_LABEL_WINDOW:
                break
            if _a_money_column_names_it(text, label.end(), number.start()):
                break
            if not _the_label_names_the_number(text, label, number.start()):
                break
            first = False
        elif (
            _a_word_intervenes(text, boundary, number.start())
            or number.start() > _line_bounds(text, boundary)[1]
            or not _a_list_separator_intervenes(text, boundary, number.start())
        ):
            break
        boundary = number.end()
        if _is_a_standalone_code(text, number) and not _already_something_else(number):
            found.append(number)
    return found


def _extract_hsn(text: str, matched: Iterable[Span]) -> tuple[tuple[RuleField, ...], list[Span]]:
    """The 4/6/8-digit codes printed with an HSN/SAC label, in document order.

    The returned codes are deduplicated by value; the consumed spans are not,
    so no occurrence of a matched code survives into stage 2's input.
    """
    fields: list[RuleField] = []
    spans: list[Span] = []
    seen: set[str] = set()
    matched = list(matched)
    for label in HSN_LABEL_PATTERN.finditer(text):
        numbers = _hsn_codes_for_label(text, label, matched)
        if not numbers:
            continue
        spans.append(Span(label.start(), label.end()))
        for number in numbers:
            spans.append(Span(number.start(), number.end()))
            if number.group(0) in seen:
                continue
            seen.add(number.group(0))
            fields.append(RuleField(number.group(0), Span(number.start(), number.end()), "regex"))
    return tuple(fields), spans


# --- entry point ------------------------------------------------------------


def extract_rules(text: str) -> RuleExtraction:
    """Extract the structurally confirmable fields from an invoice's text."""
    warnings: list[ExtractionWarning] = []
    consumed: list[Span] = []

    candidates = list(GSTIN_PATTERN.finditer(text))
    consumed.extend(Span(match.start(), match.end()) for match in candidates)
    valid, gstin_warnings = _gstin_candidates(text)
    warnings.extend(gstin_warnings)

    seller_gstin, buyer_gstin, party_warnings = _resolve_parties(text, valid)
    warnings.extend(party_warnings)

    seller_stcd, seller_stcd_warnings = _derive_stcd(seller_gstin, "SellerDtls.Stcd")
    buyer_stcd, buyer_stcd_warnings = _derive_stcd(buyer_gstin, "BuyerDtls.Stcd")
    warnings.extend(seller_stcd_warnings)
    warnings.extend(buyer_stcd_warnings)

    doc_no, doc_no_warnings = _extract_doc_no(
        text, [Span(match.start(), match.end()) for match in candidates]
    )
    warnings.extend(doc_no_warnings)
    if doc_no is not None:
        consumed.append(doc_no.span)

    # The invoice number is resolved first and its span passed on: the financial year in
    # a Tally-style serial ("SFPL/24-25/0731") is date-shaped, and taking it as the date
    # would file a wrong mandatory value and lose the date the document actually prints.
    doc_date, doc_date_warnings, consume_date = _extract_doc_date(
        text, None if doc_no is None else doc_no.span
    )
    warnings.extend(doc_date_warnings)
    if doc_date is not None and consume_date:
        consumed.append(doc_date.span)

    hsn_codes, hsn_spans = _extract_hsn(
        text,
        [Span(match.start(), match.end()) for match in candidates]
        + ([] if doc_no is None else [doc_no.span]),
    )
    consumed.extend(hsn_spans)

    return RuleExtraction(
        seller_gstin=seller_gstin,
        buyer_gstin=buyer_gstin,
        seller_stcd=seller_stcd,
        buyer_stcd=buyer_stcd,
        doc_no=doc_no,
        doc_date=doc_date,
        hsn_codes=hsn_codes,
        consumed=_merge_spans(consumed, len(text)),
        warnings=tuple(warnings),
    )
