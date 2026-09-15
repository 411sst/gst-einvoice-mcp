"""Stage 2 extraction: an LLM structures the text stage 1 could not confirm.

The model may STRUCTURE text; it may never INVENT values. That guarantee is
enforced twice, not once:

1. In the prompt (:func:`build_prompt`), stated explicitly.
2. In code, after the response returns: every value is verified against the
   source text it was supposed to come from. A value with no source in the
   input is replaced with ``None`` plus a warning, however plausible it looks.
   A hallucinated line item that survives into an INV-01 payload is exactly the
   confidently-wrong extraction this tool exists to prevent.

The source text for step 2 is the remaining text *plus* the confirmed fields
passed in as context, because a correct model may legitimately echo a field
stage 1 already established.

Documented deviations from the written contract, for build 3's LIMITATIONS.md:

1. The contract types the JSON string fields as ``string|null``, and this module
   additionally accepts an int or a float there (a model commonly returns the
   code-like ``HsnCd`` as ``7326`` rather than ``"7326"``) and renders it as
   text. Grounding still applies afterwards, so it widens what can be read,
   never what can be invented.
2. The contract prescribes two warning messages (ungrounded, and null). This
   module emits five for a dropped value, because "it did not appear in the
   source text" is the wrong diagnosis for four of the five ways one is lost:
   a numeric field holding no plain number, a string field holding a list or an
   object, a figure the document prints as a credit, and digits that occur only
   inside a confirmed identifier are all printed on the page or in the confirmed
   context. The outcome is always the contract's -- ``None`` plus exactly one
   ``check="extract_llm"`` warning quoting the value, naming the path, and
   saying it was dropped rather than kept.
3. A figure the document prints in accounting brackets (``(250.00)``), with a
   ``CR`` suffix, behind a Unicode minus (``−250.00``, U+2212, what a typeset
   PDF emits) or with a trailing minus (``250.00-``, the Tally/SAP form) is
   negative, so it grounds ``-250.00`` and never ``+250.00``. The contract's
   rule as written ("some numeric token parses to the same number") would ground
   the positive reading and silently turn a discount into a charge. A Unicode
   minus is read as a sign only when the character before it is not itself part
   of a number, so ``1500.00 − 250.00`` stays a subtraction.
   A TRAILING minus is read as a sign only when it ENDS THE LINE -- nothing may
   follow it but spaces and the line break. So ``Discount   250.00-`` is a credit
   of 250.00, while in a multi-column layout whose credit column is followed by
   another column (``Discount   250.00-   1,250.00``, how Tally prints a running
   balance beside the figure) the minus is not read as a sign at all and the
   printed credit is read as a POSITIVE CHARGE. See the consequences below: that
   limitation is kept deliberately rather than widened.
   A **detached ASCII hyphen is not read as a sign at all**. On an Indian
   invoice it is far more often a label separator (``Freight - 200.00``,
   ``CGST @ 9% - 135.00``) than a printed minus, and reading it as one does both
   harms at once: it grounds the sign-inverted ``-200.00`` with clean
   provenance, and it drops the correct ``+200.00`` the model actually read.
   The contract's own token pattern (``[-+]?[\\d,]*\\.?\\d+``) grounds ``+200.00``
   there, and a minus a document really means is attached to its digits
   (``-200.00``), which that pattern already carries.
4. The null-field warning is ``severity="info"`` for the four fields INV-01 does
   not require (``Discount``, ``CesAmt``, ``CesVal``, ``RndOffAmt``), whose
   absence is a fact about the document rather than a gap in the reading.
   Everything else, and every dropped value, stays ``severity="warning"``.
5. A kept value means "this text is printed on the document", never "this text
   belongs where the model put it" -- grounding is value-presence only, and the
   uncorroborated-assignment risk that creates runs through the party blocks and
   the row figures too (see the consequences below). An item's ``HsnCd`` is the
   one assignment this stage can warn about explicitly: stage 1 confirms the
   codes and strips them, so a row's code is grounded by that confirmed list
   alone and which code belongs to which row is the model's judgement, which the
   grounding check structurally cannot corroborate. Every such row therefore
   carries a warning saying so, the way stage 1 always warns when a party was
   assigned by position. The warning fires unless the code is printed as a
   standalone token in the text left to this stage -- a substring check would
   let the PIN ``400021`` silence the warning on an ``HsnCd`` of ``4000``.
6. The contract makes the grounding source the remaining text plus every
   confirmed value. That holds for strings, which must appear whole. Numbers are
   grounded against the remaining text ALONE, because tokenising digit runs
   *inside* a confirmed identifier turns it into a supply of numeric tokens:
   ``STP/2026/0417`` would otherwise ground a fabricated ``UnitPrice`` of
   ``417.00`` and ``27AAPFU0939F1ZV`` a ``Qty`` of ``939``, kept silently with
   ``"source": "llm"``. Every field stage 1 confirms is an identifier or a
   classification code rather than a figure: the party GSTINs, their state codes
   (``27``), the invoice number, the invoice date and the HSN codes (``7326``).
   Which field it is decides this, never whether its text parses as a number:
   Indian invoice numbers are commonly bare digits (``1042``, ``25``, ``0417``),
   and a parseability test let an amount equal to the invoice number ground
   against the invoice number. That is the confidently-wrong extraction this
   stage exists to prevent, so numeric grounding is narrowed rather than
   widened. A value lost to that narrowing is reported as such -- its digits are
   printed, inside an identifier that is not a figure -- and not as a value that
   never appeared.
7. A kept string field carries the DOCUMENT's spelling, not the model's.
   Grounding tolerates re-casing and re-spacing, so ``nos`` grounds against a
   printed ``NOS``; keeping the model's rendering would put a value in the
   payload that is not the one on the page, and GSTN enumerates ``Unit`` codes,
   so that is a real defect in a submitted payload. The model's own value is
   used only when the matched slice cannot be located. ``confidence_lookup`` is
   asked about the same matched source text either way, so a shaky OCR read
   still surfaces.
   WHICH printing of the value is matched decides which read's confidence that
   is, so the match is anchored: the value as a standalone token in the model's
   own case first, then as a standalone token in any case, and only then the
   loose unanchored search a digit-only field needs. An unanchored search takes
   whatever comes first in the document, which put ``Nos`` from ``Nosy
   Distributors`` in the payload as a ``Unit`` and recorded that word's
   confidence. Only a slice printed as a token of its own is treated as the
   document's SPELLING: a value that grounds solely inside a longer word is
   kept in the model's own rendering, because re-casing it to that fragment
   manufactures a value that is neither the model's nor any token on the page
   (an uncorroborated ``NOS`` became ``Nos``, and GSTN's UQC list has no such
   code). The lookup is still asked about the fragment, since that is the only
   text the value was grounded against.
   When the winning tier matches more than once -- the same name in a clean
   header and in a smudged Bill To block -- every printing is looked up and the
   LOWEST confidence is recorded, as ``ocr.confidence_for_span`` takes the
   minimum over a span's words. That holds of a FIGURE exactly as it holds of a
   string: one amount is routinely printed twice and in two forms, as a row's
   ``1500.00`` and a totals line's ``1,500.00``, so every matching token is
   looked up too. Recording the first, or the highest, would let a clean
   printing launder a shaky one, which is the one thing this number exists to
   prevent.
   The document's SPELLING is kept; its LAYOUT is not. The matched slice carries
   whatever line break or column padding the page put inside the value, and
   INV-01's ``Addr1``, ``PrdDesc`` and ``LglNm`` are single-line, length-capped
   fields: an ``Addr1`` holding a raw newline, or a ``PrdDesc`` reading
   ``Steel   Bracket 12mm``, is not a value an accountant can submit. The slice's
   whitespace runs are therefore collapsed to one space for the payload, which
   changes no word and no letter case. The lookup is still asked about the raw
   slice, because that is the text as the page prints it.
8. A kept ``Pin`` carries a warning unless its digits are printed as a number of
   their own in the text left to this stage. String grounding is substring-based
   by contract, so a digit-only field grounds off any longer digit run
   (``110001`` inside an account number ``1100015678``) -- and the grounding
   source includes the confirmed context, so it also grounds off a confirmed
   identifier alone (a ``Pin`` of ``27`` off the confirmed state code, a ``Pin``
   of ``400021`` off an invoice number that happens to be those digits).
   ``HsnCd`` already says so (deviation 5); ``Pin`` is the other digit-only
   string field and is mandatory in INV-01, so it says so too rather than being
   kept silently with ``"source": "llm"``. Both checks therefore ask the same
   question of the same text -- the remaining text, not the whole grounding text
   -- because a confirmed identifier corroborates that those digits are an
   identifier, never that they are this document's PIN code. The value is kept
   -- it is grounded exactly as the contract defines grounding -- and the warning
   says the corroboration is partial.
9. A value shorter than :data:`_MIN_GROUNDED_LENGTH` characters is grounded when,
   and only when, the source prints it as a STANDALONE TOKEN; it is then kept
   with a warning saying the corroboration is weak. The floor exists to stop a
   one-character value matching trivially anywhere in the text, and requiring a
   token of its own serves exactly that purpose -- it is the same question
   ``HsnCd`` and ``Pin`` already ask (deviations 5 and 8). Refusing every short
   value instead served it at a price the product cannot pay: ``Item.SlNo`` is
   mandatory in INV-01 and the great majority of Indian invoices number their
   rows ``1``-``9``, so a flat floor made Part D report ``ItemList[0].SlNo``
   missing and emit no payload at all for the ordinary case. A one-character
   match still proves very little, which is why the kept value carries a warning
   saying so -- but weak corroboration REPORTED is what this product is for, and
   destroying a correctly read value to avoid reporting it is the trade
   deviation 3 refuses. A short value the source does not print as a token of
   its own is ungrounded and dropped like any other value that cannot be found.

Consequences of the contract's grounding rules, not deviations from them,
recorded here so build 3 can publish them as known limitations:

* Grounding corroborates that a value is PRINTED, never which party or which
  row it belongs to. This is the largest of these limitations. A response that
  swaps the seller and buyer blocks, or that pulls row 2's figures onto row 1,
  passes every check: each value really is printed on the page, so all of them
  are kept with clean ``{"source": "llm", "ocr_confidence": null}`` provenance
  and no warning at all. ``HsnCd`` is not a special case of carelessness
  elsewhere -- it warns (deviation 5) only because stage 1 strips the codes,
  leaving nothing on the row to check one against; the identical risk applies
  silently to ``SellerDtls``/``BuyerDtls`` ``LglNm``, ``Addr1``, ``Loc`` and
  ``Pin``, and to every per-row figure. So no warning on ``SellerDtls.LglNm``
  means "this name is printed on the document", not "this name was corroborated
  as the seller's", and a swapped block pairs stage 1's regex-confirmed
  ``SellerDtls.Gstin`` with the buyer's name and address. Stage 1 warns loudly
  whenever a party was assigned by position; this stage cannot, because presence
  is all the contract gives it to check. Party blocks and per-row figures
  therefore need confirming against the invoice by eye.
* String grounding is substring-based, as specified, so a short value of ANY
  string field is kept when its characters occur inside something longer. A
  digit-only field grounds off a longer number: ``HsnCd`` ``"4000"`` is kept
  when the document prints the PIN ``400021``. The same value routed as a
  number is refused, because numeric grounding is token-based and ``400021`` is
  one token -- but that holds only of digits inside a longer NUMBER, not of
  digits inside a WORD (see the next point). The two digit-only string fields,
  ``HsnCd`` and ``Pin``, both warn when this happens (deviations 5 and 8).
  The short ALPHABETIC fields have no such warning: a ``Unit`` of ``NOS`` is
  kept when the page prints only ``Nosy Distributors``, and a ``SlNo`` or a
  two-word ``Loc`` can ground inside a longer word the same way. Such a value
  now enters the payload in the model's own spelling rather than the fragment's
  (deviation 7), so it is at least never rewritten into a code GSTN does not
  list -- but it is kept with ``{"source": "llm"}`` and the ``ocr_confidence``
  of the word it was found inside, and no warning. A ``Unit`` that appears
  nowhere on the page as a unit therefore needs confirming against the invoice
  by eye, exactly like the party blocks and the per-row figures.
* Numeric grounding is token-based, and the contract's token pattern
  (``[-+]?[\\d,]*\\.?\\d+``) finds a digit run wherever it sits, including glued
  to letters. A fabricated figure that happens to match the digits inside a word
  is therefore kept with clean ``{"source": "llm", "ocr_confidence": null}``
  provenance and no warning: ``Steel Bracket 12mm`` grounds a ``Qty`` of ``12``,
  ``Cable 2.5sqmm`` a ``UnitPrice`` of ``2.5``, ``Model MS150`` an amount of
  ``150``, ``A4 Paper`` a ``4`` and a vehicle number ``MH12AB1234`` both ``12``
  and ``1234``. Product dimensions, SKU and model codes, paper sizes and vehicle
  numbers are all over Indian invoices, so this is not a rare shape. The pattern
  is left exactly as the contract writes it rather than being narrowed to digits
  with no letter beside them, because that narrowing has a cost in the other
  direction that is worse: ``Rs1500.00`` and ``INR1500.00`` are ordinary
  printings of a real figure, and refusing them would DROP a correctly read
  amount and report it as one that never appeared -- inventing nothing, but
  destroying something, which is the same trade the detached-hyphen rule
  (deviation 3) refuses. Figures whose digits also occur inside a word on the
  page therefore need confirming against the invoice by eye, exactly like the
  party blocks and the per-row figures above.
* A parenthesised token that is not money at all -- the area code in
  ``Phone (022) 2222 3333``, a bracketed ``(9)`` -- is read as an accounting
  credit, so a correct positive figure the model actually read is dropped and
  the warning calls it a credit, which of that token is not true. The rule is
  kept as is because the alternative is worse: requiring a decimal point or a
  thousands separator before reading brackets as a sign would make ``(1500)``
  ground ``+1500`` instead of ``-1500``, turning a printed credit into a charge
  with clean provenance. This way errs towards dropping a figure, never towards
  inventing or inverting one, and it only bites when the figure is printed
  nowhere else on the page.
* A TRAILING MINUS carries its sign only at the END OF A LINE, so a Tally-style
  column layout reads a printed credit as a positive charge. ``250.00-`` alone
  at the end of its line grounds ``-250.00`` and never ``+250.00``; the same
  figure printed as one column among several -- ``Discount   250.00-
  1,250.00`` -- is read as ``+250.00``, kept with clean
  ``{"source": "llm", "ocr_confidence": null}`` provenance and no warning, and
  the credit is silently inverted into a charge. The pattern is deliberately NOT
  widened to accept a minus followed by a column gap, because that would read
  the minus in ``150.00-200.00`` and ``1500.00 - 250.00`` as the sign of the
  figure BEFORE it and invert a charge into a credit -- a new sign error in the
  other direction rather than a fix for this one, and on layouts (a range, a
  subtraction, a hyphen-separated label) at least as common as the multi-column
  credit. A credit column that is not the last column on its line therefore
  needs confirming against the invoice by eye, like the party blocks and the
  per-row figures above.
* A one-character value is kept on the strength of a standalone-token match
  (deviation 9), and a single character printed as a token of its own is weak
  evidence of anything: the ``1`` in an invoice's Sl column is the same character
  as the ``1`` in a quantity column, a rate band or a page number, so a wrong
  ``SlNo`` of ``1`` is corroborated by a page that prints a ``1`` anywhere at
  all. The value is reported as weakly corroborated rather than dropped, so row
  numbering needs confirming against the invoice by eye, exactly like the party
  blocks and the per-row figures above.

Nothing here constructs a client or reads an API key: ``client`` is a required
keyword argument, so the stage is fully exercisable with a fake. Only
:func:`make_client` touches ``GROQ_API_KEY``, and nothing in this module calls
it.
"""

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gst_einvoice.schema import ExtractionWarning

#: The model stage 2 calls when the caller names none. This is the model the live
#: validation actually ran against -- twelve invoices, 136 hand-transcribed value
#: checks, zero mismatches -- so it is the only one with evidence behind it. The
#: previous default, ``llama-3.3-70b-versatile``, had been retired by Groq and
#: returned HTTP 404, which broke the first run for anyone who did not know to
#: override it. ``GST_MCP_MODEL`` overrides this for a deployment, and a deployment
#: should set it: a hard-coded model identifier expires silently.
DEFAULT_MODEL = "openai/gpt-oss-120b"

#: Keys of :attr:`LlmExtraction.totals`, always all present (``None`` when unread).
TOTAL_KEYS: tuple[str, ...] = (
    "AssVal",
    "CgstVal",
    "SgstVal",
    "IgstVal",
    "CesVal",
    "TotInvVal",
    "RndOffAmt",
)

_CHECK = "extract_llm"
_ENV_VAR = "GROQ_API_KEY"
# Money compared to the paisa: 0.005 keeps 1500.00 and 1500.004 the same number.
_ABS_TOL = 0.005
# A string shorter than this matches almost any document by accident, so it
# grounds only when the source prints it as a token of its own -- and it then
# says so in a warning. Refusing it outright cost every invoice numbered 1-9 its
# mandatory Item.SlNo, and so its whole payload.
_MIN_GROUNDED_LENGTH = 2
# Indian grouping (1,23,456.78) included; commas are stripped before parsing.
_NUMBER_TOKEN = re.compile(r"[-+]?[\d,]*\.?\d+")
# The only characters a numeric string may carry around its number without
# changing what the figure means. Accounting brackets "(250.00)", a detached
# minus "- 250.00" and a "CR"/"DR" suffix all carry a sign, and reading such a
# value as its bare digits would silently flip a credit into a charge.
_SAFE_NUMBER_DECORATION = frozenset(" \t\n\r\f\v₹$%")
# The same sign, but carried by the SOURCE text rather than the model's answer:
# a figure printed in accounting brackets, suffixed "CR", behind a Unicode minus,
# or with a trailing minus is negative even though its digits are printed
# positive. ("DR" is a debit, which is what the bare digits already mean, so it
# is left alone.)
_CREDIT_OPEN = re.compile(r"\([ \t]*$")
_CREDIT_CLOSE = re.compile(r"^[ \t]*\)")
_CREDIT_SUFFIX = re.compile(r"^[ \t]*CR\b", re.IGNORECASE)
# "250.00-" (Tally/SAP), and only when the minus ENDS THE LINE: nothing may
# follow it but spaces and the line break. In "150.00-200.00" that minus belongs
# to the next figure, not to this one, and in a column layout that prints further
# columns after the credit one the sign is not read at all (deviation 3).
_CREDIT_TRAILING = re.compile(r"^-(?=[ \t]*(?:$|[\r\n]))")
# "−250.00" (U+2212, what a typeset PDF emits). Deliberately NOT the ASCII
# hyphen: detached, that character is a label separator far more often than a
# minus ("Freight - 200.00", "CGST @ 9% - 135.00"), and reading it as a sign
# both grounds the inverted figure and drops the correct positive one. An ASCII
# minus that a document really means is attached to its digits ("-200.00"), and
# _NUMBER_TOKEN already carries that sign. The sign must open the line or follow
# something that is not itself part of a number, so "1500.00 − 250.00" stays a
# subtraction.
_CREDIT_MINUS = re.compile(r"(?:(?:^|[\r\n])[ \t]*|[^\d.,\s][ \t]*)(?P<sign>−[ \t]*)$")
# INV-01 fields with a schema default (schema.py: Item.Discount, Item.CesAmt,
# ValDtls.CesVal, ValDtls.RndOffAmt). A document that prints no discount and no
# cess is an ordinary document, so their absence is reported as `info` and does
# not compete for attention with a mandatory field that could not be read.
_OPTIONAL_FIELDS = frozenset({"Discount", "CesAmt", "CesVal", "RndOffAmt"})

# (JSON key, is the field numeric), in the order fields are resolved and reported.
_ITEM_FIELDS: tuple[tuple[str, bool], ...] = (
    ("SlNo", False),
    ("PrdDesc", False),
    ("HsnCd", False),
    ("Qty", True),
    ("Unit", False),
    ("UnitPrice", True),
    ("TotAmt", True),
    ("Discount", True),
    ("AssAmt", True),
    ("GstRt", True),
    ("CgstAmt", True),
    ("SgstAmt", True),
    ("IgstAmt", True),
    ("CesAmt", True),
    ("TotItemVal", True),
)

# (attribute of LlmExtraction, response block, key in that block, INV-01 path).
_PARTY_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("seller_name", "seller", "LglNm", "SellerDtls.LglNm"),
    ("seller_addr1", "seller", "Addr1", "SellerDtls.Addr1"),
    ("seller_loc", "seller", "Loc", "SellerDtls.Loc"),
    ("seller_pin", "seller", "Pin", "SellerDtls.Pin"),
    ("buyer_name", "buyer", "LglNm", "BuyerDtls.LglNm"),
    ("buyer_addr1", "buyer", "Addr1", "BuyerDtls.Addr1"),
    ("buyer_loc", "buyer", "Loc", "BuyerDtls.Loc"),
    ("buyer_pin", "buyer", "Pin", "BuyerDtls.Pin"),
)


@dataclass(frozen=True)
class LlmItem:
    """One ``ItemList`` row as the model structured it. Every field is optional."""

    SlNo: str | None = None
    PrdDesc: str | None = None
    HsnCd: str | None = None
    Qty: float | None = None
    Unit: str | None = None
    UnitPrice: float | None = None
    TotAmt: float | None = None
    Discount: float | None = None
    AssAmt: float | None = None
    GstRt: float | None = None
    CgstAmt: float | None = None
    SgstAmt: float | None = None
    IgstAmt: float | None = None
    CesAmt: float | None = None
    TotItemVal: float | None = None


@dataclass(frozen=True)
class LlmExtraction:
    """Everything stage 2 produced, with its warnings and provenance."""

    items: tuple[LlmItem, ...]
    seller_name: str | None
    seller_addr1: str | None
    seller_loc: str | None
    seller_pin: str | None
    buyer_name: str | None
    buyer_addr1: str | None
    buyer_loc: str | None
    buyer_pin: str | None
    totals: dict[str, float | None]
    warnings: tuple[ExtractionWarning, ...]
    provenance: dict[str, dict]


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_JSON_SHAPE = """{
  "items": [
    {"SlNo": string|null, "PrdDesc": string|null, "HsnCd": string|null,
     "Qty": number|null, "Unit": string|null, "UnitPrice": number|null,
     "TotAmt": number|null, "Discount": number|null, "AssAmt": number|null,
     "GstRt": number|null, "CgstAmt": number|null, "SgstAmt": number|null,
     "IgstAmt": number|null, "CesAmt": number|null, "TotItemVal": number|null}
  ],
  "seller": {"LglNm": string|null, "Addr1": string|null, "Loc": string|null, "Pin": string|null},
  "buyer": {"LglNm": string|null, "Addr1": string|null, "Loc": string|null, "Pin": string|null},
  "totals": {"AssVal": number|null, "CgstVal": number|null, "SgstVal": number|null,
             "IgstVal": number|null, "CesVal": number|null, "TotInvVal": number|null,
             "RndOffAmt": number|null}
}"""

_HARD_CONSTRAINT = """HARD CONSTRAINT - you may STRUCTURE text, you may never INVENT values:
- Every value you output must already appear in the INVOICE TEXT below.
- Never infer, complete, correct, translate or invent a value, and never
  calculate one that is not printed.
- Any field you cannot find in the INVOICE TEXT must be null. Do not guess, and
  do not fill a field with a plausible value.
- Every value is checked against the INVOICE TEXT after you answer. A value that
  is not there is dropped and reported, so guessing only destroys information."""

# No "do not repeat them": the JSON shape below asks for each item's HsnCd, and
# the HSN codes are exactly what stage 1 confirms and strips out of the invoice
# text, so a model told not to restate them would return HsnCd null on every row
# -- a mandatory INV-01 field, obtainable from nowhere else.
_CONFIRMED_RULE = """ALREADY CONFIRMED - these fields were established deterministically before you
were called. They are settled: do not re-derive, re-check or contradict them.
They were removed from the INVOICE TEXT, so where the JSON shape below asks for
one of them - each item's HsnCd - copy the confirmed value onto the row it
belongs to, exactly as written here."""


def _confirmed_values(confirmed: dict) -> tuple[str, ...]:
    """Every confirmed value as text, flattening one level of list/tuple, skipping None."""
    values: list[str] = []
    for raw in (confirmed or {}).values():
        candidates = raw if isinstance(raw, (list, tuple)) else [raw]
        values.extend(str(item) for item in candidates if item is not None)
    return tuple(values)


def build_prompt(remaining: str, confirmed: dict) -> str:
    """The single user message: the hard constraint, the confirmed context, the text."""
    lines: list[str] = []
    for key, raw in (confirmed or {}).items():
        if raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            raw = ", ".join(str(item) for item in raw)
        lines.append(f"- {key}: {raw}")
    confirmed_block = "\n".join(lines) if lines else "- (nothing was confirmed by stage 1)"
    return (
        "You are reading one Indian GST tax invoice and returning its line-item "
        "table and party/total fields as JSON.\n\n"
        f"{_HARD_CONSTRAINT}\n\n"
        f"{_CONFIRMED_RULE}\n{confirmed_block}\n\n"
        "Return ONLY a JSON object with exactly this shape:\n"
        f"{_JSON_SHAPE}\n\n"
        "Rules for the JSON:\n"
        "- One object in \"items\" per line item printed in the table, in printed "
        "order. Never merge, split or add rows.\n"
        "- Write numbers as JSON numbers with the digits as printed, without "
        "currency symbols, thousands separators or percent signs.\n"
        "- Copy strings as printed; do not expand abbreviations or fix spelling.\n"
        "- Use null for anything the INVOICE TEXT does not contain.\n\n"
        "INVOICE TEXT (the only source you may use):\n"
        '"""\n'
        f"{remaining}\n"
        '"""'
    )


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


def _normalise(text: str) -> str:
    """Collapse every whitespace run to one space and casefold."""
    return " ".join(str(text).split()).casefold()


def _is_standalone_token(value: str, normalised_source: str) -> bool:
    """True when ``value`` is printed in the source as a token, not inside a longer one."""
    return re.search(rf"\b{re.escape(_normalise(value))}\b", normalised_source) is not None


def _look_behind(text: str, start: int) -> str:
    """The only text in front of ``start`` a credit notation can occupy.

    Rebuilding and rescanning the whole prefix for every numeric token made
    grounding quadratic in the length of the document -- a ten-page invoice spent
    seconds in :func:`_numeric_tokens` alone. Nothing further back than the
    token's own line can ever match: both look-behind patterns are anchored to
    ``$``, and :data:`_CREDIT_MINUS`'s one otherwise unbounded branch is anchored
    to a line start. ``$`` also matches immediately before a trailing newline, so
    a token that OPENS a line can still be reached by a notation at the end of
    the line above (``Discount (\\n250.00)``); that line is included and nothing
    before it can be, since ``$`` never matches further back than one newline.
    """
    line_start = text.rfind("\n", 0, start) + 1
    if line_start == start and line_start:
        line_start = text.rfind("\n", 0, start - 1) + 1
    return text[line_start:start]


def _credit_form(text: str, start: int, end: int) -> str | None:
    """The figure as printed when its decoration negates it, else None.

    ``Discount (250.00)``, ``250.00 CR``, ``−250.00`` (U+2212) and ``250.00-``
    all print a credit: the digits are positive, the notation carries the minus.
    A token that already carries its own sign is left alone -- its digits mean
    what they say. A detached ASCII hyphen (``Freight - 200.00``) is not a
    notation: it is a label separator at least as often as a minus, so the
    figure after it is read exactly as printed.

    The trailing minus is read only when it ENDS the line, so a credit column
    followed by another column reads as a charge (deviation 3).
    """
    printed = text[start:end]
    if printed[:1] in ("-", "+"):
        return None
    behind = _look_behind(text, start)
    if _CREDIT_OPEN.search(behind) and _CREDIT_CLOSE.match(text[end:]):
        return f"({printed})"
    suffix = _CREDIT_SUFFIX.match(text[end:])
    if suffix:
        return f"{printed}{suffix.group(0)}"
    trailing = _CREDIT_TRAILING.match(text[end:])
    if trailing:
        return f"{printed}{trailing.group(0)}"
    minus = _CREDIT_MINUS.search(behind)
    return f"{minus.group('sign')}{printed}" if minus else None


def _numeric_tokens(text: str) -> tuple[tuple[str, float, str | None], ...]:
    """Every numeric token as (raw token, parsed value, printed credit form or None)."""
    tokens: list[tuple[str, float, str | None]] = []
    for match in _NUMBER_TOKEN.finditer(text):
        raw = match.group(0)
        parsed = float(raw.replace(",", ""))
        tokens.append((raw, parsed, _credit_form(text, match.start(), match.end())))
    return tuple(tokens)


def _source_slices(value: str, raw_source: str) -> tuple[tuple[str, ...], bool]:
    """Every occurrence of ``value`` in ``raw_source``, from the best-anchored tier.

    Three tiers, most trustworthy first: the value printed as a standalone token
    in exactly the case the model used; the same token in any case; then the
    loose unanchored match, which is the only one that can land inside a longer
    word or number. The tiers exist because an unanchored search takes whatever
    comes FIRST in the document: with ``Nosy Distributors`` printed above the
    item table, a ``Unit`` of ``NOS`` matched the middle of that word, so the
    payload carried ``Nos`` and the confidence recorded belonged to a word that
    is not the unit at all. The loose tier is kept as the last resort because a
    digit-only field legitimately grounds inside a longer number by contract
    (``HsnCd`` and ``Pin`` both warn when it happens).

    All the occurrences of the winning tier are returned, in document order,
    with a flag saying whether that tier anchored the value as a token of its
    own. The first occurrence is the spelling that enters the payload when it
    did; the caller asks the lookup about every one of them, because the same
    string printed twice can be read at two different confidences and the lower
    one has to survive.
    """
    tokens = str(value).split()
    if not tokens:
        return (), False
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    standalone = rf"(?<!\w){pattern}(?!\w)"
    tiers = (
        (standalone, 0, True),
        (standalone, re.IGNORECASE, True),
        (pattern, re.IGNORECASE, False),
    )
    for regex, flags, anchored in tiers:
        matches = tuple(match.group(0) for match in re.finditer(regex, raw_source, flags))
        if matches:
            return matches, anchored
    return (), False


def _match_strings(
    value: str, raw_source: str, normalised_source: str
) -> tuple[tuple[str, ...], bool]:
    """The source occurrences grounding ``value``, or () when the source lacks it.

    The text returned is the source's own spelling, not the model's: grounding
    tolerates re-casing and re-spacing, and the caller both asks
    ``confidence_lookup`` about this text and puts it in the payload. Handing the
    lookup the model's rendering would lose the OCR confidence of the words
    actually read, which is the laundering Part C exists to prevent; putting the
    model's rendering in the payload would submit a value the document does not
    print, and GSTN enumerates the ``Unit`` codes.

    These are the slices exactly as printed, line breaks and column padding
    included, because that is what the lookup has to find on the page. The
    caller collapses the whitespace of the one it keeps before putting it in the
    payload.

    The second element says whether the winning tier anchored the value as a
    token of its own. It did not when the value grounds only inside a longer
    word, and the source's spelling is then a fragment of a different word --
    ``NOS`` inside ``Nosy Distributors`` -- so the caller keeps the model's
    rendering instead of writing that fragment into the payload.

    A value shorter than :data:`_MIN_GROUNDED_LENGTH` must be printed as a
    STANDALONE TOKEN: substring presence means nothing of one character, which
    is what the floor was for, and a token of its own is exactly that test
    (deviation 9). The caller keeps such a value with a warning saying the
    corroboration is weak.
    """
    normalised = _normalise(value)
    if not normalised:
        return (), False
    if len(normalised) < _MIN_GROUNDED_LENGTH:
        if not _is_standalone_token(normalised, normalised_source):
            return (), False
    elif normalised not in normalised_source:
        return (), False
    # The normalised comparison above is the authority on grounding; the slices
    # only recover the source spelling, so fall back when they cannot be located.
    matches, anchored = _source_slices(value, raw_source)
    return (matches, anchored) if matches else ((value,), False)


def _match_numbers(
    value: float, tokens: tuple[tuple[str, float, str | None], ...]
) -> tuple[str, ...]:
    """Every source figure equal to ``value``, as printed, in document order.

    A token the document prints as a credit -- ``(250.00)``, ``0.40 CR`` -- is
    negative however positive its digits look, so it grounds ``-250.00`` and
    never ``+250.00``. What comes back is the printed form, brackets and all,
    because that is the text ``confidence_lookup`` has to find on the page.

    ALL the matching printings are returned, not just the first, for the reason
    :func:`_source_slices` returns all of a string's: one figure is routinely
    printed more than once and in more than one form -- the row amount
    ``1500.00`` and the totals line ``1,500.00`` -- and the two printings can be
    read at quite different confidences. Handing the caller only the first would
    record whichever came first in the document, so a clean row could launder
    the smudged totals line the value was actually taken from.
    """
    printings: list[str] = []
    for raw, parsed, credit in tokens:
        signed = -parsed if credit is not None else parsed
        if math.isclose(signed, value, rel_tol=0, abs_tol=_ABS_TOL):
            printings.append(credit if credit is not None else raw)
    return tuple(printings)


def _credit_conflict(value: float, tokens: tuple[tuple[str, float, str | None], ...]) -> str | None:
    """The printed credit whose digits equal a positive ``value``, else None.

    This is the difference between "the model invented this figure" and "the
    model read this figure and dropped its sign". Only the second is true when
    the page prints ``(250.00)`` and the model answers ``250.00``.
    """
    if value <= 0:
        return None
    for _raw, parsed, credit in tokens:
        if credit is not None and math.isclose(parsed, value, rel_tol=0, abs_tol=_ABS_TOL):
            return credit
    return None


def _coerce_str(raw: Any) -> str | None:
    """Model output for a string field as text, or None when it is not a scalar."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, float):
        return str(int(raw)) if raw.is_integer() else str(raw)
    return None


def _coerce_number(raw: Any) -> float | None:
    """Model output for a numeric field as a float, or None when it holds no plain number.

    A string is accepted only when it holds exactly one numeric token and nothing
    around it but whitespace, a currency symbol or a percent sign. Anything else
    is refused rather than read as its bare digits: Indian invoices print credits
    in accounting brackets, so keeping ``"(250.00)"`` as ``250.00`` would invert
    the sign of a discount or a round-off and feed the wrong AssVal and
    TotInvVal onward with no warning at all.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        matches = list(_NUMBER_TOKEN.finditer(raw))
        if len(matches) != 1:
            return None
        match = matches[0]
        residue = raw[: match.start()] + raw[match.end() :]
        if any(char not in _SAFE_NUMBER_DECORATION for char in residue):
            return None
        return float(match.group(0).replace(",", ""))
    return None


def is_grounded(value: Any, source_text: str) -> bool:
    """True when ``value`` demonstrably comes from ``source_text``.

    Strings must appear as a substring once whitespace is collapsed and case is
    folded; a string shorter than two characters must appear as a token of its
    own, since a substring match means nothing of one character. Numbers must
    equal some numeric token in the source to within half a paisa, and a token
    the document prints as a credit grounds only its negative reading.
    """
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(_match_strings(value, source_text, _normalise(source_text))[0])
    if isinstance(value, (int, float)):
        return bool(_match_numbers(float(value), _numeric_tokens(source_text)))
    return False


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Source:
    """The grounding source plus the warnings and provenance being accumulated."""

    raw: str
    normalised: str
    #: ``remaining_text`` alone, normalised: what this document itself still prints.
    stripped: str
    tokens: tuple[tuple[str, float, str | None], ...]
    #: Tokens mined from the confirmed values, every one of which is an identifier
    #: rather than a figure (a GSTIN, an invoice number, a date). They never ground
    #: anything; they only tell a dropped number apart from one that appears
    #: nowhere at all.
    identifier_tokens: tuple[tuple[str, float, str | None], ...]
    confidence_lookup: Callable[[str], float | None] | None
    warnings: list[ExtractionWarning]
    provenance: dict[str, dict]


def _missing_warning(path: str) -> ExtractionWarning:
    # "no value was returned for it" rather than "the model returned null": the
    # same outcome arrives when the model omitted the key altogether, or returned
    # a whole party block as a string, and stating that it answered null would be
    # a claim about its answer that is not true.
    #
    # A document that prints no discount and no cess is an ordinary document:
    # that absence is a fact about the invoice, not a gap in the reading, so it
    # does not compete for attention with a mandatory field that could not be read.
    optional = path.rsplit(".", 1)[-1] in _OPTIONAL_FIELDS
    return ExtractionWarning(
        field=path,
        check=_CHECK,
        severity="info" if optional else "warning",
        message=(
            f"{path} was not found in the document: the model returned no value for it, "
            f"so it is left empty rather than filled with a guess."
        ),
    )


def _ungrounded_warning(path: str, value: Any) -> ExtractionWarning:
    return ExtractionWarning(
        field=path,
        check=_CHECK,
        severity="warning",
        message=(
            f'The model returned "{value}" for {path}, but that value did not appear '
            f"in the source text, so it was dropped rather than kept."
        ),
    )


def _short_value_warning(path: str, value: Any) -> ExtractionWarning:
    """Kept on a standalone-token match, which of one character is weak evidence."""
    return ExtractionWarning(
        field=path,
        check=_CHECK,
        severity="warning",
        message=(
            f'"{value}" is a single character. It is printed as a token of its own in the '
            f"document, which is why {path} was kept rather than dropped, but one character "
            f"matches almost any page by accident, so that is weak corroboration. Confirm it "
            f"against the invoice by eye -- on a line item, that means the row numbering."
        ),
    )


def _unreadable_number_warning(path: str, value: Any) -> ExtractionWarning:
    return ExtractionWarning(
        field=path,
        check=_CHECK,
        severity="warning",
        message=(
            f'The model returned "{value}" for {path}, but that is not a plain number: '
            f"an accounting bracket, a detached sign or stray text changes what the "
            f"figure means, and reading it as its bare digits could invert a credit "
            f"into a charge, so it was dropped rather than kept."
        ),
    )


def _shape_name(value: Any) -> str:
    """The JSON name for a value's shape, for a message an accountant can act on."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _wrong_shape_warning(path: str, value: Any) -> ExtractionWarning:
    """A container where text belongs. The text may be printed; the shape is not."""
    return ExtractionWarning(
        field=path,
        check=_CHECK,
        severity="warning",
        message=(
            f'The model returned a JSON {_shape_name(value)} ("{value}") for {path} where a '
            f"text value was expected. The text inside it may well be printed on the "
            f"invoice, but a value of that shape cannot be read as one, so it was dropped "
            f"rather than kept."
        ),
    )


def _credit_warning(path: str, value: Any, printed: str) -> ExtractionWarning:
    """The digits are on the page, but the page prints them as a credit."""
    return ExtractionWarning(
        field=path,
        check=_CHECK,
        severity="warning",
        message=(
            f'The model returned "{value}" for {path}, but the document prints that figure '
            f'as "{printed}", which is a credit and therefore negative. Keeping the positive '
            f"reading would turn a credit into a charge, so it was dropped rather than kept."
        ),
    )


def _identifier_only_warning(path: str, value: Any) -> ExtractionWarning:
    """The digits exist, but only inside a confirmed identifier, which is not a figure."""
    return ExtractionWarning(
        field=path,
        check=_CHECK,
        severity="warning",
        message=(
            f'The model returned "{value}" for {path}, but those digits occur only inside a '
            f"confirmed identifier such as a GSTIN, invoice number or date, which is not a "
            f"figure, so the value could not be corroborated as an amount read from the "
            f"document and was dropped rather than kept."
        ),
    )


def _substring_only_warning(path: str, value: Any) -> ExtractionWarning:
    """A digit-only field not printed as a number of its own on the page. Kept, but say so."""
    return ExtractionWarning(
        field=path,
        check=_CHECK,
        severity="warning",
        message=(
            f'"{value}" is not printed as a number of its own in the text left to this '
            f"stage: it was grounded only as part of a longer number, such as an account, "
            f"phone or invoice number, or only by a value stage 1 confirmed, which is an "
            f"identifier rather than a PIN code. Digits borrowed from another number are "
            f"not a PIN code, so confirm it against the invoice."
        ),
    )


def _assignment_warning(path: str, value: Any) -> ExtractionWarning:
    """An HSN not printed as a code on the row: which row it belongs to is a guess.

    ``info``, not ``warning``: stage 1 strips the codes it confirms, so stage 2
    structurally cannot corroborate a row assignment and this note fires on every
    successful extraction. A warning that fires on 100% of documents says nothing
    about THIS document and teaches an accountant to skim past the warning stream,
    which costs the warnings that matter their credibility. It reports a
    limitation of the two-stage design, which is what ``info`` already means here
    for the tax-split skip note and the assumed place of supply.
    """
    return ExtractionWarning(
        field=path,
        check=_CHECK,
        severity="info",
        message=(
            f'"{value}" is not printed as a code of its own in this row\'s text: it was '
            f"grounded by the HSN/SAC codes stage 1 confirmed, or by a longer number "
            f"elsewhere on the page. Which code belongs to which row is therefore the "
            f"model's judgement and could not be corroborated against the document. A code "
            f"on the wrong row changes that row's tax classification, so confirm it against "
            f"the invoice."
        ),
    )


def _lowest_confidence(
    lookup: Callable[[str], float | None] | None, texts: tuple[str, ...]
) -> float | None:
    """The lowest confidence any of ``texts`` was read at, or None when unknown.

    A string can be printed twice and read at two quite different confidences --
    a header line read cleanly, the same name in a smudged Bill To block. Taking
    the first occurrence's number would let the clean printing mask the shaky one
    and launder exactly the read this stage exists to keep visible, so the lowest
    is recorded, the way ``ocr.confidence_for_span`` returns the minimum over a
    span's words. An occurrence the lookup knows nothing about (``None``) is not
    a low confidence, it is no information, so it neither lowers nor raises what
    is recorded.
    """
    if lookup is None:
        return None
    seen = [lookup(text) for text in dict.fromkeys(texts)]
    known = [value for value in seen if value is not None]
    return min(known) if known else None


def _resolve(source: _Source, path: str, raw: Any, *, numeric: bool) -> Any:
    """Ground one field: record provenance and return it, or warn and return None."""
    if raw is None:
        source.warnings.append(_missing_warning(path))
        return None
    value = _coerce_number(raw) if numeric else _coerce_str(raw)
    if value is None:
        # Several ways to lose a value, a diagnosis for each: an unreadable
        # figure, a container where text belongs, a credit read positive, or one
        # genuinely absent. Telling the accountant the model invented a figure it
        # read correctly is the opposite diagnosis, and the wrong one to act on.
        source.warnings.append(
            _unreadable_number_warning(path, raw) if numeric else _wrong_shape_warning(path, raw)
        )
        return None
    # Every source occurrence that grounds this value, as printed: a figure and
    # a string alike are routinely printed more than once, and all of the
    # printings are asked about so the shakiest read is the one recorded.
    matches: tuple[str, ...] = ()
    anchored = True
    if numeric:
        matches = _match_numbers(value, source.tokens)
        if not matches:
            credit = _credit_conflict(value, source.tokens)
            if credit is not None:
                source.warnings.append(_credit_warning(path, raw, credit))
                return None
            if _match_numbers(value, source.identifier_tokens):
                source.warnings.append(_identifier_only_warning(path, raw))
                return None
    else:
        matches, anchored = _match_strings(value, source.raw, source.normalised)
    if not matches:
        source.warnings.append(_ungrounded_warning(path, raw))
        return None
    confidence = _lowest_confidence(source.confidence_lookup, matches)
    source.provenance[path] = {"source": "llm", "ocr_confidence": confidence}
    # A short string got here on a standalone-token match, which of one character
    # is weak evidence: keeping it silently would present "the page prints a 1
    # somewhere" as corroboration that this row is row 1 (deviation 9).
    if not numeric and len(_normalise(value)) < _MIN_GROUNDED_LENGTH:
        source.warnings.append(_short_value_warning(path, value))
    # A string enters the payload in the DOCUMENT's spelling: grounding tolerates
    # re-casing and re-spacing, and "nos" where the invoice prints "NOS" is a
    # wrong value in a payload whose Unit codes GSTN enumerates. Its LAYOUT does
    # not come with it: the slice carries whatever line break or column padding
    # sat inside the value on the page, and Addr1/PrdDesc/LglNm are single-line,
    # length-capped INV-01 fields. Collapsing the whitespace changes no word and
    # no letter case. The lookup above was asked about the raw slice, which is
    # the text as printed.
    #
    # Only a slice the source printed as a TOKEN of its own is a spelling. When
    # the value grounded solely inside a longer word, the slice is a fragment of
    # a different word -- "Nos" out of "Nosy Distributors" -- and re-casing the
    # answer to it would manufacture a value that is neither the model's nor any
    # token on the page, turning an uncorroborated Unit of "NOS" into "Nos",
    # which is not a UQC code GSTN accepts. The model's own rendering is kept
    # there; the lookup was still asked about the fragment, because that is the
    # only text the value was grounded against.
    #
    # A number is the parsed figure; the printed forms were only what the lookup
    # needed.
    if numeric:
        return value
    return " ".join((matches[0] if anchored else value).split())


def _block(data: dict, name: str) -> dict:
    """One named object from the response; an absent or malformed block reads as empty."""
    block = data.get(name)
    return block if isinstance(block, dict) else {}


def _parse(content: Any) -> tuple[dict | None, str]:
    """The response content as a JSON object, or (None, why it could not be used)."""
    if not isinstance(content, str):
        return None, f"the response content was {type(content).__name__}, not text"
    try:
        data = json.loads(content)
    except ValueError as exc:
        return None, f"it is not valid JSON ({exc})"
    if not isinstance(data, dict):
        return None, f"it is a JSON {type(data).__name__}, not a JSON object"
    return data, ""


def _empty_totals() -> dict[str, float | None]:
    return {key: None for key in TOTAL_KEYS}


def _unparsable(reason: str) -> LlmExtraction:
    """The whole-response failure: nothing extracted, one warning, nothing guessed."""
    return LlmExtraction(
        items=(),
        seller_name=None,
        seller_addr1=None,
        seller_loc=None,
        seller_pin=None,
        buyer_name=None,
        buyer_addr1=None,
        buyer_loc=None,
        buyer_pin=None,
        totals=_empty_totals(),
        warnings=(
            ExtractionWarning(
                field="ItemList",
                check=_CHECK,
                severity="warning",
                message=(
                    f"The model's response could not be parsed as a JSON object: {reason}. "
                    f"No fields were taken from it and nothing was guessed in their place."
                ),
            ),
        ),
        provenance={},
    )


def _resolve_items(raw_items: Any, source: _Source) -> tuple[LlmItem, ...]:
    """Ground every line-item row, keeping response order."""
    if raw_items is None or (isinstance(raw_items, list) and not raw_items):
        source.warnings.append(
            ExtractionWarning(
                field="ItemList",
                check=_CHECK,
                severity="warning",
                message=(
                    # "returned none" covers both ways this arrives -- an empty
                    # ItemList, and no ItemList key at all. Saying it returned an
                    # empty list would be a claim about its answer that is not
                    # true of the second.
                    "No line items were found in the document: the model returned none, "
                    "so none were invented in their place."
                ),
            )
        )
        return ()
    if not isinstance(raw_items, list):
        source.warnings.append(
            ExtractionWarning(
                field="ItemList",
                check=_CHECK,
                severity="warning",
                message=(
                    f"The model returned ItemList as a JSON {type(raw_items).__name__} "
                    f"rather than a list of rows, so no line items were taken from it."
                ),
            )
        )
        return ()

    items: list[LlmItem] = []
    for position, row in enumerate(raw_items, start=1):
        if not isinstance(row, dict):
            source.warnings.append(
                ExtractionWarning(
                    field="ItemList",
                    check=_CHECK,
                    severity="warning",
                    message=(
                        f"Row {position} of the model's ItemList was a JSON "
                        f"{type(row).__name__} rather than an object, so it was dropped."
                    ),
                )
            )
            continue
        index = len(items)
        values = {
            name: _resolve(source, f"ItemList[{index}].{name}", row.get(name), numeric=numeric)
            for name, numeric in _ITEM_FIELDS
        }
        # Stage 1 strips the HSN codes it confirmed, so a row's code is grounded
        # by that confirmed list rather than by anything left on the row. Which
        # code belongs to which row is then the model's call, and the grounding
        # check structurally cannot corroborate it -- say so instead of
        # presenting it as read from the document. The check is for the code as a
        # standalone token: a substring test lets the PIN "400021" pass off a
        # fabricated HsnCd of "4000" as read from the row.
        hsn = values["HsnCd"]
        if hsn is not None and not _is_standalone_token(hsn, source.stripped):
            source.warnings.append(_assignment_warning(f"ItemList[{index}].HsnCd", hsn))
        items.append(LlmItem(**values))
    return tuple(items)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def extract_with_llm(
    remaining_text: str,
    confirmed: dict,
    *,
    client: Any,
    model: str = DEFAULT_MODEL,
    confidence_lookup: Callable[[str], float | None] | None = None,
) -> LlmExtraction:
    """Structure ``remaining_text`` with one LLM call, then verify every value came from it.

    ``client`` has no default on purpose: this module never builds one, so the
    stage cannot reach the network by accident. One call, one answer -- a
    response that cannot be parsed is reported, not retried.
    """
    prompt = build_prompt(remaining_text, confirmed)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    # The only external boundary this module has. A provider that stops on a
    # content filter returns no choices at all; that is one more unusable
    # answer, not a crash in the middle of the pipeline.
    choices = getattr(response, "choices", None) or []
    content = choices[0].message.content if choices else None
    data, reason = _parse(content)
    if data is None:
        return _unparsable(reason)

    # A correct model may echo a confirmed field, so those values ground a string.
    # They never ground a NUMBER: every field stage 1 confirms is an identifier or
    # a classification code -- a GSTIN, an invoice number, a date, a state code, an
    # HSN code -- so mining its digit runs for numeric tokens would let a
    # fabricated amount match one of them (deviation 6).
    confirmed_values = _confirmed_values(confirmed)
    grounding_text = "\n".join((remaining_text, *confirmed_values))
    source = _Source(
        raw=grounding_text,
        normalised=_normalise(grounding_text),
        stripped=_normalise(remaining_text),
        tokens=_numeric_tokens(remaining_text),
        identifier_tokens=_numeric_tokens("\n".join(confirmed_values)),
        confidence_lookup=confidence_lookup,
        warnings=[],
        provenance={},
    )

    items = _resolve_items(data.get("items"), source)
    parties = {
        attr: _resolve(source, path, _block(data, block).get(key), numeric=False)
        for attr, block, key, path in _PARTY_FIELDS
    }
    # Pin is the other digit-only string field, and string grounding is
    # substring-based by contract: "110001" grounds against an account number
    # "1100015678". HsnCd already says when that happened; Pin is mandatory in
    # INV-01, so it says so too rather than reaching the payload silently.
    # The corroboration asked for is the same one HsnCd asks for: the value
    # printed as a token of its own in the text left to THIS stage
    # (`source.stripped`), not in the whole grounding text. Testing the whole
    # grounding text lets the confirmed context corroborate the value on its own,
    # and a confirmed value is an identifier -- an invoice number, a GSTIN, a
    # state code -- so a Pin of "27" would be "corroborated" by the confirmed
    # state code and reach the payload silently as a PIN code.
    for attr, _block_name, key, path in _PARTY_FIELDS:
        pin = parties[attr]
        if key == "Pin" and pin is not None and not _is_standalone_token(pin, source.stripped):
            source.warnings.append(_substring_only_warning(path, pin))
    raw_totals = _block(data, "totals")
    totals = {
        key: _resolve(source, f"ValDtls.{key}", raw_totals.get(key), numeric=True)
        for key in TOTAL_KEYS
    }
    return LlmExtraction(
        items=items,
        totals=totals,
        warnings=tuple(source.warnings),
        provenance=source.provenance,
        **parties,
    )


def make_client(api_key: str | None = None):
    """Build a Groq client from ``GROQ_API_KEY``. Nothing in this module calls it."""
    key = api_key or os.environ.get(_ENV_VAR)
    if not key:
        raise RuntimeError(
            f"{_ENV_VAR} is not set, so no Groq client can be built. Set {_ENV_VAR} in the "
            f"environment, or pass api_key, or inject your own client into extract_with_llm."
        )
    from groq import Groq

    return Groq(api_key=key)
