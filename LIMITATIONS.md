# Limitations

This document describes what the tool actually does, not what it was intended to do.
Every claim here traces to code in this repository or to a test in `tests/`.

Read it before trusting any output. The tool is built so that a value it cannot
corroborate is reported as absent rather than filled in, but "reported" only helps if
somebody reads the report. **The warnings block is not decoration. It is the output.**

---

## What it handles

| Input | Behaviour |
|---|---|
| PDF with a text layer | Read directly, no OCR |
| Scanned image PDF | Rasterised and OCR'd through Tesseract |
| Bare image file | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` |
| Multi-page documents | Routed **per page**, so one document can mix both routes |
| Mixed GST rates | Each line item keeps its own rate |
| Intra-state invoices | CGST + SGST |
| Inter-state invoices | IGST |
| Compensation cess | Carried into the item and the totals when printed |

Routing is per page rather than per document because a letterhead page often carries a
clean text layer while a scanned continuation page does not. A page is read natively
only if it has more than about 100 characters of text **and** at least one of GSTIN,
"tax invoice", HSN, SGST, CGST or IGST appears in it. Some text is not evidence of
usable text: a corrupted layer can carry a stray header string while the body is
garbage, so both gates must pass.

OCR costs roughly three seconds per page at the default 300 dpi, most of it deskewing.

---

## What it refuses, and why

Two kinds of document are refused outright. The tool produces no payload and explains
what it found, which field revealed it, why it cannot proceed, and what to do instead.

**Export and SEZ invoices.** Detected from phrases such as "supply meant for export",
"under LUT", "letter of undertaking", "SEZ unit" or "shipping bill". A foreign buyer has
no GSTIN by design, so a pipeline that requires one would fail every export invoice and
produce a broken payload rather than an honest refusal. For an SEZ supply the buyer does
hold a GSTIN, but the supply is zero-rated and treated as inter-state regardless of the
buyer's state, so the tax-split logic does not apply either.

**Multi-currency invoices.** Detected from `TotInvValFc`, from "foreign currency",
"exchange rate" or "conversion rate", from a currency glyph, or from an ISO currency code
printed next to a number. Every amount in this tool is assumed to be rupees. A
foreign-currency figure read as rupees is a silent misparse of the most important numbers
on the document.

Detection is deliberately conservative. A company named "Sharma Exports Pvt Ltd", the bare
words "export" or "SEZ", a "deemed export" (which is a domestic buyer with a GSTIN), and
line items reading "CAD drawings" or "SAR filing" do **not** trigger a refusal.

---

## What it flags rather than resolving

These do not stop a payload being produced. They appear in the warnings block.

**Party assignment by document position.** When no label resolves which GSTIN belongs to
the seller and which to the buyer, the tool falls back to reading order, since Indian
invoices conventionally put the seller top-left. This **always** warns, including when the
fallback lands on the right answer, because getting it backwards inverts who owes the tax.

**Ambiguous dates.** Output is always DD/MM/YYYY, and day-first is always the reading.
When both of the first two components are 12 or less, the date is genuinely ambiguous:
`03/04/2026` is either 3 April or 4 March. If another date on the document settles the
convention, the ambiguity resolves silently. If nothing settles it, the tool emits the
day-first value **and** warns, naming both readings.

**Low-confidence OCR.** Confidence is captured per word from Tesseract and mapped onto the
character spans each field was read from. `field_provenance` carries an `ocr_confidence`
for every field read from a scanned page. Where a field spans several words the **lowest**
confidence is recorded, not the average: a field is only as trustworthy as its
worst-read character. Where the same value is printed more than once, the lowest of those
printings is recorded, so a clean printing can never launder a smudged one.

**Fields that could not be grounded.** Any value the model returns that cannot be found in
the document text is replaced with null and reported. If that field is mandatory in INV-01,
no payload is produced at all.

**The invoice number read from the row below its label.** A column-header layout warns
every time, even when it takes the right value.

---

## Known approximations, and what each costs you

### The tax-split check compares registered state codes, not Place of Supply

`validate_tax_split` decides intra-state versus inter-state by comparing
`SellerDtls.Stcd` with `BuyerDtls.Stcd`. True Place of Supply is a separate concept in
GST law and is not extracted.

**Consequence:** SEZ supplies and bill-to/ship-to cases may be false-flagged. Every
warning this check emits says so in its own text. The check stands itself down entirely,
as an informational note rather than a warning, under reverse charge and when
`BuyerDtls.Pos` differs from `BuyerDtls.Stcd`, because it cannot evaluate those.

This is a deliberate simplification, not a defect.

### Place of supply is assumed, not read

`BuyerDtls.Pos` is set equal to the buyer's registered state code and recorded in
provenance as `"source": "assumed"`, with an informational note on every document.

**Consequence:** on a genuine bill-to/ship-to supply, where goods go to a different state
from the one the buyer is registered in, the place of supply is wrong and the CGST/SGST
versus IGST split follows the place of supply. Confirm it before filing.

### Two-column invoice headers defeat label matching on scans

Tesseract assigns both panels of a two-column header the same line number, so a seller
panel on the left and a buyer panel on the right merge into a single line of text.

**Consequence:** on a scanned two-panel invoice, label proximity cannot work and party
assignment falls back to document position — which always warns. Splitting lines on large
horizontal gaps was considered and rejected: table rows are also column-separated, and
splitting them would fragment the item rows the extraction stage has to read.

### Grounding proves a value is printed, never that it belongs where it was put

This is the largest limitation in the tool.

The check confirms that a value appears in the document. It cannot confirm which party or
which row it belongs to. A response that swaps the seller and buyer blocks, or pulls row
two's figures onto row one, passes every check: each value really is printed on the page.
Those values are kept with clean provenance and **no warning at all**.

**Consequence:** no warning on `SellerDtls.LglNm` means "this name is printed on the
document", not "this name was corroborated as the seller's". Party blocks and per-row
figures need checking against the invoice by eye.

The item `HsnCd` is the one assignment the tool can warn about, and does, on every row —
because the deterministic stage strips the confirmed codes, leaving nothing on the row to
check one against. That warning is not a sign that HSN is riskier than the rest. It is a
sign that the rest cannot be checked at all.

### The grounding check has three tiers, and the last one matches inside words

A value is matched, in order: as a standalone token in the model's own casing; as a
standalone token in any casing; then by unanchored substring search.

The unanchored tier is retained deliberately, because digit-only fields legitimately
ground inside longer numbers and ordinary printings like `Rs1500.00` would otherwise be
refused. Removing it would drop correctly read values and report them as absent, which
destroys information rather than inventing it — but it has a real cost.

**Consequence:** a `Unit` of `NOS` is kept when the page prints only "Nosy Distributors",
and a figure can ground against digits inside a word — "Steel Bracket 12mm" grounds a
quantity of 12, "Model MS150" an amount of 150, a vehicle number `MH12AB1234` both 12 and
1234. Product dimensions, SKU codes and vehicle numbers are common on Indian invoices, so
this is not a rare shape.

The two digit-only string fields, `HsnCd` and `Pin`, warn when they are grounded this way.
**Short alphabetic fields do not warn.** A `Unit` that appears nowhere on the page as a
unit needs checking by eye.

### Single-character values are weakly corroborated

An `SlNo` of "1" is ordinary on Indian invoices. Such a value is grounded only when the
document prints it as a token of its own, and it is then kept **with a warning** saying
the corroboration is weak.

**Consequence:** a single character proves very little. The "1" in a serial column is the
same character as the "1" in a quantity column, a rate band or a page number, so a wrong
`SlNo` of "1" is corroborated by a page that prints a "1" anywhere at all. Row numbering
needs checking by eye.

Refusing short values instead was tried and abandoned: it made the tool produce no output
at all for the ordinary case. See *How these limits were found*.

### Party labels must be printed on one line

A label is recognised only when its own words sit on a single line. An OCR-wrapped label
("Bill" on one line, "To" on the next) is not found, and party assignment falls back to
position, which warns.

**Consequence:** a wrapped label costs you a warning you would not otherwise have seen.
This is the safe direction. Matching across a line break lets an invoice title sitting
above an address line act as a label, which silently filed an Indian door number as the
invoice number until it was fixed.

### The reverse-charge detector needs an explicit answer

A reverse-charge phrase occupying its whole line is treated as a **label**, not a
declaration, and stays unresolved without an explicit Yes or No next to it. A phrase
embedded in a sentence ("This supply attracts reverse charge") remains an affirmation.

**Consequence:** a bare "Reverse charge applicable" with no answer reads as *not* reverse
charge, so the tax-split check runs and may raise a warning on a genuine reverse-charge
invoice.

That trade is deliberate. A false positive sets `RegRev` and stands the tax-split check
down for the whole invoice, silently, on a document that is not under reverse charge —
invisible harm. A false negative runs the check and shows a dismissible warning — visible
and recoverable. It matters more than it looks, because the detector re-runs on scanned
pages too.

### Some mandatory INV-01 fields are derived, not read

INV-01 requires all three tax heads on every line, a gross amount as well as a taxable
value, and a per-line total on every row. Real invoice templates print only what applies:
an intra-state invoice has no IGST column, an inter-state one has no CGST/SGST column, an
invoice with no discount prints one amount rather than two, and a single-line invoice
prints its total once at the foot rather than twice.

Where such a field is absent, the tool fills it from a rule rather than reporting the
document unreadable, and records `"source": "derived"` in provenance with an informational
note on every affected field. There are three such rules:

- the tax head that cannot apply is set to zero, from the two validated state codes;
- `TotAmt` is set to `AssAmt + Discount`, the INV-01 identity;
- `TotItemVal` is set to `AssAmt + CgstAmt + SgstAmt + IgstAmt + CesAmt + StateCesAmt +
  OthChrg` — the identity `validate_item_total` enforces — but **only** when both of these
  hold:
  1. the invoice has **exactly one line item**, and
  2. the derived value **reconciles with the printed `ValDtls.TotInvVal`**, within the same
     tolerance the validators compare on (0.05 by default).

**Why the per-line total is restricted to single-row invoices.** On a one-row invoice the
per-row total and the document total are the same number, so the printed foot corroborates
the derived value: the document does state it, once. On a multi-row invoice it does not.
Asserting a per-row total the document never prints means asserting a **split across rows**
that the document never prints either — the totals block constrains only the sum, and any
number of per-row splits add to the same sum. That is exactly the guessing this tool exists
not to do, so where there is more than one line item and `TotItemVal` is absent, the field
is reported missing and no payload is produced, however obvious the arithmetic looks.

The reconciliation condition is not a formality either. A single-row invoice whose derived
line total disagrees with its own printed total is evidence that something was misread, not
an invitation to paper over it; the rule stands down and the field is reported missing.

**Consequence:** these values were not read off the page. The tax-head derivation inherits
the Place of Supply approximation above, so confirm it for SEZ and bill-to/ship-to cases.
Derivation only happens when the surrounding block carries real content; a wholly empty
response still reports everything missing.

### Currency glyphs on scanned pages do not trigger a refusal

On a scanned page, `$`, `€`, `£` and `¥` are blanked before the multi-currency detector
runs, because Tesseract misreads the rupee glyph often enough that refusing on a bare
glyph would reject a large share of genuine scans. ISO currency codes next to a number
still refuse normally.

**Consequence:** a scanned foreign-currency invoice whose only clue is a bare glyph may be
extracted with its amounts recorded as rupees. Whenever a glyph is blanked, a warning is
emitted on `ValDtls.TotInvValFc` saying so.

### HSN codes in column-header tables are not confirmed deterministically

The deterministic stage takes an HSN or SAC code only where a label names it. In a table
where HSN is a column header and the codes sit in rows below, it confirms none, and the
codes come from the LLM stage instead.

**Consequence:** those codes carry `"source": "llm"` rather than `"source": "regex"`, and
the per-row assignment warning above applies.

---

## Validation semantics

Every arithmetic check takes a tolerance, **0.05 rupees by default**, and compares with
`abs(a - b) <= tolerance` rather than exact equality.

This is not laxity. Real invoices round to the nearest rupee, and floating-point equality
would false-flag nearly every genuine document. A delta of exactly the tolerance passes; a
tiny epsilon absorbs the float representation error that would otherwise make 0.05 fail.

Four checks run:

| Check | What it compares |
|---|---|
| `validate_item_total` | Each item's `TotItemVal` against its own components |
| `validate_invoice_total` | `TotInvVal` against the value block |
| `validate_val_dtls_sums` | Each value-block total against the sum of the item fields |
| `validate_tax_split` | The CGST/SGST versus IGST split against the state codes |

Every warning names the exact field path, such as `ItemList[2].TotItemVal`, and states the
actual value, the expected value with its formula, the delta and the tolerance. No check
ever returns a bare boolean.

---

## What "valid" does and does not mean

**A passing GSTIN checksum proves the number is well-formed. It does not prove the
registration exists, or that it is active.** The check is structural: fifteen characters in
the right shape, and a mod-36 check character that matches. Only the GST portal can tell
you whether a registration is live, and this tool never contacts it.

State code 25 is reported as *discontinued* rather than unknown, because it means either a
misread or a genuine pre-August-2020 record and the caller needs to tell those apart.
State code 28 is *legacy but valid*: pre-2014 Andhra Pradesh registrations remain
legitimate.

**A spec-conforming payload is submission-ready in shape. It is not a filed invoice, and
this tool does not produce an IRN.** An Invoice Reference Number is issued by the
government's Invoice Registration Portal after you submit the payload there. Nothing in
this repository talks to the IRP.

The payload is kept strictly free of custom fields, because the government API rejects
unknown keys. Everything the tool knows about its own work lives in `extraction_meta`
beside the payload, never inside it.

---

## How these limits were found

This section is here because it is the most useful thing for anyone deciding whether to
trust the tool.

**Nearly every real defect found across both prior builds was a fabrication path.** A date
invented out of an item row because a product description began with a month name. The
seller's own GSTIN filed as the invoice number. A hyphen in "Freight - 200.00" read as a
minus sign, so a charge was recorded as a credit. An amount grounded against the invoice
number itself, because the invoice number happened to be bare digits. None of these was a
crash or a wrong total. Each was a plausible value in a mandatory field, delivered without
a warning. That is evidence the grounding architecture was aimed at the right risk — and
evidence of how hard that risk is to see.

**An automated audit of 47 agents with adversarial refutation reported zero findings.**
Four independent reviewers examined the package along different axes, every finding was
put to three skeptics instructed to refute it, and only findings surviving a majority were
kept. Nothing survived.

**The audit was wrong.** Its severity cap had dropped 18 findings unjudged before the
refutation stage ever saw them. Three of those were real. One was that invoices with line
items numbered 1 through 9 — most Indian invoices — produced no output at all, because a
two-character minimum in the grounding rule discarded a single-digit serial number. Three
independent skeptics had refuted that finding and all three were wrong. It was caught by
running the case by hand.

The lesson is not that automated review is useless; it found a great deal. It is that a
clean report is not evidence of a clean system, and that the cases a reviewer drops are
not safer than the ones it keeps.

**The LLM stage was validated against the live API, not only against a fake.** Twelve real
invoices were run end to end covering native and scanned routes, intra-state and
inter-state, mixed rates, cess, a twelve-row table, a multi-page document mixing both
routes, and a deliberately degraded scan.

Across those runs the model returned well-formed JSON every time, never omitted a key from
the schema, never drifted over a long table, never mutated the casing or spacing of a kept
string, and never invented a value. **136 hand-transcribed value checks passed with zero
mismatches**, including through OCR on a scanned page and across a multi-page document.

Eleven of the twelve produce a payload. The twelfth is the deliberately degraded scan — 95
dpi and rotated — where OCR could not read either GSTIN. The tool reported them missing and
produced nothing, rather than guessing a registration number. That is the intended
behaviour on an unreadable document, not a failure.

Live contact revealed one thing no fake could. The first pass produced a payload for only
four of the twelve, because **INV-01 mandates fields that real invoice templates do not
print** — the tax head that cannot apply, and a gross amount on an invoice with no
discount. The model correctly answered null for a column that is not on the page, and the
pipeline correctly refused to invent it. The first two derivation rules above — the
inapplicable tax head and `TotAmt` — were added in response, and they are what took the
pass rate from four to eleven. The third, the single-row `TotItemVal`, came later, from the
same failure appearing on a document this suite's own "clean" fixture models.

---

## Deployment caveats

**Pin the model, and re-check it when your provider deprecates one.** A hard-coded model
identifier is a dependency that expires silently. Nothing warns you when a provider retires
a model name; the first sign is an HTTP 404 at call time, which reads like a broken install
or a bad API key rather than a stale constant. That misdirection costs disproportionate
time to diagnose, because you go looking at your credentials and your environment before
you think to look at a string in the source.

This is not hypothetical. The default shipped in an earlier build was
`llama-3.3-70b-versatile`, which Groq had retired, and every first run failed with a 404
until it was found. The current default, `openai/gpt-oss-120b`, is the model the live
validation above actually ran against, so it is the one with evidence behind it.

Set `GST_MCP_MODEL` in any deployment rather than relying on the library default, and add
the provider's deprecation notices to whatever you already watch for dependency changes.

---

## Known open issues

**A missing-field report is not evidence that the field is absent from the document.** It
means only that this attempt could not corroborate it. The distinction matters: if the tool
reports `ItemList[0].AssAmt` missing, do not conclude the invoice does not state a taxable
value. Look at the document.

The cause is that the model is non-deterministic even at temperature zero, so **the same
document can yield a payload on one attempt and a missing-field report on the next.** The
right thing happens on both — the field is reported missing and no payload is produced,
rather than a value being invented — but the two runs disagree.

**The rate has been measured.** Running stage 2 against `sample_invoice.pdf` 42 times, at
temperature zero, with the same prompt and the same model (`openai/gpt-oss-120b`):

| Field | Runs | Populated | Null |
| ----- | ---- | --------- | ---- |
| `ItemList[0].TotItemVal` | 42 | 28 | **14 (33%)** |
| `ValDtls.AssVal`, `CgstVal`, `SgstVal`, `IgstVal`, `TotInvVal` | 42 | 42 | 0 |
| `ItemList[0].AssAmt` | 12 | 12 | 0 |

(`AssAmt` was recorded over only the first 12 of those runs, so its sample is smaller; the
other rows cover all 42.)

Roughly one run in three failed to read a single field, with nothing about the input
changing between runs. That number is what tells you how much to trust a single run: on a
document of this shape, one attempt is not a reading of the document, it is one sample.

This is **provider-side non-determinism at temperature zero** — an artefact of mixture-of-
experts routing and request batching, not of sampling temperature and not of the prompt.
No prompt change fixes it, and `temperature` is already zero. Retrying a document that came
back with a field missing is legitimate and, at a 33% per-field failure rate, often
worthwhile. What you must not do is treat a second, fuller answer as proof the first was
faulty, or a missing-field report as proof about the invoice.

**One observation that is not characterised.** On a single run through the MCP server,
`ValDtls.AssVal`, `CgstVal` and `SgstVal` all came back null together while `TotInvVal` was
read normally. It has **never reproduced**: 66 subsequent runs against the same document,
the same stage-2 code and the same model produced it zero times. It is recorded here as
observed once and unreproduced, not as a known failure mode, because one occurrence cannot
establish a rate.

What makes it hard to dismiss as noise is that the failing run returned a *coherent partial
reading* rather than corruption: `sample_invoice.pdf` has no separate totals block — its
taxable and tax figures are printed on the item row, and the only document-level figure is
the invoice total — so declining to reuse the item row's amounts as invoice totals is a
defensible reading of that document. A rare semantic waver and rare noise that happened to
look coherent cannot be told apart from one occurrence, so neither is claimed. Nothing
derives `ValDtls.AssVal`, so if it does recur it blocks a payload.

**Clearer labels on the invoice make extraction worse, not better.** The obvious response to
a field that reads unreliably is to label it more explicitly on the document. That was
tested against `sample_invoice.pdf`, 12 runs per variant, every figure held identical and
only the labelling changed:

| Variant | `ItemList[0].AssAmt` | `ItemList[0].TotItemVal` | `ValDtls` totals |
| ------- | -------------------- | ------------------------ | ---------------- |
| **A** — as printed (item row carries bare `Taxable 6000.00`, `CGST 540.00 SGST 540.00`) | **12 / 12** | **8 / 12** | 12 / 12 |
| **B** — item row relabelled `Total Taxable Value` / `Total CGST` / `Total SGST` | 4 / 12 | 1 / 12 | 12 / 12 |
| **C** — item row left intact, separate labelled totals block added | 10 / 12 | 4 / 12 | 12 / 12 |

The document as printed outperformed both variants on every item field, and the labels
bought nothing on the totals, which were read 12/12 in all three.

The reading: **relabelling the item row as a totals block makes stage 2 stop treating it as
an item row.** Variant B's amounts, prefixed with "Total", stopped reading as an item's
amounts — `AssAmt` fell to a third of its baseline rate — while the `ValDtls` totals they now
looked like were already being read correctly without the labels. Variant C, which left the
item row alone and added a totals block beside it, was milder but still below baseline.

So do not "improve" an invoice template by adding totals labels to the line-item row in the
hope of more reliable extraction. On the evidence, that makes it worse.
