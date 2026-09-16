# GST e-invoice extraction

Turns an Indian GST tax invoice into the government's **INV-01 JSON payload**, and tells you exactly what it could not read.

Ships as an [MCP](https://modelcontextprotocol.io/) server with three tools, so an agent can parse a document, check a GSTIN, or re-validate a payload it already holds.

> **It produces a submission-ready payload, not a filed invoice.** There is no IRN here. An Invoice Reference Number is issued by the government's Invoice Registration Portal after you submit the payload to it. Nothing in this repository talks to the IRP.

------

## The idea

An extraction tool that quietly guesses is worse than one that says it cannot read a field. A hallucinated digit in a GSTIN that still passes its checksum, or a line item that was never on the page, is the failure that costs an accountant real money — and it is invisible precisely because it looks right.

So the design has one rule: **the model may structure text, it may never invent values.** That is enforced twice. The prompt says it, and then every value the model returns is checked back against the document text before it is kept. A value with no source in the document is replaced with null and reported, however plausible it looks. If that field is mandatory in INV-01, no payload is produced at all.

Everything the tool knows about its own work — how each page was read, where each field came from, what it was unsure about — travels beside the payload in `extraction_meta`, never inside it. The payload stays strictly spec-pure, because the government API rejects unknown keys.

**Read [LIMITATIONS.md](https://claude.ai/chat/LIMITATIONS.md) before trusting the output.** It is specific about what the tool cannot corroborate, and what that costs you.

------

## Install

Requires **Python 3.11 or newer** and **Tesseract OCR** as a system dependency.

```bash
# Tesseract (Windows)
winget install UB-Mannheim.TesseractOCR

# Tesseract (Debian/Ubuntu)
sudo apt-get install -y tesseract-ocr

# Tesseract (macOS)
brew install tesseract
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -e .
```

Tesseract does not need to be on `PATH`: the OCR module looks there first, then at the standard Windows install location, and raises an actionable error naming both if neither works.

### Environment

| Variable            | Required    | Purpose                                        |
| ------------------- | ----------- | ---------------------------------------------- |
| `GROQ_API_KEY`      | yes         | Stage 2 reads the line-item table through Groq |
| `GST_MCP_MODEL`     | recommended | Pin the model your key can reach               |
| `GST_MCP_TRANSPORT` | no          | `stdio` (default), `sse`, or `streamable-http` |

The default model is `openai/gpt-oss-120b`, which is what this release was validated against. It works without configuration.

> **Still set `GST_MCP_MODEL` in a deployment.** A hard-coded model identifier expires silently when the provider retires it, and the failure arrives as an HTTP 404 that reads like a bad key rather than a stale constant. Pin the model you have access to, and check it against Groq's deprecation notices.

------

## MCP client configuration

`pip install` puts a `gst-einvoice-mcp` command on your PATH, so a client only needs to name it:

```json
{
  "mcpServers": {
    "gst-einvoice": {
      "command": "gst-einvoice-mcp",
      "env": {
        "GROQ_API_KEY": "your-key-here",
        "GST_MCP_MODEL": "openai/gpt-oss-120b"
      }
    }
  }
}
```

Running from a clone rather than an install? Point `command` at the interpreter inside your virtual environment and invoke the module directly:

```json
{
  "mcpServers": {
    "gst-einvoice": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "gst_einvoice.server"],
      "env": {
        "GROQ_API_KEY": "your-key-here"
      }
    }
  }
}
```

On Windows that interpreter path ends `\.venv\Scripts\python.exe`.

### Tools

| Tool               | Takes                           | Gives back                                                   |
| ------------------ | ------------------------------- | ------------------------------------------------------------ |
| `parse_invoice`    | a file path, optional tolerance | the INV-01 payload, missing fields, refusals, and `extraction_meta` |
| `validate_gstin`   | a GSTIN                         | structure, checksum, state code, PAN, and the reason on failure |
| `validate_payload` | an INV-01 payload               | the four consistency checks over a payload you already hold  |

`parse_invoice` has **three normal outcomes**, and only the first gives you a payload:

1. **A payload plus warnings.** Usable, but the warnings say which fields were read at low OCR confidence, which were assigned by position rather than by a label, and which were derived rather than printed.
2. **No payload, `missing_fields` populated.** A mandatory field could not be read. Nothing was invented to fill the gap, which is why there is no payload.
3. **No payload, `refusals` populated.** Export, SEZ and foreign-currency invoices are refused by design, with a message saying what was detected and what to do instead.

------

## Worked example

The invoice, as a text-layer PDF:

```
TAX INVOICE
Seller: Nimbus Components Pvt Ltd
GSTIN 27AAPFU0939F1ZV
Plot 14 MIDC Andheri East
Mumbai 400093
Invoice No: INV-2026-0042
Invoice Date: 17/04/2026
Buyer: Kanchan Electricals LLP
GSTIN 27AABCB5507N1ZJ
221 Laxmi Road Shivajinagar
Pune 411005
Sl No 01 Laptop Stand
Qty 4 NOS Rate 1500.00
Taxable 6000.00 GST 18%
CGST 540.00 SGST 540.00 IGST 0.00
HSN/SAC: 8471
Goods once sold will not be taken back or exchanged
Total Invoice Value 7080.00
```

```python
from gst_einvoice.extract_llm import make_client
from gst_einvoice.pipeline import extract_invoice

# model defaults to openai/gpt-oss-120b; pass model=... to override
result = extract_invoice("sample_invoice.pdf", client=make_client())
```

### The payload

```json
{
  "Version": "1.1",
  "TranDtls": { "TaxSch": "GST", "SupTyp": "B2B" },
  "DocDtls": { "Typ": "INV", "No": "INV-2026-0042", "Dt": "17/04/2026" },
  "SellerDtls": {
    "Gstin": "27AAPFU0939F1ZV",
    "LglNm": "Nimbus Components Pvt Ltd",
    "Addr1": "Plot 14 MIDC Andheri East",
    "Loc": "Mumbai",
    "Pin": 400093,
    "Stcd": "27"
  },
  "BuyerDtls": {
    "Gstin": "27AABCB5507N1ZJ",
    "LglNm": "Kanchan Electricals LLP",
    "Addr1": "221 Laxmi Road Shivajinagar",
    "Loc": "Pune",
    "Pin": 411005,
    "Stcd": "27",
    "Pos": "27"
  },
  "ItemList": [
    {
      "SlNo": "01",
      "PrdDesc": "Laptop Stand",
      "IsServc": "N",
      "HsnCd": "8471",
      "Qty": 4.0,
      "Unit": "NOS",
      "UnitPrice": 1500.0,
      "TotAmt": 6000.0,
      "Discount": 0.0,
      "AssAmt": 6000.0,
      "GstRt": 18.0,
      "CgstAmt": 540.0,
      "SgstAmt": 540.0,
      "IgstAmt": 0.0,
      "CesAmt": 0.0,
      "StateCesAmt": 0.0,
      "OthChrg": 0.0,
      "TotItemVal": 7080.0
    }
  ],
  "ValDtls": {
    "AssVal": 6000.0, "CgstVal": 540.0, "SgstVal": 540.0, "IgstVal": 0.0,
    "CesVal": 0.0, "StCesVal": 0.0, "RndOffAmt": 0.0, "TotInvVal": 7080.0
  }
}
```

### The warnings block

This is the half most tools do not give you. Six entries, from the run above, all `info`:

```
[info]    extract_llm  ItemList[0].HsnCd
    "8471" is not printed as a code of its own in this row's text: it was grounded by
    the HSN/SAC codes stage 1 confirmed, or by a longer number elsewhere on the page.
    Which code belongs to which row is therefore the model's judgement and could not
    be corroborated against the document. A code on the wrong row changes that row's
    tax classification, so confirm it against the invoice.

[info]    extract_llm  ItemList[0].Discount
    ItemList[0].Discount was not found in the document: the model returned no value
    for it, so it is left empty rather than filled with a guess.

[info]    extract_llm  ItemList[0].CesAmt        (same wording)
[info]    extract_llm  ValDtls.CesVal            (same wording)
[info]    extract_llm  ValDtls.RndOffAmt         (same wording)

[info]    pipeline     BuyerDtls.Pos
    BuyerDtls.Pos (place of supply) was not read from the document -- build 2 does
    not extract it -- so it was assumed equal to the buyer's registered state code
    (27). A genuine bill-to/ship-to supply, where the goods go to a different state
    from the one the buyer is registered in, has a different place of supply, and
    the CGST/SGST-versus-IGST split follows the place of supply. Confirm it against
    the document before filing.
```

Nothing in that list means the payload is wrong. Each one names something the tool could not corroborate, so you know where to look. The four arithmetic validators raised nothing, which is what silence from them means.

On an invoice whose template omits a column — an intra-state invoice with no IGST column, or one that prints a taxable value but no separate gross — you will also see a `pipeline` note saying the field was derived rather than read, and `field_provenance` will record it as `"source": "derived"`. Five rules can do this: the tax head that cannot apply is set to zero from the two state codes; a row's gross is filled from its taxable value and discount, and a row's taxable value from its gross, each the other's exact inverse and never both on one row; the per-line total on a **single-line** invoice that prints its total only at the foot is filled from the INV-01 item identity; and a document total the model dropped is filled from its row counterpart, again only on a single-line invoice. The last two fire only when the result reconciles with the printed `ValDtls.TotInvVal`, and a derived value never feeds another derivation across the row/totals boundary. On a multi-line invoice a missing `TotItemVal` or document total is still reported missing and no payload is produced — see [LIMITATIONS.md](LIMITATIONS.md) for why.

### Provenance

`extraction_meta.field_provenance` carries an entry for **every** field in the payload — 45 for this invoice — saying which stage produced it and, for a scanned page, the OCR confidence of the text it was read from:

```json
{
  "SellerDtls.Gstin":  { "source": "regex",   "ocr_confidence": null },
  "SellerDtls.Stcd":   { "source": "derived", "ocr_confidence": null },
  "ItemList[0].PrdDesc": { "source": "llm",   "ocr_confidence": null },
  "ItemList[0].IsServc": { "source": "derived", "ocr_confidence": null },
  "BuyerDtls.Pos":     { "source": "assumed", "ocr_confidence": null }
}
```

On a scanned page the same fields carry real numbers — 0.86 to 0.96 on a clean 300 dpi render — and the **lowest** confidence across a field's words is the one recorded.

| `source`  | Meaning                                                      |
| --------- | ------------------------------------------------------------ |
| `regex`   | Confirmed deterministically, structurally certain            |
| `llm`     | Structured by the model, then verified against the document text |
| `derived` | Follows by rule from values that were read; not printed on the page |
| `assumed` | Neither read nor derived — an assumption the tool names explicitly |

------

## Development

```bash
pytest -q -W error
```

1617 tests across ten modules, passing with warnings treated as errors. The LLM stage takes an injected client, so the whole suite runs with no API key and no network.

The suite passes on both 3.11 and 3.13; 3.11 is the floor because that is the lowest version the whole dependency set resolves on, and it was verified by running the suite there rather than assumed.

### Testing local changes through an MCP client

**The MCP server runs whatever is installed in `site-packages`, not your working tree.** An
MCP client launches the server through the `gst-einvoice-mcp` entry point, which resolves to
the installed distribution. Editing a file in the repo changes nothing that the server
serves until you reinstall, and there is no error to tell you — the tool call succeeds and
returns the old behaviour.

This is easy to lose an hour to. During the 0.1.2 work the repo carried a new derivation
rule while the server was still serving 0.1.1, so a tool call against the new code silently
exercised the old path and appeared to show the change had not worked.

Either reinstall after every change:

```bash
pip install -e .        # editable, so later edits are picked up on server restart
```

or point the MCP client's `command` at the repo's own virtualenv instead of a user-level
install, so the server and the tests run the same code:

```jsonc
// in your MCP client config
"command": "C:\\path\\to\\repo\\.venv\\Scripts\\gst-einvoice-mcp.exe"
```

Either way, **restart the MCP server after changing code** — a running server holds the
modules it imported at start-up. If a change seems to have no effect, check which copy is
being served before looking for the bug in your code.

| Module             | What it does                                                 |
| ------------------ | ------------------------------------------------------------ |
| `gstin.py`         | Structure and mod-36 checksum                                |
| `state_codes.py`   | State code table, including discontinued 25 and legacy 28    |
| `schema.py`        | INV-01 pydantic models, `extra="forbid"` throughout          |
| `validators.py`    | The four arithmetic and tax-split checks                     |
| `ingest.py`        | Per-page routing and the detect-and-refuse rules             |
| `ocr.py`           | Tesseract with per-word confidence mapped onto character spans |
| `extract_rules.py` | Deterministic extraction: GSTINs, parties, number, date, HSN |
| `extract_llm.py`   | The LLM stage and the grounding check                        |
| `pipeline.py`      | End-to-end assembly                                          |
| `server.py`        | The MCP server                                               |

## Licence note

This project depends on **PyMuPDF**, which is AGPL-3.0. That is a deliberate choice, made because PyMuPDF opens image files directly as one-page documents and gave more reliable text-layer detection than the alternatives. If you intend to distribute this tool as part of a closed-source product, check that licence first.
