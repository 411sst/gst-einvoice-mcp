"""MCP server exposing the GST e-invoice pipeline as three focused tools.

Three tools rather than one, because an agent often wants a single answer without
paying for a document parse: checking a GSTIN it already holds, or re-running the
arithmetic checks over a payload it assembled itself.

The model used for stage 2 comes from ``GST_MCP_MODEL`` when set, falling back to
``extract_llm.DEFAULT_MODEL``. Groq retires model names, so a deployment pins the
one it has access to rather than relying on the library default.
"""

import json
import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from gst_einvoice.extract_llm import DEFAULT_MODEL, make_client
from gst_einvoice.gstin import validate_gstin as _validate_gstin
from gst_einvoice.pipeline import extract_invoice
from gst_einvoice.schema import Invoice
from gst_einvoice.state_codes import lookup_state_code
from gst_einvoice.validators import validate_invoice as _validate_invoice

#: Stage 2's model. Set GST_MCP_MODEL to pin the one your Groq key can reach.
MODEL = os.environ.get("GST_MCP_MODEL") or DEFAULT_MODEL

server = MCPServer(
    name="gst-einvoice",
    instructions=(
        "Extracts Indian GST tax invoices into the government's INV-01 JSON schema, and "
        "validates GSTINs and INV-01 payloads. Every tool reports what it could not read "
        "rather than filling it in: an absent field comes back null with a warning, never "
        "as a plausible-looking value. parse_invoice can legitimately return no payload at "
        "all, either because the document is out of scope or because a mandatory field was "
        "unreadable. Read the warnings, not just the payload."
    ),
)


def _warning_dicts(warnings) -> list[dict[str, str]]:
    return [
        {"field": w.field, "severity": w.severity, "check": w.check, "message": w.message}
        for w in warnings
    ]


@server.tool(
    name="parse_invoice",
    title="Parse an Indian GST tax invoice into an INV-01 payload",
    description=(
        "Read one Indian GST tax invoice (PDF or image) and return the government's INV-01 "
        "JSON payload for it, together with the evidence behind every field.\n"
        "\n"
        "Accepts .pdf, .png, .jpg, .jpeg, .tif, .tiff and .bmp. Pages that carry a text "
        "layer are read directly; scanned pages go through OCR, which is slower (budget a "
        "few seconds per scanned page).\n"
        "\n"
        "Returns an object with four keys:\n"
        "  invoice          the INV-01 payload, or null when none could be produced\n"
        "  missing_fields   INV-01 paths that are mandatory but could not be read\n"
        "  refusals         why the document is out of scope, when it is\n"
        "  extraction_meta  per-page read method, per-field provenance, and warnings\n"
        "\n"
        "THREE OUTCOMES ARE NORMAL, AND ONLY THE FIRST GIVES YOU A PAYLOAD.\n"
        "1. A payload plus warnings. Usable, but read the warnings: they say which fields "
        "were read at low OCR confidence, which were assigned by document position rather "
        "than by a label, and which were derived rather than printed.\n"
        "2. No payload, with missing_fields populated. A mandatory field could not be read "
        "from the document. Nothing was invented to fill the gap, which is why there is no "
        "payload at all.\n"
        "3. No payload, with refusals populated. Export and SEZ invoices and foreign-currency "
        "invoices are refused by design; the refusal message says what was detected, which "
        "field revealed it, and what to do instead.\n"
        "\n"
        "The payload is submission-ready in shape, but this tool does NOT file it and does "
        "NOT return an IRN. An IRN is issued by the government's Invoice Registration Portal "
        "after you submit the payload there.\n"
        "\n"
        "Requires a Groq API key in the environment; stage 2 uses an LLM to read the line-item "
        "table, and every value it returns is checked back against the document text before "
        "it is kept."
    ),
)
def parse_invoice(path: str, tolerance: float = 0.05) -> dict[str, Any]:
    """Run the full pipeline over one invoice file.

    Args:
        path: Filesystem path to the invoice (PDF or image).
        tolerance: Rupee tolerance for the arithmetic checks. Defaults to 0.05.
    """
    result = extract_invoice(path, client=make_client(), model=MODEL, tolerance=tolerance)
    invoice = result.extraction.invoice if result.extraction else None
    return {
        "invoice": json.loads(invoice.model_dump_json(exclude_none=True)) if invoice else None,
        "missing_fields": list(result.missing_fields),
        "refusals": [
            {"kind": r.kind, "field": r.field, "evidence": r.evidence, "message": r.message}
            for r in result.refusals
        ],
        "extraction_meta": {
            "pages": [{"page": p.page, "method": p.method} for p in result.meta.pages],
            "field_provenance": result.meta.field_provenance,
            "warnings": _warning_dicts(result.meta.warnings),
        },
    }


@server.tool(
    name="validate_gstin",
    title="Validate a GSTIN's structure and check digit",
    description=(
        "Check one GSTIN (the 15-character Indian GST registration number) against its "
        "structural rules and its mod-36 check character, without touching any document.\n"
        "\n"
        "Returns is_valid, structural_ok, checksum_ok, the state code and its name, the "
        "embedded PAN, and a human-readable reason when it fails. checksum_ok is null, not "
        "false, when the structure itself failed: the checksum was never evaluated, and "
        "reporting two failures where one check ran would be misleading.\n"
        "\n"
        "State code 25 resolves to a distinct 'discontinued' status rather than 'unknown', "
        "because a 25 means either a misread or a genuine pre-2020 record, and the caller "
        "needs to tell those apart. State code 28 is legacy-but-valid: pre-2014 Andhra "
        "Pradesh registrations remain legitimate.\n"
        "\n"
        "IMPORTANT: a passing checksum proves the number is well-formed, NOT that the "
        "registration exists or is currently active. Only the GST portal can tell you that."
    ),
)
def validate_gstin(gstin: str) -> dict[str, Any]:
    """Validate a GSTIN structurally and by checksum.

    Args:
        gstin: The 15-character GSTIN to check.
    """
    outcome = _validate_gstin(gstin)
    payload: dict[str, Any] = {
        "is_valid": outcome.is_valid,
        "structural_ok": outcome.structural_ok,
        "checksum_ok": outcome.checksum_ok,
        "state_code": outcome.extracted_state_code,
        "pan": outcome.extracted_pan,
        "reason": outcome.reason,
    }
    if outcome.extracted_state_code is not None:
        state = lookup_state_code(outcome.extracted_state_code)
        payload["state"] = {
            "code": state.code,
            "name": state.name,
            "status": state.status,
            "is_valid": state.is_valid,
            "note": state.note,
        }
    return payload


@server.tool(
    name="validate_payload",
    title="Run the arithmetic and tax-split checks over an INV-01 payload",
    description=(
        "Take an INV-01 payload you already have and run the same consistency checks the "
        "parser runs, without re-reading any document. Useful for a payload you assembled "
        "yourself, or one you edited after parsing.\n"
        "\n"
        "Four checks run: each item's total against its own components; the invoice total "
        "against the value block; each value-block total against the sum of the item fields; "
        "and the CGST/SGST-versus-IGST split against the two parties' state codes.\n"
        "\n"
        "Comparisons use a rupee tolerance (0.05 by default), never exact equality, because "
        "real invoices round to the nearest rupee and exact comparison would flag almost "
        "every genuine document.\n"
        "\n"
        "Returns valid (true when nothing was flagged), the warning list with the exact field "
        "path for each, and schema_error when the payload does not fit INV-01 at all. The "
        "tax-split check reports itself as skipped, as an informational note rather than a "
        "warning, under reverse charge and when the place of supply differs from the buyer's "
        "registered state, because it cannot evaluate those cases."
    ),
)
def validate_payload(invoice: dict[str, Any], tolerance: float = 0.05) -> dict[str, Any]:
    """Validate an already-assembled INV-01 payload.

    Args:
        invoice: The INV-01 payload as a JSON object.
        tolerance: Rupee tolerance for the arithmetic checks. Defaults to 0.05.
    """
    try:
        parsed = Invoice.model_validate(invoice)
    except Exception as exc:
        return {
            "valid": False,
            "schema_error": str(exc),
            "warnings": [],
        }
    warnings = _validate_invoice(parsed, tolerance)
    return {
        "valid": not any(w.severity == "warning" for w in warnings),
        "schema_error": None,
        "warnings": _warning_dicts(warnings),
    }


def main() -> None:
    """Run the server.

    Defaults to stdio, the transport an MCP client launches locally. Over an HTTP
    transport the host and port come from the environment: the library defaults to
    127.0.0.1:8000, which a container platform cannot reach, so a deployment must
    bind 0.0.0.0 and whatever port the platform assigns in ``PORT``.
    """
    transport = os.environ.get("GST_MCP_TRANSPORT", "stdio")
    if transport in ("sse", "streamable-http"):
        server.run(
            transport=transport,
            host=os.environ.get("GST_MCP_HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8000")),
        )
        return
    server.run(transport=transport)


if __name__ == "__main__":
    main()
