"""Tests for the MCP server layer.

The server is a thin adapter: it must not re-implement anything, and it must not
flatten the distinction between "no payload because the document is out of scope"
and "no payload because a mandatory field was unreadable". Both come back with
invoice=None, and only refusals/missing_fields tell them apart.
"""

import asyncio
import json
import os

import pymupdf
import pytest

from gst_einvoice import server
from gst_einvoice.schema import Invoice

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

CLEAN_TEXT = (
    "TAX INVOICE\n"
    "Seller: Nimbus Components Pvt Ltd\n"
    "GSTIN 27AAPFU0939F1ZV\n"
    "Plot 14 MIDC Andheri East\n"
    "Mumbai 400093\n"
    "Invoice No: INV-2026-0042\n"
    "Invoice Date: 17/04/2026\n"
    "Bill To: Kanchan Electricals LLP\n"
    "GSTIN 27AABCB5507N1ZJ\n"
    "221 Laxmi Road Shivajinagar\n"
    "Pune 411005\n"
    "Sl 1 Laptop Stand HSN/SAC: 8471 Qty 4 NOS Rate 1500.00\n"
    "Taxable 6000.00  GST 18%  CGST 540.00  SGST 540.00  IGST 0.00  Line Total 7080.00\n"
    "Taxable 6000.00  CGST 540.00  SGST 540.00  IGST 0.00\n"
    "Total Invoice Value 7080.00\n"
)

# Transcribed by hand from CLEAN_TEXT above, not produced by the module under test.
VALID_PAYLOAD = {
    "Version": "1.1",
    "TranDtls": {"TaxSch": "GST", "SupTyp": "B2B"},
    "DocDtls": {"Typ": "INV", "No": "INV-2026-0042", "Dt": "17/04/2026"},
    "SellerDtls": {"Gstin": "27AAPFU0939F1ZV", "LglNm": "Nimbus Components Pvt Ltd",
                   "Addr1": "Plot 14 MIDC Andheri East", "Loc": "Mumbai",
                   "Pin": 400093, "Stcd": "27"},
    "BuyerDtls": {"Gstin": "27AABCB5507N1ZJ", "LglNm": "Kanchan Electricals LLP",
                  "Addr1": "221 Laxmi Road Shivajinagar", "Loc": "Pune",
                  "Pin": 411005, "Stcd": "27", "Pos": "27"},
    "ItemList": [{"SlNo": "1", "PrdDesc": "Laptop Stand", "IsServc": "N", "HsnCd": "8471",
                  "Qty": 4.0, "Unit": "NOS", "UnitPrice": 1500.0, "TotAmt": 6000.0,
                  "Discount": 0, "AssAmt": 6000.0, "GstRt": 18.0, "CgstAmt": 540.0,
                  "SgstAmt": 540.0, "IgstAmt": 0.0, "CesAmt": 0, "StateCesAmt": 0,
                  "OthChrg": 0, "TotItemVal": 7080.0}],
    "ValDtls": {"AssVal": 6000.0, "CgstVal": 540.0, "SgstVal": 540.0, "IgstVal": 0.0,
                "CesVal": 0, "StCesVal": 0, "RndOffAmt": 0, "TotInvVal": 7080.0},
}

LLM_RESPONSE = {
    "items": [{"SlNo": "1", "PrdDesc": "Laptop Stand", "HsnCd": "8471", "Qty": 4,
               "Unit": "NOS", "UnitPrice": 1500.00, "TotAmt": 6000.00, "Discount": None,
               "AssAmt": 6000.00, "GstRt": 18, "CgstAmt": 540.00, "SgstAmt": 540.00,
               "IgstAmt": 0.00, "CesAmt": None, "TotItemVal": 7080.00}],
    "seller": {"LglNm": "Nimbus Components Pvt Ltd", "Addr1": "Plot 14 MIDC Andheri East",
               "Loc": "Mumbai", "Pin": "400093"},
    "buyer": {"LglNm": "Kanchan Electricals LLP", "Addr1": "221 Laxmi Road Shivajinagar",
              "Loc": "Pune", "Pin": "411005"},
    "totals": {"AssVal": 6000.00, "CgstVal": 540.00, "SgstVal": 540.00, "IgstVal": 0.00,
               "CesVal": None, "TotInvVal": 7080.00, "RndOffAmt": None},
}


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def create(self, **kwargs):
        self.calls += 1
        body = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return _Response(body)


class _Chat:
    def __init__(self, payload):
        self.completions = _Completions(payload)


class FakeClient:
    """Stands in for a Groq client. No network, no key."""

    def __init__(self, payload):
        self.chat = _Chat(payload)


@pytest.fixture
def fake_groq(monkeypatch):
    """Point the server's client factory at a fake, so no test needs a key."""
    holder = {}

    def install(payload):
        client = FakeClient(payload)
        holder["client"] = client
        monkeypatch.setattr(server, "make_client", lambda *a, **k: client)
        return client

    return install


def _text_pdf(path, text):
    doc = pymupdf.open()
    doc.new_page().insert_text((40, 50), text, fontsize=10)
    doc.save(str(path))
    doc.close()
    return str(path)


# --------------------------------------------------------------------------- #
# The three tools are registered, with descriptions an agent can act on
# --------------------------------------------------------------------------- #


def _listed_tools():
    """The tool list as an MCP client actually receives it."""
    return asyncio.run(server.server.list_tools())


def test_all_three_tools_are_registered():
    names = {t.name for t in _listed_tools()}
    assert names == {"parse_invoice", "validate_gstin", "validate_payload"}


def test_each_tool_advertises_the_arguments_it_takes():
    by_name = {t.name: t for t in _listed_tools()}
    assert sorted(by_name["parse_invoice"].input_schema["properties"]) == ["path", "tolerance"]
    assert by_name["parse_invoice"].input_schema["required"] == ["path"]
    assert by_name["validate_gstin"].input_schema["required"] == ["gstin"]
    assert by_name["validate_payload"].input_schema["required"] == ["invoice"]


def test_the_server_instructions_warn_that_a_payload_is_not_guaranteed():
    text = (server.server.instructions or "").lower()
    assert "null with a warning" in text
    assert "no payload" in text


@pytest.mark.parametrize(
    "name,must_mention",
    [
        # An agent decides whether to call from the description alone, so the
        # descriptions carry the facts that change what a caller does next.
        ("parse_invoice", ["INV-01", "missing_fields", "refusals", "IRN", "warnings"]),
        ("validate_gstin", ["checksum", "state code", "active"]),
        ("validate_payload", ["tolerance", "field path", "skipped"]),
    ],
)
def test_tool_descriptions_state_what_the_caller_needs(name, must_mention):
    tool = next(t for t in _listed_tools() if t.name == name)
    text = (tool.description or "")
    for phrase in must_mention:
        assert phrase.lower() in text.lower(), f"{name} description omits {phrase!r}"


def test_parse_invoice_description_says_a_payload_is_not_guaranteed():
    tool = next(t for t in _listed_tools() if t.name == "parse_invoice")
    text = (tool.description or "").lower()
    assert "three outcomes" in text
    assert "does not return an irn" in text


# --------------------------------------------------------------------------- #
# parse_invoice
# --------------------------------------------------------------------------- #


def test_parse_invoice_returns_the_payload_and_its_evidence(tmp_path, fake_groq):
    fake_groq(LLM_RESPONSE)
    out = server.parse_invoice(_text_pdf(tmp_path / "clean.pdf", CLEAN_TEXT))
    assert out["invoice"] is not None
    assert out["missing_fields"] == []
    assert out["refusals"] == []
    # Hand-transcribed from CLEAN_TEXT, not read back from the module.
    assert out["invoice"]["DocDtls"]["No"] == "INV-2026-0042"
    assert out["invoice"]["DocDtls"]["Dt"] == "17/04/2026"
    assert out["invoice"]["SellerDtls"]["Stcd"] == "27"
    assert out["invoice"]["BuyerDtls"]["Stcd"] == "27"
    assert out["invoice"]["ValDtls"]["TotInvVal"] == 7080.0
    assert out["invoice"]["ItemList"][0]["HsnCd"] == "8471"
    meta = out["extraction_meta"]
    assert meta["pages"] == [{"page": 1, "method": "native"}]
    assert meta["field_provenance"]["SellerDtls.Gstin"]["source"] == "regex"
    assert all({"field", "severity", "check", "message"} == set(w) for w in meta["warnings"])


def test_parse_invoice_is_json_serialisable(tmp_path, fake_groq):
    """The transport serialises the result, so nothing may be a pydantic object."""
    fake_groq(LLM_RESPONSE)
    out = server.parse_invoice(_text_pdf(tmp_path / "clean.pdf", CLEAN_TEXT))
    json.dumps(out)


def test_parse_invoice_reports_a_missing_mandatory_field_without_a_payload(tmp_path, fake_groq):
    response = json.loads(json.dumps(LLM_RESPONSE))
    response["buyer"]["LglNm"] = None
    fake_groq(response)
    out = server.parse_invoice(_text_pdf(tmp_path / "nolglnm.pdf", CLEAN_TEXT))
    assert out["invoice"] is None
    assert "BuyerDtls.LglNm" in out["missing_fields"]
    assert out["refusals"] == []


def test_parse_invoice_surfaces_a_refusal_and_never_calls_the_model(tmp_path, fake_groq):
    client = fake_groq(LLM_RESPONSE)
    text = CLEAN_TEXT.replace("Total Invoice Value 7080.00", "Grand Total USD 2950.00")
    out = server.parse_invoice(_text_pdf(tmp_path / "usd.pdf", text))
    assert out["invoice"] is None
    assert [r["kind"] for r in out["refusals"]] == ["multi_currency"]
    assert out["refusals"][0]["field"] == "ValDtls.TotInvValFc"
    assert client.chat.completions.calls == 0


def test_a_refusal_and_an_unreadable_field_are_distinguishable(tmp_path, fake_groq):
    """Both give invoice=None. Only refusals/missing_fields say which happened."""
    response = json.loads(json.dumps(LLM_RESPONSE))
    response["buyer"]["LglNm"] = None
    fake_groq(response)
    unreadable = server.parse_invoice(_text_pdf(tmp_path / "a.pdf", CLEAN_TEXT))
    text = CLEAN_TEXT.replace("Total Invoice Value 7080.00", "Grand Total USD 2950.00")
    refused = server.parse_invoice(_text_pdf(tmp_path / "b.pdf", text))
    assert unreadable["invoice"] is refused["invoice"] is None
    assert unreadable["missing_fields"] and not unreadable["refusals"]
    assert refused["refusals"] and not refused["missing_fields"]


def test_parse_invoice_passes_the_tolerance_through(tmp_path, fake_groq):
    response = json.loads(json.dumps(LLM_RESPONSE))
    response["totals"]["TotInvVal"] = 7085.00
    text = CLEAN_TEXT.replace("Total Invoice Value 7080.00", "Total Invoice Value 7085.00")
    fake_groq(response)
    strict = server.parse_invoice(_text_pdf(tmp_path / "t1.pdf", text))
    assert any(w["check"] == "validate_invoice_total" for w in strict["extraction_meta"]["warnings"])
    fake_groq(response)
    loose = server.parse_invoice(_text_pdf(tmp_path / "t2.pdf", text), tolerance=100.0)
    assert not any(w["check"] == "validate_invoice_total"
                   for w in loose["extraction_meta"]["warnings"])


# --------------------------------------------------------------------------- #
# validate_gstin
# --------------------------------------------------------------------------- #


def test_validate_gstin_accepts_a_real_one():
    out = server.validate_gstin("27AAPFU0939F1ZV")
    assert out["is_valid"] is True
    assert out["structural_ok"] is True
    assert out["checksum_ok"] is True
    assert out["state_code"] == "27"
    assert out["pan"] == "AAPFU0939F"
    assert out["reason"] is None
    assert out["state"]["name"] == "Maharashtra"
    assert out["state"]["status"] == "active"


def test_validate_gstin_reports_a_checksum_failure_with_the_state_still_known():
    # Structurally fine, wrong check character: build 1's ground truth.
    out = server.validate_gstin("29AAACT2727Q1ZW")
    assert out["is_valid"] is False
    assert out["structural_ok"] is True
    assert out["checksum_ok"] is False
    assert "checksum" in out["reason"].lower()
    assert out["state"]["name"] == "Karnataka"


def test_a_structural_failure_leaves_checksum_ok_null_not_false():
    out = server.validate_gstin("27AAPFU0939F1AV")
    assert out["structural_ok"] is False
    assert out["checksum_ok"] is None
    assert out["state_code"] is None
    assert "state" not in out


def test_validate_gstin_reports_state_25_as_discontinued():
    out = server.validate_gstin("25AAPFU0939F1ZG")
    assert out["state"]["status"] == "discontinued"
    assert out["state"]["is_valid"] is False
    assert "2020" in out["state"]["note"]


def test_validate_gstin_reports_state_28_as_legacy_but_valid():
    out = server.validate_gstin("28AAPFU0939F1ZK")
    assert out["state"]["status"] == "legacy"
    assert out["state"]["is_valid"] is True


def test_validate_gstin_is_json_serialisable():
    json.dumps(server.validate_gstin("27AAPFU0939F1ZV"))
    json.dumps(server.validate_gstin("nonsense"))


# --------------------------------------------------------------------------- #
# validate_payload
# --------------------------------------------------------------------------- #


def test_validate_payload_passes_a_consistent_invoice():
    out = server.validate_payload(VALID_PAYLOAD)
    assert out["valid"] is True
    assert out["schema_error"] is None
    assert [w for w in out["warnings"] if w["severity"] == "warning"] == []


def test_validate_payload_flags_an_inconsistent_total_with_its_field_path():
    payload = json.loads(json.dumps(VALID_PAYLOAD))
    payload["ValDtls"]["TotInvVal"] = 7085.00
    out = server.validate_payload(payload)
    assert out["valid"] is False
    flagged = [w for w in out["warnings"] if w["check"] == "validate_invoice_total"]
    assert len(flagged) == 1
    assert flagged[0]["field"] == "ValDtls.TotInvVal"


def test_validate_payload_honours_the_tolerance():
    payload = json.loads(json.dumps(VALID_PAYLOAD))
    payload["ValDtls"]["TotInvVal"] = 7080.02
    assert server.validate_payload(payload)["valid"] is True
    payload["ValDtls"]["TotInvVal"] = 7085.00
    assert server.validate_payload(payload)["valid"] is False
    assert server.validate_payload(payload, tolerance=100.0)["valid"] is True


def test_validate_payload_reports_a_schema_error_rather_than_raising():
    out = server.validate_payload({"not": "an invoice"})
    assert out["valid"] is False
    assert out["schema_error"]
    assert out["warnings"] == []


def test_validate_payload_rejects_an_unknown_field_inside_the_invoice():
    """The payload goes to a government API that refuses unknown keys."""
    payload = json.loads(json.dumps(VALID_PAYLOAD))
    payload["ValDtls"]["MadeUpField"] = 1
    out = server.validate_payload(payload)
    assert out["valid"] is False
    assert out["schema_error"]


def test_validate_payload_reports_a_skip_as_info_not_a_warning():
    payload = json.loads(json.dumps(VALID_PAYLOAD))
    payload["TranDtls"]["RegRev"] = "Y"
    out = server.validate_payload(payload)
    skips = [w for w in out["warnings"] if w["field"] == "TranDtls.RegRev"]
    assert len(skips) == 1
    assert skips[0]["severity"] == "info"
    assert out["valid"] is True


def test_validate_payload_round_trips_what_parse_invoice_produced(tmp_path, fake_groq):
    """The two tools agree: a payload the parser emits validates on its own."""
    fake_groq(LLM_RESPONSE)
    parsed = server.parse_invoice(_text_pdf(tmp_path / "clean.pdf", CLEAN_TEXT))
    assert parsed["invoice"] is not None
    out = server.validate_payload(parsed["invoice"])
    assert out["schema_error"] is None
    assert out["valid"] is True


def test_the_hand_transcribed_payload_is_a_real_inv01_invoice():
    """Guards the fixture itself: a typo here would weaken every test above."""
    Invoice.model_validate(VALID_PAYLOAD)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_model_is_configurable_and_falls_back_to_the_library_default():
    from gst_einvoice.extract_llm import DEFAULT_MODEL

    assert server.MODEL == (os.environ.get("GST_MCP_MODEL") or DEFAULT_MODEL)


@pytest.fixture
def captured_run(monkeypatch):
    """Record what main() would pass to run(), without opening a socket."""
    calls = []
    monkeypatch.setattr(server.server, "run", lambda **kw: calls.append(kw))
    return calls


def test_main_defaults_to_stdio_with_no_host_or_port(monkeypatch, captured_run):
    monkeypatch.delenv("GST_MCP_TRANSPORT", raising=False)
    server.main()
    assert captured_run == [{"transport": "stdio"}]


@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
def test_an_http_transport_binds_all_interfaces_and_the_assigned_port(
    transport, monkeypatch, captured_run
):
    """The library default is 127.0.0.1:8000, which a container platform cannot reach.

    Render and every other platform hand the port in PORT and expect 0.0.0.0, so a
    deployment that took the defaults would come up healthy and be unreachable.
    """
    monkeypatch.setenv("GST_MCP_TRANSPORT", transport)
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.delenv("GST_MCP_HOST", raising=False)
    server.main()
    assert captured_run == [{"transport": transport, "host": "0.0.0.0", "port": 10000}]


def test_the_http_host_can_be_overridden(monkeypatch, captured_run):
    monkeypatch.setenv("GST_MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("GST_MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9001")
    server.main()
    assert captured_run[0]["host"] == "127.0.0.1"
    assert captured_run[0]["port"] == 9001


def test_the_http_app_exposes_only_the_mcp_route():
    """render.yaml sets no healthCheckPath because there is no /health to point it at."""
    app = server.server.streamable_http_app()
    paths = {getattr(r, "path", None) for r in app.routes}
    assert paths == {"/mcp"}
