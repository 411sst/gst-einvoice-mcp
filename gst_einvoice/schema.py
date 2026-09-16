"""INV-01 e-invoice models (GSTN schema v1.1) and the extraction-result wrapper.

Field names are the official GSTN abbreviations, exactly as they appear in the
INV-01 JSON schema (``TranDtls``, ``SupTyp``, ``LglNm`` ...), not readable
aliases. Only the mandatory blocks are modelled.

Every model forbids unknown keys (``extra="forbid"``), so the invoice payload
can never carry a field the government API would reject. Extraction metadata
(page info, provenance, warnings) lives beside the invoice in
``ExtractionResult``, never inside it.

The models carry data; they do not validate formats (GSTIN pattern, DD/MM/YYYY
dates, PIN length). Reporting such problems is the warnings block's job.

Submission form: ``invoice.model_dump(exclude_none=True)`` -- optional strings
that are absent are omitted rather than sent as null. Optional numerics default
to 0 (never None) so downstream arithmetic needs no null guards.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TranDtls(BaseModel):
    """Transaction details."""

    model_config = ConfigDict(extra="forbid")

    TaxSch: Literal["GST"] = "GST"
    SupTyp: Literal["B2B", "SEZWP", "SEZWOP", "EXPWP", "EXPWOP", "DEXP"]
    RegRev: Literal["Y", "N"] | None = None
    IgstOnIntra: Literal["Y", "N"] | None = None


class DocDtls(BaseModel):
    """Document details. ``Dt`` is DD/MM/YYYY per GSTN; format is not enforced here."""

    model_config = ConfigDict(extra="forbid")

    Typ: Literal["INV", "CRN", "DBN"]
    No: str
    Dt: str


class PartyDtls(BaseModel):
    """Fields shared by seller and buyer blocks."""

    model_config = ConfigDict(extra="forbid")

    Gstin: str
    LglNm: str
    TrdNm: str | None = None
    Addr1: str
    Addr2: str | None = None
    Loc: str
    Pin: int
    Stcd: str


class SellerDtls(PartyDtls):
    """Seller details."""


class BuyerDtls(PartyDtls):
    """Buyer details; ``Pos`` (place of supply state code) is mandatory."""

    Pos: str


class Item(BaseModel):
    """One ``ItemList`` entry."""

    model_config = ConfigDict(extra="forbid")

    SlNo: str
    PrdDesc: str
    IsServc: Literal["Y", "N"]
    HsnCd: str
    Qty: float
    Unit: str
    UnitPrice: float
    TotAmt: float
    Discount: float = 0
    AssAmt: float
    GstRt: float
    CgstAmt: float
    SgstAmt: float
    IgstAmt: float
    CesAmt: float = 0
    StateCesAmt: float = 0
    OthChrg: float = 0
    TotItemVal: float


class ValDtls(BaseModel):
    """Invoice-level value totals."""

    model_config = ConfigDict(extra="forbid")

    AssVal: float
    CgstVal: float
    SgstVal: float
    IgstVal: float
    CesVal: float = 0
    StCesVal: float = 0
    RndOffAmt: float = 0
    TotInvVal: float


class Invoice(BaseModel):
    """The INV-01 payload. ``Version`` is the schema's mandatory top-level version string."""

    model_config = ConfigDict(extra="forbid")

    Version: str = "1.1"
    TranDtls: TranDtls
    DocDtls: DocDtls
    SellerDtls: SellerDtls
    BuyerDtls: BuyerDtls
    ItemList: list[Item] = Field(min_length=1)
    ValDtls: ValDtls


class ExtractionWarning(BaseModel):
    """One problem found while extracting or validating an invoice."""

    model_config = ConfigDict(extra="forbid")

    field: str
    message: str
    severity: Literal["warning", "info"] = "warning"
    check: str


class PageMeta(BaseModel):
    """How one page (1-based) of the source document was read."""

    model_config = ConfigDict(extra="forbid")

    page: int
    method: Literal["native", "ocr_required"]


class ExtractionMeta(BaseModel):
    """Everything about the extraction that is not part of the INV-01 payload."""

    model_config = ConfigDict(extra="forbid")

    pages: list[PageMeta] = Field(default_factory=list)
    field_provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    #: How many stage-2 attempts this result took. 1 for almost every run; 2 when
    #: the pipeline retried a no-payload, no-refusal outcome. Recorded so a change
    #: in the underlying failure rate can never hide behind a silent retry.
    attempts: int = 1


class ExtractionResult(BaseModel):
    """Tool output: a spec-pure invoice plus its extraction metadata."""

    model_config = ConfigDict(extra="forbid")

    invoice: Invoice
    extraction_meta: ExtractionMeta
