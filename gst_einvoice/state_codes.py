"""GST state codes (the first two digits of a GSTIN) and a lookup that
distinguishes active, legacy, discontinued and unknown codes."""

import re
from dataclasses import dataclass
from typing import Literal

STATE_CODES: dict[str, str] = {
    "01": "J&K",
    "02": "Himachal Pradesh",
    "03": "Punjab",
    "04": "Chandigarh",
    "05": "Uttarakhand",
    "06": "Haryana",
    "07": "Delhi",
    "08": "Rajasthan",
    "09": "Uttar Pradesh",
    "10": "Bihar",
    "11": "Sikkim",
    "12": "Arunachal Pradesh",
    "13": "Nagaland",
    "14": "Manipur",
    "15": "Mizoram",
    "16": "Tripura",
    "17": "Meghalaya",
    "18": "Assam",
    "19": "West Bengal",
    "20": "Jharkhand",
    "21": "Odisha",
    "22": "Chhattisgarh",
    "23": "Madhya Pradesh",
    "24": "Gujarat",
    "26": "Dadra & Nagar Haveli and Daman & Diu",
    "27": "Maharashtra",
    "28": "Andhra Pradesh (legacy, pre-2014)",
    "29": "Karnataka",
    "30": "Goa",
    "31": "Lakshadweep",
    "32": "Kerala",
    "33": "Tamil Nadu",
    "34": "Puducherry",
    "35": "Andaman & Nicobar",
    "36": "Telangana",
    "37": "Andhra Pradesh (current)",
    "38": "Ladakh",
    "97": "Other Territory / UN bodies",
    "99": "Foreign (OIDAR)",
}

_DISCONTINUED_NOTE = (
    "State code 25 (Daman & Diu) was discontinued in 2020 when it merged into "
    "26 (Dadra & Nagar Haveli and Daman & Diu); code-26 GSTINs took effect "
    "from 1 August 2020. Treat a 25 as either an OCR misread (25 vs 26, 28 or "
    "35) or a stale record from before the switch-over. To tell them apart, "
    "check the document date: on a document dated on or before 31 July 2020 a "
    "25 can be genuine; on a document dated 1 August 2020 or later it cannot "
    "be a genuine code-25 registration, so either the digits were misread "
    "(re-read the GSTIN; the true code may be 26, 28 or 35) or the supplier is "
    "still printing their pre-merger GSTIN (ask them for their migrated "
    "code-26 GSTIN). Then verify the GSTIN checksum: a misread digit usually "
    "breaks it, so a passing checksum points to a genuine pre-merger record "
    "before the switch-over, or to a supplier still printing the ceased GSTIN "
    "after it."
)

_LEGACY_NOTE = (
    "State code 28 is the pre-2014 Andhra Pradesh code from before the "
    "Telangana split. Registrations issued under it remain legitimate. "
    "Current Andhra Pradesh registrations use 37 (Telangana uses 36)."
)


@dataclass(frozen=True)
class StateCodeResult:
    """Outcome of a state code lookup. is_valid is True only for active and legacy codes."""

    code: str
    name: str | None
    status: Literal["active", "legacy", "discontinued", "unknown"]
    is_valid: bool
    note: str | None


def lookup_state_code(code: str) -> StateCodeResult:
    """Resolve a two-digit GST state code; surrounding whitespace is ignored."""
    stripped = code.strip()
    if not re.fullmatch(r"[0-9]{2}", stripped):
        return StateCodeResult(
            code=stripped,
            name=None,
            status="unknown",
            is_valid=False,
            note=f"state codes are exactly two ASCII digits 0-9, got {stripped!r}",
        )
    if stripped == "25":
        return StateCodeResult(
            code=stripped,
            name="Daman & Diu",
            status="discontinued",
            is_valid=False,
            note=_DISCONTINUED_NOTE,
        )
    if stripped == "28":
        return StateCodeResult(
            code=stripped,
            name=STATE_CODES[stripped],
            status="legacy",
            is_valid=True,
            note=_LEGACY_NOTE,
        )
    name = STATE_CODES.get(stripped)
    if name is None:
        return StateCodeResult(
            code=stripped,
            name=None,
            status="unknown",
            is_valid=False,
            note=f"no GST state code {stripped}",
        )
    return StateCodeResult(
        code=stripped, name=name, status="active", is_valid=True, note=None
    )
