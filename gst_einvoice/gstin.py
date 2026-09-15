"""GSTIN validation: structural rules plus the mod-36 check character."""

import re
from dataclasses import dataclass

_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The 15-character structure, one rule per segment so a failure can name its position.
# (first position, last position, pattern, rule text); positions are 1-based, inclusive.
_SEGMENTS = (
    (1, 2, r"[0-9]{2}", "positions 1-2 must be digits (state code)"),
    (3, 7, r"[A-Z]{5}", "positions 3-7 must be letters"),
    (8, 11, r"[0-9]{4}", "positions 8-11 must be digits"),
    (12, 12, r"[A-Z]", "position 12 must be a letter"),
    (13, 13, r"[0-9A-Z]", "position 13 must be alphanumeric"),
    (14, 14, r"Z", "position 14 must be the literal 'Z'"),
    (15, 15, r"[0-9A-Z]", "position 15 must be an alphanumeric check digit"),
)


@dataclass(frozen=True)
class GstinValidation:
    """Outcome of validate_gstin. checksum_ok is None when the structural check failed."""

    is_valid: bool
    structural_ok: bool
    checksum_ok: bool | None
    extracted_state_code: str | None
    extracted_pan: str | None
    reason: str | None


def compute_check_digit(first14: str) -> str:
    """Return the mod-36 check character for the first 14 characters of a GSTIN."""
    if not isinstance(first14, str) or len(first14) != 14 or any(c not in _ALPHABET for c in first14):
        raise ValueError("first14 must be exactly 14 characters drawn from 0-9 and A-Z")
    total = 0
    for index, char in enumerate(first14):
        factor = 1 if index % 2 == 0 else 2  # position 1 x1, position 2 x2, ...
        quotient, remainder = divmod(_ALPHABET.index(char) * factor, 36)
        total += quotient + remainder
    return _ALPHABET[(36 - total % 36) % 36]


def validate_gstin(gstin: str) -> GstinValidation:
    """Validate a GSTIN structurally and by checksum. Never raises on bad input."""
    if not isinstance(gstin, str):
        return _structural_failure(f"expected a string, got {type(gstin).__name__}")
    gstin = gstin.strip()
    if len(gstin) != 15:
        return _structural_failure(f"length must be 15 characters, got {len(gstin)}")
    for first, last, pattern, rule in _SEGMENTS:
        segment = gstin[first - 1 : last]
        if re.fullmatch(pattern, segment):
            continue
        # If the segment passes case-insensitively, the defect is case alone: name the first
        # lowercase position rather than claiming 'aapfu' is not letters. re.ASCII keeps the
        # match to a-z, so look-alikes such as 'ı' or 'ſ' fall through to the rule text.
        if re.fullmatch(pattern, segment, re.IGNORECASE | re.ASCII):
            offset = next(i for i, char in enumerate(segment) if char.islower())
            return _structural_failure(
                f"position {first + offset} must be uppercase, got lowercase '{segment[offset]}'"
            )
        return _structural_failure(f"{rule}, got '{segment}'")

    state_code, pan = gstin[:2], gstin[2:12]
    computed = compute_check_digit(gstin[:14])
    if computed != gstin[14]:
        return GstinValidation(
            is_valid=False,
            structural_ok=True,
            checksum_ok=False,
            extracted_state_code=state_code,
            extracted_pan=pan,
            reason=f"checksum mismatch: computed check character '{computed}' but position 15 is '{gstin[14]}'",
        )
    return GstinValidation(
        is_valid=True,
        structural_ok=True,
        checksum_ok=True,
        extracted_state_code=state_code,
        extracted_pan=pan,
        reason=None,
    )


def _structural_failure(reason: str) -> GstinValidation:
    return GstinValidation(
        is_valid=False,
        structural_ok=False,
        checksum_ok=None,
        extracted_state_code=None,
        extracted_pan=None,
        reason=reason,
    )
