"""Tests for gst_einvoice.gstin: structure, mod-36 checksum, and the GstinValidation contract."""

import dataclasses

import pytest

from gst_einvoice.gstin import GstinValidation, compute_check_digit, validate_gstin

VALID = ["27AAPFU0939F1ZV", "07AAGFF2194N1Z1", "24AAACC1206D1ZM", "09AAACH7409R1ZZ"]

# Structurally fine, but position 15 is not the check character the algorithm produces.
CHECKSUM_FAILS = [
    ("29AAACT2727Q1ZW", "S"),
    ("33AAACT2727Q1ZS", "3"),
    ("36AAACR5055K1Z5", "8"),
    ("19AABCT1332L1ZU", "B"),
]


# --- valid GSTINs -----------------------------------------------------------


@pytest.mark.parametrize("gstin", VALID)
def test_valid_gstin(gstin):
    result = validate_gstin(gstin)
    assert result == GstinValidation(
        is_valid=True,
        structural_ok=True,
        checksum_ok=True,
        extracted_state_code=gstin[:2],
        extracted_pan=gstin[2:12],
        reason=None,
    )


def test_valid_gstin_extracts_state_code_and_pan():
    result = validate_gstin("27AAPFU0939F1ZV")
    assert result.extracted_state_code == "27"
    assert result.extracted_pan == "AAPFU0939F"


# Position 13 is alphanumeric, but every ground-truth fixture carries '1' there; pin the
# letter branch and the digit '0'. Check characters were computed by hand from the spec.
@pytest.mark.parametrize("gstin, expected_check", [("27AAPFU0939FAZM", "M"), ("27AAPFU0939F0ZW", "W")])
def test_position_13_letter_and_digit_are_valid(gstin, expected_check):
    assert compute_check_digit(gstin[:14]) == expected_check
    result = validate_gstin(gstin)
    assert result.is_valid is True
    assert result.structural_ok is True
    assert result.checksum_ok is True
    assert result.reason is None


# --- checksum failures ------------------------------------------------------


@pytest.mark.parametrize("gstin, expected_check", CHECKSUM_FAILS)
def test_checksum_failure(gstin, expected_check):
    result = validate_gstin(gstin)
    assert result.is_valid is False
    assert result.structural_ok is True
    assert result.checksum_ok is False
    assert result.extracted_state_code == gstin[:2]
    assert result.extracted_pan == gstin[2:12]
    assert result.reason == (
        f"checksum mismatch: computed check character '{expected_check}' but position 15 is '{gstin[14]}'"
    )


def test_checksum_failure_reason_exact_text():
    assert validate_gstin("29AAACT2727Q1ZW").reason == (
        "checksum mismatch: computed check character 'S' but position 15 is 'W'"
    )


# --- compute_check_digit ----------------------------------------------------


@pytest.mark.parametrize("gstin", VALID)
def test_compute_check_digit_matches_valid_gstins(gstin):
    assert compute_check_digit(gstin[:14]) == gstin[14]


@pytest.mark.parametrize("gstin, expected_check", CHECKSUM_FAILS)
def test_compute_check_digit_for_checksum_failures(gstin, expected_check):
    assert compute_check_digit(gstin[:14]) == expected_check


def test_compute_check_digit_worked_example():
    # 27AAPFU0939F1Z: values 2,7,10,10,25,15,30,0,9,3,9,15,1,35 with factors 1,2,1,2,...
    # products 2,14,10,20,25,30,30,0,9,6,9,30,1,70 -> quotient+remainder sums to 221
    # 221 mod 36 = 5 -> (36 - 5) mod 36 = 31 -> 'V'
    assert compute_check_digit("27AAPFU0939F1Z") == "V"


@pytest.mark.parametrize(
    "bad",
    [
        "27AAPFU0939F1",  # 13 chars
        "27AAPFU0939F1ZV",  # 15 chars
        "",
        "27aapfu0939f1z",  # lowercase
        "27AAPFU0939F-Z",  # non-alphanumeric
        "27AAPFU0939F1 ",  # whitespace
        None,
        12345678901234,
    ],
)
def test_compute_check_digit_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        compute_check_digit(bad)


# --- structural failures ----------------------------------------------------


def assert_structural_failure(result, reason):
    assert result == GstinValidation(
        is_valid=False,
        structural_ok=False,
        checksum_ok=None,
        extracted_state_code=None,
        extracted_pan=None,
        reason=reason,
    )


@pytest.mark.parametrize(
    "gstin, reason",
    [
        ("27AAPFU0939F1Z", "length must be 15 characters, got 14"),
        ("27AAPFU0939F1ZVV", "length must be 15 characters, got 16"),
        ("", "length must be 15 characters, got 0"),
        ("A7AAPFU0939F1ZV", "positions 1-2 must be digits (state code), got 'A7'"),
        ("271B2CD0939F1ZV", "positions 3-7 must be letters, got '1B2CD'"),
        ("27AAPFU09A9F1ZV", "positions 8-11 must be digits, got '09A9'"),
        ("27AAPFU093911ZV", "position 12 must be a letter, got '1'"),
        ("27AAPFU0939F-ZV", "position 13 must be alphanumeric, got '-'"),
        ("27AAPFU0939F1AV", "position 14 must be the literal 'Z', got 'A'"),
        ("27AAPFU0939F1Z-", "position 15 must be an alphanumeric check digit, got '-'"),
    ],
)
def test_structural_failure(gstin, reason):
    assert_structural_failure(validate_gstin(gstin), reason)


def test_structural_failure_reports_first_failing_rule():
    # Both positions 1-2 and position 14 are wrong; the earliest rule is reported.
    assert validate_gstin("A7AAPFU0939F1AV").reason == (
        "positions 1-2 must be digits (state code), got 'A7'"
    )


def test_structural_failure_does_not_evaluate_checksum():
    # Structurally broken, so checksum_ok is None (not evaluated), never False.
    result = validate_gstin("27AAPFU0939F1AV")
    assert result.structural_ok is False
    assert result.checksum_ok is None


def test_lowercase_is_structural_failure_and_reason_says_so():
    result = validate_gstin("27aapfu0939f1zv")  # uppercase form is valid
    assert_structural_failure(result, "position 3 must be uppercase, got lowercase 'a'")


# A lowercase letter is only reported as a case defect when upper-casing it would satisfy
# the segment rule; the first offending position is named.
@pytest.mark.parametrize(
    "gstin, reason",
    [
        ("27AAPFU0939F1Zv", "position 15 must be uppercase, got lowercase 'v'"),
        ("27AaPFU0939F1ZV", "position 4 must be uppercase, got lowercase 'a'"),
        ("27AAPFU0939f1ZV", "position 12 must be uppercase, got lowercase 'f'"),
        ("27AAPFU0939FaZV", "position 13 must be uppercase, got lowercase 'a'"),
        ("27AAPFU0939F1zV", "position 14 must be uppercase, got lowercase 'z'"),
    ],
)
def test_lowercase_letter_is_reported_by_position(gstin, reason):
    assert_structural_failure(validate_gstin(gstin), reason)


def test_lowercase_letter_in_a_digit_slot_reports_the_digit_rule():
    # Upper-casing '2a' would not repair it, so the digit rule is the defect, not the case.
    assert_structural_failure(
        validate_gstin("2aAAPFU0939F1ZV"), "positions 1-2 must be digits (state code), got '2a'"
    )


# Non-ASCII look-alikes (as pasted from a PDF) fail the ASCII-only segment ranges and are
# reported by the segment rule; the case-only branch must never fire for them.
@pytest.mark.parametrize(
    "gstin, reason",
    [
        ("27АAPFU0939F1ZV", "positions 3-7 must be letters, got 'АAPFU'"),  # Cyrillic A
        ("27AAPFU0939F1ΖV", "position 14 must be the literal 'Z', got 'Ζ'"),  # Greek Zeta
        ("２７AAPFU0939F1ZV", "positions 1-2 must be digits (state code), got '２７'"),  # fullwidth
        ("27ıAPFU0939F1ZV", "positions 3-7 must be letters, got 'ıAPFU'"),  # 'ı'.upper() == 'I'
        ("27AAPFU0939F1Zа", "position 15 must be an alphanumeric check digit, got 'а'"),  # Cyrillic a
    ],
)
def test_non_ascii_lookalike_is_a_structural_failure(gstin, reason):
    assert_structural_failure(validate_gstin(gstin), reason)


# --- input handling ---------------------------------------------------------


@pytest.mark.parametrize("padded", [" 27AAPFU0939F1ZV", "27AAPFU0939F1ZV ", "\t27AAPFU0939F1ZV\n", "  27AAPFU0939F1ZV  "])
def test_surrounding_whitespace_is_stripped(padded):
    assert validate_gstin(padded).is_valid is True


def test_internal_whitespace_is_not_stripped():
    assert_structural_failure(validate_gstin("27AAPFU 0939F1ZV"), "length must be 15 characters, got 16")


def test_whitespace_only_is_length_failure():
    assert_structural_failure(validate_gstin("   "), "length must be 15 characters, got 0")


@pytest.mark.parametrize("value, type_name", [(None, "NoneType"), (27, "int"), (b"27AAPFU0939F1ZV", "bytes"), (["27AAPFU0939F1ZV"], "list")])
def test_non_string_is_structural_failure_not_exception(value, type_name):
    assert_structural_failure(validate_gstin(value), f"expected a string, got {type_name}")


# --- GstinValidation contract -----------------------------------------------


def test_result_has_exactly_the_contract_fields():
    assert [f.name for f in dataclasses.fields(GstinValidation)] == [
        "is_valid",
        "structural_ok",
        "checksum_ok",
        "extracted_state_code",
        "extracted_pan",
        "reason",
    ]


def test_result_is_frozen():
    result = validate_gstin("27AAPFU0939F1ZV")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.is_valid = False


def test_reason_is_none_only_when_valid():
    assert validate_gstin("27AAPFU0939F1ZV").reason is None
    assert validate_gstin("29AAACT2727Q1ZW").reason is not None
    assert validate_gstin("27AAPFU0939F1AV").reason is not None
