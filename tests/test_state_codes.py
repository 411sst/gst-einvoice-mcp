"""Tests for gst_einvoice.state_codes, pinning the spec: codes 01-38 (minus the
discontinued 25) plus 97 and 99, with 25 and 28 handled as special cases."""

import dataclasses

import pytest

from gst_einvoice.state_codes import STATE_CODES, StateCodeResult, lookup_state_code

# The spec list, transcribed independently of the module so the dict is pinned
# name-for-name rather than compared against itself.
SPEC_CODES: dict[str, str] = {
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

ACTIVE_CODES = sorted(code for code in SPEC_CODES if code != "28")
ALL_TWO_DIGIT = [f"{n:02d}" for n in range(100)]


# --- STATE_CODES dict -------------------------------------------------------


def test_state_codes_matches_spec_exactly():
    assert STATE_CODES == SPEC_CODES


def test_state_codes_has_39_entries():
    # 01-38 is 38 codes, minus discontinued 25, plus 97 and 99.
    assert len(STATE_CODES) == 39


def test_code_25_is_not_in_state_codes():
    assert "25" not in STATE_CODES


def test_state_codes_keys_are_two_ascii_digits():
    for code in STATE_CODES:
        assert len(code) == 2
        assert code.isascii() and code.isdigit()


@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("01", "J&K"),
        ("26", "Dadra & Nagar Haveli and Daman & Diu"),
        ("28", "Andhra Pradesh (legacy, pre-2014)"),
        ("37", "Andhra Pradesh (current)"),
        ("97", "Other Territory / UN bodies"),
        ("99", "Foreign (OIDAR)"),
    ],
)
def test_spec_spellings_preserved_verbatim(code, name):
    assert STATE_CODES[code] == name


# --- StateCodeResult dataclass ---------------------------------------------


def test_result_is_frozen_dataclass_with_exact_fields():
    assert dataclasses.is_dataclass(StateCodeResult)
    assert [f.name for f in dataclasses.fields(StateCodeResult)] == [
        "code",
        "name",
        "status",
        "is_valid",
        "note",
    ]
    result = lookup_state_code("29")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.code = "30"  # type: ignore[misc]


def test_status_field_is_the_four_literals():
    hints = StateCodeResult.__annotations__
    assert set(hints["status"].__args__) == {
        "active",
        "legacy",
        "discontinued",
        "unknown",
    }


# --- active codes -----------------------------------------------------------


@pytest.mark.parametrize("code", ACTIVE_CODES)
def test_active_codes_resolve(code):
    result = lookup_state_code(code)
    assert result == StateCodeResult(
        code=code, name=SPEC_CODES[code], status="active", is_valid=True, note=None
    )


def test_active_code_count():
    assert len(ACTIVE_CODES) == 38


# --- code 28: legacy but valid ----------------------------------------------


def test_28_is_legacy_and_valid():
    result = lookup_state_code("28")
    assert result.code == "28"
    assert result.status == "legacy"
    assert result.is_valid is True
    assert result.name == "Andhra Pradesh (legacy, pre-2014)"


def test_28_note_explains_telangana_split_and_current_code():
    note = lookup_state_code("28").note
    assert note is not None
    assert "Telangana" in note
    assert "legitimate" in note
    assert "37" in note
    assert "2014" in note


# --- code 25: discontinued, not unknown ------------------------------------


def test_25_is_discontinued_not_unknown():
    result = lookup_state_code("25")
    assert result.code == "25"
    assert result.status == "discontinued"
    assert result.status != "unknown"
    assert result.is_valid is False
    assert result.name == "Daman & Diu"


def test_25_note_explains_merger_and_how_to_disambiguate():
    note = lookup_state_code("25").note
    assert note is not None
    assert "26" in note
    assert "Dadra & Nagar Haveli and Daman & Diu" in note
    assert "2020" in note
    assert "OCR" in note
    assert "stale" in note
    assert "date" in note
    assert "checksum" in note


def test_25_note_names_the_confusable_digits():
    note = lookup_state_code("25").note
    assert note is not None
    for confusable in ("26", "28", "35"):
        assert confusable in note


def test_25_note_gives_the_real_switch_over_date():
    # Code-25 GSTINs stayed in use until 31 July 2020; code-26 GSTINs took
    # effect 1 August 2020 (Trade Notice 28/2020-21, 13 July 2020). A blanket
    # "dated 2020 or later is a misread" rule would wrongly reject genuine
    # January-July 2020 documents, so the note must carry the actual cut-over.
    note = lookup_state_code("25").note
    assert note is not None
    assert "1 August 2020" in note
    assert "31 July 2020" in note
    assert "cannot legitimately" not in note


def test_25_note_does_not_diagnose_a_post_cutover_25_as_a_misread():
    # A 25 on a document dated after the switch-over is not necessarily a
    # misread: suppliers kept printing their ceased code-25 GSTIN on invoices
    # after 1 August 2020, and in build 1 (native text, no OCR) a misread is
    # not even possible. The note must say the code cannot be genuine and name
    # both causes, not guess one of them.
    note = lookup_state_code("25").note
    assert note is not None
    assert "is a misread" not in note
    assert "cannot be a genuine code-25 registration" in note
    assert "still printing" in note
    assert "pre-merger" in note
    assert "migrated code-26 GSTIN" in note
    # A misread 25 may really be 28 or 35, whose suppliers never had a
    # code-26 GSTIN, so the note must not say one is needed "either way".
    assert "either way" not in note


# --- whitespace handling ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_code", "expected_status"),
    [
        ("  29 ", "29", "active"),
        ("\t25\n", "25", "discontinued"),
        (" 28", "28", "legacy"),
        ("97\r\n", "97", "active"),
        ("  39  ", "39", "unknown"),
    ],
)
def test_surrounding_whitespace_is_stripped(raw, expected_code, expected_status):
    result = lookup_state_code(raw)
    assert result.code == expected_code
    assert result.status == expected_status


def test_internal_whitespace_is_not_a_code():
    result = lookup_state_code("2 9")
    assert result.status == "unknown"
    assert result.code == "2 9"
    assert result.name is None
    assert result.is_valid is False


# --- unknown: two digits not in the list -----------------------------------


@pytest.mark.parametrize("code", ["00", "39", "40", "50", "96", "98"])
def test_unlisted_two_digit_codes_are_unknown(code):
    result = lookup_state_code(code)
    assert result == StateCodeResult(
        code=code,
        name=None,
        status="unknown",
        is_valid=False,
        note=f"no GST state code {code}",
    )


# --- unknown: not exactly two digits ---------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "1",
        "9",
        "123",
        "029",
        "2A",
        "ab",
        "-1",
        "+9",
        "2.",
        "２９",  # fullwidth digits: str.isdigit() is True, but these are not ASCII
        "٢٩",  # Arabic-Indic digits
        "²⁹",  # superscripts
        "KA",
        "Karnataka",
    ],
)
def test_non_two_digit_inputs_are_unknown(raw):
    result = lookup_state_code(raw)
    assert result.status == "unknown"
    assert result.name is None
    assert result.is_valid is False
    assert result.code == raw.strip()
    assert result.note is not None
    # "ASCII" matters: fullwidth/Arabic-Indic inputs are two digits by
    # str.isdigit(), so the note must say which digits are required.
    assert "two ASCII digits" in result.note


def test_non_two_digit_note_echoes_the_offending_input():
    assert "'2A'" in lookup_state_code("2A").note


# --- exhaustive invariants over every two-digit string ---------------------


@pytest.mark.parametrize("code", ALL_TWO_DIGIT)
def test_is_valid_iff_active_or_legacy(code):
    result = lookup_state_code(code)
    assert result.is_valid is (result.status in ("active", "legacy"))


@pytest.mark.parametrize("code", ALL_TWO_DIGIT)
def test_name_is_present_iff_not_unknown(code):
    result = lookup_state_code(code)
    assert (result.name is not None) is (result.status != "unknown")


@pytest.mark.parametrize("code", ALL_TWO_DIGIT)
def test_note_is_none_iff_active(code):
    result = lookup_state_code(code)
    assert (result.note is None) is (result.status == "active")


def test_status_distribution_over_all_two_digit_strings():
    statuses = [lookup_state_code(code).status for code in ALL_TWO_DIGIT]
    assert statuses.count("active") == 38
    assert statuses.count("legacy") == 1
    assert statuses.count("discontinued") == 1
    assert statuses.count("unknown") == 60
    assert set(statuses) == {"active", "legacy", "discontinued", "unknown"}


def test_only_28_is_legacy_and_only_25_is_discontinued():
    assert [c for c in ALL_TWO_DIGIT if lookup_state_code(c).status == "legacy"] == ["28"]
    assert [
        c for c in ALL_TWO_DIGIT if lookup_state_code(c).status == "discontinued"
    ] == ["25"]


def test_active_set_is_state_codes_minus_28():
    active = {c for c in ALL_TWO_DIGIT if lookup_state_code(c).status == "active"}
    assert active == set(STATE_CODES) - {"28"}
