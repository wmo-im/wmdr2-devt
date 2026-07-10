from __future__ import annotations

from typing import Any

import pytest

import convert_wmdr10_json_to_wmdr2_json as converter

OBSERVED_12006 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
OBSERVED_179 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179"
GEOMETRY_POINT = "http://codes.wmo.int/wmdr/Geometry/point"


@pytest.mark.parametrize(("raw", "expected"), [
    (None, []),
    ([], []),
    (["a", "b"], ["a", "b"]),
    (("a", "b"), [("a", "b")]),
    ("a", ["a"]),
    (0, [0]),
    (False, [False]),
    ({"x": 1}, [{"x": 1}]),
])
def test_as_list(raw: Any, expected: list[Any]) -> None:
    assert converter._as_list(raw) == expected


@pytest.mark.parametrize(("values", "expected"), [
    ((None, "", [], {}, "first"), "first"),
    ((0, "fallback"), 0),
    ((False, "fallback"), False),
    (([], {}, None), None),
    (("", "x", "y"), "x"),
])
def test_first_non_empty(values: tuple[Any, ...], expected: Any) -> None:
    assert converter._first_non_empty(*values) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ({"a": None, "b": "", "c": [], "d": {}, "e": [None, "", [], {}, 1]}, {"e": [None, 1]}),
    ({"temporalGeometry": {"methods": [[], ["gps"]]}}, {"temporalGeometry": {"methods": [[], ["gps"]]}}),
    ([None, "", [], {}, "x"], [None, "x"]),
    ({"a": {"b": {"c": "value"}}}, {"a": {"b": {"c": "value"}}}),
])
def test_clean_none(raw: Any, expected: Any) -> None:
    assert converter._clean_none(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    (" a b!c ", "a-b-c"),
    ("Station/ABC # 1", "Station/ABC-#-1"),
    ("", "record"),
    ("a---b", "a---b"),
    ("0-20008-0-THE", "0-20008-0-THE"),
])
def test_sanitize_id(raw: str, expected: str) -> None:
    assert converter._sanitize_id(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("A b/c", "a-b-c"),
    ("---", "value"),
    ("Hello__World", "hello-world"),
    ("0-20008-0-THE", "0-20008-0-the"),
    ("Maker--Model", "maker-model"),
])
def test_slug(raw: str, expected: str) -> None:
    assert converter._slug(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("http://codes.wmo.int/wmdr/FacilityType/landFixed", "landFixed"),
    ("abc#def", "def"),
    ("(unknown)", "unknown"),
    ("<http://x/y>", "y"),
    ("", None),
    (None, None),
])
def test_last_segment(raw: Any, expected: str | None) -> None:
    assert converter._last_segment(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("12006", 12006),
    ("001", 1),
    ("http://codes.wmo.int/wmdr/unit/mm", "mm"),
    ("(unknown)", "unknown"),
    ("abc", "abc"),
    (5, 5),
    ({"href": "http://codes.wmo.int/wmdr/LevelOfData/level1"}, "level1"),
])
def test_normalize_code_value(raw: Any, expected: Any) -> None:
    assert converter._normalize_code_value(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("http://codes.wmo.int/wmdr/unit/mm", "mm"),
    ("https://codes.wmo.int/wmdr/FacilityType/landFixed", "landFixed"),
    ("http://example.org/not-wmdr/value", "http://example.org/not-wmdr/value"),
    ({"href": "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"}, "GBON"),
    (12006, 12006),
])
def test_compact_wmdr_code_value(raw: Any, expected: Any) -> None:
    assert converter._compact_wmdr_code_value(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("20200102_test", "2020-01-02"),
    ("20200102", "2020-01-02"),
    ("2020-01-02Z", "2020-01-02"),
    ("2020-01-02T03:04:05Z", "2020-01-02"),
    ("..", ".."),
    (None, None),
    ("not-a-date", "not-a-date"),
])
def test_normalize_date_value(raw: Any, expected: str | None) -> None:
    assert converter._normalize_date_value(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("20200102_test", "2020-01-02T00:00:00Z"),
    ("20200102", "2020-01-02T00:00:00Z"),
    ("2020-01-02", "2020-01-02T00:00:00Z"),
    ("2020-01-02Z", "2020-01-02T00:00:00Z"),
    ("2020-01-02T03:04:05Z", "2020-01-02T03:04:05Z"),
])
def test_normalize_record_datetime(raw: str, expected: str) -> None:
    assert converter._normalize_record_datetime(raw) == expected


@pytest.mark.parametrize(("start", "end", "resolution", "expected"), [
    (None, None, None, None),
    ("2020-01-01", None, None, {"interval": ["2020-01-01", ".."]}),
    (None, "2022-01-01", None, {"interval": ["..", "2022-01-01"]}),
    ("20200101", "20201231", None, {"interval": ["2020-01-01", "2020-12-31"]}),
    ("2020-01-01", None, "day", {"interval": ["2020-01-01", ".."], "resolution": "P1D"}),
    ("2020-01-01", "2020-01-02", "hour", {"interval": ["2020-01-01", "2020-01-02"], "resolution": "PT1H"}),
])
def test_time_interval(start: Any, end: Any, resolution: str | None, expected: dict[str, Any] | None) -> None:
    assert converter._time_interval(start, end, resolution=resolution) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("46 7 100", [7.0, 46.0, 100.0]),
    ({"pos": "46, 7, 100.4"}, [7.0, 46.0, 100.4]),
    ({"coordinates": [7, 46, 100]}, [7, 46, 100]),
    ("46 7", [7.0, 46.0]),
    ("nonsense", None),
    (None, None),
])
def test_parse_pos_lon_lat_z(raw: Any, expected: list[Any] | None) -> None:
    assert converter._parse_pos_lon_lat_z(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    (OBSERVED_12006, "atmosphere"),
    ("http://codes.wmo.int/wmdr/ObservedVariableTerrestrial/12", "terrestrial"),
    ("http://codes.wmo.int/wmdr/ObservedVariableOcean/1", "ocean"),
    ("not-a-code", None),
    (None, None),
])
def test_observed_domain_from_observed_variable(raw: Any, expected: str | None) -> None:
    assert converter._observed_domain_from_observed_variable(raw) == expected


@pytest.mark.parametrize(("values", "expected"), [
    (["A", "A", None, "http://x/B", "C", ""], ["A", "B", "C"]),
    (["http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"], ["GBON"]),
    ([None, "", []], []),
    (["  A B  "], ["A B"]),
])
def test_keywords_from_values(values: list[Any], expected: list[str]) -> None:
    assert converter._keywords_from_values(values) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    (["http://codes.wmo.int/wmdr/ResponsiblePartyRole/owner", {"href": "http://codes.wmo.int/wmdr/ResponsiblePartyRole/operator"}, "owner"], ["operator", "owner"]),
    ("gmxCodelists.xml#CI_RoleCode", []),
    ({"@codeList": "gmxCodelists.xml#CI_RoleCode", "@codeListValue": "pointOfContact"}, ["pointOfContact"]),
    (None, []),
])
def test_normalize_roles(raw: Any, expected: list[str]) -> None:
    assert converter._normalize_roles(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("true", True), ("1", True), ("yes", True), ("Y", True),
    ("false", False), ("0", False), ("no", False), ("N", False),
    ("maybe", None), (None, None), (True, True), (False, False),
])
def test_parse_bool(raw: Any, expected: bool | None) -> None:
    assert converter._parse_bool(raw) is expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("7", "07:00:00"),
    ("7:5", "07:05:00"),
    ("23:59:59", "23:59:59"),
    ("25:99:99", "23:59:59"),
    ("abc", "abc"),
    ("06:00Z", "06:00:00"),
])
def test_normalize_diurnal_time(raw: Any, expected: str) -> None:
    assert converter._normalize_diurnal_time(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("day", "P1D"), ("daily", "P1D"), ("hour", "PT1H"), ("hourly", "PT1H"),
    ("minute", "PT1M"), ("PT10M", "PT10M"), ("P1D", "P1D"), (None, None),
])
def test_normalize_time_resolution(raw: Any, expected: str | None) -> None:
    assert converter._normalize_time_resolution(raw) == expected


@pytest.mark.parametrize(("seconds", "expected"), [
    (60, "PT1M"),
    (3600, "PT1H"),
    (43200, "PT12H"),
    (86400, "P1D"),
    (3661, "PT1H1M1S"),
    (0, None),
])
def test_iso_duration_from_seconds(seconds: int, expected: str | None) -> None:
    assert converter._iso_duration_from_seconds(seconds) == expected


@pytest.mark.parametrize(("coverage", "expected"), [
    ({"startHour": "6", "endHour": "18"}, {"start": "0001-01-01T06:00:00", "duration": "PT12H"}),
    ({"startHour": "23", "endHour": "1"}, {"start": "0001-01-01T23:00:00", "duration": "PT2H"}),
    ({"startTime": "7:30", "endTime": "9:00"}, {"start": "0001-01-01T07:30:00", "duration": "PT1H30M"}),
    ({"startHour": "0", "endHour": "0"}, {"start": "0001-01-01T00:00:00", "duration": "P1D"}),
    ({}, {}),
])
def test_diurnal_coverage_fields(coverage: dict[str, Any], expected: dict[str, str]) -> None:
    assert converter._diurnal_coverage_fields(coverage) == expected


@pytest.mark.parametrize(("raw", "kind", "expected_keys"), [
    ({"temporalSamplingInterval": "PT10M"}, "observing", {"wmo.int:samplingFrequency": "PT10M"}),
    ({"temporalReportingInterval": "PT1H"}, "reporting", {"wmo.int:aggregationInterval": "PT1H"}),
    ({"duration": "PT2H"}, "shared", {"duration": "PT2H"}),
    ({"diurnalBaseTime": "6"}, "shared", {"wmo.int:diurnalBaseTime": "06:00:00"}),
    ({"frequency": "PT5M"}, "observing", {"wmo.int:samplingFrequency": "PT5M"}),
    ({"frequency": "PT1H"}, "reporting", {"wmo.int:aggregationInterval": "PT1H"}),
])
def test_normalize_schedule_object(raw: dict[str, Any], kind: str, expected_keys: dict[str, Any]) -> None:
    schedule = converter._normalize_schedule_object(raw, kind=kind)
    assert schedule is not None
    assert schedule["@type"] == "Event"
    assert schedule["uid"].startswith("schedule_")
    for key, expected in expected_keys.items():
        assert schedule[key] == expected


def test_facility_temporal_geometry_entries_are_sorted_and_deduplicated() -> None:
    facility = {
        "geospatialLocation": {"geoLocation": "46 7 100", "beginPosition": "2021-01-01"},
        "geospatialLocationHistory": [
            {"geoLocation": "45 6 99", "beginPosition": "2020-01-01"},
            {"geoLocation": "45 6 99", "beginPosition": "2020-01-01"},
        ],
    }
    assert converter._facility_temporal_geometry_entries(facility) == [
        {"coordinates": [6.0, 45.0, 99.0], "date": "2020-01-01"},
        {"coordinates": [7.0, 46.0, 100.0], "date": "2021-01-01"},
    ]


@pytest.mark.parametrize(("entries", "expected"), [
    ([{"coordinates": [7, 46], "date": "2020-01-01"}], None),
    ([{"coordinates": [7, 46], "date": "2020-01-01", "methods": ["gps"]}], {"type": "MovingPoint", "coordinates": [[7, 46]], "dates": ["2020-01-01"], "methods": [["gps"]]}),
    ([{"coordinates": [6, 45], "date": "2020-01-01"}, {"coordinates": [7, 46], "date": None}], {"type": "MovingPoint", "coordinates": [[6, 45], [7, 46]], "dates": ["2020-01-01", ".."]}),
])
def test_temporal_geometry_extension(entries: list[dict[str, Any]], expected: dict[str, Any] | None) -> None:
    assert converter._temporal_geometry_extension(entries) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("wsi:0-20008-0-THE", "0-20008-0-THE"),
    ("facility:0-20008-0-THE", "0-20008-0-THE"),
    ("wigos:0-20008-0-THE", "0-20008-0-THE"),
    ("0-20008-0-THE", "0-20008-0-THE"),
    ("  0-20008-0-THE  ", "0-20008-0-THE"),
])
def test_normalize_facility_wsi(raw: str, expected: str) -> None:
    assert converter._normalize_facility_wsi(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ({"value": "2.5", "@uom": "m"}, {"value": 2.5, "uom": "m"}),
    ({"#text": "0", "uom": "m"}, {"value": 0.0, "uom": "m"}),
    ("3.5", {"value": 3.5}),
    (0, {"value": 0.0}),
    ("not numeric", {"value": "not numeric"}),
])
def test_quantity(raw: Any, expected: dict[str, Any] | None) -> None:
    assert converter._quantity(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("+1-608-262-0436", "+16082620436"),
    ("+33(0)476824239", "+33476824239"),
    ("+41(081)4175137", "+41814175137"),
    ("0041587654048", "+41587654048"),
    ("0041814175157", "+41814175157"),
    ("0722782074", "0722782074"),
    ("254723521586", "254723521586"),
])
def test_normalize_phone_value_keeps_contact_schema_strict(raw: str, expected: str) -> None:
    assert converter._normalize_phone_value(raw) == expected
