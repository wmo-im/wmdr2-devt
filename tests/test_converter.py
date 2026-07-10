from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import convert_wmdr10_json_to_wmdr2_json as converter

OBSERVED_12006 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
OBSERVED_179 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179"
GEOMETRY_POINT = "http://codes.wmo.int/wmdr/Geometry/point"


def _walk_dicts(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _contact(role: str) -> dict[str, Any]:
    return {
        "organisationName": "Example Met Service",
        "contactInfo": {
            "address": {"electronicMailAddress": "ops@example.org"},
            "phone": {"voice": "+41 1 234 56 78"},
        },
        "role": role,
    }


def _payload() -> dict[str, Any]:
    contact_base = {
        "organisationName": "Example Met Service",
        "contactInfo": {
            "address": {"electronicMailAddress": "ops@example.org"},
            "phone": {"voice": "+41 1 234 56 78"},
        },
    }
    return {
        "header": {"dateStamp": "2020-01-02"},
        "facility": {
            "identifier": "wsi:0-20008-0-THE",
            "name": "Thessaloniki",
            "geospatialLocation": "40.5 22.9 10",
            "beginPosition": "2020-01-01",
            "contact": contact_base | {"role": "owner"},
        },
        "observationSeries": [
            {
                "observedProperty": OBSERVED_12006,
                "observedGeometry": GEOMETRY_POINT,
                "contact": contact_base | {"role": "operator"},
                "deployments": [
                    {
                        "beginPosition": "2020-01-01",
                        "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
                        "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading",
                        "referenceSurface": "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround",
                        "heightAboveLocalReferenceSurface": {"@uom": "m", "#text": "2.0"},
                        "manufacturer": "Maker",
                        "model": "Model",
                        "serialNumber": "SN1",
                        "dataGeneration": {
                            "beginPosition": "2020-01-01",
                            "sampling": {
                                "samplingStrategy": "continuous",
                                "temporalSamplingInterval": "PT10M",
                            },
                            "reporting": {
                                "internationalExchange": "true",
                                "temporalReportingInterval": "PT1H",
                                "uom": "http://codes.wmo.int/wmdr/unit/mm",
                            },
                            "coverage": {"diurnalBaseTime": "7:5"},
                            "contact": contact_base | {"role": "reportingAuthority"},
                        },
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        (["a", "b"], ["a", "b"]),
        ("a", ["a"]),
    ],
)
def test_as_list_normalizes_scalars_lists_and_none(raw: Any, expected: list[Any]) -> None:
    assert converter._as_list(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("day", "P1D"),
        ("hour", "PT1H"),
        ("PT10M", "PT10M"),
        (None, None),
    ],
)
def test_time_resolution_is_normalized_to_iso_duration(raw: Any, expected: str | None) -> None:
    assert converter._normalize_time_resolution(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7", "07:00:00"),
        ("7:5", "07:05:00"),
        ("23:59:59", "23:59:59"),
        ("25:99:99", "23:59:59"),
        ("abc", "abc"),
    ],
)
def test_normalize_diurnal_time_handles_compact_clock_values(raw: str, expected: str) -> None:
    assert converter._normalize_diurnal_time(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("wsi:0-20008-0-THE", "0-20008-0-THE"),
        ("facility:0-20008-0-THE", "0-20008-0-THE"),
        ("record:0-20008-0-THE", "0-20008-0-THE"),
        ("0-20008-0-THE", "0-20008-0-THE"),
    ],
)
def test_facility_identifier_is_bare_wsi(raw: str, expected: str) -> None:
    assert converter._normalize_facility_wsi(raw) == expected


def test_convert_payload_emits_current_v031_shape() -> None:
    record = converter.convert_payload(_payload(), source_name="20200102_0-20008-0-THE")
    props = record["properties"]
    observation = props["observationSeries"][0]
    cfg = observation["observingConfigurations"][0]
    reporting = observation["reportingProcedures"][0]

    assert record["id"] == "0-20008-0-THE"
    assert record["conformsTo"] == ["http://wigos.wmo.int/spec/wmdr/2/conf/core"]
    assert record["time"] == {"interval": ["2020-01-01", ".."], "resolution": "P1D"}
    assert observation["id"] == "observationSeries:12006"
    assert observation["observedProperty"] == 12006
    assert observation["observedGeometry"] == "point"
    assert observation["observedDomain"] == {"domain": "atmosphere"}

    assert "observingLocation" not in cfg
    assert "temporalGeometry" not in cfg
    assert "keywords" not in cfg
    assert cfg["time"] == {"interval": ["2020-01-01", ".."]}
    assert cfg["observingMethod"] == 266
    assert cfg["sourceOfObservation"] == "automaticReading"
    assert cfg["referenceSurface"] == "localGround"
    assert cfg["verticalDistanceFromReferenceSurface"] == {"value": 2.0, "uom": "m"}
    assert cfg["serialNumber"] == "SN1"
    assert cfg["instrument"].startswith("instrument:")

    observing = observation["observingProcedures"][0]
    schedules = {schedule["uid"]: schedule for schedule in props["schedules"]}

    assert "time" not in reporting
    assert "temporalReportingInterval" not in reporting
    assert reporting["internationalExchange"] is True
    assert reporting["uom"] == "mm"
    assert reporting["reportingSchedules"][0] in schedules
    reporting_schedule = schedules[reporting["reportingSchedules"][0]]
    assert reporting_schedule["wmo.int:aggregationInterval"] == "PT1H"
    assert reporting_schedule["wmo.int:diurnalBaseTime"] == "07:05:00"
    assert "duration" not in reporting_schedule

    assert observing["time"] == {"interval": ["2020-01-01", ".."]}
    assert observing["strategy"] == "continuous"
    assert observing["observingSchedules"][0] in schedules
    assert observing["observingSchedules"] == reporting["reportingSchedules"]
    assert schedules[observing["observingSchedules"][0]]["wmo.int:samplingFrequency"] == "PT10M"

    assert props["contacts"] == [
        {
            "identifier": "contact:ops@example.org",
            "organization": "Example Met Service",
            "emails": [{"value": "ops@example.org"}],
            "phones": [{"value": "+4112345678"}],
        }
    ]
    assert props["contactAssignments"] == [{"contact": "contact:ops@example.org", "roles": ["owner"]}]
    assert observation["contactAssignments"] == [{"contact": "contact:ops@example.org", "roles": ["operator"]}]
    assert reporting["contactAssignments"] == [
        {"contact": "contact:ops@example.org", "roles": ["reportingAuthority"]}
    ]
    forbidden = {"beginPosition", "endPosition", "validFrom", "validTo"}
    assert all(forbidden.isdisjoint(item) for item in _walk_dicts(record))


def test_normalize_existing_record_converts_all_source_temporal_keys() -> None:
    legacy = {
        "type": "Feature",
        "id": "wsi:0-TEST",
        "geometry": {"type": "Point", "coordinates": [7.0, 46.0, 100]},
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "time": {"interval": ["2020-01-01", ".."], "resolution": "day"},
        "properties": {
            "type": "facility",
            "title": "Test",
            "programAffiliations": [
                {"program": "GBON", "beginPosition": "2020-01-01", "endPosition": "2022-01-01"}
            ],
            "observationSeries": [
                {
                    "id": "observationSeries:test",
                    "beginPosition": "2020-02-01",
                    "endPosition": "2023-02-01",
                    "observingConfigurations": [
                        {
                            "beginPosition": "2020-03-01",
                            "endPosition": "2024-03-01",
                            "observingLocation": {
                                "beginPosition": "2020-04-01",
                                "endPosition": "2025-04-01",
                                "referenceSurface": "localGround",
                            },
                        }
                    ],
                }
            ],
        },
    }
    record = converter.normalize_wmdr2_record(legacy)
    props = record["properties"]
    series = props["observationSeries"][0]
    cfg = series["observingConfigurations"][0]
    assert props["programAffiliations"][0]["time"] == {"interval": ["2020-01-01", "2022-01-01"]}
    assert series["time"] == {"interval": ["2020-02-01", "2023-02-01"]}
    assert cfg["time"] == {"interval": ["2020-03-01", "2024-03-01"]}
    assert cfg["referenceSurface"] == "localGround"
    forbidden = {"beginPosition", "endPosition", "validFrom", "validTo"}
    assert all(forbidden.isdisjoint(item) for item in _walk_dicts(record))


def test_normalize_existing_record_renames_configuration_temporal_geometry_to_geometry() -> None:
    legacy = {
        "type": "Feature",
        "id": "0-TEST",
        "geometry": {"type": "Point", "coordinates": [7.0, 46.0, 100]},
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "properties": {
            "type": "facility",
            "title": "Test",
            "observationSeries": [
                {
                    "id": "observationSeries:test",
                    "observingConfigurations": [
                        {
                            "observingMethod": {"nilReason": "unknown"},
                            "time": {"interval": ["2020-01-01", ".."]},
                            "temporalGeometry": {
                                "type": "MovingPoint",
                                "coordinates": [[7.0, 46.0, 100], [7.1, 46.1, 101]],
                                "dates": ["2020-01-01", "2021-01-01"],
                            },
                            "keywords": ["legacy-keyword"],
                        }
                    ],
                }
            ],
        },
    }
    record = converter.normalize_wmdr2_record(legacy)
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    assert "temporalGeometry" not in cfg
    assert "keywords" not in cfg
    assert cfg["geometry"] == {"type": "Point", "coordinates": [7.1, 46.1, 101.0]}


def test_normalize_existing_record_promotes_observing_location_and_time() -> None:
    legacy = {
        "type": "Feature",
        "id": "wsi:0-TEST",
        "geometry": {"type": "Point", "coordinates": [7.0, 46.0, 100]},
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "time": {"interval": ["2020-01-01", ".."], "resolution": "day"},
        "properties": {
            "type": "facility",
            "title": "Test",
            "observationSeries": [
                {
                    "id": "observationSeries:test",
                    "observingConfigurations": [
                        {
                            "validFrom": "2020-01-01",
                            "observingLocation": {
                                "referenceSurface": "localGround",
                                "verticalDistanceFromReferenceSurface": {"value": 2, "uom": "m"},
                            },
                        }
                    ],
                }
            ],
        },
    }
    record = converter.normalize_wmdr2_record(legacy)
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    assert record["id"] == "0-TEST"
    assert record["time"]["resolution"] == "P1D"
    assert cfg["time"] == {"interval": ["2020-01-01", ".."]}
    assert cfg["referenceSurface"] == "localGround"
    assert "validFrom" not in cfg
    assert "observingLocation" not in cfg


def test_normalize_existing_record_removes_time_from_reporting_procedures() -> None:
    legacy = {
        "type": "Feature",
        "id": "0-TEST",
        "geometry": {"type": "Point", "coordinates": [7.0, 46.0, 100]},
        "properties": {
            "type": "facility",
            "title": "Test",
            "observationSeries": [
                {
                    "id": "observationSeries:test",
                    "reportingProcedures": [
                        {
                            "time": {"interval": ["2020-01-01", ".."]},
                            "internationalExchange": True,
                            "reportingSchedules": ["schedule_abc"],
                        }
                    ],
                }
            ],
            "schedules": [{"uid": "schedule_abc", "@type": "Event", "wmo.int:aggregationInterval": "PT1H"}],
        },
    }
    record = converter.normalize_wmdr2_record(legacy)
    reporting = record["properties"]["observationSeries"][0]["reportingProcedures"][0]
    assert "time" not in reporting
    assert "temporalReportingInterval" not in reporting
    assert reporting["reportingSchedules"] == ["schedule_abc"]
    schedule = {item["uid"]: item for item in record["properties"]["schedules"]}["schedule_abc"]
    assert schedule["wmo.int:aggregationInterval"] == "PT1H"
    assert "duration" not in schedule


def test_contact_registry_does_not_freeze_contextual_roles() -> None:
    record = converter.convert_payload(
        {
            "facility": {"identifier": "0-TEST", "name": "Test", "contact": _contact("owner")},
            "observationSeries": [{"observedProperty": OBSERVED_179, "contact": _contact("operator")}],
        }
    )
    contact = record["properties"]["contacts"][0]
    assert "roles" not in contact
    assert record["properties"]["contactAssignments"][0]["roles"] == ["owner"]
    assert record["properties"]["observationSeries"][0]["contactAssignments"][0]["roles"] == ["operator"]


def test_convert_file_writes_json_and_reports_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "input.json"
    target = tmp_path / "out" / "output.json"
    source.write_text(json.dumps({"facility": {"identifier": "0-TEST", "name": "Test"}}), encoding="utf-8")

    written = converter.convert_file(source, target)

    assert written == target
    assert target.exists()
    assert "wrote" in capsys.readouterr().out

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20200102_test", "2020-01-02"),
        ("20200102", "2020-01-02"),
        ("2020-01-02T03:04:05Z", "2020-01-02"),
        ("..", ".."),
        (None, None),
    ],
)
def test_normalize_date_value(raw: Any, expected: str | None) -> None:
    assert converter._normalize_date_value(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20200102_test", "2020-01-02T00:00:00Z"),
        ("2020-01-02", "2020-01-02T00:00:00Z"),
        ("2020-01-02T03:04:05Z", "2020-01-02T03:04:05Z"),
    ],
)
def test_normalize_record_datetime(raw: str, expected: str) -> None:
    assert converter._normalize_record_datetime(raw) == expected


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (None, None, None),
        ("2020-01-01", None, {"interval": ["2020-01-01", ".."]}),
        (None, "2022-01-01", {"interval": ["..", "2022-01-01"]}),
        ("20200101", "20201231", {"interval": ["2020-01-01", "2020-12-31"]}),
    ],
)
def test_time_interval(start: Any, end: Any, expected: dict[str, Any] | None) -> None:
    assert converter._time_interval(start, end) == expected


def test_time_interval_converts_resolution_to_iso_duration() -> None:
    assert converter._time_interval("2020-01-01", None, resolution="day") == {
        "interval": ["2020-01-01", ".."],
        "resolution": "P1D",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("46 7 100", [7.0, 46.0, 100]),
        ({"pos": "46, 7, 100.4"}, [7.0, 46.0, 100.4]),
        ({"coordinates": [7, 46, 100]}, [7, 46, 100]),
        ("nonsense", None),
    ],
)
def test_parse_pos_lon_lat_z_converts_wmdr_lat_lon_to_geojson_lon_lat(raw: Any, expected: list[Any] | None) -> None:
    assert converter._parse_pos_lon_lat_z(raw) == expected


def test_facility_temporal_geometry_entries_are_sorted_and_deduplicated() -> None:
    facility = {
        "geospatialLocation": {"geoLocation": "46 7 100", "beginPosition": "2021-01-01"},
        "geospatialLocationHistory": [
            {"geoLocation": "45 6 99", "beginPosition": "2020-01-01"},
            {"geoLocation": "45 6 99", "beginPosition": "2020-01-01"},
        ],
    }
    assert converter._facility_temporal_geometry_entries(facility) == [
        {"coordinates": [6.0, 45.0, 99], "date": "2020-01-01"},
        {"coordinates": [7.0, 46.0, 100], "date": "2021-01-01"},
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (OBSERVED_12006, "atmosphere"),
        ("http://codes.wmo.int/wmdr/ObservedVariableTerrestrial/12", "terrestrial"),
        ("not-a-code", None),
    ],
)
def test_observed_domain_from_observed_variable(raw: str, expected: str | None) -> None:
    assert converter._observed_domain_from_observed_variable(raw) == expected


def test_observation_title_uses_code_list_label_when_available() -> None:
    converter.CODE_LIST_LABELS.clear()
    converter.CODE_LIST_LABELS.update({"ObservedVariableAtmosphere": {"12006": "Air temperature"}})
    assert converter._format_observation_title(OBSERVED_12006, GEOMETRY_POINT) == (
        "domain: atmosphere; geometry: point; variable: 12006 Air temperature"
    )
    converter.CODE_LIST_LABELS.clear()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://codes.wmo.int/wmdr/unit/mm", "mm"),
        ("https://codes.wmo.int/wmdr/FacilityType/landFixed", "landFixed"),
        ("http://example.org/not-wmdr/value", "http://example.org/not-wmdr/value"),
    ],
)
def test_compact_wmdr_code_value_only_compacts_wmdr_urls(raw: str, expected: str) -> None:
    assert converter._compact_wmdr_code_value(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("maybe", None),
    ],
)
def test_parse_bool_accepts_common_wmdr_boolean_spellings(raw: str, expected: bool | None) -> None:
    assert converter._parse_bool(raw) is expected


def test_normalize_roles_extracts_code_values_and_drops_generic_iso_reference() -> None:
    assert set(converter._normalize_roles(
        [
            "http://codes.wmo.int/wmdr/ResponsiblePartyRole/owner",
            {"href": "http://codes.wmo.int/wmdr/ResponsiblePartyRole/operator"},
            "gmxCodelists.xml#CI_RoleCode",
        ]
    )) == {"owner", "operator"}


def test_normalize_ogc_contact_uses_ogc_email_and_phone_objects() -> None:
    contact = converter._normalize_ogc_contact(
        {
            "organisationName": "Example Org",
            "individualName": "Jane Doe",
            "contactInfo": {
                "address": {"electronicMailAddress": "jane@example.org"},
                "phone": {"voice": "+41 1 234 56 78"},
            },
            "role": "owner",
        }
    )
    assert contact == {
        "name": "Jane Doe",
        "organization": "Example Org",
        "emails": [{"value": "jane@example.org"}],
        "phones": [{"value": "+4112345678"}],
        "roles": ["owner"],
    }


def test_quantity_accepts_wmdr10_text_value_and_uom() -> None:
    assert converter._quantity({"@uom": "m", "#text": "2.0"}) == {"value": 2.0, "uom": "m"}


def test_normalize_code_or_nil_reason_keeps_explicit_unknown_as_nil_reason() -> None:
    assert converter._normalize_code_or_nil_reason("unknown") == {"nilReason": "unknown"}
    assert converter._normalize_code_or_nil_reason({"nilReason": "withheld"}) == {"nilReason": "withheld"}


def test_normalizer_accepts_contact_references_alias_but_outputs_contact_assignments() -> None:
    record = {
        "type": "Feature",
        "id": "0-20000-0-TEST",
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "geometry": {"type": "Point", "coordinates": [7, 46, 100]},
        "time": {"interval": ["2020-01-01", ".."], "resolution": "day"},
        "properties": {
            "type": "facility",
            "title": "Test",
            "contactReferences": [
                {"contact": "contact:ops@example.org", "roles": ["owner"]}
            ],
            "contacts": [{"identifier": "contact:ops@example.org", "emails": ["ops@example.org"]}],
        },
    }
    normalized = converter.normalize_wmdr2_record(record)
    assert "contactReferences" not in normalized["properties"]
    assert normalized["properties"]["contactAssignments"] == [
        {"contact": "contact:ops@example.org", "roles": ["owner"]}
    ]
    assert normalized["properties"]["contacts"] == [
        {"identifier": "contact:ops@example.org", "emails": [{"value": "ops@example.org"}]}
    ]


def test_reporting_procedure_matches_v031_uml_attributes_and_uses_reusable_schedule() -> None:
    record = converter.convert_payload(
        {
            "facility": {"identifier": "0-TEST", "name": "Test"},
            "observationSeries": [
                {
                    "observedProperty": OBSERVED_179,
                    "dataGeneration": {
                        "beginPosition": "2020-01-01",
                        "reporting": {
                            "strategy": "routine",
                            "internationalExchange": True,
                            "dataFormat": ["bufr", "json"],
                            "dataPolicy": "http://codes.wmo.int/wmdr/DataPolicy/noLimitation",
                            "levelOfData": "http://codes.wmo.int/wmdr/LevelOfData/level1",
                            "numberOfObservationsInReportingInterval": 6,
                            "referenceDatum": "WGS84",
                            "referenceTimeSource": ["UTC", {"href": "http://codes.wmo.int/wmdr/ReferenceTimeSource/gps"}],
                            "spatialReportingInterval": "point",
                            "timeStampMeaning": "beginning",
                            "timeliness": "PT30M",
                            "uom": "http://codes.wmo.int/wmdr/unit/K",
                            "temporalReportingInterval": "PT1H",
                            "diurnalBaseTime": "6",
                        },
                    },
                }
            ],
        }
    )
    procedure = record["properties"]["observationSeries"][0]["reportingProcedures"][0]
    assert set(procedure) >= {
        "strategy",
        "internationalExchange",
        "dataFormat",
        "dataPolicy",
        "levelOfData",
        "numberOfObservationsInReportingInterval",
        "referenceDatum",
        "referenceTimeSource",
        "spatialReportingInterval",
        "timeStampMeaning",
        "timeliness",
        "uom",
        "reportingSchedules",
    }
    assert "time" not in procedure
    assert "temporalReportingInterval" not in procedure
    assert procedure["dataPolicy"] == "noLimitation"
    assert procedure["levelOfData"] == "level1"
    assert procedure["referenceTimeSource"] == ["UTC", "gps"]
    schedule_uid = procedure["reportingSchedules"][0]
    schedule = {item["uid"]: item for item in record["properties"]["schedules"]}[schedule_uid]
    assert schedule["wmo.int:aggregationInterval"] == "PT1H"
    assert schedule["wmo.int:diurnalBaseTime"] == "06:00:00"
    assert "duration" not in schedule



def test_diurnal_coverage_sets_dummy_start_duration_and_shared_schedule() -> None:
    record = converter.convert_payload(
        {
            "facility": {"identifier": "0-TEST", "name": "Test"},
            "observationSeries": [
                {
                    "observedProperty": OBSERVED_179,
                    "dataGeneration": {
                        "beginPosition": "2020-01-01",
                        "sampling": {
                            "samplingStrategy": "continuous",
                            "temporalSamplingInterval": "PT10M",
                        },
                        "reporting": {
                            "internationalExchange": True,
                            "temporalReportingInterval": "PT1H",
                            "uom": "K",
                        },
                        "coverage": {
                            "startHour": "6",
                            "startMinute": "0",
                            "endHour": "18",
                            "endMinute": "0",
                            "diurnalBaseTime": "6",
                        },
                    },
                }
            ],
        }
    )
    props = record["properties"]
    series = props["observationSeries"][0]
    observing = series["observingProcedures"][0]
    reporting = series["reportingProcedures"][0]
    assert observing["observingSchedules"] == reporting["reportingSchedules"]
    schedule = {item["uid"]: item for item in props["schedules"]}[observing["observingSchedules"][0]]
    assert schedule["start"] == "0001-01-01T06:00:00"
    assert schedule["duration"] == "PT12H"
    assert schedule["wmo.int:samplingFrequency"] == "PT10M"
    assert schedule["wmo.int:aggregationInterval"] == "PT1H"
    assert schedule["wmo.int:diurnalBaseTime"] == "06:00:00"


def test_observing_and_reporting_procedures_can_use_different_schedules() -> None:
    payload = {
        "facility": {"identifier": "0-20000-0-TEST", "name": "Test", "geospatialLocation": "46 7 500"},
        "observationSeries": [
            {
                "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
                "dataGeneration": {
                    "beginPosition": "2020-01-01",
                    "observingSchedule": {
                        "start": "0001-01-01T00:00:00",
                        "wmo.int:samplingFrequency": "PT10M",
                    },
                    "reporting": {
                        "internationalExchange": True,
                        "reportingSchedule": {
                            "start": "0001-01-01T06:00:00",
                            "wmo.int:aggregationInterval": "PT1H",
                            "wmo.int:diurnalBaseTime": "06:00:00",
                        },
                    },
                },
            }
        ],
    }

    record = converter.convert_payload(payload, source_name="0-20000-0-TEST")
    series = record["properties"]["observationSeries"][0]
    observing_ref = series["observingProcedures"][0]["observingSchedules"][0]
    reporting_ref = series["reportingProcedures"][0]["reportingSchedules"][0]

    assert observing_ref != reporting_ref
    schedules = {item["uid"]: item for item in record["properties"]["schedules"]}
    assert schedules[observing_ref]["wmo.int:samplingFrequency"] == "PT10M"
    assert "wmo.int:aggregationInterval" not in schedules[observing_ref]
    assert schedules[reporting_ref]["wmo.int:aggregationInterval"] == "PT1H"
    assert schedules[reporting_ref]["wmo.int:diurnalBaseTime"] == "06:00:00"


def test_observing_and_reporting_procedures_reuse_same_schedule_when_source_schedule_is_shared() -> None:
    payload = {
        "facility": {"identifier": "0-20000-0-TEST", "name": "Test", "geospatialLocation": "46 7 500"},
        "observationSeries": [
            {
                "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
                "dataGeneration": {
                    "beginPosition": "2020-01-01",
                    "temporalSamplingInterval": "PT10M",
                    "reporting": {
                        "internationalExchange": True,
                        "temporalReportingInterval": "PT1H",
                    },
                    "coverage": {
                        "startHour": "0",
                        "startMinute": "0",
                        "endHour": "23",
                        "endMinute": "59",
                    },
                },
            }
        ],
    }

    record = converter.convert_payload(payload, source_name="0-20000-0-TEST")
    series = record["properties"]["observationSeries"][0]
    observing_ref = series["observingProcedures"][0]["observingSchedules"][0]
    reporting_ref = series["reportingProcedures"][0]["reportingSchedules"][0]

    assert observing_ref == reporting_ref


def test_facility_description_is_ogc_string_when_source_has_temporal_objects() -> None:
    payload = {
        "facility": {
            "identifier": "0-20000-0-TEST",
            "name": "Test",
            "geospatialLocation": "46 7 500",
            "description": [
                {"description": "Earlier description", "beginPosition": "2020-01-01"},
                {"description": "Later description", "beginPosition": "2021-01-01"},
            ],
        }
    }

    record = converter.convert_payload(payload, source_name="0-20000-0-TEST")

    assert record["properties"]["description"] == "Earlier description\n\nLater description"


def test_program_affiliation_without_explicit_time_is_not_emitted() -> None:
    payload = {
        "facility": {
            "identifier": "0-20000-0-TEST",
            "name": "Test",
            "geospatialLocation": "46 7 500",
            "programAffiliations": ["GBON", {"program": "GOS"}],
        }
    }

    record = converter.convert_payload(payload, source_name="0-20000-0-TEST")

    assert "programAffiliations" not in record["properties"]


def test_program_affiliation_with_explicit_time_is_emitted_as_temporal_object() -> None:
    payload = {
        "facility": {
            "identifier": "0-20000-0-TEST",
            "name": "Test",
            "geospatialLocation": "46 7 500",
            "programAffiliations": [
                {
                    "program": "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON",
                    "beginPosition": "2020-01-01",
                }
            ],
        }
    }

    record = converter.convert_payload(payload, source_name="0-20000-0-TEST")

    assert record["properties"]["programAffiliations"] == [
        {"program": "GBON", "time": {"interval": ["2020-01-01", ".."]}}
    ]


def test_facility_territory_is_temporal_array_when_source_has_time() -> None:
    record = converter.convert_record({
        "facility": {
            "identifier": "0-20000-0-06650",
            "name": "Example",
            "territory": {
                "territoryName": "CHE",
                "beginPosition": "1864-01-01",
                "endPosition": "..",
            },
        }
    })

    territory = record["properties"]["territory"]
    assert territory == [{"territory": "CHE", "time": {"interval": ["1864-01-01", ".."]}}]


def test_facility_territory_without_time_is_not_emitted() -> None:
    record = converter.convert_record({
        "facility": {
            "identifier": "0-20000-0-06650",
            "name": "Example",
            "territory": "CHE",
        }
    })

    assert "territory" not in record["properties"]

def test_facility_title_is_first_name_and_additional_titles_hold_aliases() -> None:
    record = {
        "facility": {
            "identifier": "0-756-1-387493",
            "name": ["Flüela permafrost", "Flüelapass"],
            "title": "Legacy title should not override name",
            "beginPosition": "2002-10-01",
        }
    }
    converted = converter.convert_record(record)
    assert converted["properties"]["title"] == "Flüela permafrost"
    assert converted["properties"]["additionalTitles"] == ["Flüelapass", "Legacy title should not override name"]


def test_facility_additional_ids_are_emitted_for_additional_wsi_values() -> None:
    record = {
        "facility": {
            "identifier": "0-20000-0-06725,0-756-1-Blatten",
            "name": "Blatten",
            "beginPosition": "2000-08-17",
        }
    }
    converted = converter.convert_record(record)
    assert converted["id"] == "0-20000-0-06725"
    assert converted["properties"]["additionalIds"] == ["0-756-1-Blatten"]
    assert "identifiers" not in converted["properties"]



def test_facility_id_is_first_wsi_from_identifier_list() -> None:
    record = {
        "facility": {
            "identifier": ["0-20000-0-06725", "0-756-1-Blatten"],
            "wigosStationIdentifier": "0-999-0-SHOULD-NOT-WIN",
            "name": "Blatten",
            "beginPosition": "2000-08-17",
        }
    }
    converted = converter.convert_record(record)
    assert converted["id"] == "0-20000-0-06725"
    assert converted["properties"]["additionalIds"] == ["0-756-1-Blatten", "0-999-0-SHOULD-NOT-WIN"]


def test_normalize_existing_record_uses_name_before_title() -> None:
    record = {
        "type": "Feature",
        "id": "0-20008-0-MKN",
        "geometry": None,
        "properties": {
            "type": "facility",
            "name": ["Mt. Kenya", "Mount Kenya"],
            "title": "Legacy title",
            "time": {"interval": ["2009-01-06", ".."], "resolution": "P1D"},
        },
    }
    normalized = converter.normalize_wmdr2_record(record)
    assert normalized["properties"]["title"] == "Mt. Kenya"
    assert normalized["properties"]["additionalTitles"] == ["Mount Kenya", "Legacy title"]
    assert "name" not in normalized["properties"]

def test_normalize_existing_record_converts_title_list_to_title_and_additional_titles() -> None:
    record = {
        "type": "Feature",
        "id": "0-20008-0-MKN",
        "geometry": None,
        "properties": {
            "type": "facility",
            "title": ["Mt. Kenya", "Mount Kenya"],
            "time": {"interval": ["2009-01-06", ".."], "resolution": "P1D"},
        },
    }
    normalized = converter.normalize_wmdr2_record(record)
    assert normalized["properties"]["title"] == "Mt. Kenya"
    assert normalized["properties"]["additionalTitles"] == ["Mount Kenya"]


def test_normalize_existing_record_converts_legacy_identifiers_to_additional_ids() -> None:
    record = {
        "type": "Feature",
        "id": "0-20000-0-06725",
        "geometry": None,
        "properties": {
            "type": "facility",
            "title": "Blatten",
            "identifiers": ["0-20000-0-06725", "0-756-1-Blatten"],
            "time": {"interval": ["2000-08-17", ".."], "resolution": "P1D"},
        },
    }
    normalized = converter.normalize_wmdr2_record(record)
    assert normalized["properties"]["additionalIds"] == ["0-756-1-Blatten"]
    assert "identifiers" not in normalized["properties"]

