from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "wmdr2-record-feature.schema.json"

TERRITORY = "http://codes.wmo.int/wmdr/TerritoryName/CHE"
PROGRAM = "http://codes.wmo.int/wmdr/ProgramAffiliation/GAW"
CLIMATE_ZONE = "http://codes.wmo.int/wmdr/ClimateZone/tropicalWet"
SURFACE_COVER = "http://codes.wmo.int/wmdr/SurfaceCover/grass"
SURFACE_SCHEME = "http://codes.wmo.int/wmdr/SurfaceCoverClassification/LCZ"
SURFACE_ROUGHNESS = "http://codes.wmo.int/wmdr/SurfaceRoughness/open"
TOPOGRAPHY = "http://codes.wmo.int/wmdr/LocalTopography/flat"
OFFICIAL_STATUS = "http://codes.wmo.int/wmdr/StatusOfObservation/operational"

TIME = {"interval": ["2020-01-01", ".."]}


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validator_for(schema: dict, def_name: str) -> Draft202012Validator:
    wrapper = {
        "$schema": schema["$schema"],
        "$ref": f"#/$defs/{def_name}",
        "$defs": schema["$defs"],
    }
    return Draft202012Validator(wrapper, format_checker=FormatChecker())


def is_valid(schema: dict, def_name: str, value: object) -> bool:
    return not list(validator_for(schema, def_name).iter_errors(value))


def test_date_field_no_longer_requires_time(schema: dict) -> None:
    assert "required" not in schema["$defs"]["dateField"]


@pytest.mark.parametrize("value", [{}, {"resolution": "PT1H"}])
def test_time_object_rejects_objects_without_temporal_anchor(schema: dict, value: object) -> None:
    assert not is_valid(schema, "timeObject", value)


@pytest.mark.parametrize(
    "value",
    [
        {"date": "2020-01-01"},
        {"timestamp": "2020-01-01T12:00:00Z"},
        {"interval": ["2020-01-01", ".."]},
        {"interval": ["2020-01-01", ".."], "resolution": "PT1H"},
    ],
)
def test_time_object_accepts_real_temporal_content(schema: dict, value: object) -> None:
    assert is_valid(schema, "timeObject", value)


def test_territory_requires_payload_but_not_time(schema: dict) -> None:
    assert is_valid(schema, "territory", {"territory": TERRITORY})
    assert is_valid(schema, "territory", {"territory": TERRITORY, "time": TIME})
    assert not is_valid(schema, "territory", {})
    assert not is_valid(schema, "territory", {"time": TIME})
    assert not is_valid(schema, "territory", {"territory": "CHE"})


def test_program_affiliation_requires_program_but_not_time(schema: dict) -> None:
    assert is_valid(schema, "programAffiliation", {"program": PROGRAM})
    assert is_valid(schema, "programAffiliation", {"program": PROGRAM, "time": TIME})
    assert not is_valid(schema, "programAffiliation", {})
    assert not is_valid(schema, "programAffiliation", {"time": TIME})


@pytest.mark.parametrize("value", [{}, {"time": TIME}])
def test_environment_rejects_empty_or_time_only_occurrences(schema: dict, value: object) -> None:
    assert not is_valid(schema, "environment", value)


@pytest.mark.parametrize(
    "value",
    [
        {"climateZone": CLIMATE_ZONE},
        {"surfaceRoughness": SURFACE_ROUGHNESS},
        {"population": [1000, None]},
        {"perimeter_km": [None, 10]},
        {"topographyBathymetry": {"localTopography": TOPOGRAPHY}},
        {"surfaceCover": {"value": SURFACE_COVER, "scheme": SURFACE_SCHEME}},
        {"climateZone": CLIMATE_ZONE, "time": TIME},
    ],
)
def test_environment_accepts_semantic_payload_without_requiring_time(
    schema: dict, value: object
) -> None:
    assert is_valid(schema, "environment", value)


@pytest.mark.parametrize(
    "value",
    [{"population": [None, None]}, {"perimeter_km": [None, None]}],
)
def test_environment_null_only_numeric_ranges_are_not_semantic_payload(
    schema: dict, value: object
) -> None:
    assert not is_valid(schema, "environment", value)


def test_surface_cover_requires_value_and_classification_scheme(schema: dict) -> None:
    valid = {"value": SURFACE_COVER, "scheme": SURFACE_SCHEME}
    assert is_valid(schema, "surfaceCover", valid)
    assert not is_valid(schema, "surfaceCover", {"value": SURFACE_COVER})
    assert not is_valid(schema, "surfaceCover", {"scheme": SURFACE_SCHEME})
    assert not is_valid(
        schema, "surfaceCover", {"value": "grass", "scheme": SURFACE_SCHEME}
    )


def test_topography_bathymetry_requires_at_least_one_component(schema: dict) -> None:
    assert not is_valid(schema, "topographyBathymetry", {})
    for key in (
        "localTopography",
        "relativeElevation",
        "topographicContext",
        "altitudeOrDepth",
    ):
        assert is_valid(
            schema,
            "topographyBathymetry",
            {key: f"http://codes.wmo.int/wmdr/{key}/example"},
        )
    assert not is_valid(
        schema, "topographyBathymetry", {"localTopography": "flat"}
    )


def test_observing_procedure_requires_time_and_schedule(schema: dict) -> None:
    valid = {"time": TIME, "observingSchedules": ["schedule_example"]}
    assert is_valid(schema, "observingProcedure", valid)
    assert not is_valid(schema, "observingProcedure", {"time": TIME})
    assert not is_valid(
        schema, "observingProcedure", {"observingSchedules": ["schedule_example"]}
    )


def test_official_status_requires_status_and_time(schema: dict) -> None:
    assert is_valid(
        schema,
        "officialStatus",
        {"officialStatus": OFFICIAL_STATUS, "time": TIME},
    )
    assert not is_valid(schema, "officialStatus", {"time": TIME})
    assert not is_valid(
        schema, "officialStatus", {"officialStatus": OFFICIAL_STATUS}
    )


@pytest.mark.parametrize(
    "value",
    [
        {"uid": "schedule_example"},
        {"uid": "schedule_example", "start": "0001-01-01"},
        {
            "uid": "schedule_example",
            "start": "0001-01-01",
            "wmo.int:diurnalBaseTime": "00:00:00",
        },
        {
            "uid": "schedule_example",
            "start": "0001-01-01",
            "recurrenceOverrides": {"x": {}},
        },
        {
            "uid": "schedule_example",
            "start": "0001-01-01",
            "recurrenceRules": [{}],
        },
        {
            "uid": "schedule_example",
            "start": "0001-01-01",
            "recurrenceRules": [{"frequency": "sometimes"}],
        },
    ],
)
def test_schedule_rejects_identifier_or_anchor_without_cadence(
    schema: dict, value: object
) -> None:
    assert not is_valid(schema, "schedule", value)


@pytest.mark.parametrize(
    "value",
    [
        {
            "uid": "schedule_duration",
            "start": "0001-01-01T06:00:00",
            "duration": "PT12H",
        },
        {
            "uid": "schedule_recurrence",
            "start": "0001-01-01",
            "recurrenceRules": [{"frequency": "daily"}],
        },
        {
            "uid": "schedule_sampling",
            "start": "0001-01-01",
            "wmo.int:samplingFrequency": "PT10M",
        },
        {
            "uid": "schedule_reporting",
            "start": "0001-01-01",
            "wmo.int:aggregationInterval": "PT1H",
        },
    ],
)
def test_schedule_accepts_start_plus_meaningful_schedule_semantics(
    schema: dict, value: object
) -> None:
    assert is_valid(schema, "schedule", value)


def test_schedule_wmo_intervals_are_iso8601_durations(schema: dict) -> None:
    assert not is_valid(
        schema,
        "schedule",
        {
            "uid": "schedule_bad",
            "start": "0001-01-01",
            "wmo.int:samplingFrequency": "hourly",
        },
    )
