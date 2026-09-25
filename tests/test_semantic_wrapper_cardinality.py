from __future__ import annotations

import pytest

from schema_registry import validator_for_def


CLIMATE_ZONE = "http://codes.wmo.int/wmdr/ClimateZone/tropicalWet"
SURFACE_COVER = "http://codes.wmo.int/wmdr/SurfaceCover/grass"
SURFACE_SCHEME = "http://codes.wmo.int/wmdr/SurfaceCoverClassification/LCZ"
SURFACE_ROUGHNESS = "http://codes.wmo.int/wmdr/SurfaceRoughness/open"
TOPOGRAPHY = "http://codes.wmo.int/wmdr/LocalTopography/flat"
OFFICIAL_STATUS = "http://codes.wmo.int/wmdr/StatusOfObservation/operational"

TIME = {"interval": ["2020-01-01", ".."]}


def concept(uri: str) -> dict[str, str]:
    return {"id": uri}


def is_valid(def_name: str, value: object) -> bool:
    validator = validator_for_def("wmdr2-common.schema.json", def_name)
    return not list(validator.iter_errors(value))


def test_date_field_does_not_require_time() -> None:
    assert is_valid("dateField", {})


@pytest.mark.parametrize("value", [{}, {"resolution": "PT1H"}, None])
def test_time_object_rejects_values_without_temporal_anchor(value: object) -> None:
    assert not is_valid("timeObject", value)


@pytest.mark.parametrize(
    "value",
    [
        {"date": "2020-01-01"},
        {"timestamp": "2020-01-01T12:00:00Z"},
        {"interval": ["2020-01-01", ".."]},
        {"interval": ["2020-01-01", ".."], "resolution": "PT1H"},
    ],
)
def test_time_object_accepts_real_temporal_content(value: object) -> None:
    assert is_valid("timeObject", value)


@pytest.mark.parametrize("value", [{}, {"time": TIME}])
def test_environment_rejects_empty_or_time_only_occurrences(value: object) -> None:
    assert not is_valid("environment", value)


@pytest.mark.parametrize(
    "value",
    [
        {"climateZone": concept(CLIMATE_ZONE)},
        {"surfaceRoughness": concept(SURFACE_ROUGHNESS)},
        {"population": [1000, None]},
        {"perimeter_km": [None, 10]},
        {"topographyBathymetry": {"localTopography": concept(TOPOGRAPHY)}},
        {
            "surfaceCover": {
                "value": concept(SURFACE_COVER),
                "scheme": concept(SURFACE_SCHEME),
            }
        },
        {"climateZone": concept(CLIMATE_ZONE), "time": TIME},
    ],
)
def test_environment_accepts_semantic_payload_without_requiring_time(
    value: object,
) -> None:
    assert is_valid("environment", value)


@pytest.mark.parametrize(
    "value",
    [{"population": [None, None]}, {"perimeter_km": [None, None]}],
)
def test_environment_null_only_numeric_ranges_are_not_semantic_payload(
    value: object,
) -> None:
    assert not is_valid("environment", value)


def test_surface_cover_requires_value_and_classification_scheme() -> None:
    valid = {
        "value": concept(SURFACE_COVER),
        "scheme": concept(SURFACE_SCHEME),
    }
    assert is_valid("surfaceCover", valid)
    assert not is_valid("surfaceCover", {"value": concept(SURFACE_COVER)})
    assert not is_valid("surfaceCover", {"scheme": concept(SURFACE_SCHEME)})
    assert not is_valid(
        "surfaceCover",
        {"value": "grass", "scheme": concept(SURFACE_SCHEME)},
    )


def test_topography_bathymetry_requires_at_least_one_component() -> None:
    assert not is_valid("topographyBathymetry", {})
    for key in (
        "localTopography",
        "relativeElevation",
        "topographicContext",
        "altitudeOrDepth",
    ):
        assert is_valid(
            "topographyBathymetry",
            {key: concept(f"http://codes.wmo.int/wmdr/{key}/example")},
        )
    assert not is_valid(
        "topographyBathymetry",
        {"localTopography": "flat"},
    )


def test_observing_procedure_requires_time_and_schedule() -> None:
    valid = {"time": TIME, "observingSchedules": ["schedule_example"]}
    assert is_valid("observingProcedure", valid)
    assert not is_valid("observingProcedure", {"time": TIME})
    assert not is_valid(
        "observingProcedure",
        {"observingSchedules": ["schedule_example"]},
    )


def test_official_status_requires_status_and_time() -> None:
    assert is_valid(
        "officialStatus",
        {"officialStatus": concept(OFFICIAL_STATUS), "time": TIME},
    )
    assert not is_valid("officialStatus", {"time": TIME})
    assert not is_valid(
        "officialStatus",
        {"officialStatus": concept(OFFICIAL_STATUS)},
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
    value: object,
) -> None:
    assert not is_valid("schedule", value)


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
    value: object,
) -> None:
    assert is_valid("schedule", value)


def test_schedule_wmo_intervals_are_iso8601_durations() -> None:
    assert not is_valid(
        "schedule",
        {
            "uid": "schedule_bad",
            "start": "0001-01-01",
            "wmo.int:samplingFrequency": "hourly",
        },
    )
