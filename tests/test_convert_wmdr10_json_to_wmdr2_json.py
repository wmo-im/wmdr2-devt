from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "convert_wmdr10_json_to_wmdr2_json.py"

OBSERVED_179 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179"
OBSERVED_12006 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
GEOMETRY_POINT = "http://codes.wmo.int/wmdr/Geometry/point"
GAW_REGIONAL = "http://codes.wmo.int/wmdr/ProgramAffiliation/GAWregional"
GBON = "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"


def _load_converter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_wmdr10_json_to_wmdr2_json", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_converter()


def _minimal_facility() -> dict[str, Any]:
    return {
        "identifier": "0-TEST",
        "name": "TEST",
        "geospatialLocation": "46 7 100",
    }


def _data_generation_with_coverage_and_reporting() -> dict[str, Any]:
    return {
        "beginPosition": "2020-01-01T00:00:00Z",
        "sampling": {
            "samplingStrategy": "continuous",
            "temporalSamplingInterval": "PT2S",
            "samplingTimePeriod": "PT2S",
        },
        "reporting": {
            "internationalExchange": "true",
            "temporalReportingInterval": "PT1H",
            "timeliness": "PT30M",
            "uom": "http://codes.wmo.int/wmdr/unit/mm",
            "dataPolicy": {
                "dataPolicy": "http://codes.wmo.int/wmdr/DataPolicy/noLimitation",
                "attribution": {"originator": {"role": None}},
            },
            "levelOfData": "http://codes.wmo.int/wmdr/LevelOfData/level1",
        },
        "coverage": {
            "startMonth": "1",
            "endMonth": "12",
            "startWeekday": "1",
            "endWeekday": "7",
            "startHour": "0",
            "endHour": "23",
            "startMinute": "0",
            "endMinute": "59",
            "diurnalBaseTime": "00:00:00Z",
        },
    }


def _observation_with_deployment(*, duplicate_data_generation: bool = False) -> dict[str, Any]:
    data_generation = [_data_generation_with_coverage_and_reporting()]
    if duplicate_data_generation:
        data_generation.append(_data_generation_with_coverage_and_reporting())
    return {
        "observedProperty": OBSERVED_12006,
        "type": GEOMETRY_POINT,
        "programAffiliation": [
            {"programAffiliation": GAW_REGIONAL, "beginPosition": "2020-01-01T00:00:00Z"},
            {"href": GBON, "beginPosition": "2021-01-01T00:00:00Z"},
            {"programAffiliation": GAW_REGIONAL, "beginPosition": "2022-01-01T00:00:00Z"},
        ],
        "deployments": [
            {
                "id": "dep1",
                "beginPosition": "2020-01-01T00:00:00Z",
                "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading",
                "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
                "referenceSurface": "http://codes.wmo.int/wmdr/LocalReferenceSurface/localGround",
                "verticalDistanceFromReferenceSurface": 2.0,
                "manufacturer": "Maker",
                "model": "Model",
                "verticalRange": {"min": 0, "max": 30},
                "observedProperty": [OBSERVED_12006, "local free-text variable"],
                "observedGeometry": GEOMETRY_POINT,
                "serialNumber": "SN1",
                "dataGeneration": data_generation,
            }
        ],
    }


def test_observation_title_uses_label_and_domain_when_labels_available() -> None:
    module.CODE_LIST_LABELS.clear()
    module.CODE_LIST_LABELS.update({"ObservedVariableAtmosphere": {"179": "Cloud amount"}})
    title = module._format_observation_title(OBSERVED_179)
    assert title == "domain: atmosphere; variable: 179 Cloud amount"


def test_observation_title_includes_geometry_when_available() -> None:
    module.CODE_LIST_LABELS.clear()
    module.CODE_LIST_LABELS.update({"ObservedVariableAtmosphere": {"179": "Cloud amount"}})
    title = module._format_observation_title(OBSERVED_179, GEOMETRY_POINT)
    assert title == "domain: atmosphere; geometry: point; variable: 179 Cloud amount"


def test_observation_program_affiliations_are_plain_unique_code_values() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [_observation_with_deployment()],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    observation = record["properties"]["observationSeries"][0]
    assert observation["programAffiliation"] == ["GAWregional", "GBON"]
    assert "programAffiliations" not in observation


def test_facility_temporal_program_affiliation_keeps_temporal_metadata() -> None:
    facility = _minimal_facility() | {
        "programAffiliation": [
            {
                "programAffiliation": GAW_REGIONAL,
                "beginPosition": "2020-01-01T00:00:00Z",
                "programSpecificFacilityId": "GAW-TEST",
                "reportingStatus": [
                    {
                        "reportingStatus": "http://codes.wmo.int/wmdr/ReportingStatus/operational",
                        "beginPosition": "2020-01-01T00:00:00Z",
                    }
                ],
            }
        ]
    }
    record = module.build_facility_feature(
        facility,
        [],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    assert record["properties"]["programAffiliation"] == [
        {
            "programAffiliation": "GAWregional",
            "reportingStatus": "operational",
            "date": "2020-01-01",
            "programSpecificFacilityId": "GAW-TEST",
        }
    ]
    assert "temporalProgramAffiliation" not in record["properties"]



def test_build_facility_feature_uses_current_core_model() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [_observation_with_deployment()],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    props = record["properties"]
    observation = props["observationSeries"][0]
    deployment_ref = observation["observingConfigurations"][0]["deployment"]
    deployment = props["deployments"][0]

    assert record["type"] == "Feature"
    assert record["conformsTo"] == ["http://wigos.wmo.int/spec/wmdr/2/conf/core"]
    assert record["geometry"] == {"type": "Point", "coordinates": [7.0, 46.0, 100]}
    assert "wmdr2" not in props
    assert "themes" not in props
    assert "externalIds" not in props
    assert props["deployments"]
    assert observation["observedProperty"] == 12006
    assert observation["observedGeometry"] == "point"
    assert observation["observedFeature"] == {"domain": "atmosphere"}
    assert "domain" not in observation
    assert "description" not in observation
    assert observation["referenceSurface"] == "localGround"
    assert observation["verticalDistanceFromReferenceSurface"] == {"value": 2.0}
    assert props["deployments"][0]["serialNumber"] == "SN1"
    assert props["deployments"][0]["instrument"].startswith("instrument:")
    assert deployment["date"] == "2020-01-01"
    assert deployment_ref == props["deployments"][0]["id"]
    assert "deployments" not in observation
    assert deployment["serialNumber"] == "SN1"
    assert deployment["instrument"].startswith("instrument:")
    assert props["instruments"][0]["verticalRange"] == {"min": 0.0, "max": 30.0}

def test_reporting_is_reusable_and_history_keeps_date_reference_and_uom() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [_observation_with_deployment()],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    props = record["properties"]
    reporting_defs = props["reporting"]
    reporting_history = props["observationSeries"][0]["reporting"]

    assert len(reporting_defs) == 1
    reporting_id = reporting_defs[0]["id"]
    assert reporting_id.startswith("reporting:")
    assert reporting_defs[0] == {
        "id": reporting_id,
        "internationalExchange": True,
        "temporalAggregate": "PT1H",
        "timeliness": "PT30M",
        "levelOfData": "level1",
        "dataPolicy": {"dataPolicy": "noLimitation"},
    }
    assert reporting_history == [
        {"date": "2020-01-01", "strategy": "unknown", "uom": "mm", "reporting": reporting_id}
    ]

def test_duplicate_temporal_observing_schedule_references_are_removed() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [_observation_with_deployment(duplicate_data_generation=True)],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    schedules = record["properties"]["schedules"]
    observing_procedures = record["properties"]["observationSeries"][0]["observingProcedures"]
    assert schedules
    assert len(observing_procedures) == len({(ref["date"], ref["strategy"], tuple(ref["observingSchedules"])) for ref in observing_procedures})
    assert all(
        schedule_uid in {schedule["uid"] for schedule in schedules}
        for procedure in observing_procedures
        for schedule_uid in procedure["observingSchedules"]
    )

def test_facility_environment_wraps_environmental_histories_as_arrays_of_objects() -> None:
    facility = _minimal_facility() | {
        "climateZone": {
            "climateZone": "http://codes.wmo.int/wmdr/ClimateZone/Cfb",
            "beginPosition": "1980-01-01T00:00:00Z",
        },
        "surfaceCover": {
            "surfaceCover": "http://codes.wmo.int/wmdr/SurfaceCover/urbanBuiltup",
            "beginPosition": "1981-01-01T00:00:00Z",
        },
        "population": {
            "population": [100, None],
            "perimeter_km": [10, 50],
            "beginPosition": "1990-01-01T00:00:00Z",
        },
        "surfaceRoughness": {
            "surfaceRoughness": "http://codes.wmo.int/wmdr/SurfaceRoughness/rough",
            "beginPosition": "1991-01-01T00:00:00Z",
        },
        "topographyBathymetry": {
            "localTopography": "flat",
        },
    }
    record = module.build_facility_feature(
        facility,
        [],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    props = record["properties"]
    environment = props["environment"]
    assert "environment" in props
    assert "temporalClimateZone" not in props
    assert "temporalSurfaceCover" not in props
    assert "localTopography" not in props
    assert {item["date"]: item for item in environment} == {
        "1980-01-01": {"date": "1980-01-01", "climateZone": "Cfb"},
        "1981-01-01": {"date": "1981-01-01", "surfaceCover": "urbanBuiltup"},
        "1990-01-01": {"date": "1990-01-01", "population": [100.0, None], "perimeter_km": [10.0, 50.0]},
        "1991-01-01": {"date": "1991-01-01", "surfaceRoughness": "rough"},
        "..": {"date": "..", "topographyBathymetry": {"localTopography": "flat"}},
    }

# ---------------------------------------------------------------------------
# Additional regression coverage restored for the drop-in package
# ---------------------------------------------------------------------------

import json
import tempfile

import pytest


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        (["a", "b"], ["a", "b"]),
        ("a", ["a"]),
    ],
)
def test_as_list_normalizes_scalars_lists_and_none(raw: Any, expected: list[Any]) -> None:
    assert module._as_list(raw) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((None, "", [], {}, "first"), "first"),
        ((0, "fallback"), 0),
        ((False, "fallback"), False),
    ],
)
def test_first_non_empty_preserves_falsey_but_meaningful_values(values: tuple[Any, ...], expected: Any) -> None:
    assert module._first_non_empty(*values) == expected


def test_clean_none_removes_empty_members_but_preserves_nulls_inside_arrays() -> None:
    assert module._clean_none({"a": None, "b": "", "c": [], "d": {}, "e": [None, "", [], {}, 1]}) == {
        "e": [None, 1]
    }


def test_preserve_and_restore_null_sentinel_roundtrip() -> None:
    payload = {"a": [None, "x"], "b": {"c": None}}
    assert module._restore_null_sentinel(module._preserve_nulls(payload)) == payload


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" a b!c ", "a-b-c"),
        ("Station/ABC # 1", "Station/ABC-#-1"),
        ("", "record"),
    ],
)
def test_sanitize_id(raw: str, expected: str) -> None:
    assert module._sanitize_id(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A b/c", "a-b-c"),
        ("---", "value"),
        ("Hello__World", "hello-world"),
    ],
)
def test_slug(raw: str, expected: str) -> None:
    assert module._slug(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://codes.wmo.int/wmdr/FacilityType/landFixed", "landFixed"),
        ("abc#def", "def"),
        ("(unknown)", "unknown"),
        ("<http://x/y>", "y"),
        ("", None),
    ],
)
def test_last_segment(raw: str, expected: str | None) -> None:
    assert module._last_segment(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12006", 12006),
        ("http://codes.wmo.int/wmdr/unit/mm", "mm"),
        ("(unknown)", "unknown"),
        ("abc", "abc"),
        (5, 5),
    ],
)
def test_normalize_code_value(raw: Any, expected: Any) -> None:
    assert module._normalize_code_value(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://codes.wmo.int/wmdr/unit/mm", "mm"),
        ("https://codes.wmo.int/wmdr/FacilityType/landFixed", "landFixed"),
        ("http://example.org/not-wmdr/value", "http://example.org/not-wmdr/value"),
    ],
)
def test_compact_wmdr_code_value_only_compacts_wmdr_urls(raw: str, expected: str) -> None:
    assert module._compact_wmdr_code_value(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20200102_test", "2020-01-02"),
        ("20200102", "2020-01-02"),
        ("2020-01-02Z", "2020-01-02"),
        ("2020-01-02T03:04:05Z", "2020-01-02"),
        ("..", ".."),
        (None, None),
    ],
)
def test_normalize_date_value(raw: Any, expected: str | None) -> None:
    assert module._normalize_date_value(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20200102_test", "2020-01-02T00:00:00Z"),
        ("20200102", "2020-01-02T00:00:00Z"),
        ("2020-01-02", "2020-01-02T00:00:00Z"),
        ("2020-01-02Z", "2020-01-02T00:00:00Z"),
        ("2020-01-02T03:04:05Z", "2020-01-02T03:04:05Z"),
    ],
)
def test_normalize_record_datetime(raw: str, expected: str) -> None:
    assert module._normalize_record_datetime(raw) == expected


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (None, None, None),
        ("2020-01-01", None, {"interval": ["2020-01-01", ".."]}),
        (None, "2022-01-01", {"interval": ["..", "2022-01-01"]}),
        ("20200101", "20201231", {"interval": ["2020-01-01", "2020-12-31"]}),
    ],
)
def test_time_interval_uses_daily_resolution_and_open_marker(start: Any, end: Any, expected: dict[str, Any] | None) -> None:
    assert module._time_interval(start, end) == expected


def test_time_interval_can_include_explicit_resolution() -> None:
    assert module._time_interval("2020-01-01", None, resolution="day") == {
        "interval": ["2020-01-01", ".."],
        "resolution": "day",
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
    assert module._parse_pos_lon_lat_z(raw) == expected


def test_facility_temporal_geometry_entries_are_sorted_and_deduplicated() -> None:
    facility = {
        "geospatialLocation": {"geoLocation": "46 7 100", "beginPosition": "2021-01-01"},
        "geospatialLocationHistory": [
            {"geoLocation": "45 6 99", "beginPosition": "2020-01-01"},
            {"geoLocation": "45 6 99", "beginPosition": "2020-01-01"},
        ],
    }
    assert module._facility_temporal_geometry_entries(facility) == [
        {"coordinates": [6.0, 45.0, 99], "date": "2020-01-01"},
        {"coordinates": [7.0, 46.0, 100], "date": "2021-01-01"},
    ]


def test_temporal_geometry_extension_omits_single_coordinate_without_methods() -> None:
    assert module._temporal_geometry_extension([{"coordinates": [7, 46], "date": "2020-01-01"}]) is None


def test_temporal_geometry_extension_emits_single_coordinate_when_methods_are_present() -> None:
    assert module._temporal_geometry_extension(
        [
            {
                "coordinates": [7, 46],
                "date": "2020-01-01",
                "methods": ["gps"],
            }
        ]
    ) == {
        "type": "MovingPoint",
        "coordinates": [[7, 46]],
        "dates": ["2020-01-01"],
        "methods": [["gps"]],
    }


def test_temporal_geometry_extension_emits_two_or_more_coordinates_without_methods() -> None:
    assert module._temporal_geometry_extension(
        [
            {"coordinates": [6, 45], "date": "2020-01-01"},
            {"coordinates": [7, 46], "date": None},
        ]
    ) == {"type": "MovingPoint", "coordinates": [[6, 45], [7, 46]], "dates": ["2020-01-01", ".."]}


def test_facility_geometry_uses_latest_temporal_geometry_entry() -> None:
    assert module._facility_geometry_from_entries(
        [
            {"coordinates": [6, 45], "date": "2020-01-01"},
            {"coordinates": [7, 46, 100], "date": "2021-01-01"},
        ]
    ) == {"type": "Point", "coordinates": [7, 46, 100]}


def test_geopositioning_methods_are_extracted_from_exact_wmdr10_key() -> None:
    item = {
        "geopositioningMethod": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
    }
    assert module._geopositioning_methods(item) == ["gps"]


def test_geopositioning_methods_ignore_unrecognised_spelling_variants() -> None:
    item = {
        "geoPositioningMethod": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
    }
    assert module._geopositioning_methods(item) == []


def test_temporal_geometry_entries_include_geopositioning_methods_when_present() -> None:
    facility = {
        "geospatialLocation": {
            "geoLocation": "46 7 100",
            "beginPosition": "2021-01-01",
            "geopositioningMethod": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
        },
        "geospatialLocationHistory": [
            {"geoLocation": "45 6 99", "beginPosition": "2020-01-01"},
        ],
    }
    assert module._facility_temporal_geometry_entries(facility) == [
        {"coordinates": [6.0, 45.0, 99], "date": "2020-01-01"},
        {"coordinates": [7.0, 46.0, 100], "date": "2021-01-01", "methods": ["gps"]},
    ]


def test_temporal_geometry_extension_emits_aligned_methods_only_when_present() -> None:
    assert module._temporal_geometry_extension(
        [
            {"coordinates": [6, 45], "date": "2020-01-01"},
            {"coordinates": [7, 46], "date": "2021-01-01", "methods": ["gps"]},
        ]
    ) == {
        "type": "MovingPoint",
        "coordinates": [[6, 45], [7, 46]],
        "dates": ["2020-01-01", "2021-01-01"],
        "methods": [[], ["gps"]],
    }


def test_temporal_geometry_methods_empty_slots_survive_clean_none() -> None:
    payload = {
        "temporalGeometry": {
            "type": "MovingPoint",
            "coordinates": [[6, 45], [7, 46]],
            "dates": ["2020-01-01", "2021-01-01"],
            "methods": [[], ["gps"]],
        }
    }
    assert module._clean_none(payload) == payload


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (OBSERVED_12006, "atmosphere"),
        ("http://codes.wmo.int/wmdr/ObservedVariableTerrestrial/12", "terrestrial"),
        ("not-a-code", None),
    ],
)
def test_observed_domain_from_observed_variable(raw: str, expected: str | None) -> None:
    assert module._observed_domain_from_observed_variable(raw) == expected


def test_format_observation_title_without_label_uses_code_only() -> None:
    module.CODE_LIST_LABELS.clear()
    assert module._format_observation_title(OBSERVED_12006, GEOMETRY_POINT) == (
        "domain: atmosphere; geometry: point; variable: 12006"
    )


def test_keywords_from_values_flatten_normalize_and_deduplicate() -> None:
    assert module._keywords_from_values(["A", "A", None, "http://x/B", "C", ""]) == ["A", "B", "C"]


def test_normalize_display_text_collapses_whitespace_and_unknown_tokens() -> None:
    assert module._normalize_display_text("  A\n  B\tC  ") == "A B C"
    assert module._normalize_display_text("(unknown)") == "unknown"


def test_normalize_description_value_prefers_textual_fields_and_drops_metadata() -> None:
    assert module._normalize_description_value({"@gml:id": "x", "description": "  Some text  "}) == "Some text"
    assert module._normalize_description_value({"value": "Fallback text"}) == "Fallback text"


def test_normalize_roles_extracts_code_values_from_mixed_inputs() -> None:
    assert module._normalize_roles(
        [
            "http://codes.wmo.int/wmdr/ResponsiblePartyRole/owner",
            {"href": "http://codes.wmo.int/wmdr/ResponsiblePartyRole/operator"},
            "owner",
        ]
    ) == ["owner", "operator"]


def test_normalize_contact_returns_public_contact_and_discovery_contact() -> None:
    public, discovery = module._normalize_contact(
        {
            "organisationName": "Org",
            "individualName": "Jane Doe",
            "contactInfo": {"address": {"electronicMailAddress": "jane@example.org"}},
            "role": "http://codes.wmo.int/wmdr/ResponsiblePartyRole/owner",
        }
    )
    assert public == {"name": "Jane Doe", "organization": "Org", "emails": [{"value": "jane@example.org"}], "roles": ["owner"]}
    assert discovery == public


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
    assert module._parse_bool(raw) is expected


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
def test_normalize_diurnal_time(raw: str, expected: str) -> None:
    assert module._normalize_diurnal_time(raw) == expected


def test_normalize_temporal_values_produces_array_of_objects_not_object_of_arrays() -> None:
    assert module._normalize_temporal_climate_zone(
        [
            {"climateZone": "http://codes.wmo.int/wmdr/ClimateZone/Cfb", "beginPosition": "2020-01-01T00:00:00Z"},
            "http://codes.wmo.int/wmdr/ClimateZone/Af",
        ]
    ) == [
        {"climateZone": "Cfb", "date": "2020-01-01"},
        {"climateZone": "Af", "date": ".."},
    ]


def test_surface_cover_preserves_classification_inside_each_temporal_object() -> None:
    assert module._normalize_temporal_surface_cover(
        {
            "surfaceCover": "http://codes.wmo.int/wmdr/SurfaceCover/grassland",
            "surfaceClassification": {"href": "http://codes.wmo.int/wmdr/SurfaceClassification/local"},
            "beginPosition": "2020-01-01",
        }
    ) == [{"surfaceCover": "grassland", "date": "2020-01-01", "surfaceClassification": "local"}]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("100 200", [100.0, 200.0]),
        ("100,200", [100.0, 200.0]),
        ([1, 2], [1.0, 2.0]),
        ("100", [100.0, None]),
        ("not numeric", None),
    ],
)
def test_parse_two_value_number_array(raw: Any, expected: list[float | None] | None) -> None:
    assert module._parse_two_value_number_array(raw) == expected


def test_parse_two_value_perimeter_array_defaults_missing_values() -> None:
    assert module._parse_two_value_perimeter_array(None) == [10.0, 50.0]
    assert module._parse_two_value_perimeter_array([5, None]) == [5.0, 50.0]


def test_normalize_temporal_population_uses_population_perimeter_and_dates() -> None:
    assert module._normalize_temporal_population(
        {"population": "100", "perimeter_km": [10, 50], "beginPosition": "1990-01-01T00:00:00Z"}
    ) == [{"population": [100.0, None], "perimeter_km": [10.0, 50.0], "dates": ["1990-01-01", ".."]}]


def test_normalize_temporal_surface_roughness_is_array_of_objects() -> None:
    assert module._normalize_temporal_surface_roughness(
        {"surfaceRoughness": "http://codes.wmo.int/wmdr/SurfaceRoughness/rough", "beginPosition": "1991-01-01"}
    ) == [{"surfaceRoughness": "rough", "date": "1991-01-01"}]


def test_environment_uses_topography_bathymetry_object() -> None:
    environment = module._normalize_environment(
        {
            "topographyBathymetry": {
                "localTopography": {"value": "flat", "beginPosition": "2020-01-01"},
                "relativeElevation": {"value": "ridge", "beginPosition": "2021-01-01"},
                "topographicContext": {"value": "valley", "beginPosition": "2022-01-01"},
                "altitudeOrDepth": {"value": 123, "beginPosition": "2023-01-01"},
            }
        }
    )
    assert environment == [
        {
            "date": "..",
            "topographyBathymetry": {
                "localTopography": "flat",
                "relativeElevation": "ridge",
                "topographicContext": "valley",
                "altitudeOrDepth": 123,
            },
        }
    ]

def test_program_affiliation_values_accept_program_affiliation_href_and_string() -> None:
    assert module._program_affiliation_values({"programAffiliation": GAW_REGIONAL}) == ["GAWregional"]
    assert module._program_affiliation_values({"href": GBON}) == ["GBON"]
    assert module._program_affiliation_values(GBON) == ["GBON"]


def test_normalize_program_affiliations_deduplicates_plain_values() -> None:
    assert module._normalize_program_affiliations(
        [
            {"programAffiliation": GAW_REGIONAL},
            {"href": GBON},
            {"programAffiliation": GAW_REGIONAL},
        ]
    ) == ["GAWregional", "GBON"]


def test_temporal_program_affiliation_expands_reporting_status_history() -> None:
    assert module._normalize_temporal_program_affiliation(
        {
            "programAffiliation": GAW_REGIONAL,
            "beginPosition": "2020-01-01",
            "programSpecificFacilityId": "GAW-1",
            "reportingStatus": [
                {"reportingStatus": "http://codes.wmo.int/wmdr/ReportingStatus/operational", "beginPosition": "2020-01-01"},
                {"reportingStatus": "http://codes.wmo.int/wmdr/ReportingStatus/closed", "beginPosition": "2021-01-01"},
            ],
        }
    ) == [
        {
            "programAffiliation": "GAWregional",
            "reportingStatus": "operational",
            "date": "2020-01-01",
            "programSpecificFacilityId": "GAW-1",
        },
        {
            "programAffiliation": "GAWregional",
            "reportingStatus": "closed",
            "date": "2021-01-01",
            "programSpecificFacilityId": "GAW-1",
        },
    ]


def test_facility_reporting_status_is_only_kept_under_program_affiliation() -> None:
    facility = _minimal_facility() | {
        "reportingStatus": {
            "reportingStatus": "http://codes.wmo.int/wmdr/ReportingStatus/operational",
            "beginPosition": "2020-01-01",
        }
    }
    record = module.build_facility_feature(
        facility,
        [],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    assert "temporalReportingStatus" not in record["properties"]


def test_temporal_instrument_operating_status_is_array_of_objects() -> None:
    assert module._normalize_temporal_instrument_operating_status(
        {"instrumentOperatingStatus": "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational", "beginPosition": "2020-01-01"}
    ) == [{"instrumentOperatingStatus": "operational", "date": "2020-01-01"}]


def test_temporal_official_status_maps_wmdr10_boolean_values() -> None:
    assert module._normalize_temporal_official_status(
        {"officialStatus": True, "beginPosition": "2020-01-01"}
    ) == [{"officialStatus": "primary", "date": "2020-01-01"}]
    assert module._normalize_temporal_official_status(
        {"officialStatus": False, "beginPosition": "2020-01-01"}
    ) == [{"officialStatus": "additional", "date": "2020-01-01"}]
    assert module._normalize_temporal_official_status(
        None,
        fallback_date="2020-01-01",
        default_unknown=True,
    ) == [{"officialStatus": "unknown", "date": "2020-01-01"}]


def test_temporal_data_policy_retains_policy_and_attribution_with_dates() -> None:
    assert module._normalize_temporal_data_policy(
        {"dataPolicy": "http://codes.wmo.int/wmdr/DataPolicy/noLimitation", "attribution": {"originator": "Org"}, "beginPosition": "2020-01-01"}
    ) == [{"dataPolicy": "noLimitation", "attribution": {"originator": "Org"}, "date": "2020-01-01"}]


def test_facility_set_refs_are_plural_id_references() -> None:
    assert module._facility_set_refs(["GAW", {"facilitySet": "GBON"}, "GAW"]) == ["facilitySet:GAW", "facilitySet:GBON"]


def test_facility_set_catalog_uses_id_title_description_shape() -> None:
    assert module.facility_set_catalog_entry("GAW", description="Global Atmosphere Watch") == {
        "id": "facilitySet:GAW",
        "title": "GAW",
        "description": "Global Atmosphere Watch",
    }
    assert module.facility_set_catalog(["GAW"]) == {
        "facilitySets": [{"id": "facilitySet:GAW", "title": "GAW"}]
    }


def test_instrument_id_prefers_explicit_identifier_and_generated_ids_are_stable() -> None:
    raw = {"manufacturer": "Maker", "model": "Model"}
    generated = module._instrument_record_id(raw, facility_id="0-TEST")
    assert isinstance(generated, str)
    assert generated.startswith("instrument:")
    assert generated == module._instrument_record_id({"manufacturer": "Maker", "model": "Model"}, facility_id="0-TEST")
    assert module._instrument_record_id({"manufacturer": "unknown", "model": None}, facility_id="0-TEST") is None




def test_observing_configurations_are_dated_on_observation_series_and_capabilities_on_instrument() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [_observation_with_deployment()],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    instrument = record["properties"]["instruments"][0]
    observation = record["properties"]["observationSeries"][0]
    assert instrument["observingMethods"] == [266]
    assert observation["observingConfigurations"] == [{"date": "2020-01-01", "deployment": "deployment:dep1", "observingMethod": 266}]


def test_observation_series_observing_configuration_uses_nil_reason_when_absent_without_bloating_instrument() -> None:
    observation = _observation_with_deployment()
    observation["deployments"][0].pop("observingMethod", None)
    record = module.build_facility_feature(
        _minimal_facility(),
        [observation],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    instrument = record["properties"]["instruments"][0]
    observation_series = record["properties"]["observationSeries"][0]
    assert "observingMethods" not in instrument
    assert observation_series["observingConfigurations"] == [{"date": "..", "observingMethod": {"nilReason": "unknown"}}]




def test_observation_series_observing_configuration_converts_unknown_string_to_nil_reason() -> None:
    observation = _observation_with_deployment()
    observation["deployments"][0]["observingMethod"] = "unknown"
    record = module.build_facility_feature(
        _minimal_facility(),
        [observation],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    instrument = record["properties"].get("instruments", [])[0]
    observation_series = record["properties"]["observationSeries"][0]
    assert "observingMethods" not in instrument
    assert observation_series["observingConfigurations"] == [
        {"date": "2020-01-01", "deployment": "deployment:dep1", "observingMethod": {"nilReason": "unknown"}}
    ]

def test_observation_series_observing_configuration_preserves_explicit_nil_reason() -> None:
    observation = _observation_with_deployment()
    observation["deployments"][0]["observingMethod"] = {"nilReason": "withheld"}
    record = module.build_facility_feature(
        _minimal_facility(),
        [observation],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    instrument = record["properties"]["instruments"][0]
    observation_series = record["properties"]["observationSeries"][0]
    assert "observingMethods" not in instrument
    assert observation_series["observingConfigurations"] == [{"date": "2020-01-01", "deployment": "deployment:dep1", "observingMethod": {"nilReason": "withheld"}}]


def test_observation_series_can_record_consecutive_observing_methods_from_deployments() -> None:
    first = _observation_with_deployment()
    second_deployment = dict(first["deployments"][0])
    second_deployment["id"] = "dep2"
    second_deployment["beginPosition"] = "2001-01-01"
    second_deployment["observingMethod"] = "http://codes.wmo.int/wmdr/ObservingMethod/267"
    first["deployments"].append(second_deployment)
    record = module.build_facility_feature(
        _minimal_facility(),
        [first],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    observation_series = record["properties"]["observationSeries"][0]
    assert observation_series["observingConfigurations"] == [
        {"date": "2020-01-01", "deployment": "deployment:dep1", "observingMethod": 266},
        {"date": "2001-01-01", "deployment": "deployment:dep2", "observingMethod": 267},
    ]


def test_observing_configuration_method_alone_does_not_create_instrument_catalogue_entry() -> None:
    observation = {
        "observedProperty": OBSERVED_12006,
        "deployments": [
            {
                "id": "dep1",
                "beginPosition": "1980-01-01T00:00:00Z",
                "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
            }
        ],
    }
    record = module.build_facility_feature(
        _minimal_facility(),
        [observation],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    props = record["properties"]
    assert "instruments" not in props
    assert props["observationSeries"][0]["observingConfigurations"] == [
        {"date": "1980-01-01", "deployment": "deployment:dep1", "observingMethod": 266}
    ]

def test_normalize_instrument_includes_optional_title_and_description_when_available() -> None:
    instrument = module._normalize_instrument(
        {
            "manufacturer": "Maker",
            "model": "Model",
            "instrumentTitle": "Instrument title",
            "instrumentDescription": {"description": "Instrument description"},
        },
        facility_id="0-TEST",
    )
    assert instrument is not None
    assert instrument["title"] == "Instrument title"
    assert instrument["description"] == "Instrument description"
    assert instrument["manufacturer"] == "Maker"
    assert instrument["model"] == "Model"


def test_normalize_vertical_range_accepts_object_and_flat_min_max_fields() -> None:
    assert module._normalize_vertical_range({"verticalRange": {"min": "0", "max": 30}}) == {"min": 0.0, "max": 30.0}
    assert module._normalize_vertical_range({"verticalRangeMinimum": 5, "verticalRangeMaximum": "50"}) == {
        "min": 5.0,
        "max": 50.0,
    }
    assert module._normalize_vertical_range({"verticalRange": {"min": 0}}) is None


def test_normalize_instrument_includes_vertical_range_when_available() -> None:
    instrument = module._normalize_instrument(
        {
            "manufacturer": "Maker",
            "model": "Model",
            "verticalRange": {"min": 0, "max": 30},
        },
        facility_id="0-TEST",
    )
    assert instrument is not None
    assert instrument["verticalRange"] == {"min": 0.0, "max": 30.0}


def test_normalize_instrument_observed_property_accepts_codes_and_free_text() -> None:
    assert module._normalize_instrument_observed_property(
        {
            "observedProperty": [
                "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
                {"href": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"},
                {"description": "locally defined aerosol metric"},
            ]
        }
    ) == [12006, "locally defined aerosol metric"]


def test_normalize_instrument_observed_geometry_compacts_geometry_code() -> None:
    assert module._normalize_instrument_observed_geometry(
        {"observedGeometry": "http://codes.wmo.int/wmdr/Geometry/point"}
    ) == "point"
    assert module._normalize_instrument_observed_geometry({"observedGeometry": {"href": GEOMETRY_POINT}}) == "point"


def test_normalize_instrument_includes_observed_property_and_geometry_when_available() -> None:
    instrument = module._normalize_instrument(
        {
            "manufacturer": "Maker",
            "model": "Model",
            "observedProperty": [OBSERVED_12006, "local free-text variable"],
            "observedGeometry": GEOMETRY_POINT,
        },
        facility_id="0-TEST",
    )
    assert instrument is not None
    assert instrument["observedProperty"] == [12006, "local free-text variable"]
    assert instrument["observedGeometry"] == "point"


def test_observed_property_alone_is_enough_to_create_an_instrument_record() -> None:
    instrument = module._normalize_instrument(
        {"observedProperty": [OBSERVED_12006]},
        facility_id="0-TEST",
    )
    assert instrument is not None
    assert instrument["id"].startswith("instrument:")
    assert instrument["observedProperty"] == [12006]


def test_vertical_range_alone_is_enough_to_create_an_instrument_record() -> None:
    instrument = module._normalize_instrument(
        {"verticalRangeMin": 10, "verticalRangeMax": 500},
        facility_id="0-TEST",
    )
    assert instrument is not None
    assert instrument["id"].startswith("instrument:")
    assert instrument["verticalRange"] == {"min": 10.0, "max": 500.0}


def test_deployment_temporal_serial_number_is_array_of_objects() -> None:
    assert module._deployment_temporal_serial_number({"serialNumber": "SN1", "beginPosition": "2020-01-01"}) == [
        {"serialNumber": "SN1", "date": "2020-01-01"}
    ]


def test_jscalendar_schedule_uses_v024_sampling_and_aggregation_extensions() -> None:
    event = module._jscalendar_observing_schedule(
        {
            "beginPosition": "2020-01-01T00:00:00Z",
            "temporalSamplingInterval": "PT1H",
            "reporting": {"temporalReportingInterval": "PT1H"},
            "diurnalBaseTime": "7:5",
        }
    )
    assert event is not None
    assert event["@type"] == "Event"
    assert event["wmi.int:samplingFrequency"] == "PT1H"
    assert event["wmo.int:aggregationInterval"] == "PT1H"
    assert "wmo.int:aggregation" not in event
    assert event["uid"].startswith("schedule_")


def test_register_observing_schedule_refs_deduplicates_references_and_registry_entries() -> None:
    registry: dict[str, dict[str, Any]] = {}
    refs = module._register_observing_schedule_refs(
        [
            [
                {"beginPosition": "2020-01-01", "temporalSamplingInterval": "PT1H"},
                {"beginPosition": "2020-01-01", "temporalSamplingInterval": "PT1H"},
            ]
        ],
        schedule_registry=registry,
    )
    assert refs is not None
    assert len(refs) == 1
    assert list(registry) == [refs[0]["schedule"]]


def test_normalize_observation_reporting_registers_reusable_definitions() -> None:
    registry: dict[str, dict[str, Any]] = {}
    reporting = module._normalize_observation_reporting(
        [
            {
                "beginPosition": "2020-01-01",
                "temporalSamplingInterval": "PT1H",
                "reporting": {
                    "internationalExchange": "true",
                    "temporalReportingInterval": "PT1H",
                    "uom": "http://codes.wmo.int/wmdr/unit/mm",
                    "timeliness": "PT30M",
                },
            },
            {
                "beginPosition": "2021-01-01",
                "temporalSamplingInterval": "PT10M",
                "reporting": {
                    "internationalExchange": "false",
                    "temporalReportingInterval": "PT10M",
                    "uom": None,
                },
            },
        ],
        reporting_registry=registry,
    )
    assert reporting is not None
    assert len(registry) == 2
    assert reporting[0]["date"] == "2020-01-01"
    assert reporting[0]["strategy"] == "unknown"
    assert reporting[0]["uom"] == "mm"
    assert reporting[0]["reporting"] in registry
    assert registry[reporting[0]["reporting"]]["internationalExchange"] is True
    assert registry[reporting[0]["reporting"]]["temporalAggregate"] == "PT1H"
    assert registry[reporting[0]["reporting"]]["timeliness"] == "PT30M"
    assert reporting[1]["date"] == "2021-01-01"
    assert reporting[1]["strategy"] == "unknown"
    assert reporting[1]["reporting"] in registry
    assert registry[reporting[1]["reporting"]]["internationalExchange"] is False
    assert registry[reporting[1]["reporting"]]["temporalAggregate"] == "PT10M"


def test_two_observations_can_reuse_the_same_reporting_definition() -> None:
    obs1 = _observation_with_deployment()
    obs2 = _observation_with_deployment() | {"observedProperty": OBSERVED_179}
    record = module.build_facility_feature(
        _minimal_facility(),
        [obs1, obs2],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    props = record["properties"]
    assert len(props["reporting"]) == 1
    reporting_id = props["reporting"][0]["id"]
    assert all(
        observation["reporting"][0]["reporting"] == reporting_id
        for observation in props["observationSeries"]
    )


def test_normalize_observation_links_to_embedded_deployments_by_id() -> None:
    observation = module._normalize_observation(_observation_with_deployment(), index=1, facility_id="0-TEST")
    assert "deployments" not in observation
    assert observation["observingConfigurations"][0]["deployment"] == "deployment:dep1"
    assert observation["programAffiliation"] == ["GAWregional", "GBON"]

def test_convert_payload_accepts_split_payload_shape() -> None:
    converted = module.convert_payload(
        {
            "header": {"dateStamp": "2020-01-02"},
            "facility": _minimal_facility(),
            "observationSeries": [_observation_with_deployment()],
            "deployments": [],
        },
        source_name="20200102_0-TEST",
    )
    assert converted["id"] == "facility:0-TEST"
    assert converted["properties"]["created"] == "2020-01-02T00:00:00Z"
    assert converted["properties"]["observationSeries"][0]["programAffiliation"] == ["GAWregional", "GBON"]


def test_convert_payload_emits_temporal_geometry_methods_from_geopositioning_method() -> None:
    converted = module.convert_payload(
        {
            "facility": {
                "identifier": "0-TEST",
                "name": "TEST",
                "geospatialLocation": {
                    "geoLocation": "46 7 100",
                    "beginPosition": "2021-01-01",
                    "geopositioningMethod": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
                },
                "geospatialLocationHistory": [
                    {"geoLocation": "45 6 99", "beginPosition": "2020-01-01"},
                ],
            },
            "observationSeries": [],
        },
        source_name="0-TEST",
    )
    assert converted["temporalGeometry"] == {
        "type": "MovingPoint",
        "coordinates": [[6.0, 45.0, 99], [7.0, 46.0, 100]],
        "dates": ["2020-01-01", "2021-01-01"],
        "methods": [[], ["gps"]],
    }


def test_convert_payload_emits_single_temporal_geometry_when_method_is_present() -> None:
    converted = module.convert_payload(
        {
            "facility": {
                "identifier": "0-TEST",
                "name": "TEST",
                "geospatialLocation": {
                    "geoLocation": "46 7 100",
                    "beginPosition": "2021-01-01",
                    "geopositioningMethod": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
                },
            },
            "observationSeries": [],
        },
        source_name="0-TEST",
    )
    assert converted["geometry"] == {"type": "Point", "coordinates": [7.0, 46.0, 100]}
    assert converted["temporalGeometry"] == {
        "type": "MovingPoint",
        "coordinates": [[7.0, 46.0, 100]],
        "dates": ["2021-01-01"],
        "methods": [["gps"]],
    }


def test_convert_group_uses_source_name_as_fallback_facility_when_missing() -> None:
    converted = module.convert_group({"observationSeries": []}, source_name="fallback-record")
    assert converted["id"] == "facility:fallback-record"
    assert converted["properties"]["title"] == "fallback-record"


def test_convert_file_writes_json_output(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    target_dir = tmp_path / "out"
    source.write_text(json.dumps({"facility": _minimal_facility(), "observationSeries": []}), encoding="utf-8")
    output_path = module.convert_file(source, target_dir)
    converted = json.loads(output_path.read_text(encoding="utf-8"))
    assert converted["id"] == "facility:0-TEST"


def test_load_code_list_labels_reads_csv_mapping(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text("domain,code,label\nObservedVariableAtmosphere,12006,Air temperature\n", encoding="utf-8")
    assert module._load_code_list_labels({"codeListLabels": {"files": [str(labels)]}}, base_dir=tmp_path) == {
        "ObservedVariableAtmosphere": {"12006": "Air temperature"}
    }


def test_configured_empty_discovery_policy_suppresses_keywords() -> None:
    policy = module._normalize_discovery_policy({"discovery": {"facility": {"keywords": []}}})
    assert policy["facility"]["keywords"] == []

    record = module.build_facility_feature(
        _minimal_facility(),
        [],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    assert record["properties"]["keywords"] == ["0-TEST", "TEST"]

    old_policy = getattr(module, "DISCOVERY_POLICY")
    try:
        setattr(module, "DISCOVERY_POLICY", policy)
        record = module.build_facility_feature(
            _minimal_facility(),
            [],
            [],
            {},
            source_name="20200101_0-TEST",
        )
        assert "keywords" not in record["properties"]
    finally:
        setattr(module, "DISCOVERY_POLICY", old_policy)


def test_configured_discovery_policy_can_explicitly_keep_facility_keywords() -> None:
    policy = module._normalize_discovery_policy(
        {"discovery": {"facility": {"keywords": ["identifier", "name"]}}}
    )
    old_policy = getattr(module, "DISCOVERY_POLICY")
    try:
        setattr(module, "DISCOVERY_POLICY", policy)
        record = module.build_facility_feature(
            _minimal_facility(),
            [],
            [],
            {},
            source_name="20200101_0-TEST",
        )
        assert record["properties"]["keywords"] == ["0-TEST", "TEST"]
    finally:
        setattr(module, "DISCOVERY_POLICY", old_policy)


def test_main_runs_catalogue_post_processing_when_enabled(tmp_path: Path) -> None:
    source = tmp_path / "wmdr10"
    target = tmp_path / "wmdr2"
    catalogue_records = target / "catalogue_representation"
    catalogues = target / "catalogues"
    source.mkdir()

    payload = {
        "header": {"dateStamp": "2020-01-02"},
        "facility": _minimal_facility()
        | {
            "contact": {
                "individualName": "Jane Smith",
                "organisationName": "Example Org",
                "contactInfo": {
                    "address": {"electronicMailAddress": "jane.smith@example.org"},
                    "phone": {"voice": "+41 1 234 56 78"},
                },
                "role": "http://codes.wmo.int/wmdr/ResponsiblePartyRole/owner",
            }
        },
        "observationSeries": [_observation_with_deployment()],
        "deployments": [],
    }
    (source / "record.json").write_text(json.dumps(payload), encoding="utf-8")

    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "convert_wmdr10_json_to_wmdr2_json:",
                f"  source: {source.as_posix()}",
                f"  target: {target.as_posix()}",
                "  recursive: true",
                "  catalogues:",
                "    enabled: true",
                f"    records_path: {catalogue_records.as_posix()}",
                f"    contacts_path: {(catalogues / 'contacts.json').as_posix()}",
                f"    instruments_path: {(catalogues / 'instruments.json').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    module.main(["--config", str(config)])
    module.main(["--config", str(config)])

    embedded = json.loads((target / "record.json").read_text(encoding="utf-8"))
    rewritten = json.loads((catalogue_records / "record.json").read_text(encoding="utf-8"))
    contacts = json.loads((catalogues / "contacts.json").read_text(encoding="utf-8"))["contacts"]
    instruments = json.loads((catalogues / "instruments.json").read_text(encoding="utf-8"))["instruments"]

    assert embedded["properties"]["instruments"]
    assert "instruments" not in rewritten["properties"]
    assert rewritten["properties"]["contacts"] == [
        {
            "identifier": "contact:jane.smith@example.org",
            "name": "Jane Smith",
            "organization": "Example Org",
            "roles": ["owner"],
            "links": [
                {
                    "rel": "about",
                    "href": "../catalogues/contacts.json#contact:jane.smith@example.org",
                    "type": "application/json",
                }
            ],
        }
    ]
    assert [contact["identifier"] for contact in contacts] == ["contact:jane.smith@example.org"]
    assert contacts[0]["phones"] == [{"value": "+41 1 234 56 78"}]
    assert len(instruments) == 1
    deployment_ref = rewritten["properties"]["observationSeries"][0]["observingConfigurations"][0]["deployment"]
    assert rewritten["properties"]["deployments"][0]["instrument"] == instruments[0]["id"]
    assert deployment_ref == rewritten["properties"]["deployments"][0]["id"]
    assert "deployments" not in rewritten["properties"]["observationSeries"][0]

def test_catalogues_source_config_key_is_obsolete(tmp_path: Path) -> None:
    source = tmp_path / "wmdr10"
    target = tmp_path / "wmdr2"
    source.mkdir()
    (source / "record.json").write_text(json.dumps({"facility": _minimal_facility()}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "convert_wmdr10_json_to_wmdr2_json:",
                f"  source: {source.as_posix()}",
                f"  target: {target.as_posix()}",
                "  catalogues:",
                "    enabled: true",
                "    source: old-derived-source",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="catalogues.source is obsolete"):
        module.main(["--config", str(config)])


def test_observation_uses_observed_property_and_domain_object() -> None:
    observation = module._normalize_observation(_observation_with_deployment(), index=1, facility_id="0-TEST")
    assert observation["observedProperty"] == OBSERVED_12006
    assert "observedVariable" not in observation
    assert "observedDomain" not in observation
    assert observation["observedFeature"] == {"domain": "atmosphere"}
    assert "domain" not in observation

def test_domain_object_accepts_future_domain_feature_fields() -> None:
    observation = module._normalize_observation(
        {
            "observedProperty": OBSERVED_12006,
            "observedDomain": {
                "domainFeature": "near-surface-air",
                "featureName": "2 m air",
            },
        },
        index=1,
        facility_id="0-TEST",
    )
    assert observation["observedFeature"] == {
        "domain": "atmosphere",
        "domainFeature": "near-surface-air",
        "featureName": "2 m air",
    }

def test_deployment_vertical_distance_is_quantity_from_height_above_local_reference_surface() -> None:
    observation = module._normalize_observation(
        {
            "observedProperty": OBSERVED_12006,
            "deployments": [
                {
                    "id": "dep1",
                    "heightAboveLocalReferenceSurface": {"@uom": "m", "#text": "0.0"},
                }
            ],
        },
        index=1,
        facility_id="0-TEST",
    )
    assert observation["verticalDistanceFromReferenceSurface"] == {"value": 0.0, "uom": "m"}

def test_deployment_official_status_defaults_to_unknown_when_absent_and_maps_booleans() -> None:
    missing = module._normalize_historical_official_status(
        {},
        fallback_date="2020-01-01",
    )
    assert missing is None

    primary = module._normalize_historical_official_status(
        {"id": "dep1", "beginPosition": "2020-01-01", "officialStatus": True},
        fallback_date="2020-01-01",
    )
    assert primary == [{"officialStatus": "primary", "date": "2020-01-01"}]

    additional = module._normalize_historical_official_status(
        {"id": "dep1", "beginPosition": "2020-01-01", "officialStatus": False},
        fallback_date="2020-01-01",
    )
    assert additional == [{"officialStatus": "additional", "date": "2020-01-01"}]

def test_deployment_temporal_geometry_uses_same_moving_point_model_as_facility() -> None:
    deployments = module._normalize_deployment(
        {
            "id": "dep1",
            "geospatialLocation": {
                "geoLocation": "46 7 100",
                "beginPosition": "2021-01-01",
                "geopositioningMethod": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
            },
        },
        index=1,
        facility_id="0-TEST",
        schedule_registry={},
    )
    assert deployments == [
        {
            "id": "deployment:dep1",
            "date": "2021-01-01",
            "geometry": {"type": "Point", "coordinates": [7.0, 46.0, 100]},
        }
    ]

def test_observation_accepts_legacy_observed_geometry_type_but_outputs_observed_geometry() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [
            {
                "observedProperty": OBSERVED_12006,
                "observedGeometryType": GEOMETRY_POINT,
            }
        ],
        [],
        {},
        source_name="20200101_0-TEST",
    )
    observation = record["properties"]["observationSeries"][0]
    assert observation["observedGeometry"] == "point"
    assert "observedGeometryType" not in observation
